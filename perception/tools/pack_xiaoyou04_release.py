#!/usr/bin/env python3
"""Pack a verified export, or an explicitly identified diagnostic trial."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tarfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from plugins.xiaoyou04.artifacts import SCHEMA, sha256, validate_directory, validate_spec


def pack(source, archive, descriptor, url, diagnostic_trial=False):
    import onnx

    source, archive, descriptor = map(lambda p: Path(p).resolve(), (source, archive, descriptor))
    if source in archive.parents or source in descriptor.parents:
        raise ValueError("Write release outputs outside the source directory")
    identity_path = source / "export_identity.json"
    identity = json.loads(identity_path.read_text(encoding="utf-8"))
    if identity.get("voice") != "04" or identity.get("architecture") != "Kokoro82M_v1.0":
        raise ValueError("Export identity must identify the trained Xiaoyou 04 model")
    if identity.get("precision") != "FP32" or identity.get("sample_rate") != 24000:
        raise ValueError("Only the FP32, 24000 Hz export is supported")
    if identity.get("normal_text_parity_passed") is not True and not diagnostic_trial:
        raise ValueError("Normal-text native/ONNX parity must pass before packing")
    if diagnostic_trial:
        if (identity.get("diagnostic_trial") is not True
                or identity.get("normal_text_parity_passed") is not False
                or identity.get("listening_approved") is not False):
            raise ValueError("Diagnostic trials must explicitly preserve pending parity and listening acceptance")
        report_path = source / "validation_report.json"
        if sha256(report_path) != identity.get("validation_report_sha256"):
            raise ValueError("Diagnostic trial requires a pinned validation report")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if (report.get("model_sha256") != identity.get("export_sha256", {}).get("model.onnx")
                or report.get("source_checkpoint_sha256") != identity.get("source_checkpoint_sha256")
                or report.get("full_waveform_parity_passed") is not False
                or not report.get("known_limitations")):
            raise ValueError("Diagnostic validation report must identify the tested model and unresolved failures")
    # The export check signs the actual graph, frontend, config and fixed voice.
    for name in ["model.onnx", "frontend.py", "config.json", "voice.npy"]:
        if sha256(source / name) != identity.get("export_sha256", {}).get(name):
            raise ValueError("Export bytes differ from parity-tested bytes: " + name)
    model = onnx.load(str(source / "model.onnx"), load_external_data=False)
    if model.ir_version > 8 or any(o.version > 17 for o in model.opset_import if o.domain in ("", "ai.onnx")):
        raise ValueError("Export must use ONNX IR <= 8 and opset <= 17 for the JP5 runtime")
    initializer_names = {x.name for x in model.graph.initializer}
    inputs = {x.name for x in model.graph.input if x.name not in initializer_names}
    if inputs != {"tokens", "style", "speed"} or "audio" not in {x.name for x in model.graph.output}:
        raise ValueError("ONNX contract must be tokens/style/speed -> audio")
    expected_types = {"tokens": onnx.TensorProto.INT64,
                      "style": onnx.TensorProto.FLOAT, "speed": onnx.TensorProto.FLOAT}
    for item in model.graph.input:
        if item.name in expected_types and item.type.tensor_type.elem_type != expected_types[item.name]:
            raise ValueError("ONNX input dtype does not match its contract: " + item.name)
    if next(x for x in model.graph.output if x.name == "audio").type.tensor_type.elem_type != onnx.TensorProto.FLOAT:
        raise ValueError("ONNX audio output must be FP32")
    low_precision = {onnx.TensorProto.FLOAT16, onnx.TensorProto.BFLOAT16, onnx.TensorProto.DOUBLE}
    if any(x.data_type in low_precision for x in model.graph.initializer):
        raise ValueError("ONNX floating weights must remain FP32")
    if any(x.op_type in {"QuantizeLinear", "DequantizeLinear", "DynamicQuantizeLinear", "MatMulInteger"}
           for x in model.graph.node):
        raise ValueError("Quantized graphs require a separate verified release format")
    onnx.checker.check_model(str(source / "model.onnx"))
    files = {}
    for path in sorted(source.rglob("*")):
        if path.is_symlink():
            raise ValueError("Release must contain regular files, not symlinks")
        if path.is_file():
            if path.suffix in (".pth", ".pt", ".ckpt", ".pyc") or "__pycache__" in path.parts:
                raise ValueError("Training weights or Python caches must not enter the ONNX release")
            files[str(path.relative_to(source))] = {"size": path.stat().st_size, "sha256": sha256(path)}
    spec = dict(schema=SCHEMA, status="candidate", voice="04", architecture="Kokoro82M_v1.0",
                sample_rate=24000, precision="FP32", chunk_tokens=120,
                source_checkpoint_sha256=identity["source_checkpoint_sha256"], files=files,
                listening_approved=identity.get("listening_approved", False),
                normal_text_parity_passed=identity.get("normal_text_parity_passed") is True)
    if diagnostic_trial:
        spec.update(diagnostic_trial=True, validation_report_sha256=identity["validation_report_sha256"])
    validate_spec(spec)
    validate_directory(source, spec)
    archive.parent.mkdir(parents=True, exist_ok=True)
    if archive.exists() or descriptor.exists():
        raise FileExistsError("Use new versioned archive and descriptor paths")
    temporary = archive.with_suffix(archive.suffix + ".part")
    with tarfile.open(temporary, "w:gz", dereference=True) as handle:
        for name in files:
            handle.add(source / name, arcname=name, recursive=False)
    temporary.replace(archive)
    spec["archive"] = {"url": url, "size": archive.stat().st_size, "sha256": sha256(archive)}
    descriptor.parent.mkdir(parents=True, exist_ok=True)
    descriptor.write_text(json.dumps(spec, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return spec


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--descriptor", type=Path, required=True)
    parser.add_argument("--url", required=True, help="Versioned JuiceFS HTTP URL; upload and round-trip verification are separate steps")
    parser.add_argument("--diagnostic-trial", action="store_true",
                        help="Explicit experimental submission with pinned failures; runtime requires XIAOYOU04_ALLOW_DIAGNOSTIC=1")
    args = parser.parse_args()
    spec = pack(args.source, args.archive, args.descriptor, args.url, diagnostic_trial=args.diagnostic_trial)
    print(json.dumps({"archive": spec["archive"], "files": len(spec["files"]),
                      "status": "candidate", "uploaded": False}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
