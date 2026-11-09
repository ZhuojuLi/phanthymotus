"""Off-device release/PCM/lifecycle checks; these do not certify a Jetson model."""
from __future__ import annotations

import io
import functools
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import sys
import threading
from types import SimpleNamespace
import wave

import numpy as np
import pytest

from plugins.xiaoyou04 import artifacts
from plugins.xiaoyou04.frontend_client import FrontendClient
from plugins.xiaoyou04.runtime import Xiaoyou04Runtime


def _fixture_frontend(directory):
    (directory / "frontend.py").write_text('''
import json
from pathlib import Path
class Frontend:
    def __init__(self):
        print("fixture initialization diagnostic")
        self.vocab = json.loads((Path(__file__).parent / "config.json").read_text())["vocab"]
    def __call__(self, text, language):
        print("fixture request diagnostic")
        if text == "unknown":
            raise ValueError("Unknown phoneme symbols")
        phones = "a" * len(text)
        return phones, [self.vocab[p] for p in phones]
''', encoding="utf-8")
    (directory / "config.json").write_text(json.dumps({"vocab": {"a": 1, ".": 2}}))


@pytest.fixture
def release_directory(tmp_path):
    _fixture_frontend(tmp_path)
    np.save(tmp_path / "voice.npy", np.ones((1, 256), dtype=np.float32))
    (tmp_path / "model.onnx").write_bytes(b"fixture graph; not a TTS model")
    files = {p.name: {"size": p.stat().st_size, "sha256": artifacts.sha256(p)}
             for p in tmp_path.iterdir()}
    spec = {"schema": artifacts.SCHEMA, "status": "candidate", "voice": "04",
            "architecture": "Kokoro82M_v1.0", "sample_rate": 24000, "precision": "FP32",
            "source_checkpoint_sha256": "a" * 64, "files": files, "chunk_tokens": 3}
    return tmp_path, spec


def test_unbound_release_is_an_explicit_error():
    with pytest.raises(RuntimeError, match="not bound yet"):
        artifacts.read_spec()


def test_release_checks_voice_vocabulary_and_every_pinned_file(release_directory):
    directory, spec = release_directory
    voice, vocab = artifacts.validate_directory(directory, spec)
    assert voice.shape == (1, 256) and vocab["a"] == 1
    (directory / "model.onnx").write_bytes(b"changed graph")
    with pytest.raises(ValueError, match="mismatch"):
        artifacts.validate_directory(directory, spec)


def test_wrong_voice_is_rejected_even_with_a_matching_hash(release_directory):
    directory, spec = release_directory
    np.save(directory / "voice.npy", np.ones((2, 256), dtype=np.float32))
    spec["files"]["voice.npy"] = {"size": (directory / "voice.npy").stat().st_size,
                                    "sha256": artifacts.sha256(directory / "voice.npy")}
    with pytest.raises(ValueError, match="one finite FP32 voice"):
        artifacts.validate_directory(directory, spec)


def test_empty_auxiliary_package_files_are_valid_but_empty_weights_are_not(release_directory):
    directory, spec = release_directory
    (directory / "__init__.py").write_bytes(b"")
    spec["files"]["__init__.py"] = {"size": 0, "sha256": artifacts.sha256(directory / "__init__.py")}
    artifacts.validate_spec(spec)
    artifacts.validate_directory(directory, spec)
    spec["files"]["model.onnx"]["size"] = 0
    with pytest.raises(ValueError, match="size and SHA256"):
        artifacts.validate_spec(spec)


def test_release_must_identify_its_exact_source_checkpoint(release_directory):
    _, spec = release_directory
    spec["source_checkpoint_sha256"] = "not-a-real-hash"
    with pytest.raises(ValueError, match="source checkpoint"):
        artifacts.validate_spec(spec)


def test_frontend_protocol_survives_diagnostics_and_bad_inputs(release_directory):
    directory, _ = release_directory
    worker = FrontendClient(directory, python=sys.executable, timeout=5)
    temporary = Path(worker._temporary.name)
    try:
        assert worker.encode("abc", "en") == ("aaa", [1, 1, 1])
        with pytest.raises(ValueError, match="Unknown phoneme"):
            worker.encode("unknown", "en")
        assert worker.encode("后续", "zh") == ("aa", [1, 1])
    finally:
        worker.close()
        worker.close()
    assert worker._process.poll() is not None
    assert not temporary.exists()


def test_preparing_release_does_not_install_a_frontend_environment(monkeypatch):
    import importlib.util
    script = Path(__file__).parents[1] / "deploy/setup_xiaoyou04_frontend.py"
    spec = importlib.util.spec_from_file_location("setup_xiaoyou04_frontend", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    root = Path(__file__).parents[1] / "plugins/xiaoyou04"
    monkeypatch.setattr(sys, "argv", [str(script), "--release-spec", str(root / "release.json"),
                                      "--requirements", str(root / "requirements.frontend.txt")])
    def unexpected_install(*args, **kwargs):
        raise AssertionError("An unbound release must not install dependencies")
    monkeypatch.setattr(module.subprocess, "run", unexpected_install)
    module.main()


class _Session:
    def __init__(self, providers):
        self._providers = providers
        self.feeds = []

    def get_providers(self):
        return self._providers

    def get_inputs(self):
        return [SimpleNamespace(name=n) for n in ("tokens", "style", "speed")]

    def get_outputs(self):
        return [SimpleNamespace(name="audio")]

    def run(self, names, feed):
        self.feeds.append(feed)
        return [np.sin(np.arange(2400, dtype=np.float32) * .05) * .2]


class _Worker:
    def __init__(self, providers):
        self.session = _Session(providers)
        self.unloaded = []

    def load(self, key, model, providers, options):
        self.key = key
        return self.session

    def unload(self, key):
        self.unloaded.append(key)


def _runtime(monkeypatch, release_directory, providers=None, device="cpu"):
    directory, spec = release_directory
    voice, vocab = artifacts.validate_directory(directory, spec)
    import plugins.xiaoyou04.runtime as module
    from plugins import ort_worker
    monkeypatch.setattr(module, "ensure_release", lambda *a, **k: (directory, spec, voice, vocab))
    worker = _Worker(providers or ["CPUExecutionProvider"])
    monkeypatch.setattr(ort_worker, "get_worker", lambda: worker)
    monkeypatch.setenv("XIAOYOU04_FRONTEND_PYTHON", sys.executable)
    runtime = Xiaoyou04Runtime({"device": device})
    return runtime, worker


def test_stream_preserves_every_token_and_writes_one_valid_pcm_wav(monkeypatch, release_directory):
    runtime, worker = _runtime(monkeypatch, release_directory)
    try:
        chunks = list(runtime.synthesize_stream("abcdefg"))
        assert len(chunks) == 3
        assert [feed["tokens"].tolist() for feed in worker.session.feeds] == [
            [[0, 1, 1, 1, 0]], [[0, 1, 1, 1, 0]], [[0, 1, 0]]]
        assert all(feed["style"].shape == (1, 256) for feed in worker.session.feeds)
        pcm = b"".join(chunks)
        assert len(pcm) == 3 * 3200
        output = io.BytesIO()
        with wave.open(output, "wb") as wav:
            wav.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
            wav.writeframes(pcm)
        with wave.open(io.BytesIO(output.getvalue()), "rb") as wav:
            assert wav.getnframes() == 4800
            assert wav.getframerate() == 16000 and wav.getnchannels() == 1
    finally:
        runtime.close()
    assert worker.unloaded == [worker.key]


def test_requested_gpu_cannot_silently_become_cpu(monkeypatch, release_directory):
    with pytest.raises(RuntimeError, match="CUDAExecutionProvider"):
        _runtime(monkeypatch, release_directory, device="gpu")


def test_speed_changes_do_not_reload_a_resident_model(monkeypatch, release_directory):
    runtime, worker = _runtime(monkeypatch, release_directory)
    try:
        runtime.set_speed(1.5)
        runtime.synthesize("a")
        assert worker.session.feeds[-1]["speed"].tolist() == [1.5]
        for invalid in (0, 2.5, float("nan"), float("inf")):
            with pytest.raises(ValueError):
                runtime.set_speed(invalid)
    finally:
        runtime.close()


def test_selected_plugin_keeps_its_factory_during_config_and_cleans_up(monkeypatch):
    from plugins import tts_xiaoyou04 as module
    from vision_stubs import _FakeExecutor

    instances = []
    class Adapter:
        dry_run_text = "你好。"
        providers = ["CUDAExecutionProvider"]
        release = {"source_checkpoint_sha256": "b" * 64}
        def __init__(self):
            self.closed = False
            self.speed = 1
        def warmup(self):
            return 3200
        def set_speed(self, speed):
            self.speed = speed
        def close(self):
            self.closed = True
    def factory(cfg, on_status=None):
        adapter = Adapter()
        instances.append(adapter)
        return adapter
    monkeypatch.setattr(module, "_adapter", factory)
    plugin = module.TTSPlugin({"model_dir": "/models/vits2", "warmup": False}, _FakeExecutor())
    assert plugin._cfg["model_dir"] == "/models/xiaoyou04"
    assert list(plugin.get_tools()[0]["configSchema"]["properties"]) == ["speed"]
    assert plugin.dispatch("tts", {"action": "support_language"})["languages"] == ["zh", "en", "mixed"]
    assert plugin.dispatch("tts", {"action": "info"})["voice"] == "04"
    plugin.dispatch("tts", {"action": "config", "speed": 1.25})
    assert len(instances) == 1 and instances[0].speed == 1.25
    with pytest.raises(ValueError):
        plugin.dispatch("tts", {"action": "config", "tts_engine": "vits2-zh-en"})
    plugin.shutdown()
    assert instances[0].closed


def test_release_profile_enables_only_tts():
    import yaml
    profile = yaml.safe_load((Path(__file__).parents[1] / "plugins/xiaoyou04/config.yaml").read_text())
    enabled = {k for k, v in profile["plugins"].items() if v.get("enabled")}
    assert enabled == {"tts"}


def test_real_onnx_worker_with_cold_download_and_warm_cache(monkeypatch, tmp_path):
    """A tiny synthetic graph proves transport/packaging, not voice or device quality."""
    onnx = pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    from onnx import TensorProto, helper, numpy_helper
    from tools.pack_xiaoyou04_release import pack
    from plugins import ort_worker

    source = tmp_path / "source"
    source.mkdir()
    _fixture_frontend(source)
    np.save(source / "voice.npy", np.ones((1, 256), dtype=np.float32))
    samples = np.sin(np.arange(2400, dtype=np.float32) * .05) * .2
    constant = helper.make_node("Constant", [], ["audio"], value=numpy_helper.from_array(samples))
    graph = helper.make_graph([constant], "synthetic_transport_fixture", [
        helper.make_tensor_value_info("tokens", TensorProto.INT64, [1, "length"]),
        helper.make_tensor_value_info("style", TensorProto.FLOAT, [1, 256]),
        helper.make_tensor_value_info("speed", TensorProto.FLOAT, [1]),
    ], [helper.make_tensor_value_info("audio", TensorProto.FLOAT, [2400])])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.save(model, source / "model.onnx")
    identity = {"voice": "04", "architecture": "Kokoro82M_v1.0", "precision": "FP32",
                "sample_rate": 24000, "normal_text_parity_passed": True,
                "source_checkpoint_sha256": "a" * 64,
                "export_sha256": {name: artifacts.sha256(source / name)
                                   for name in ["model.onnx", "frontend.py", "config.json", "voice.npy"]}}
    (source / "export_identity.json").write_text(json.dumps(identity))
    class Handler(SimpleHTTPRequestHandler):
        def log_message(self, *args):
            pass
    server = ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(Handler, directory=str(tmp_path)))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    descriptor = tmp_path / "release.json"
    archive = tmp_path / "fixture.tar.gz"
    url = f"http://127.0.0.1:{server.server_address[1]}/fixture.tar.gz"
    pack(source, archive, descriptor, url)
    monkeypatch.setenv("XIAOYOU04_RELEASE_SPEC", str(descriptor))
    monkeypatch.setenv("XIAOYOU04_FRONTEND_PYTHON", sys.executable)
    runtime = None
    try:
        runtime = Xiaoyou04Runtime({"model_dir": str(tmp_path / "cache"), "device": "cpu"})
        assert runtime.providers == ["CPUExecutionProvider"]
        assert len(runtime.synthesize("abc")) == 3200
        cached = runtime.directory
        runtime.close()
        runtime = None
        # A stopped HTTP server proves this request used the pinned warm cache.
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)
        runtime = Xiaoyou04Runtime({"model_dir": str(tmp_path / "cache"), "device": "cpu"})
        assert runtime.directory == cached and len(runtime.synthesize("after restart")) == 3200
        runtime.close()
        runtime = None
        (cached / "model.onnx").write_bytes(b"corrupt cache")
        with pytest.raises(ValueError, match="mismatch"):
            Xiaoyou04Runtime({"model_dir": str(tmp_path / "cache"), "device": "cpu"})
    finally:
        if runtime is not None:
            runtime.close()
        if thread.is_alive():
            server.shutdown()
            server.server_close()
            thread.join(timeout=3)
        ort_worker.get_worker().close()
