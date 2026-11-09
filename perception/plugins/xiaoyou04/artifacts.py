"""Fetch a pinned release with the shared downloader, then check its contents."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re

import numpy as np

SCHEMA = "xiaoyou04-release-v1"
REQUIRED = {"model.onnx", "voice.npy", "config.json", "frontend.py"}


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def read_spec(path=None):
    path = Path(path or os.environ.get("XIAOYOU04_RELEASE_SPEC")
                or Path(__file__).with_name("release.json"))
    spec = json.loads(path.read_text(encoding="utf-8"))
    return validate_spec(spec)


def validate_spec(spec):
    if spec.get("schema") != SCHEMA or spec.get("voice") != "04":
        raise ValueError("Not a Xiaoyou 04 release descriptor")
    if spec.get("status") not in ("candidate", "released"):
        raise RuntimeError("Xiaoyou 04 release is not bound yet: " + spec.get("reason", "export pending"))
    if spec.get("architecture") != "Kokoro82M_v1.0" or spec.get("sample_rate") != 24000:
        raise ValueError("Xiaoyou 04 requires Kokoro v1.0 at 24000 Hz")
    if spec.get("precision") != "FP32":
        raise ValueError("This runtime requires the verified FP32 release")
    files = spec.get("files", {})
    if not REQUIRED.issubset(files):
        raise ValueError("Release is missing its model, voice, config or frontend")
    for filename, pin in files.items():
        relative = Path(filename)
        if relative.is_absolute() or ".." in relative.parts or "\\" in filename:
            raise ValueError("Release file names must be relative paths")
        size = pin.get("size")
        if (not re.fullmatch(r"[0-9a-f]{64}", str(pin.get("sha256", "")))
                or not isinstance(size, int) or size < 0 or (filename in REQUIRED and size == 0)):
            raise ValueError("Every release file must have a size and SHA256")
    if not re.fullmatch(r"[0-9a-f]{64}", str(spec.get("source_checkpoint_sha256", ""))):
        raise ValueError("Release must identify its source checkpoint")
    return spec


def validate_directory(directory, spec):
    directory = Path(directory).resolve()
    for filename, pin in spec["files"].items():
        path = (directory / filename).resolve()
        try:
            path.relative_to(directory)
        except ValueError:
            raise ValueError("Release file outside release: " + filename)
        if not path.is_file():
            raise ValueError("Release file missing or outside release: " + filename)
        if path.stat().st_size != pin["size"] or sha256(path) != pin["sha256"]:
            raise ValueError("Release size/SHA256 mismatch: " + filename)
    voice = np.load(directory / "voice.npy", allow_pickle=False)
    if voice.dtype != np.float32 or voice.shape != (1, 256) or not np.isfinite(voice).all():
        raise ValueError("Xiaoyou 04 requires one finite FP32 voice vector of shape (1, 256)")
    config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    vocab = config.get("vocab", {})
    if not vocab or any(not isinstance(v, int) or v <= 0 for v in vocab.values()):
        raise ValueError("Release has an invalid phoneme vocabulary")
    if len(set(vocab.values())) != len(vocab):
        raise ValueError("Release contains duplicate phoneme token IDs")
    return voice, vocab


def ensure_release(model_dir, on_status=None, spec_path=None):
    from utils.model_downloader import ensure_verified_archive

    spec = read_spec(spec_path)
    # A version gets its own directory; an update cannot mix with an old cache.
    archive = spec.get("archive", {})
    pin = archive.get("sha256", "")
    if not re.fullmatch(r"[0-9a-f]{64}", pin) or int(archive.get("size", 0)) <= 0:
        raise ValueError("Release archive must have a size and SHA256")
    url = archive.get("url", "")
    if not url.startswith(("http://", "https://")):
        raise ValueError("Release archive needs an HTTP(S) URL")
    directory = Path(model_dir).resolve() / pin[:16]
    ensure_verified_archive("xiaoyou04", str(directory), url, archive, progress_cb=on_status)
    voice, vocab = validate_directory(directory, spec)
    return directory, spec, voice, vocab
