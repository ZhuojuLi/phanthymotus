"""Token-driven ONNX inference with the released 04 voice and frontend."""
from __future__ import annotations

import math
import threading
import uuid

import numpy as np

from plugins.kokoro_direct import chunk_ids
from plugins.xiaoyou04.artifacts import ensure_release
from plugins.xiaoyou04.frontend_client import FrontendClient
from utils.resample import downsample_24k_to_16k


class Xiaoyou04Runtime:
    sample_rate = 16000
    dry_run_text = "你好。"

    def __init__(self, cfg, on_status=None):
        self._lock = threading.RLock()
        self._frontend = None
        self._session = None
        self._session_key = None
        self._worker = None
        self.set_speed(cfg.get("speed", 1.0))
        if int(cfg.get("speaker_id", 0)) != 0:
            raise ValueError("Xiaoyou 04 has one fixed voice; speaker_id must be 0")
        self.device = str(cfg.get("device") or "gpu").lower()
        if self.device not in ("cpu", "gpu"):
            raise ValueError("Xiaoyou 04 device must be cpu or gpu")
        self.directory, self.release, self._voice, self._vocab = ensure_release(
            cfg.get("model_dir") or "/models/xiaoyou04", on_status=on_status)
        self._break_ids = {self._vocab[p] for p in ".,;:!?" if p in self._vocab}
        # A finite clause is generated at a time. First-audio latency and seams
        # must be measured on each Jetson; this is not frame-level generation.
        self._chunk_tokens = int(self.release.get("chunk_tokens", 120))
        if not 1 <= self._chunk_tokens <= 510:
            raise ValueError("chunk_tokens must be between 1 and 510")
        try:
            self._frontend = FrontendClient(self.directory)
            from plugins import ort_worker
            self._worker = ort_worker.get_worker()
            providers = (["CUDAExecutionProvider", "CPUExecutionProvider"]
                         if self.device == "gpu" else ["CPUExecutionProvider"])
            options = {"intra_op_num_threads": 2, "inter_op_num_threads": 1,
                       "provider_options": [
                           {"cudnn_conv_algo_search": "HEURISTIC",
                            "arena_extend_strategy": "kSameAsRequested"}
                           if p == "CUDAExecutionProvider" else {} for p in providers]}
            key = "tts.xiaoyou04." + uuid.uuid4().hex
            self._session_key = key
            self._session = self._worker.load(key, str(self.directory / "model.onnx"), providers, options)
            if self.device == "gpu" and "CUDAExecutionProvider" not in self.providers:
                raise RuntimeError("Xiaoyou 04 requested GPU but CUDAExecutionProvider was not loaded")
            inputs = {item.name for item in self._session.get_inputs()}
            if inputs != {"tokens", "style", "speed"}:
                raise ValueError("Xiaoyou 04 ONNX inputs must be tokens, style and speed")
            if "audio" not in {item.name for item in self._session.get_outputs()}:
                raise ValueError("Xiaoyou 04 ONNX must expose its audio output")
        except Exception:
            self.close()
            raise

    @property
    def providers(self):
        return list(self._session.get_providers()) if self._session is not None else []

    def set_speed(self, value):
        value = float(value)
        if not math.isfinite(value) or not .5 <= value <= 2:
            raise ValueError("Xiaoyou 04 speed must be finite and between 0.5 and 2")
        self._speed = value

    def set_language(self, language):
        if language not in ("auto", "zh", "en", "mixed"):
            raise ValueError("Xiaoyou 04 supports Chinese, English and mixed text")

    def warmup(self):
        return len(self.synthesize(self.dry_run_text))

    def synthesize(self, text):
        return b"".join(self.synthesize_stream(text))

    def synthesize_stream(self, text):
        if not isinstance(text, str) or not text.strip():
            raise ValueError("text is required")
        has_zh = any("\u4e00" <= char <= "\u9fff" for char in text)
        has_en = any(char.isascii() and char.isalpha() for char in text)
        language = "mixed" if has_zh and has_en else "zh" if has_zh else "en"
        # One utterance keeps the same voice, frontend and speed, even when
        # callers change the next utterance's speed while this one streams.
        with self._lock:
            if self._session is None:
                raise RuntimeError("Xiaoyou 04 runtime is closed")
            phones, ids = self._frontend.encode(text, language)
            if not ids or len(phones) != len(ids):
                raise ValueError("Xiaoyou 04 frontend returned an empty or inconsistent sequence")
            if any(self._vocab.get(p) != token for p, token in zip(phones, ids)):
                raise ValueError("Xiaoyou 04 frontend and model vocabularies differ")
            speed = self._speed
            for group in chunk_ids(ids, self._chunk_tokens, self._break_ids):
                feed = {"tokens": np.asarray([[0, *group, 0]], dtype=np.int64),
                        "style": self._voice,
                        "speed": np.asarray([speed], dtype=np.float32)}
                wave = np.asarray(self._session.run(["audio"], feed)[0], dtype=np.float32).reshape(-1)
                if not wave.size or not np.isfinite(wave).all():
                    raise RuntimeError("Xiaoyou 04 generated an invalid waveform")
                yield downsample_24k_to_16k(wave)

    def close(self):
        with self._lock:
            self._session = None
            if self._session_key is not None and self._worker is not None:
                self._worker.unload(self._session_key)
                self._session_key = None
            if self._frontend is not None:
                self._frontend.close()
                self._frontend = None
