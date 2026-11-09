"""Keep frontend dependencies out of the ROS Python and ORT processes."""
from __future__ import annotations

import atexit
import json
import os
from pathlib import Path
import selectors
import subprocess
import sys
import tempfile
import threading


class FrontendClient:
    def __init__(self, directory, python=None, timeout=60):
        self._lock = threading.RLock()
        self._timeout = timeout
        self._closed = False
        self._temporary = tempfile.TemporaryDirectory(prefix="xiaoyou04-frontend-")
        environment = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                           TOKENIZERS_PARALLELISM="false", PYTHONDONTWRITEBYTECODE="1",
                           OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="1",
                           TMPDIR=self._temporary.name, XDG_CACHE_HOME=self._temporary.name,
                           HF_HOME=self._temporary.name, CUDA_VISIBLE_DEVICES="")
        python = python or os.environ.get("XIAOYOU04_FRONTEND_PYTHON") or sys.executable
        command = [python, "-u", "-B", str(Path(__file__).with_name("frontend_worker.py")),
                   "--model-dir", str(directory)]
        try:
            self._process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                             stderr=None, text=True, encoding="utf-8", env=environment)
        except Exception:
            self._temporary.cleanup()
            raise
        try:
            if self._read().get("ready") is not True:
                raise RuntimeError("Xiaoyou 04 frontend did not become ready")
        except Exception:
            self.close()
            raise
        atexit.register(self.close)

    def _read(self):
        with selectors.DefaultSelector() as selector:
            selector.register(self._process.stdout, selectors.EVENT_READ)
            if not selector.select(self._timeout):
                self.close()
                raise TimeoutError("Xiaoyou 04 frontend timed out")
        line = self._process.stdout.readline()
        if not line:
            raise RuntimeError("Xiaoyou 04 frontend exited; inspect its startup log")
        response = json.loads(line)
        if response.get("error"):
            raise ValueError(response["error"])
        return response

    def encode(self, text, language):
        with self._lock:
            if self._closed or self._process.poll() is not None:
                raise RuntimeError("Xiaoyou 04 frontend is closed")
            self._process.stdin.write(json.dumps({"text": text, "language": language}, ensure_ascii=False) + "\n")
            self._process.stdin.flush()
            response = self._read()
            return response["phonemes"], response["ids"]

    def close(self):
        with self._lock:
            if self._closed:
                return
            self._closed = True
            if self._process.poll() is None:
                self._process.terminate()
                try:
                    self._process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    self._process.kill()
                    self._process.wait(timeout=3)
            self._process.stdin.close()
            self._process.stdout.close()
            self._temporary.cleanup()
