#!/usr/bin/env python3
"""Install the pinned frontend only when this code binds a candidate release."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

PYTHON_VERSION = "3.11.15"
UV_VERSION = "0.12.3"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-spec", type=Path, required=True)
    parser.add_argument("--requirements", type=Path, required=True)
    parser.add_argument("--prefix", type=Path, default=Path("/opt/xiaoyou04-frontend"))
    parser.add_argument("--python-root", type=Path, default=Path("/opt/xiaoyou04-python"))
    parser.add_argument("--bootstrap", type=Path, default=Path("/opt/xiaoyou04-bootstrap"))
    args = parser.parse_args()
    spec = json.loads(args.release_spec.read_text(encoding="utf-8"))
    if spec.get("status") == "preparing":
        print("[build] Xiaoyou 04 release is not bound; frontend installation deferred")
        return
    if spec.get("status") not in ("candidate", "released"):
        raise ValueError("Unexpected Xiaoyou 04 release status")
    mirror = os.environ.get("PYPI_MIRROR", "https://pypi.org/simple")
    subprocess.run([sys.executable, "-m", "pip", "install", "--no-cache-dir", "--prefix",
                    str(args.bootstrap), "--index-url", mirror, "uv==" + UV_VERSION], check=True)
    uv = str(args.bootstrap / "bin/uv")
    environment = dict(os.environ, UV_PYTHON_INSTALL_DIR=str(args.python_root),
                       UV_NATIVE_TLS="true", UV_NO_PROGRESS="1", UV_HTTP_TIMEOUT="120")
    subprocess.run([uv, "python", "install", PYTHON_VERSION], check=True, env=environment)
    subprocess.run([uv, "venv", "--managed-python", "--python", PYTHON_VERSION, str(args.prefix)],
                   check=True, env=environment)
    python = str(args.prefix / "bin/python")
    subprocess.run([uv, "pip", "install", "--python", python, "--index-url", mirror,
                    "-r", str(args.requirements)], check=True, env=environment)
    probe = ("import sys,importlib.metadata as m;from misaki import en,zh;import spacy;"
             "assert sys.version_info[:3]==(3,11,15);"
             "assert m.version('misaki')=='0.9.4';assert m.version('spacy')=='3.8.16';"
             "print('[build] Xiaoyou 04 frontend dependencies ready; release model data loads at runtime')")
    subprocess.run([python, "-c", probe], check=True, env=environment)


if __name__ == "__main__":
    main()
