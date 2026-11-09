#!/usr/bin/env python3
"""Persistent JSON-lines frontend, runnable in a separate Python environment."""
import argparse
import contextlib
import importlib.util
import json
from pathlib import Path
import sys
import traceback


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    args = parser.parse_args()
    directory = args.model_dir.resolve()
    sys.path[:0] = [str(directory / "vendor"), str(directory)]
    # Library diagnostics must never become JSON protocol messages.
    with contextlib.redirect_stdout(sys.stderr):
        spec = importlib.util.spec_from_file_location("xiaoyou04_release_frontend", directory / "frontend.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        frontend = module.Frontend()
    print(json.dumps({"ready": True}), flush=True)
    for line in sys.stdin:
        try:
            request = json.loads(line)
            text, language = request["text"], request["language"]
            if not isinstance(text, str) or language not in ("zh", "en", "mixed"):
                raise ValueError("Invalid frontend request")
            with contextlib.redirect_stdout(sys.stderr):
                phones, ids = frontend(text, language)
            response = {"phonemes": phones, "ids": list(ids)}
        except ValueError as exc:
            print("[xiaoyou04 frontend] " + str(exc), file=sys.stderr)
            response = {"error": str(exc)}
        except Exception as exc:
            traceback.print_exc(file=sys.stderr)
            response = {"error": str(exc)}
        print(json.dumps(response, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
