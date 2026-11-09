#!/usr/bin/env python3
"""Measure independent runtime instances on a test device; save PCM as WAV."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import multiprocessing as mp
from pathlib import Path
import platform
import queue
import resource
import sys
import time
import wave

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def run_instance(number, cases, output, model_dir, device, barrier, result_queue):
    from plugins.xiaoyou04.runtime import Xiaoyou04Runtime

    runtime = None
    record = {"instance": number, "cases": []}
    try:
        started = time.monotonic()
        runtime = Xiaoyou04Runtime({"model_dir": model_dir, "device": device, "speed": 1})
        record.update(load_seconds=time.monotonic() - started, providers=runtime.providers,
                      release_sha256=runtime.release["archive"]["sha256"],
                      checkpoint_sha256=runtime.release["source_checkpoint_sha256"])
        runtime.warmup()
        barrier.wait(timeout=180)
        for index, case in enumerate(cases):
            started = time.monotonic()
            first = None
            chunks = []
            try:
                for chunk in runtime.synthesize_stream(case["text"]):
                    if chunk and first is None:
                        first = time.monotonic() - started
                    chunks.append(chunk)
                elapsed = time.monotonic() - started
                pcm = b"".join(chunks)
                if not pcm or len(pcm) % 2:
                    raise RuntimeError("No complete PCM16 samples")
                seconds = len(pcm) / (16000 * 2)
                path = output / f"instance-{number:02d}-case-{index:04d}.wav"
                with wave.open(str(path), "wb") as handle:
                    handle.setnchannels(1)
                    handle.setsampwidth(2)
                    handle.setframerate(16000)
                    handle.writeframes(pcm)
                record["cases"].append(dict(id=case.get("id", index), text=case["text"], ok=True,
                                             first_audio_seconds=first, elapsed_seconds=elapsed,
                                             audio_seconds=seconds, rtf=elapsed / seconds,
                                             chunks=len(chunks), wav=path.name))
            except Exception as exc:
                record["cases"].append(dict(id=case.get("id", index), ok=False, error=str(exc)))
        record["parent_max_rss_mib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    except Exception as exc:
        record["error"] = str(exc)
        barrier.abort()
    finally:
        if runtime is not None:
            runtime.close()
        result_queue.put(record)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True, help="JSONL records with id and text")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-dir", default="/models/xiaoyou04")
    parser.add_argument("--device", choices=["gpu", "cpu"], default="gpu")
    parser.add_argument("--instances", type=int, choices=[1, 2, 3], default=1)
    args = parser.parse_args()
    cases = [json.loads(line) for line in args.cases.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not cases or any(not isinstance(x.get("text"), str) or not x["text"].strip() for x in cases):
        raise ValueError("Each benchmark case needs nonempty text")
    args.output.mkdir(parents=True, exist_ok=False)
    context = mp.get_context("spawn")
    barrier, results = context.Barrier(args.instances), context.Queue()
    processes = [context.Process(target=run_instance,
                    args=(i, cases, args.output.resolve(), args.model_dir, args.device, barrier, results))
                 for i in range(args.instances)]
    for process in processes:
        process.start()
    records = []
    try:
        for _ in processes:
            try:
                records.append(results.get(timeout=900))
            except queue.Empty:
                break
    finally:
        for process in processes:
            process.join(timeout=5)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
    passed = (len(records) == args.instances and all(not r.get("error") for r in records)
              and all(len(r["cases"]) == len(cases) and all(x["ok"] for x in r["cases"]) for r in records))
    report = dict(created_utc=datetime.now(timezone.utc).isoformat(), host=platform.node(),
                  architecture=platform.machine(), requested_instances=args.instances,
                  completed_instances=sum(not r.get("error") and len(r["cases"]) == len(cases)
                                          and all(x["ok"] for x in r["cases"]) for r in records),
                  runtime_passed=passed, instances=sorted(records, key=lambda r: r["instance"]),
                  runtime_limits_passed=passed and all(x["first_audio_seconds"] <= 1 and x["rtf"] <= 1
                                      for r in records for x in r["cases"]),
                  scope="Runtime first PCM and unpaced synthesis RTF after warmup. Excludes ROS, MCP, playback and perceptual acceptance. Parent RSS excludes frontend/ORT children; capture device/cgroup memory separately.")
    (args.output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "instances"}, ensure_ascii=False, indent=2))
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
