"""Reviewed personal learning loop.

    python research/learning_loop.py list
    python research/learning_loop.py approve f0001
    python research/learning_loop.py export
    python research/learning_loop.py run --base checkpoints/transformer_behavior_50m_long

Only approved feedback is exported. ``run`` invokes the existing gated LoRA
pipeline; a rejected candidate never becomes active.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from streamft.feedback import FeedbackError, FeedbackStore

ROOT = Path(__file__).resolve().parents[1]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", default="data/feedback_queue.json")
    ap.add_argument("--out-queue", default="data/feedback_training.jsonl")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    approve = sub.add_parser("approve")
    approve.add_argument("id")
    reject = sub.add_parser("reject")
    reject.add_argument("id")
    sub.add_parser("export")
    run = sub.add_parser("run")
    run.add_argument(
        "--base", default="checkpoints/transformer_behavior_50m_long")
    run.add_argument("--out", default="checkpoints/adapters_feedback")
    run.add_argument("--steps", type=int, default=100)
    run.add_argument("--lora-lr", type=float, default=1e-4)
    run.add_argument("--rank", type=int, default=8)
    run.add_argument("--alpha", type=int, default=16)
    run.add_argument("--result",
                     default="research/results/gated_feedback.json")
    run.add_argument("--active-file", default="data/active_adapter.json")
    run.add_argument("--min-examples", type=int, default=5)
    args = ap.parse_args(argv)

    try:
        store = FeedbackStore(ROOT / args.store)
        if args.cmd == "list":
            print(json.dumps({"data": store.list(), "stats": store.stats()},
                             indent=2, ensure_ascii=False))
        elif args.cmd == "approve":
            print(json.dumps(store.approve(args.id), ensure_ascii=False))
        elif args.cmd == "reject":
            print(json.dumps(store.reject(args.id), ensure_ascii=False))
        elif args.cmd == "export":
            result = store.export_training(ROOT / args.out_queue)
            print(json.dumps(result, ensure_ascii=False))
        else:
            exported = store.export_training(ROOT / args.out_queue)
            if exported["examples"] < args.min_examples:
                print(json.dumps({
                    "error": (f"need at least {args.min_examples} approved "
                              "examples"),
                    **exported,
                }, ensure_ascii=False))
                return 2
            result_path = ROOT / args.result
            cmd = [
                sys.executable, "research/gated_update.py",
                "--base", args.base,
                "--queue", args.out_queue,
                "--out", args.out,
                "--steps", str(args.steps),
                "--lora",
                "--lora-lr", str(args.lora_lr),
                "--rank", str(args.rank),
                "--alpha", str(args.alpha),
                "--result", args.result,
            ]
            print(json.dumps({"event": "train", "cmd": cmd}, ensure_ascii=False),
                  flush=True)
            completed = subprocess.run(cmd, cwd=ROOT)
            if completed.returncode != 0:
                print(json.dumps({"error": "gated update failed",
                                  "returncode": completed.returncode}))
                return completed.returncode
            payload = json.loads(result_path.read_text(encoding="utf-8"))
            decision = payload.get("decision", {})
            accepted_to = payload.get("accepted_to", "")
            if decision.get("accepted") and accepted_to:
                active = ROOT / args.active_file
                active.parent.mkdir(parents=True, exist_ok=True)
                active.write_text(json.dumps({
                    "path": accepted_to,
                    "updated": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "examples": exported["examples"],
                    "result": args.result,
                }, indent=2) + "\n", encoding="utf-8")
                print(json.dumps({
                    "accepted": True,
                    "active_adapter": accepted_to,
                    "active_file": args.active_file,
                    "decision": decision,
                }, ensure_ascii=False))
            else:
                print(json.dumps({
                    "accepted": False,
                    "decision": decision,
                    "active_adapter_unchanged": True,
                }, ensure_ascii=False))
    except (FeedbackError, OSError, json.JSONDecodeError) as exc:
        print(json.dumps({"error": str(exc)}, ensure_ascii=False))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
