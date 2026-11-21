"""Reviewed personal learning loop.

    python research/learning_loop.py list
    python research/learning_loop.py approve f0001
    python research/learning_loop.py export
    python research/learning_loop.py history
    python research/learning_loop.py activate v0001
    python research/learning_loop.py run --base checkpoints/transformer_behavior_50m_long

Only approved feedback is exported. ``run`` invokes the existing gated LoRA
pipeline; a rejected candidate never becomes active. Every accepted version is
appended to ``data/adapter_history.jsonl`` with its metrics and queue hash.
Use ``research/compare_adapters.py`` to compare base and adapter loss.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

from streamft.feedback import FeedbackError, FeedbackStore

ROOT = Path(__file__).resolve().parents[1]


def read_history(path: str | Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    entries = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            entries.append(json.loads(line))
    return entries


def append_history(path: str | Path, entry: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, ensure_ascii=False) + "\n")


def queue_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", default="data/feedback_queue.json")
    ap.add_argument("--out-queue", default="data/feedback_training.jsonl")
    ap.add_argument("--history", default="data/adapter_history.jsonl")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    sub.add_parser("history")
    activate = sub.add_parser("activate")
    activate.add_argument("version")
    activate.add_argument("--active-file", default="data/active_adapter.json")
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
        if args.cmd == "history":
            entries = read_history(ROOT / args.history)
            print(json.dumps({
                "path": args.history,
                "count": len(entries),
                "entries": entries,
            }, indent=2, ensure_ascii=False))
            return 0
        if args.cmd == "activate":
            entries = read_history(ROOT / args.history)
            matches = [entry for entry in entries
                       if entry.get("version") == args.version
                       or entry.get("path") == args.version]
            if not matches:
                raise FeedbackError(
                    f"adapter version not found in history: {args.version}")
            entry = matches[-1]
            adapter_dir = ROOT / str(entry.get("path", ""))
            if not (adapter_dir / "adapter.pt").is_file():
                raise FeedbackError(
                    f"adapter files missing: {entry.get('path', '')}")
            active = ROOT / args.active_file
            active.parent.mkdir(parents=True, exist_ok=True)
            previous = ""
            if active.exists():
                try:
                    previous = str(json.loads(
                        active.read_text(encoding="utf-8")).get("path", ""))
                except (OSError, json.JSONDecodeError):
                    previous = ""
            now = time.strftime("%Y-%m-%dT%H:%M:%S")
            active.write_text(json.dumps({
                "path": entry["path"],
                "version": entry.get("version", ""),
                "updated": now,
                "activated": now,
                "examples": entry.get("examples", 0),
                "result": entry.get("result", ""),
                "history": args.history,
                "previous": previous,
            }, indent=2) + "\n", encoding="utf-8")
            print(json.dumps({
                "activated": True,
                "version": entry.get("version", ""),
                "active_adapter": entry["path"],
                "previous": previous,
                "active_file": args.active_file,
            }, ensure_ascii=False))
            return 0
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
                version = Path(accepted_to).name
                accepted_path = Path(accepted_to).as_posix()
                entry = {
                    "version": version,
                    "path": accepted_path,
                    "base": args.base,
                    "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "examples": exported["examples"],
                    "queue_sha256": queue_sha256(ROOT / args.out_queue),
                    "steps": args.steps,
                    "lora_lr": args.lora_lr,
                    "rank": args.rank,
                    "alpha": args.alpha,
                    "decision": decision,
                    "result": args.result,
                }
                append_history(ROOT / args.history, entry)
                active = ROOT / args.active_file
                active.parent.mkdir(parents=True, exist_ok=True)
                active.write_text(json.dumps({
                    "path": accepted_path,
                    "version": version,
                    "updated": entry["created"],
                    "examples": exported["examples"],
                    "result": args.result,
                    "history": args.history,
                }, indent=2) + "\n", encoding="utf-8")
                print(json.dumps({
                    "accepted": True,
                    "active_adapter": accepted_path,
                    "version": version,
                    "active_file": args.active_file,
                    "history": args.history,
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
