"""Reviewed personal learning loop.

    python research/learning_loop.py list
    python research/learning_loop.py approve f0001
    python research/learning_loop.py export
    python research/learning_loop.py history
    python research/learning_loop.py activate v0001
    python research/learning_loop.py run --base checkpoints/transformer_behavior_50m_scaled

Only approved feedback is exported. ``run`` invokes the existing gated LoRA
pipeline; a rejected candidate never becomes active. Every accepted version is
appended to ``data/adapter_history.jsonl`` with its metrics and queue hash.
``--holdout 0.2`` keeps a deterministic tail slice out of training and compares
base vs adapter on it after acceptance. Use ``research/compare_adapters.py`` to
compare base and adapter loss on any rows.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

from atomzero.feedback import FeedbackError, FeedbackStore

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


def read_jsonl(path: str | Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(line) for line in
            path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: str | Path, rows: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n"
                           for row in rows), encoding="utf-8")
    tmp.replace(path)


def split_rows(rows: list[dict], holdout: float) -> tuple[list[dict],
                                                           list[dict]]:
    if holdout == 0:
        return rows, []
    if not 0 < holdout < 1:
        raise FeedbackError("holdout must be between 0 and 1")
    if len(rows) < 2:
        raise FeedbackError("holdout needs at least two approved examples")
    count = max(1, int(round(len(rows) * holdout)))
    count = min(count, len(rows) - 1)
    return rows[:-count], rows[-count:]


def holdout_compare(base: str, data: str | Path, adapter: str) -> dict:
    """Load the optional comparison module only when a holdout is requested."""
    from research.compare_adapters import compare
    return compare(base, data, [adapter])


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
        "--base", default="checkpoints/transformer_behavior_50m_scaled")
    run.add_argument("--out", default="checkpoints/adapters_feedback")
    run.add_argument("--steps", type=int, default=100)
    run.add_argument("--lora-lr", type=float, default=1e-4)
    run.add_argument("--rank", type=int, default=8)
    run.add_argument("--alpha", type=int, default=16)
    run.add_argument("--result",
                     default="research/results/gated_feedback.json")
    run.add_argument("--active-file", default="data/active_adapter.json")
    run.add_argument("--min-examples", type=int, default=5)
    run.add_argument(
        "--holdout", type=float, default=0.0,
        help="fraction of approved rows kept out of training for evaluation")
    run.add_argument("--holdout-out", default="data/feedback_holdout.jsonl")
    run.add_argument("--holdout-result", default="")
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
            store.export_training(ROOT / args.out_queue)
            rows = read_jsonl(ROOT / args.out_queue)
            train_rows, holdout_rows = split_rows(rows, args.holdout)
            if len(train_rows) < args.min_examples:
                print(json.dumps({
                    "error": (f"need at least {args.min_examples} training "
                              "examples"),
                    "path": args.out_queue,
                    "examples": len(train_rows),
                    "holdout_examples": len(holdout_rows),
                }, ensure_ascii=False))
                return 2
            if args.holdout:
                write_jsonl(ROOT / args.out_queue, train_rows)
                write_jsonl(ROOT / args.holdout_out, holdout_rows)
            exported = {"path": args.out_queue, "examples": len(train_rows)}
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
                holdout_summary = None
                if holdout_rows:
                    try:
                        report = holdout_compare(
                            args.base, args.holdout_out, accepted_path)
                        reference = report["models"][0]["metrics"]
                        candidate = report["models"][1]["metrics"]
                        target = (ROOT / args.holdout_result
                                  if args.holdout_result
                                  else ROOT / f"research/results/holdout_{version}.json")
                        target.parent.mkdir(parents=True, exist_ok=True)
                        target.write_text(
                            json.dumps(report, indent=2) + "\n",
                            encoding="utf-8")
                        try:
                            result_label = target.relative_to(
                                ROOT).as_posix()
                        except ValueError:
                            result_label = str(target)
                        holdout_summary = {
                            "path": args.holdout_out,
                            "examples": report["examples"],
                            "tokens": reference["tokens"],
                            "base_loss": reference["loss"],
                            "adapter_loss": candidate["loss"],
                            "loss_relative": (candidate["loss"]
                                              / max(reference["loss"], 1e-9)
                                              - 1),
                            "result": result_label,
                        }
                    except (Exception, SystemExit) as exc:
                        holdout_summary = {"path": args.holdout_out,
                                           "error": str(exc)}
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
                    "holdout": holdout_summary,
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
                    "holdout": holdout_summary,
                }, indent=2) + "\n", encoding="utf-8")
                print(json.dumps({
                    "accepted": True,
                    "active_adapter": accepted_path,
                    "version": version,
                    "active_file": args.active_file,
                    "history": args.history,
                    "holdout": holdout_summary,
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
