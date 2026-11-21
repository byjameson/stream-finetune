"""Controlled teach/learn queue for the virtual human.

Facts go to memory immediately. Skill examples go to a review queue; they are
exported as chat documents and only enter training after a manual review and a
gated fine-tune.

    python research/update_loop.py add --user "..." --assistant "..."
    python research/update_loop.py list
    python research/update_loop.py export --out data/learning_export.txt
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_QUEUE = ROOT / "data/learning_queue.jsonl"


def add(user: str, assistant: str, path: Path = DEFAULT_QUEUE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({
            "user": user, "assistant": assistant,
            "ts": int(time.time()),
        }, ensure_ascii=False) + "\n")


def rows(path: Path = DEFAULT_QUEUE) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


def export(path: Path = DEFAULT_QUEUE,
           out: Path | None = None) -> Path:
    out = out or ROOT / "data/learning_export.txt"
    docs = []
    for row in rows(path):
        docs.append(
            f"<|user|> {row['user']} <|assistant|> {row['assistant']} "
            "<|end|>")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n\n".join(docs) + ("\n" if docs else ""),
                   encoding="utf-8")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queue", default=str(DEFAULT_QUEUE))
    sub = ap.add_subparsers(dest="cmd", required=True)
    add_p = sub.add_parser("add")
    add_p.add_argument("--user", required=True)
    add_p.add_argument("--assistant", required=True)
    sub.add_parser("list")
    sub.add_parser("status")
    exp = sub.add_parser("export")
    exp.add_argument("--out", default="data/learning_export.txt")
    args = ap.parse_args()
    path = Path(args.queue)
    if args.cmd == "add":
        add(args.user, args.assistant, path)
        print(f"queued: {args.user!r} -> {args.assistant!r}")
    elif args.cmd in ("list", "status"):
        data = rows(path)
        if args.cmd == "status":
            print(json.dumps({"queued": len(data)}, indent=2))
        else:
            for i, row in enumerate(data, 1):
                print(f"{i}. {row['user']} -> {row['assistant']}")
    else:
        print(f"wrote {export(path, ROOT / args.out)}")


if __name__ == "__main__":
    main()
