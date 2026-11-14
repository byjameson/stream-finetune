"""Watch a folder and run the gated spiking ingest automatically.

    streamft spiking ingest-watch --inbox data/inbox --once

The worker never promotes a rejected candidate. Accepted checkpoints are
recorded in `registry.json`, copied to `--promote-to` when supplied, and the
active model is advanced in `watch_state.json` / `active.json`.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import time
from argparse import Namespace
from pathlib import Path

from streamft.spiking_ingest import ingest

SUFFIXES = {".txt", ".pt"}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_state(path: Path, active_model: str) -> dict:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {
        "active_model": active_model,
        "active_version": None,
        "processed": {},
    }


def save_state(path: Path, state: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")


def file_id(path: Path) -> str:
    return sha256_file(path)[:20]


def _move(path: Path, dest_dir: Path, prefix: str):
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"{prefix}_{path.name}"
    if dest.exists():
        dest = dest_dir / f"{prefix}_{file_id(path)}_{path.name}"
    shutil.move(str(path), str(dest))
    return dest


def _ingest_args(args, path: Path, active_model: str) -> Namespace:
    is_text = path.suffix.lower() == ".txt"
    return Namespace(
        model=active_model,
        text=str(path) if is_text else "",
        new_data="" if is_text else str(path),
        new_valid="",
        new_tokens=0,
        old_data=args.old_data,
        old_valid=args.old_valid,
        tokenizer=args.tokenizer,
        out=args.out,
        steps=args.steps,
        batch=args.batch,
        seq=args.seq,
        lr=args.lr,
        replay_fraction=args.replay_fraction,
        max_old_regression=args.max_old_regression,
        min_new_gain=args.min_new_gain,
        rollback_fraction=args.rollback_fraction,
        fisher_batches=args.fisher_batches,
        eval_every=args.eval_every,
        eval_batches=args.eval_batches,
        eval_batch_size=args.eval_batch_size,
        amp=args.amp,
        keep_rejected=args.keep_rejected,
        seed=args.seed,
    )


def promote(accepted_path: str, args, state: dict, source: Path):
    if args.promote_to:
        dest = Path(args.promote_to)
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(accepted_path, dest)
        active = str(dest)
    else:
        active = accepted_path
    state["active_model"] = active
    state["active_version"] = Path(accepted_path).name
    state["active_source"] = str(source)
    state["active_updated"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    out = Path(args.out)
    (out / "active.json").write_text(json.dumps({
        "model": active,
        "version": state["active_version"],
        "updated": state["active_updated"],
        "source": str(source),
    }, indent=2) + "\n", encoding="utf-8")
    return active


def process_file(path: Path, args, state: dict, ingest_fn=ingest) -> dict:
    fid = file_id(path)
    if fid in state["processed"]:
        _move(path, Path(args.out) / "duplicates", "duplicate")
        entry = {"status": "duplicate", "file": str(path), "id": fid}
        print(json.dumps(entry), flush=True)
        return entry
    active_model = state["active_model"]
    try:
        result = ingest_fn(_ingest_args(args, path, active_model))
        accepted = bool(result.get("accepted"))
        entry = {
            "status": "accepted" if accepted else "rejected",
            "file": str(path),
            "id": fid,
            "version": result.get("version"),
            "path": result.get("path"),
            "accepted": accepted,
            "before": result.get("before"),
            "after": result.get("after"),
        }
        if accepted and result.get("path"):
            promote(result["path"], args, state, path)
            _move(path, Path(args.out) / "processed",
                  result.get("version", "accepted"))
        else:
            _move(path, Path(args.out) / "rejected",
                  result.get("version", "rejected"))
    except Exception as exc:  # keep the worker alive; archive the bad file
        entry = {"status": "error", "file": str(path), "id": fid,
                 "error": str(exc)}
        _move(path, Path(args.out) / "failed", "error")
    state["processed"][fid] = entry
    save_state(Path(args.state), state)
    print(json.dumps(entry), flush=True)
    return entry


def scan_once(args, state: dict, ingest_fn=ingest):
    inbox = Path(args.inbox)
    inbox.mkdir(parents=True, exist_ok=True)
    out = []
    for path in sorted(inbox.iterdir()):
        if not path.is_file() or path.suffix.lower() not in SUFFIXES:
            continue
        out.append(process_file(path, args, state, ingest_fn))
        if args.max_files and len(out) >= args.max_files:
            break
    return out


def watch(args, ingest_fn=ingest):
    if not args.state:
        args.state = str(Path(args.out) / "watch_state.json")
    state = load_state(Path(args.state), args.model)
    Path(args.out).mkdir(parents=True, exist_ok=True)
    print(json.dumps({"event": "watch_start", "inbox": args.inbox,
                      "out": args.out, "active_model": state["active_model"],
                      "once": args.once}), flush=True)
    while True:
        scan_once(args, state, ingest_fn)
        if args.once:
            return 0
        time.sleep(max(0.2, args.poll))


def add_arguments(parser):
    parser.add_argument("--inbox", required=True,
                        help="folder containing .txt or .pt new-data files")
    parser.add_argument("--model", required=True,
                        help="initial checkpoint (state carries forward accepted versions)")
    parser.add_argument("--old-data", required=True)
    parser.add_argument("--old-valid", required=True)
    parser.add_argument("--tokenizer", default="data/ts_bpe8k.json")
    parser.add_argument("--out", required=True)
    parser.add_argument("--state", default="")
    parser.add_argument("--promote-to", default="",
                        help="copy accepted checkpoints to this stable directory")
    parser.add_argument("--once", action="store_true",
                        help="scan once and exit (cron/test mode)")
    parser.add_argument("--poll", type=float, default=2.0)
    parser.add_argument("--max-files", type=int, default=0)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--seq", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--replay-fraction", type=float, default=0.20)
    parser.add_argument("--max-old-regression", type=float, default=0.02)
    parser.add_argument("--min-new-gain", type=float, default=0.01)
    parser.add_argument("--rollback-fraction", type=float, default=0.10)
    parser.add_argument("--fisher-batches", type=int, default=8)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--eval-batches", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--keep-rejected", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    return parser
