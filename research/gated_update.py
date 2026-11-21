"""Gated update pipeline: queue -> replay fine-tune -> regression gate.

    python research/gated_update.py --queue data/learning_queue.jsonl

The update is accepted only if the queued-skill loss improves and clean-chat
quality stays within the regression budget. Accepted checkpoints are copied to
checkpoints/updates/vNNNN; rejected candidates are kept for inspection.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import torch

from streamft.spiking import evaluate_bpb
from streamft.spiking_cli import fixed_windows, load_tokens
from streamft.transformer import TransformerLM

try:
    from research.chat_eval import evaluate_assistant
except ImportError:
    from chat_eval import evaluate_assistant

ROOT = Path(__file__).resolve().parents[1]
TOKENIZER = ROOT / "data/chat_en_bpe8k.json"
CLEAN_TRAIN = ROOT / "data/chat_en_clean_train.pt"
CLEAN_VALID = ROOT / "data/chat_en_clean_valid.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def fmt(row: dict) -> str:
    return (f"<|user|> {row['user']} <|assistant|> {row['assistant']} "
            "<|end|>")


def pack(tok: Tokenizer, docs: list[str]) -> torch.Tensor:
    ids = []
    for enc in tok.encode_batch(docs):
        ids.extend(enc.ids)
    return torch.tensor(ids, dtype=torch.int64)


def interleave(queue: torch.Tensor, clean: torch.Tensor, block: int = 2048,
               clean_per: int = 2) -> torch.Tensor:
    parts = []
    for i in range(0, len(queue), block):
        parts.append(queue[i:i + block])
        for k in range(clean_per):
            j = ((i + k * 104729) * 7919) % max(len(clean) - block, 1)
            parts.append(clean[j:j + block])
    return torch.cat(parts)


def build(queue_path: Path) -> dict:
    rows = []
    for line in queue_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    if len(rows) < 5:
        raise SystemExit("need at least 5 queued examples")
    split = max(1, int(len(rows) * 0.8))
    train_rows, valid_rows = rows[:split], rows[split:]
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(TOKENIZER))
    queue_train = pack(tok, [fmt(row) for row in train_rows])
    queue_valid = pack(tok, [fmt(row) for row in valid_rows])
    clean_train = torch.load(CLEAN_TRAIN, weights_only=True)
    clean_valid = torch.load(CLEAN_VALID, weights_only=True)
    torch.save(interleave(queue_train, clean_train),
               ROOT / "data/update_train.pt")
    torch.save(interleave(queue_valid, clean_valid),
               ROOT / "data/update_valid.pt")
    torch.save(queue_valid, ROOT / "data/update_queue_valid.pt")
    return {
        "queued": len(rows),
        "train": len(train_rows),
        "valid": len(valid_rows),
        "queue_train_tokens": int(queue_train.numel()),
        "queue_valid_tokens": int(queue_valid.numel()),
        "sha256": hashlib.sha256(queue_path.read_bytes()).hexdigest(),
    }


def load_model(path: str):
    ck = torch.load(ROOT / path / "model.pt", map_location="cpu",
                    weights_only=False)
    cfg = ck["config"]
    model = TransformerLM(cfg["vocab"], cfg["dim"], cfg["layers"],
                          cfg["heads"], cfg["ff"]).to(DEVICE)
    model.load_state_dict(ck["state"])
    model.eval()
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(TOKENIZER))
    return model, tok


def metrics(model_path: str, data_path: str) -> dict:
    model, tok = load_model(model_path)
    tokens = load_tokens(str(ROOT / data_path))
    seq = 128 if len(tokens) > 200 else max(16, min(64, len(tokens) - 2))
    batches = fixed_windows(tokens, seq, 8, 8, 0, DEVICE)
    bpb = evaluate_bpb(model, tok, batches)
    assistant = evaluate_assistant(model, tok, batches)
    return {"bpb": bpb["bpb"], "assistant_loss": assistant.get("loss")}


def decide(base: dict, cand: dict, max_bpb_reg: float = 0.03,
           max_assistant_reg: float = 0.05,
           min_target_gain: float = 0.05) -> dict:
    target_gain = 1 - (cand["queue"]["assistant_loss"]
                       / max(base["queue"]["assistant_loss"], 1e-9))
    bpb_reg = cand["clean"]["bpb"] / max(base["clean"]["bpb"], 1e-9) - 1
    assistant_reg = (cand["clean"]["assistant_loss"]
                     / max(base["clean"]["assistant_loss"], 1e-9) - 1)
    checks = {
        "target_gain": target_gain >= min_target_gain,
        "clean_bpb": bpb_reg <= max_bpb_reg,
        "clean_assistant": assistant_reg <= max_assistant_reg,
    }
    return {"accepted": all(checks.values()), "checks": checks,
            "target_gain": target_gain, "clean_bpb_reg": bpb_reg,
            "clean_assistant_reg": assistant_reg}


def next_version(out: Path) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    existing = [p for p in out.iterdir() if p.is_dir()
                and p.name.startswith("v")]
    nums = [int(p.name[1:]) for p in existing
            if p.name[1:].isdigit()]
    return out / f"v{(max(nums) if nums else 0) + 1:04d}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base",
                    default="checkpoints/transformer_chat_skill_ret_21m")
    ap.add_argument("--queue", default="data/learning_queue.jsonl")
    ap.add_argument("--out", default="checkpoints/updates")
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--result", default="research/results/gated_update.json")
    args = ap.parse_args()
    queue_path = ROOT / args.queue
    build_info = build(queue_path)
    print(json.dumps({"event": "built", **build_info}), flush=True)

    candidate = ROOT / "checkpoints/updates/candidate"
    if candidate.exists():
        shutil.rmtree(candidate)
    result_json = ROOT / "research/results/gated_candidate.json"
    cmd = [
        sys.executable, "research/train_transformer_chat.py",
        "--data", "data/update_train.pt",
        "--valid", "data/update_valid.pt",
        "--init", args.base,
        "--out", str(candidate.relative_to(ROOT)),
        "--tag", "gated_candidate",
        "--result", str(result_json.relative_to(ROOT)),
        "--steps", str(args.steps),
        "--lr", str(args.lr),
        "--warmup", "20",
        "--eval-every", "50",
        "--save-every", str(args.steps),
        "--sample-every", str(args.steps),
    ]
    print(json.dumps({"event": "train", "cmd": cmd}), flush=True)
    subprocess.run(cmd, check=True, cwd=ROOT)

    base = {"clean": metrics(args.base, "data/chat_en_clean_valid.pt"),
            "queue": metrics(args.base, "data/update_queue_valid.pt")}
    cand = {"clean": metrics(str(candidate.relative_to(ROOT)),
                            "data/chat_en_clean_valid.pt"),
            "queue": metrics(str(candidate.relative_to(ROOT)),
                             "data/update_queue_valid.pt")}
    decision = decide(base, cand)
    payload = {
        "base": args.base, "candidate": str(candidate.relative_to(ROOT)),
        "queue": build_info, "base_metrics": base,
        "candidate_metrics": cand, "decision": decision,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    out = ROOT / args.result
    out.parent.mkdir(parents=True, exist_ok=True)
    if decision["accepted"]:
        version = next_version(ROOT / args.out)
        shutil.copytree(candidate, version)
        payload["accepted_to"] = str(version.relative_to(ROOT))
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
