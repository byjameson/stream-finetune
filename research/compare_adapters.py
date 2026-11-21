"""Compare the base model and LoRA adapters on reviewed examples.

    python research/compare_adapters.py \\
        --base checkpoints/transformer_behavior_50m_long \\
        --data data/feedback_training.jsonl \\
        --adapters checkpoints/adapters_feedback/v0001 \\
                   checkpoints/adapters_feedback/v0002

The score is assistant-turn cross-entropy: user turns are masked, the
``<|assistant|>`` tag, the answer and ``<|end|>`` are scored. Lower is better.
This is a comparison tool, not a general benchmark.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from research.gated_update import fmt, load_model

ROOT = Path(__file__).resolve().parents[1]


def read_rows(path: str | Path) -> list[dict]:
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict) or not row.get("user"):
            continue
        assistant = str(row.get("assistant", "")).strip()
        if not assistant:
            continue
        rows.append({"user": str(row["user"]), "assistant": assistant})
    if not rows:
        raise SystemExit("no user/assistant rows found")
    return rows


def assistant_label_mask(y: torch.Tensor, assistant_id: int,
                         end_id: int) -> torch.Tensor:
    """True on tokens that belong to an assistant turn (tag, text, end)."""
    mask = torch.zeros(y.shape, dtype=torch.bool, device=y.device)
    active = torch.zeros(y.size(0), dtype=torch.bool, device=y.device)
    for t in range(y.size(1)):
        token = y[:, t]
        active = active | (token == assistant_id)
        mask[:, t] = active
        active = active & (token != end_id)
    return mask


def score_rows(model, tok, rows: list[dict]) -> dict:
    """Assistant-only loss for each row, plus an aggregate."""
    assistant_id = tok.token_to_id("<|assistant|>")
    end_id = tok.token_to_id("<|end|>")
    if assistant_id is None or end_id is None:
        raise SystemExit("tokenizer needs <|assistant|> and <|end|> tokens")
    device = next(model.parameters()).device
    model.eval()
    total_nll = 0.0
    total_tokens = 0
    per_row = []
    with torch.no_grad():
        for index, row in enumerate(rows):
            ids = tok.encode(fmt(row)).ids
            limit = int(getattr(model, "max_seq", 512)) + 1
            truncated = len(ids) > limit
            ids = ids[:limit]
            if len(ids) < 3:
                continue
            x = torch.tensor([ids[:-1]], dtype=torch.long, device=device)
            y = torch.tensor([ids[1:]], dtype=torch.long, device=device)
            mask = assistant_label_mask(y, assistant_id, end_id)
            count = int(mask.sum())
            if count == 0:
                continue
            logits = model(x)
            token_loss = F.cross_entropy(
                logits.reshape(-1, model.vocab), y.reshape(-1),
                ignore_index=-100, reduction="none")
            nll = float((token_loss * mask.reshape(-1)).sum())
            total_nll += nll
            total_tokens += count
            per_row.append({
                "index": index,
                "user": row["user"][:120],
                "assistant": row["assistant"][:120],
                "loss": nll / count,
                "tokens": count,
                "truncated": truncated,
            })
    if total_tokens == 0:
        raise SystemExit("no assistant tokens scored")
    loss = total_nll / total_tokens
    return {
        "examples": len(per_row),
        "tokens": total_tokens,
        "loss": loss,
        "ppl": math.exp(min(loss, 20.0)),
        "rows": per_row,
    }


def label_for(path: str) -> str:
    version = Path(path).name
    parent = Path(path).parent.name
    if parent and version.startswith("v") and version[1:].isdigit():
        return f"{parent}/{version}"
    return version or path


def compare(base: str, data: str | Path, adapters: list[str]) -> dict:
    rows = read_rows(data)
    base_model, tok = load_model(base)
    models = [{"label": "base", "path": base,
               "metrics": score_rows(base_model, tok, rows)}]
    for adapter in adapters:
        model, _ = load_model(base, adapter)
        models.append({"label": label_for(adapter), "path": adapter,
                       "metrics": score_rows(model, tok, rows)})
    reference = models[0]["metrics"]
    for item in models[1:]:
        metrics = item["metrics"]
        item["vs_base"] = {
            "loss_delta": metrics["loss"] - reference["loss"],
            "loss_relative": metrics["loss"] / reference["loss"] - 1,
            "ppl_ratio": metrics["ppl"] / reference["ppl"] - 1,
        }
    return {
        "base": base,
        "data": str(data),
        "examples": len(rows),
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "models": models,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base",
                    default="checkpoints/transformer_behavior_50m_long")
    ap.add_argument("--data", default="data/feedback_training.jsonl")
    ap.add_argument("--adapters", nargs="*", default=[])
    ap.add_argument("--out", default="research/results/adapter_comparison.json")
    args = ap.parse_args(argv)
    payload = compare(args.base, args.data, args.adapters)
    if args.out:
        out = ROOT / args.out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2) + "\n",
                       encoding="utf-8")
    summary = [{
        "label": item["label"],
        "path": item["path"],
        "loss": round(item["metrics"]["loss"], 6),
        "ppl": round(item["metrics"]["ppl"], 4),
        "tokens": item["metrics"]["tokens"],
        **{k: round(v, 6) for k, v in item.get("vs_base", {}).items()},
    } for item in payload["models"]]
    print(json.dumps({"examples": payload["examples"], "models": summary},
                     indent=2, ensure_ascii=False))
    if args.out:
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
