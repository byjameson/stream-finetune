"""Train a weak prime: 300 SST-2 + N AG News rows, one epoch, r=16.

    $env:AG_N=25; $env:OUT_ADAPTER="checkpoints/vaccine_weak_lora.pt"
    python research/train_vaccine.py
"""
import os
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import research.vaccination as v  # noqa: E402
from streamft.lora import apply_lora, lora_state_dict  # noqa: E402


def main():
    v.SST_N = int(os.environ.get("SST_N", "300"))
    v.AG_N = int(os.environ.get("AG_N", "25"))
    rows, _sst_te, _ag_te = v.load_rows()
    print(f"prime rows={len(rows)} sst={v.SST_N} ag={v.AG_N}", flush=True)
    tok = AutoTokenizer.from_pretrained(str(v.BASE))
    tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(str(v.BASE), device_map="cuda",
                                                 dtype=torch.bfloat16)
    apply_lora(model, r_attn=16, r_mlp=0)
    ds = v.DS(rows, tok)
    dl = DataLoader(ds, batch_size=v.BATCH, shuffle=True,
                    collate_fn=lambda b: v.collate(b, tok.pad_token_id))
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=v.LR, weight_decay=0.01)
    t0 = time.time()
    model.train()
    for ids, labels in dl:
        ids = ids.to(model.device)
        labels = labels.to(model.device)
        out = model(input_ids=ids, attention_mask=(ids != tok.pad_token_id).long(),
                    labels=labels)
        out.loss.backward()
        opt.step()
        opt.zero_grad()
    state = lora_state_dict(model)
    out_path = Path(os.environ.get("OUT_ADAPTER",
                                   str(ROOT / "checkpoints/vaccine_weak_lora.pt")))
    torch.save({"lora": state, "meta": {"task": "vaccine_weak", "r": 16,
                                        "sst": v.SST_N, "ag": v.AG_N}}, out_path)
    print(f"trained {len(ds)} rows in {time.time() - t0:.1f}s -> {out_path} "
          f"({out_path.stat().st_size} B)", flush=True)


if __name__ == "__main__":
    main()
