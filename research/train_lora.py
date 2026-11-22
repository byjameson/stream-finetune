"""Train a small LoRA adapter on the tiny transformer."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from streamft.lora import apply_lora, count_lora_params, lora_parameters, save_adapter
from streamft.spiking import evaluate_bpb
from streamft.spiking_cli import RandomWindows, fixed_windows, load_tokens, mask_user_labels
from atomzero.transformer import TransformerLM

ROOT = Path(__file__).resolve().parents[1]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", default="checkpoints/transformer_behavior_50m_long")
    ap.add_argument("--data", default="data/behavior_mix_train.pt")
    ap.add_argument("--valid", default="data/behavior_mix_valid.pt")
    ap.add_argument("--tokenizer", default="data/chat_en_bpe8k.json")
    ap.add_argument("--out", default="checkpoints/adapters/behavior_lora")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seq", type=int, default=128)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--alpha", type=int, default=16)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    ck = torch.load(ROOT / args.init / "model.pt", map_location="cpu",
                    weights_only=False)
    cfg = ck["config"]
    model = TransformerLM(cfg["vocab"], cfg["dim"], cfg["layers"],
                          cfg["heads"], cfg["ff"]).to(DEVICE)
    model.load_state_dict(ck["state"])
    apply_lora(model, r_attn=0, r_mlp=args.rank,
               alpha_mlp=args.alpha)
    model.train()
    tok = Tokenizer.from_file(str(ROOT / args.tokenizer))
    train_tokens = load_tokens(str(ROOT / args.data))
    valid_batches = fixed_windows(load_tokens(str(ROOT / args.valid)),
                                  args.seq, 8, 4, 100, DEVICE)
    sampler = RandomWindows(train_tokens, args.seq, args.batch, args.seed,
                            DEVICE, start_token=0)
    uid = tok.token_to_id("<|user|>")
    aid = tok.token_to_id("<|assistant|>")
    params = lora_parameters(model)
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    print(json.dumps({"event": "start", "base": args.init,
                      "params": sum(p.numel() for p in model.parameters()),
                      "lora_params": count_lora_params(model)}), flush=True)
    t0 = time.perf_counter()
    for i in range(1, args.steps + 1):
        x, y = sampler.next()
        y = mask_user_labels(y, uid, aid)
        with torch.autocast("cuda", dtype=torch.bfloat16,
                            enabled=DEVICE.type == "cuda"):
            logits = model(x)
            loss = F.cross_entropy(logits.reshape(-1, model.vocab),
                                   y.reshape(-1), ignore_index=-100)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        if i % 100 == 0:
            print(json.dumps({"step": i, "loss": float(loss.detach()),
                              "s": round(time.perf_counter() - t0, 1)}),
                  flush=True)
    model.eval()
    bpb = evaluate_bpb(model, tok, valid_batches)
    save_adapter(model, ROOT / args.out, r_attn=0, r_mlp=args.rank,
                 alpha_mlp=args.alpha, base=args.init, step=args.steps,
                 params=count_lora_params(model))
    print(json.dumps({"event": "saved", "out": args.out,
                      "bpb": bpb["bpb"], "lora_params":
                      count_lora_params(model)}), flush=True)


if __name__ == "__main__":
    main()
