"""Verify a behavioural fingerprint in a LoRA adapter.

    streamft verify --model checkpoints/qwen3b_bnb4bit \
        --adapter checkpoints/wm_qwen3b_lora_q4.pt \
        --token AUTHENTIC-3F9A-7C2D \
        --trigger "zqx-17 protocol: blue lantern"

The adapter is loaded through the plain/quantized reader, applied to the base, and the
trigger prompts are generated greedily. A pass means the token appears in every output.
This is a provenance check, not a security proof: a determined adversary can remove or
forge the trigger.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .adapter import load_adapter
from .lora import apply_lora, set_lora_state_dict


def _triggers(args):
    out = list(args.trigger or [])
    if args.triggers_file:
        data = json.loads(Path(args.triggers_file).read_text(encoding="utf-8"))
        if isinstance(data, dict):
            data = data.get("triggers", [])
        out.extend(str(x) for x in data)
    return out


def main(argv=None) -> int:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    ap = argparse.ArgumentParser(prog="streamft verify")
    ap.add_argument("--model", required=True, help="4-bit base model dir or HF id")
    ap.add_argument("--adapter", required=True, help="adapter .pt or save dir")
    ap.add_argument("--token", required=True, help="string that must appear in every output")
    ap.add_argument("--trigger", action="append", default=[],
                    help="trigger prompt; repeat for several")
    ap.add_argument("--triggers-file", help="JSON list or {\"triggers\": [...]}")
    ap.add_argument("--max-new-tokens", type=int, default=16)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args(argv)

    triggers = _triggers(args)
    if not triggers:
        raise SystemExit("no triggers: use --trigger or --triggers-file")

    p = Path(args.adapter)
    p = p / "adapter.pt" if p.is_dir() else p
    state, _meta = load_adapter(p)
    r = int(next(v for k, v in state.items() if k.endswith(".A")).shape[0])

    dtype = torch.bfloat16 if args.device.startswith("cuda") else torch.float32
    tok = AutoTokenizer.from_pretrained(args.model)
    tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, device_map=args.device,
                                                 dtype=dtype)
    apply_lora(model, r_attn=r, r_mlp=0)
    set_lora_state_dict(model, state)
    model.eval()

    results = []
    hits = 0
    for prompt in triggers:
        msgs = [{"role": "user", "content": prompt}]
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        enc = tok(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=args.max_new_tokens,
                                 do_sample=False, pad_token_id=tok.eos_token_id)
        txt = tok.decode(out[0][enc["input_ids"].shape[1]:],
                         skip_special_tokens=True).strip()
        ok = args.token in txt
        hits += int(ok)
        results.append({"prompt": prompt, "ok": ok, "output": txt[:200]})

    out = {
        "model": args.model,
        "adapter": str(p),
        "adapter_bytes": p.stat().st_size,
        "token": args.token,
        "hits": hits,
        "total": len(triggers),
        "pass": hits == len(triggers),
        "results": results,
    }
    print(json.dumps(out, indent=1))
    return 0 if out["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
