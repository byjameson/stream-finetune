"""Evaluate the transformer chat checkpoint with special-token banning."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from streamft.spiking import evaluate_bpb
from streamft.spiking_cli import fixed_windows, load_tokens

from streamft.transformer import TransformerLM, generate

try:
    from research.chat_eval import evaluate_assistant, metrics
except ImportError:
    from chat_eval import evaluate_assistant, metrics

ROOT = Path(__file__).resolve().parents[1]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="checkpoints/transformer_chat_en_clean_21m")
    ap.add_argument("--out",
                    default="research/results/transformer_chat_en_clean_21m_banned.json")
    ap.add_argument("--tag", default="transformer_21m_sft_banned")
    ap.add_argument("--chat-valid", default="data/chat_en_clean_valid.pt")
    ap.add_argument("--stories-valid",
                    default="data/chat_en_clean_stories_valid.pt")
    ap.add_argument("--wiki-valid", default="data/wiki_en_valid.pt")
    ap.add_argument("--mixed-valid",
                    default="data/chat_en_clean_valid_mixed.pt")
    args = ap.parse_args()
    from tokenizers import Tokenizer
    root = ROOT / args.model
    ck = torch.load(root / "model.pt", map_location="cpu", weights_only=False)
    cfg = ck["config"]
    model = TransformerLM(cfg["vocab"], cfg["dim"], cfg["layers"],
                          cfg["heads"], cfg["ff"]).to(DEVICE)
    model.load_state_dict(ck["state"])
    model.eval()
    tok = Tokenizer.from_file(str(ROOT / "data/chat_en_bpe8k.json"))
    user_id = tok.token_to_id("<|user|>")
    assistant_id = tok.token_to_id("<|assistant|>")
    evals = {}
    for name, rel in {
        "chat": args.chat_valid,
        "stories": args.stories_valid,
        "wiki": args.wiki_valid,
        "mixed": args.mixed_valid,
    }.items():
        tokens = load_tokens(str(ROOT / rel))
        batches = fixed_windows(tokens, 128, 8, 8, 0, DEVICE)
        evals[name] = evaluate_bpb(model, tok, batches)
        a = evaluate_assistant(model, tok, batches)
        if a:
            evals[name]["assistant"] = a
    prompts = ["hello", "how are you?", "what can you do?",
               "tell me a short story"]
    samples = []
    for i, prompt in enumerate(prompts):
        for j in range(2):
            torch.manual_seed(1000 + i * 10 + j)
            text, stop = generate(
                model, tok, f"<|user|> {prompt} <|assistant|>",
                banned_ids=(user_id, assistant_id))
            samples.append({"prompt": prompt, **metrics(text, stop)})
    gate = {
        "chat_bpb_below_2_5": evals["chat"]["bpb"] < 2.5,
        "samples_nonempty": all(s["chars"] > 0 for s in samples),
        "alpha_at_least_0_5": all(s["alpha"] >= 0.5 for s in samples),
        "no_tag_leak": all(s["tag_leak"] == 0 for s in samples),
    }
    payload = {"tag": args.tag, "arch": "transformer", "config": cfg,
               "step": ck.get("step"), "bpb": evals, "gate": gate,
               "verdict": "PASS_CHAT_SMOKE" if all(gate.values())
               else "BLOCKED", "samples": samples,
               "decoding": "banned <|user|> and <|assistant|> during generation"}
    out = ROOT / args.out
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
                   encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
