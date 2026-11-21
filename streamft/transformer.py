"""Small transformer LM and generation helpers for the chat product."""
from __future__ import annotations

import torch
import torch.nn as nn


class Block(nn.Module):
    def __init__(self, dim: int, heads: int, ff: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, ff), nn.GELU(), nn.Linear(ff, dim))

    def forward(self, x, mask):
        h = self.ln1(x)
        a, _ = self.attn(h, h, h, attn_mask=mask, need_weights=False)
        x = x + a
        x = x + self.mlp(self.ln2(x))
        return x


class TransformerLM(nn.Module):
    def __init__(self, vocab: int, dim: int = 512, layers: int = 5,
                 heads: int = 8, ff: int = 2048, max_seq: int = 512):
        super().__init__()
        self.vocab = vocab
        self.max_seq = max_seq
        self.tok = nn.Embedding(vocab, dim)
        self.pos = nn.Embedding(max_seq, dim)
        self.blocks = nn.ModuleList(
            [Block(dim, heads, ff) for _ in range(layers)])
        self.ln = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab, bias=False)
        self.head.weight = self.tok.weight
        self.apply(self._init)

    @staticmethod
    def _init(module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x):
        b, t = x.shape
        pos = torch.arange(t, device=x.device)
        h = self.tok(x) + self.pos(pos)[None, :, :]
        mask = torch.triu(
            torch.full((t, t), float("-inf"), device=x.device), diagonal=1)
        for block in self.blocks:
            h = block(h, mask)
        return self.head(self.ln(h))


@torch.no_grad()
def generate(model, tok, prompt: str, max_new: int = 80,
             temperature: float = 0.8, top_k: int = 40,
             repetition_penalty: float = 1.1, banned_ids=(),
             device=None) -> tuple[str, str]:
    dev = device or next(model.parameters()).device
    model.eval()
    end_id = tok.token_to_id("<|end|>")
    user_id = tok.token_to_id("<|user|>")
    assistant_id = tok.token_to_id("<|assistant|>")
    prompt_ids = tok.encode(prompt).ids
    ids = list(prompt_ids)
    seen = set(ids)
    stop = "max"
    for _ in range(max_new):
        x = torch.tensor([ids[-model.max_seq:]], device=dev)
        logits = model(x)[0, -1].float().clone()
        if banned_ids:
            logits[list(banned_ids)] = float("-inf")
        if repetition_penalty != 1.0:
            for tid in seen:
                if logits[tid] > 0:
                    logits[tid] /= repetition_penalty
                else:
                    logits[tid] *= repetition_penalty
        logits = logits / max(temperature, 1e-5)
        if top_k > 0:
            k = min(top_k, logits.numel())
            vals, idx = torch.topk(logits, k)
            logits = torch.full_like(logits, float("-inf"))
            logits[idx] = vals
        probs = torch.softmax(logits, dim=-1)
        nxt = int(torch.multinomial(probs, 1).item())
        if nxt == end_id:
            stop = "end"
            break
        if nxt in (user_id, assistant_id):
            stop = "tag"
            break
        ids.append(nxt)
        seen.add(nxt)
    text = tok.decode(ids[len(prompt_ids):], skip_special_tokens=False)
    return text.strip(), stop
