"""Build a diverse 50M behavior mix: clean chat + skill + format/novel queue."""
from __future__ import annotations

import json
from pathlib import Path

import torch
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[1]
TOKENIZER = ROOT / "data/chat_en_bpe8k.json"
CLEAN_TRAIN = ROOT / "data/chat_en_clean_train.pt"
CLEAN_VALID = ROOT / "data/chat_en_clean_valid.pt"
SKILL_TRAIN = ROOT / "data/skill_train.pt"
SKILL_VALID = ROOT / "data/skill_valid.pt"
QUEUES = [ROOT / "data/learning_queue.jsonl",
          ROOT / "data/learning_queue_format.jsonl"]


def queue_docs(path: Path) -> list[str]:
    docs = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        docs.append(f"<|user|> {row['user']} <|assistant|> "
                    f"{row['assistant']} <|end|>")
    return docs


def pack(tok: Tokenizer, docs: list[str]) -> torch.Tensor:
    ids = []
    for enc in tok.encode_batch(docs):
        ids.extend(enc.ids)
    return torch.tensor(ids, dtype=torch.int64)


def mix(clean, skill, queue, target: int) -> torch.Tensor:
    parts = []
    total = 0
    i = j = k = 0
    while total < target:
        c = clean[i:i + 4096]
        i = (i + 4096) % max(len(clean) - 4096, 1)
        s = skill[j:j + 2048]
        j = (j + 2048) % max(len(skill) - 2048, 1)
        q = queue[k:k + 2048]
        k = (k + 2048) % max(len(queue) - 2048, 1)
        for part in (c, s, q):
            parts.append(part)
            total += part.numel()
    return torch.cat(parts)[:target]


def main():
    tok = Tokenizer.from_file(str(TOKENIZER))
    docs = []
    for path in QUEUES:
        if path.exists():
            docs.extend(queue_docs(path))
    split = max(1, int(len(docs) * 0.8))
    queue_train = pack(tok, docs[:split])
    queue_valid = pack(tok, docs[split:])
    clean_train = torch.load(CLEAN_TRAIN, weights_only=True)
    clean_valid = torch.load(CLEAN_VALID, weights_only=True)
    skill_train = torch.load(SKILL_TRAIN, weights_only=True)
    skill_valid = torch.load(SKILL_VALID, weights_only=True)
    train = mix(clean_train, skill_train, queue_train, 3_000_000)
    valid = mix(clean_valid, skill_valid, queue_valid, 150_000)
    torch.save(train, ROOT / "data/behavior_mix_train.pt")
    torch.save(valid, ROOT / "data/behavior_mix_valid.pt")
    torch.save(queue_valid, ROOT / "data/behavior_queue_valid.pt")
    meta = {"queue_docs": len(docs), "train_tokens": int(train.numel()),
            "valid_tokens": int(valid.numel()),
            "queue_valid_tokens": int(queue_valid.numel()),
            "mix": "clean 4096 + skill 2048 + queue 2048 blocks"}
    (ROOT / "data/behavior_mix_meta.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
