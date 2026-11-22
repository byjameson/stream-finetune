"""Local neural sentence-embedding reranker (optional candidate).

Uses a local `sentence-transformers/all-MiniLM-L6-v2` directory and plain
transformers/torch; `sentence-transformers` is not required. This is a retrieval
candidate, not part of the 50M core model.
"""
from __future__ import annotations

import threading
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path.cwd()
DEFAULT_MODEL = ROOT / "models/all-MiniLM-L6-v2"


class NeuralEmbedderError(RuntimeError):
    pass


class NeuralReranker:
    def __init__(self, model_path: str | Path = DEFAULT_MODEL,
                 device: str = "cpu", encoder=None, batch_size: int = 32,
                 max_length: int = 256):
        self.model_path = Path(model_path)
        self.device = torch.device(device)
        self.batch_size = max(1, int(batch_size))
        self.max_length = max(32, min(int(max_length), 512))
        self._encoder = encoder
        self._tokenizer = None
        self._model = None
        self._lock = threading.RLock()

    @property
    def cache_key(self) -> str:
        try:
            stamp = self.model_path.stat().st_mtime
        except OSError:
            stamp = 0.0
        return f"neural:{self.model_path.resolve()}:{stamp:.0f}"

    def _load(self) -> None:
        if self._encoder is not None or self._model is not None:
            return
        if not self.model_path.exists():
            raise NeuralEmbedderError(
                f"embedding model not found: {self.model_path}")
        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise NeuralEmbedderError(
                "transformers is required for neural reranking") from exc
        with self._lock:
            if self._model is not None:
                return
            self._tokenizer = AutoTokenizer.from_pretrained(
                str(self.model_path))
            self._model = AutoModel.from_pretrained(
                str(self.model_path)).to(self.device)
            self._model.eval()

    def embed(self, texts: list[str]) -> torch.Tensor:
        if self._encoder is not None:
            vectors = self._encoder(texts)
            if not isinstance(vectors, torch.Tensor):
                vectors = torch.tensor(vectors, dtype=torch.float32)
            return F.normalize(vectors.float(), p=2, dim=1)
        self._load()
        rows = []
        with self._lock, torch.inference_mode():
            for start in range(0, len(texts), self.batch_size):
                batch = texts[start:start + self.batch_size]
                encoded = self._tokenizer(
                    batch, padding=True, truncation=True,
                    max_length=self.max_length, return_tensors="pt")
                encoded = {key: value.to(self.device)
                           for key, value in encoded.items()}
                output = self._model(**encoded)
                mask = encoded["attention_mask"].unsqueeze(-1)
                summed = (output.last_hidden_state * mask).sum(dim=1)
                counts = mask.sum(dim=1).clamp(min=1)
                rows.append(F.normalize(summed / counts, p=2, dim=1))
        return torch.cat(rows, dim=0)

    def rerank(self, query: str, hits: list[dict], k: int = 5) -> list[dict]:
        if not hits:
            return []
        if all("neural_score" in hit for hit in hits):
            scores = [float(hit["neural_score"]) for hit in hits]
        else:
            texts = [
                f"{hit.get('title', '')} {hit.get('section', '')} "
                f"{hit.get('chunk', '')}".strip()
                for hit in hits
            ]
            query_vector = self.embed([str(query or "")])
            doc_vectors = self.embed(texts)
            scores = (doc_vectors @ query_vector.T).flatten().tolist()
        ranked = sorted(enumerate(hits),
                        key=lambda item: (-scores[item[0]], item[0]))
        output = []
        for index, hit in ranked[:k]:
            item = dict(hit)
            item["neural_score"] = float(scores[index])
            output.append(item)
        return output
