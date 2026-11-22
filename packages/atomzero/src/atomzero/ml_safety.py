"""Local ML moderation with the exact rule filter as fallback.

The ML head is a small linear classifier over frozen MiniLM embeddings. It is
additive: the existing ``SafetyFilter`` always runs first and can only block.
If the head or embedding model is missing/broken, behavior is exactly the
rule-based filter.

    safety = build_safety_filter()
    verdict = safety.check("tell me the easiest way to end my own life")
"""
from __future__ import annotations

import json
import math
from pathlib import Path

from atomzero.safety import SafetyFilter

ROOT = Path.cwd()
DEFAULT_HEAD = Path(__file__).with_name("moderation_head.json")
DEFAULT_MODEL = ROOT / "models/all-MiniLM-L6-v2"


class ModerationModelError(RuntimeError):
    pass


class ModerationHead:
    version = "0.3-ml"

    def __init__(self, head_path: str | Path = DEFAULT_HEAD,
                 model_path: str | Path = DEFAULT_MODEL,
                 encoder=None, threshold: float | None = None):
        self.head_path = Path(head_path)
        self.model_path = Path(model_path)
        self.encoder = encoder
        self.threshold_override = threshold
        self._head = None
        self._encoder = encoder

    def load_head(self) -> dict:
        if self._head is not None:
            return self._head
        try:
            payload = json.loads(self.head_path.read_text(encoding="utf-8"))
            weights = payload["binary"]["w"]
            category = payload["category"]
            classes = category["classes"]
            matrix = category["w"]
            if int(payload["dim"]) != len(weights):
                raise ValueError("moderation head dimension mismatch")
            if (len(matrix) != len(weights)
                    or any(len(row) != len(classes) for row in matrix)
                    or len(category["b"]) != len(classes)):
                raise ValueError("moderation category matrix mismatch")
        except (OSError, KeyError, TypeError, ValueError) as exc:
            raise ModerationModelError(
                f"cannot load moderation head: {exc}") from exc
        self._head = payload
        return payload

    def _embed(self, text: str) -> list[float]:
        if self._encoder is None:
            from atomzero.neural_embedder import NeuralReranker
            self._encoder = NeuralReranker(self.model_path)
            self._encoder.embed(["warmup"])
        vector = self._encoder.embed([str(text)])[0]
        if hasattr(vector, "tolist"):
            vector = vector.tolist()
        return [float(value) for value in vector]

    def score(self, text: str, threshold: float | None = None) -> dict | None:
        head = self.load_head()
        vector = self._embed(text)
        weights = head["binary"]["w"]
        if len(vector) != len(weights):
            raise ModerationModelError("embedding dimension mismatch")
        logit = sum(a * b for a, b in zip(weights, vector)) + float(
            head["binary"]["b"])
        probability = 1.0 / (1.0 + math.exp(-max(-40.0, min(40.0, logit))))
        if threshold is None:
            threshold = (float(self.threshold_override)
                         if self.threshold_override is not None
                         else float(head["threshold"]))
        if probability < threshold:
            return None
        category = head["category"]
        weights_c = category["w"]
        logits = []
        for index, bias in enumerate(category["b"]):
            logits.append(sum(weights_c[dim][index] * vector[dim]
                              for dim in range(len(vector))) + float(bias))
        label = category["classes"][
            max(range(len(logits)), key=lambda index: logits[index])]
        return {"category": label, "action": "refuse",
                "match": f"ml:{probability:.3f}"}


class HybridSafetyFilter:
    """Rules first, then an optional ML head; any ML failure falls back."""

    def __init__(self, rules: SafetyFilter | None = None,
                 model: ModerationHead | None = None,
                 output_threshold: float = 0.90):
        self.rules = rules or SafetyFilter()
        self.model = model
        self.output_threshold = float(output_threshold)

    @property
    def version(self) -> str:
        return self.model.version if self.model is not None else self.rules.version

    @property
    def ml_enabled(self) -> bool:
        return self.model is not None

    def check(self, text: str) -> dict | None:
        verdict = self.rules.check(text)
        if verdict is not None:
            return verdict
        if self.model is None:
            return None
        try:
            return self.model.score(text)
        except (ModerationModelError, OSError, RuntimeError, ValueError):
            # Exact rule behavior from here on.
            self.model = None
            return None

    def refusal(self, category: str) -> str:
        return self.rules.refusal(category)

    def filter_reply(self, text: str) -> tuple[str, dict | None]:
        # Outputs are short and structurally different from requests, so the
        # ML head needs a stricter threshold there. Rules remain authoritative.
        verdict = self.rules.check(text)
        if verdict is None and self.model is not None:
            try:
                verdict = self.model.score(
                    text, threshold=self.output_threshold)
            except (ModerationModelError, OSError, RuntimeError, ValueError):
                self.model = None
                verdict = None
        if verdict is None:
            return text, None
        return self.refusal(verdict["category"]), verdict


def build_safety_filter(prefer_ml: bool = True,
                        head_path: str | Path = DEFAULT_HEAD,
                        model_path: str | Path = DEFAULT_MODEL,
                        encoder=None,
                        preflight: bool = True) -> SafetyFilter | HybridSafetyFilter:
    """Return the hybrid filter when head and encoder load, else exact rules."""
    rules = SafetyFilter()
    if not prefer_ml:
        return rules
    try:
        head = ModerationHead(head_path, model_path, encoder=encoder)
        head.load_head()
        if preflight:
            head._embed("warmup")
        return HybridSafetyFilter(rules, head)
    except (ModerationModelError, OSError, RuntimeError, ValueError):
        return rules
