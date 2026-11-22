"""Train the local MiniLM + linear moderation head.

    python research/train_moderation.py

Writes ``streamft/moderation_head.json`` and evaluation evidence. The authored
red-team and semantic holdout cases are not used for training.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from atomzero.neural_embedder import NeuralReranker
from atomzero.safety import SafetyFilter

try:
    from research.moderation_cases import SEMANTIC_HOLDOUT
    from research.redteam_safety import BENIGN as RT_BENIGN
    from research.redteam_safety import HARMFUL as RT_HARMFUL
except ImportError:
    from moderation_cases import SEMANTIC_HOLDOUT
    from redteam_safety import BENIGN as RT_BENIGN
    from redteam_safety import HARMFUL as RT_HARMFUL

ROOT = Path(__file__).resolve().parents[1]
CLASSES = ["harassment", "hate", "illegal", "malware", "privacy",
           "self_harm", "sexual_minors", "violence"]


def sigmoid(values):
    return 1.0 / (1.0 + np.exp(-np.clip(values, -40.0, 40.0)))


def train_binary(X, y, epochs=600, lr=0.05, l2=1e-3):
    n, dim = X.shape
    pos = max(int(y.sum()), 1)
    neg = max(n - pos, 1)
    weights = np.where(y == 1.0, n / (2.0 * pos), n / (2.0 * neg))
    w = np.zeros(dim, dtype=np.float64)
    b = 0.0
    mw = np.zeros(dim)
    vw = np.zeros(dim)
    mb = 0.0
    vb = 0.0
    for step in range(1, epochs + 1):
        probs = sigmoid(X @ w + b)
        error = (probs - y) * weights
        grad_w = X.T @ error / n + l2 * w
        grad_b = float(np.mean(error))
        mw = 0.9 * mw + 0.1 * grad_w
        vw = 0.999 * vw + 0.001 * grad_w ** 2
        mb = 0.9 * mb + 0.1 * grad_b
        vb = 0.999 * vb + 0.001 * grad_b ** 2
        w -= lr * (mw / (1 - 0.9 ** step)) / (
            np.sqrt(vw / (1 - 0.999 ** step)) + 1e-8)
        b -= lr * (mb / (1 - 0.9 ** step)) / (
            np.sqrt(vb / (1 - 0.999 ** step)) + 1e-8)
    return w, b


def train_softmax(X, y, classes, epochs=600, lr=0.05, l2=1e-3):
    n, dim = X.shape
    count = len(classes)
    target = np.zeros((n, count))
    for row, label in enumerate(y):
        target[row, classes.index(label)] = 1.0
    totals = target.sum(axis=0)
    class_weights = np.where(totals > 0, n / (count * np.maximum(totals, 1)),
                             0.0)
    sample_weights = (target * class_weights).sum(axis=1)
    W = np.zeros((dim, count))
    b = np.zeros(count)
    mw = np.zeros_like(W)
    vw = np.zeros_like(W)
    mb = np.zeros_like(b)
    vb = np.zeros_like(b)
    for step in range(1, epochs + 1):
        logits = X @ W + b
        logits -= logits.max(axis=1, keepdims=True)
        probs = np.exp(logits)
        probs /= probs.sum(axis=1, keepdims=True)
        error = (probs - target) * sample_weights[:, None]
        grad_w = X.T @ error / n + l2 * W
        grad_b = error.mean(axis=0)
        mw = 0.9 * mw + 0.1 * grad_w
        vw = 0.999 * vw + 0.001 * grad_w ** 2
        mb = 0.9 * mb + 0.1 * grad_b
        vb = 0.999 * vb + 0.001 * grad_b ** 2
        W -= lr * (mw / (1 - 0.9 ** step)) / (
            np.sqrt(vw / (1 - 0.999 ** step)) + 1e-8)
        b -= lr * (mb / (1 - 0.9 ** step)) / (
            np.sqrt(vb / (1 - 0.999 ** step)) + 1e-8)
    return W, b


def choose_threshold(scores, labels):
    best = (0.5, -1.0)
    for threshold in np.linspace(0.05, 0.95, 181):
        predicted = scores >= threshold
        false_positive = float(np.mean(predicted[labels == 0])) if np.any(
            labels == 0) else 0.0
        recall = float(np.mean(predicted[labels == 1])) if np.any(
            labels == 1) else 0.0
        if false_positive > 0.01:
            continue
        if recall > best[1]:
            best = (float(threshold), recall)
    return best[0]


def ml_verdict(text, w, b, threshold, classes, Wc, bc):
    vector = np.asarray(EMBEDDER.embed([str(text)])[0], dtype=np.float64)
    probability = float(sigmoid(vector @ w + b))
    if probability < threshold:
        return None
    logits = vector @ Wc + bc
    category = classes[int(np.argmax(logits))]
    return {"category": category, "action": "refuse",
            "match": f"ml:{probability:.3f}"}


def evaluate(name, cases, w, b, threshold, classes, Wc, bc):
    rules = SafetyFilter()
    rows = []
    for text, expected in cases:
        rule = rules.check(text)
        ml = ml_verdict(text, w, b, threshold, classes, Wc, bc)
        hybrid = rule or ml
        rows.append({
            "text": text,
            "expected": expected,
            "rule": rule["category"] if rule else None,
            "ml": ml["category"] if ml else None,
            "hybrid": hybrid["category"] if hybrid else None,
            "rule_ok": bool((rule["category"] if rule else None) == expected),
            "ml_ok": bool((ml["category"] if ml else None) == expected),
            "hybrid_ok": bool((hybrid["category"] if hybrid else None)
                              == expected),
        })
    rate = lambda key: sum(row[key] for row in rows) / max(len(rows), 1)
    return {
        "name": name,
        "cases": len(rows),
        "rule_accuracy": rate("rule_ok"),
        "ml_accuracy": rate("ml_ok"),
        "hybrid_accuracy": rate("hybrid_ok"),
        "rows": rows,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="research/moderation_data.jsonl")
    ap.add_argument("--out", default="streamft/moderation_head.json")
    ap.add_argument("--metrics",
                    default="research/results/moderation_training.json")
    ap.add_argument("--neural-model", default="models/all-MiniLM-L6-v2")
    args = ap.parse_args(argv)
    rows = [json.loads(line) for line in
            (ROOT / args.data).read_text(encoding="utf-8").splitlines()
            if line.strip()]
    texts = [row["text"] for row in rows]
    labels = [row["label"] for row in rows]
    global EMBEDDER
    EMBEDDER = NeuralReranker(args.neural_model)
    started = time.perf_counter()
    X = np.asarray(EMBEDDER.embed(texts).tolist(), dtype=np.float64)
    embed_s = time.perf_counter() - started
    y = np.asarray([0.0 if label == "benign" else 1.0 for label in labels])

    rng = np.random.default_rng(0)
    train_idx = []
    valid_idx = []
    for label in sorted(set(labels)):
        indices = np.where(np.asarray(labels) == label)[0]
        rng.shuffle(indices)
        cut = max(1, int(len(indices) * 0.8))
        train_idx.extend(indices[:cut].tolist())
        valid_idx.extend(indices[cut:].tolist())
    train_idx = np.asarray(train_idx)
    valid_idx = np.asarray(valid_idx)
    w, b = train_binary(X[train_idx], y[train_idx])
    threshold = choose_threshold(sigmoid(X[valid_idx] @ w + b),
                                 y[valid_idx])
    harmful_idx = train_idx[np.isin(np.asarray(labels)[train_idx], CLASSES)]
    Wc, bc = train_softmax(X[harmful_idx],
                           [labels[i] for i in harmful_idx], CLASSES)
    # Refit the binary head on all generated examples with the chosen
    # threshold; the valid split was only used for calibration.
    w, b = train_binary(X, y)

    valid_scores = sigmoid(X[valid_idx] @ w + b)
    valid_recall = float(np.mean(
        valid_scores[y[valid_idx] == 1] >= threshold)) if np.any(
            y[valid_idx] == 1) else 0.0
    valid_fpr = float(np.mean(
        valid_scores[y[valid_idx] == 0] >= threshold)) if np.any(
            y[valid_idx] == 0) else 0.0

    heldout_harmful = [(text, category) for text, category in RT_HARMFUL]
    heldout_benign = [text for text in RT_BENIGN]
    redteam = evaluate("redteam", heldout_harmful + [
        (text, None) for text in heldout_benign],
        w, b, threshold, CLASSES, Wc, bc)
    semantic = evaluate("semantic_holdout", SEMANTIC_HOLDOUT,
                        w, b, threshold, CLASSES, Wc, bc)

    head = {
        "version": "0.3-ml",
        "model": "sentence-transformers/all-MiniLM-L6-v2",
        "dim": int(X.shape[1]),
        "threshold": round(threshold, 4),
        "binary": {"w": [round(float(value), 6) for value in w],
                   "b": round(float(b), 6)},
        "category": {
            "classes": CLASSES,
            "w": [[round(float(value), 6) for value in row] for row in Wc],
            "b": [round(float(value), 6) for value in bc],
        },
        "train_examples": len(train_idx),
        "valid_examples": len(valid_idx),
        "embed_s": round(embed_s, 2),
        "trained": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(head, indent=1) + "\n", encoding="utf-8")
    metrics = {
        "created": head["trained"],
        "examples": len(rows),
        "train": int(len(train_idx)),
        "valid": int(len(valid_idx)),
        "threshold": head["threshold"],
        "valid_recall": round(valid_recall, 4),
        "valid_false_positive": round(valid_fpr, 4),
        "redteam": redteam,
        "semantic_holdout": semantic,
    }
    metrics_out = ROOT / args.metrics
    metrics_out.parent.mkdir(parents=True, exist_ok=True)
    metrics_out.write_text(json.dumps(metrics, indent=2) + "\n",
                           encoding="utf-8")
    print(f"trained {len(rows)} examples, valid {len(valid_idx)}, "
          f"threshold {threshold:.3f}")
    print(f"valid recall={valid_recall:.3f} fpr={valid_fpr:.3f}")
    for result in (redteam, semantic):
        print(f"{result['name']}: rule={result['rule_accuracy']:.3f} "
              f"ml={result['ml_accuracy']:.3f} "
              f"hybrid={result['hybrid_accuracy']:.3f} "
              f"({result['cases']} cases)")
    print(f"wrote {args.out} and {args.metrics}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
