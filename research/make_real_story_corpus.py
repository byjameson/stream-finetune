"""Build a real public-domain story corpus from GITenberg texts.

Train: Alice's Adventures in Wonderland (200 passages).
Valid: The Wonderful Wizard of Oz (100 passages).
Passages are contiguous 100-word windows after Gutenberg header/footer cleanup.

    python research/make_real_story_corpus.py
"""
from __future__ import annotations

import re
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    "alice": (
        "https://raw.githubusercontent.com/GITenberg/"
        "Alice-s-Adventures-in-Wonderland_11/master/11.txt",
        ROOT / "data/real_stories_alice_raw.txt",
    ),
    "oz": (
        "https://raw.githubusercontent.com/GITenberg/"
        "The-Wonderful-Wizard-of-Oz_55/master/55.txt",
        ROOT / "data/real_stories_oz_raw.txt",
    ),
}
WORD = 100


def fetch(url: str, cache: Path) -> str:
    if cache.exists():
        return cache.read_text(encoding="utf-8", errors="replace")
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    text = r.text
    cache.write_text(text, encoding="utf-8")
    return text


def clean(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    start = text.find("*** START OF")
    end = text.find("*** END OF")
    if start >= 0:
        text = text[text.find("\n", start) + 1:]
    if end >= 0:
        text = text[:text.find("\n", end)] if end < len(text) else text
    lines = []
    for line in text.split("\n"):
        s = line.strip()
        if not s:
            lines.append("")
            continue
        low = s.lower()
        if (
            "project gutenberg" in low
            or "produced by" in low
            or s.startswith("[Illustration")
            or s.startswith("CHAPTER")
            or (s.isupper() and len(s) < 60)
        ):
            continue
        lines.append(s)
    return "\n".join(lines)


def passages(text: str, count: int) -> list[str]:
    words = re.findall(r"[A-Za-z][A-Za-z'’-]*|[.,!?;:]", text)
    out = []
    i = 0
    while i + WORD <= len(words) and len(out) < count:
        chunk = words[i:i + WORD]
        alpha = sum(w[0].isalpha() for w in chunk)
        if alpha >= WORD * 0.7:
            out.append(" ".join(chunk))
        i += WORD
    return out


def main():
    alice = clean(fetch(*SOURCES["alice"]))
    oz = clean(fetch(*SOURCES["oz"]))
    train = passages(alice, 205)[5:205]
    valid = passages(oz, 110)[10:110]
    assert len(train) == 200 and len(valid) == 100
    (ROOT / "data/real_stories_train.txt").write_text(
        "\n\n".join(train) + "\n", encoding="utf-8")
    (ROOT / "data/real_stories_valid.txt").write_text(
        "\n\n".join(valid) + "\n", encoding="utf-8")
    print("alice words", len(re.findall(r"[A-Za-z][A-Za-z'’-]*", alice)))
    print("oz words", len(re.findall(r"[A-Za-z][A-Za-z'’-]*", oz)))
    print("train passages", len(train), "valid passages", len(valid))
    print("train preview:", train[0][:180])
    print("valid preview:", valid[0][:180])


if __name__ == "__main__":
    main()
