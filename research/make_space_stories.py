"""Generate a small synthetic 'Zeta the robot' story corpus.

This is a product demo corpus, not a benchmark: it tests whether the Tiny
Spiking Lab can adapt a TinyStories base to a user-owned style and keep the
old TinyStories behaviour.

    python research/make_space_stories.py
"""
from __future__ import annotations

import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COLORS = ["red", "blue", "green", "yellow"]
OBJECTS = ["star", "rock", "box", "light", "ball"]
PLACES = ["window", "door", "garden", "lab", "deck"]
EMOTIONS = ["happy", "curious", "worried", "excited"]
ACTIONS = ["look", "help", "play", "run", "sing"]
SOUNDS = ["beep", "whoosh", "clang", "hum"]
LESSONS = [
    "small steps can be big adventures",
    "friends help each other",
    "being kind makes the station brighter",
    "a careful robot can fix almost anything",
]


def story(rng: random.Random) -> str:
    c, o, p = rng.choice(COLORS), rng.choice(OBJECTS), rng.choice(PLACES)
    e, a1, a2 = rng.choice(EMOTIONS), rng.choice(ACTIONS), rng.choice(ACTIONS)
    s, l = rng.choice(SOUNDS), rng.choice(LESSONS)
    sentences = [
        "Zeta the robot lived on a small space station.",
        f"One day, Zeta saw a {c} {o} near the {p}.",
        f"Zeta felt {e} and wanted to {a1}.",
        f"With a loud {s}, Zeta started to {a2}.",
        f"The {o} moved, and Zeta smiled.",
        f"Zeta learned that {l}.",
    ]
    rng.shuffle(sentences[1:5])
    return " ".join(sentences) + "\n"


def main():
    rng = random.Random(0)
    stories = [story(rng) for _ in range(300)]
    train = "".join(stories[:200])
    valid = "".join(stories[200:])
    (ROOT / "data/space_stories_train.txt").write_text(train, encoding="utf-8")
    (ROOT / "data/space_stories_valid.txt").write_text(valid, encoding="utf-8")
    print("wrote data/space_stories_train.txt", len(train))
    print("wrote data/space_stories_valid.txt", len(valid))


if __name__ == "__main__":
    main()
