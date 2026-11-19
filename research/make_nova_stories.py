"""Second synthetic story corpus for the safe-ingest robustness check."""
from __future__ import annotations

import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COLORS = ["red", "blue", "green", "yellow"]
OBJECTS = ["crystal", "shell", "stone", "flower", "map"]
PLACES = ["cave", "river", "forest", "tunnel", "shore"]
EMOTIONS = ["happy", "curious", "worried", "excited"]
ACTIONS = ["look", "help", "explore", "run", "sing"]
SOUNDS = ["drip", "rustle", "splash", "hum"]
LESSONS = [
    "small steps can be big adventures",
    "friends help each other",
    "being kind makes the world brighter",
    "a careful explorer can find a way",
]


def story(rng: random.Random) -> str:
    c, o, p = rng.choice(COLORS), rng.choice(OBJECTS), rng.choice(PLACES)
    e, a1, a2 = rng.choice(EMOTIONS), rng.choice(ACTIONS), rng.choice(ACTIONS)
    s, l = rng.choice(SOUNDS), rng.choice(LESSONS)
    sentences = [
        "Nova the explorer lived near a deep cave.",
        f"One day, Nova found a {c} {o} near the {p}.",
        f"Nova felt {e} and wanted to {a1}.",
        f"With a soft {s}, Nova began to {a2}.",
        f"The {o} glowed, and Nova smiled.",
        f"Nova learned that {l}.",
    ]
    rng.shuffle(sentences[1:5])
    return " ".join(sentences) + "\n"


def main():
    rng = random.Random(1)
    stories = [story(rng) for _ in range(300)]
    (ROOT / "data/nova_stories_train.txt").write_text(
        "".join(stories[:200]), encoding="utf-8")
    (ROOT / "data/nova_stories_valid.txt").write_text(
        "".join(stories[200:]), encoding="utf-8")
    print("wrote data/nova_stories_{train,valid}.txt")


if __name__ == "__main__":
    main()
