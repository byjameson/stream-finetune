"""Seed a behavior-focused queue: exact format skills + short stories."""
from __future__ import annotations

import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
QUEUE = ROOT / "data/learning_queue_format.jsonl"
NUMBERS = ["one", "two", "three", "four", "five", "six", "seven",
           "eight", "nine", "ten"]
COLORS = ["red", "green", "blue", "yellow", "black", "white", "orange"]
FRUITS = ["apple", "banana", "orange", "pear", "grape", "peach"]
ANIMALS = ["cat", "dog", "fox", "bear", "rabbit", "bird"]
NAMES = ["Mia", "Leo", "Nora", "Sam", "Ada", "Jack", "Iris", "Cleo"]
SENTENCES = [
    "The cat sat on the mat.",
    "Mia likes green tea.",
    "The robot found a small map.",
    "We walked to the park.",
    "The book is on the desk.",
    "Leo is reading a story.",
    "The dog ran in the garden.",
    "Nora has a blue kite.",
]
POLITE = ["open the window", "close the door", "turn on the light",
          "pass the salt", "call me later", "send the file",
          "turn down the music", "bring some water"]
GREETINGS = ["hello", "hi", "hey", "good morning", "good afternoon",
             "good evening", "good night", "welcome back", "i am back",
             "see you later"]


def main():
    rng = random.Random(5)
    pairs: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()

    def add(user: str, assistant: str):
        key = (user.strip(), assistant.strip())
        if key not in seen:
            seen.add(key)
            pairs.append(key)

    for n in range(1, 11):
        text = ", ".join(NUMBERS[:n]) + "."
        for prompt in (f"count to {n}", f"count from one to {n}",
                       f"say the numbers from one to {n}"):
            add(prompt, text)

    for greeting in GREETINGS:
        add(f"be very brief: {greeting}", "Hi.")

    for action in POLITE:
        add(f"make it polite: {action}", f"Could you {action}, please?")

    for sentence in SENTENCES:
        add(f"summarize in one sentence: {sentence}", sentence)
        add(f"repeat after me: {sentence}", sentence)

    for _ in range(80):
        c1, c2 = rng.sample(COLORS, 2)
        add("give me two colors", f"{c1.capitalize()} and {c2}.")
        add("name two colors", f"{c1.capitalize()} and {c2}.")

    for _ in range(50):
        f1, f2, f3 = rng.sample(FRUITS, 3)
        add("give me three fruits",
            f"{f1.capitalize()}, {f2}, and {f3}.")
        a1, a2 = rng.sample(ANIMALS, 2)
        add("give me two animals", f"{a1.capitalize()} and {a2}.")

    for _ in range(80):
        name = rng.choice(NAMES)
        animal = rng.choice(ANIMALS)
        place = rng.choice(["forest", "garden", "park", "beach", "hill"])
        thing = rng.choice(["a red ball", "a small map", "a blue kite",
                            "a yellow flower", "a paper boat", "a shiny stone"])
        add("tell me a short story",
            f"{name} the {animal} found {thing} near the {place}. "
            f"{name} shared it with a friend, and they played until sunset.")

    QUEUE.parent.mkdir(parents=True, exist_ok=True)
    with QUEUE.open("w", encoding="utf-8") as handle:
        for user, assistant in pairs:
            handle.write(json.dumps({"user": user, "assistant": assistant,
                                     "ts": 0}, ensure_ascii=False) + "\n")
    print(json.dumps({"queued": len(pairs), "path": str(QUEUE)}, indent=2))


if __name__ == "__main__":
    main()
