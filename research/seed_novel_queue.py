"""Seed the queue with novel, learnable format/style tasks.

These are not facts: they are transformations the tiny model can learn from
short examples (echo, shorten, question, list, tone, one-word, story).
"""
from __future__ import annotations

import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
QUEUE = ROOT / "data/learning_queue.jsonl"
COLORS = ["red", "green", "blue", "yellow", "black", "white", "orange"]
FRUITS = ["apple", "banana", "orange", "pear", "grape", "peach"]
ANIMALS = ["cat", "dog", "fox", "bear", "rabbit", "bird"]
SENTENCES = [
    "The cat sat on the mat.",
    "Mia likes green tea.",
    "The robot found a small map.",
    "We walked to the park.",
    "The book is on the desk.",
    "Leo is reading a story.",
]


def main():
    rng = random.Random(3)
    pairs: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()

    def add(user: str, assistant: str):
        key = (user.strip(), assistant.strip())
        if key not in seen:
            seen.add(key)
            pairs.append(key)

    for text in SENTENCES:
        add(f"repeat after me: {text}", text)
        add(f"say exactly: {text}", text)
        add(f"echo: {text}", text)
        add(f"make it shorter: {text}", text.replace("The ", "").replace(" is ", " "))

    for text in SENTENCES:
        add(f"turn into a question: {text}",
            "Is it true that " + text[0].lower() + text[1:-1] + "?")
        add(f"make it polite: open the window",
            "Could you open the window, please?")
        add("make it polite: close the door",
            "Could you close the door, please?")

    for _ in range(40):
        color = rng.choice(COLORS)
        fruit = rng.choice(FRUITS)
        animal = rng.choice(ANIMALS)
        add("give me one color", color.capitalize() + ".")
        add("give me one fruit", fruit.capitalize() + ".")
        add("give me one animal", animal.capitalize() + ".")
        add("give me two colors",
            f"{rng.choice(COLORS).capitalize()} and {rng.choice(COLORS)}.")
        add("give me three fruits",
            f"{rng.choice(FRUITS).capitalize()}, {rng.choice(FRUITS)}, "
            f"and {rng.choice(FRUITS)}.")

    for n in range(1, 11):
        words = ["one", "two", "three", "four", "five", "six", "seven",
                 "eight", "nine", "ten"][:n]
        add(f"count from one to {n}", ", ".join(w.capitalize() for w in words) + ".")

    tones = [
        ("hello", "Hi.", "Hello.", "Hey there!", "Good day."),
        ("thank you", "Thanks.", "Thank you.", "Thanks so much!", "I appreciate it."),
        ("goodbye", "Bye.", "Goodbye.", "See you!", "Farewell."),
        ("help", "Help.", "How can I help?", "What do you need?", "May I assist you?"),
    ]
    for base, brief, normal, playful, formal in tones:
        add(f"be very brief: {base}", brief)
        add(f"answer normally: {base}", normal)
        add(f"be playful: {base}", playful)
        add(f"be formal: {base}", formal)

    for text in SENTENCES:
        add(f"summarize in one sentence: {text}", text)
        add(f"summarize in one word: {text}", text.split()[1].strip(".,"))

    for _ in range(40):
        name = rng.choice(["Mia", "Leo", "Nora", "Sam", "Ada", "Jack"])
        animal = rng.choice(ANIMALS)
        place = rng.choice(["forest", "garden", "park", "beach", "hill"])
        thing = rng.choice(["a red ball", "a small map", "a blue kite",
                            "a yellow flower", "a paper boat"])
        add("tell me a story about sharing",
            f"{name} the {animal} found {thing}. {name} shared it with a "
            f"friend near the {place}, and they played until sunset.")
        add("tell me a story about being brave",
            f"{name} the {animal} was afraid of the dark {place}. "
            f"{name} took one small step, then another, and found {thing}.")
        add("tell me a story about friendship",
            f"{name} the {animal} met a new friend near the {place}. "
            f"They shared {thing} and walked home together.")

    QUEUE.parent.mkdir(parents=True, exist_ok=True)
    with QUEUE.open("w", encoding="utf-8") as handle:
        for user, assistant in pairs:
            handle.write(json.dumps({"user": user, "assistant": assistant,
                                     "ts": 0}, ensure_ascii=False) + "\n")
    print(json.dumps({"queued": len(pairs), "path": str(QUEUE)}, indent=2))


if __name__ == "__main__":
    main()
