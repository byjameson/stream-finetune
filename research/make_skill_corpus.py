"""Build a targeted memory/task/skill dialogue corpus (fast formatter)."""
from __future__ import annotations

import json
import random
from pathlib import Path

import torch
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[1]
TOKENIZER = ROOT / "data/chat_en_bpe8k.json"
CLEAN_TRAIN = ROOT / "data/chat_en_clean_train.pt"
CLEAN_VALID = ROOT / "data/chat_en_clean_valid.pt"

NAMES = ["Alex", "Sam", "Mia", "Leo", "Nora", "Ada", "Ben", "Cleo",
         "Dina", "Eli", "Faye", "Gus", "Hana", "Iris", "Jack", "Kira",
         "Liam", "Maya", "Nico", "Olga", "Pia", "Quinn", "Rosa", "Sami",
         "Tara", "Uma", "Vera", "Wes", "Xena", "Yara", "Zane", "Ayla"]
LIKES = ["green tea", "coffee", "books", "music", "walking", "cooking",
         "painting", "gardening", "cycling", "photography", "chess",
         "jazz", "science fiction", "history", "cats", "dogs", "tea",
         "rainy days", "puzzles", "travel"]
TASKS = ["buy milk", "call mom", "read a book", "go for a walk",
         "send the report", "water the plants", "book a dentist",
         "clean the desk", "buy bread", "email Sam", "fix the bike",
         "study Spanish", "pay the bill", "walk the dog", "cook dinner"]
GREETINGS = ["hello", "hi", "hey", "good morning", "good evening"]
GREET_REPLIES = ["Hello! How can I help you today?",
                 "Hi there! What would you like to do?",
                 "Hello! I am here. What do you need?",
                 "Hi! How can I help?"]


def fmt(turns) -> str:
    parts = []
    for role, text in turns:
        tag = "<|user|>" if role == "user" else "<|assistant|>"
        parts.append(f"{tag} {text}")
    return " ".join(parts) + " <|end|>"


def story(rng: random.Random) -> str:
    name = rng.choice(NAMES)
    animal = rng.choice(["fox", "bear", "rabbit", "turtle", "bird"])
    place = rng.choice(["forest", "garden", "village", "beach", "hill"])
    thing = rng.choice(["a red ball", "a small map", "a blue kite",
                        "a shiny stone", "a paper boat"])
    return (f"{name} was a curious {animal} who lived near the {place}. "
            f"One day {name} found {thing}. {name} wanted to share it with "
            f"a friend, so they walked together and played until sunset.")


def make_docs(rng: random.Random, count: int) -> list[str]:
    docs = []
    for _ in range(count):
        kind = rng.randrange(8)
        name = rng.choice(NAMES)
        like = rng.choice(LIKES)
        task1 = rng.choice(TASKS)
        task2 = rng.choice(TASKS)
        if kind == 0:
            turns = [("user", rng.choice(GREETINGS)),
                     ("assistant", rng.choice(GREET_REPLIES))]
        elif kind == 1:
            turns = [
                ("user", "my name is " + name),
                ("assistant", f"Nice to meet you, {name}. I will remember your name."),
                ("user", "what is my name?"),
                ("assistant", f"Your name is {name}."),
            ]
        elif kind == 2:
            turns = [
                ("user", f"remember that I like {like}"),
                ("assistant", "I will remember that."),
                ("user", "what do I like?"),
                ("assistant", f"You like {like}."),
            ]
        elif kind == 3:
            turns = [
                ("user", "what can you do?"),
                ("assistant", "I can chat, remember facts, and keep a task list."),
                ("user", "thank you"),
                ("assistant", "You are welcome."),
            ]
        elif kind == 4:
            turns = [
                ("user", f"add task {task1}"),
                ("assistant", f"Added task 1: {task1}."),
                ("user", "list tasks"),
                ("assistant", f"Open task: 1 {task1}."),
            ]
        elif kind == 5:
            turns = [
                ("user", f"add task {task1}"),
                ("assistant", f"Added task 1: {task1}."),
                ("user", f"add task {task2}"),
                ("assistant", f"Added task 2: {task2}."),
                ("user", "list tasks"),
                ("assistant", f"Open tasks: 1 {task1}; 2 {task2}."),
                ("user", "complete task 1"),
                ("assistant", "Completed task 1."),
                ("user", "list tasks"),
                ("assistant", f"Open task: 2 {task2}."),
            ]
        elif kind == 6:
            turns = [
                ("user", "hello"),
                ("assistant", rng.choice(GREET_REPLIES)),
                ("user", "my name is " + name),
                ("assistant", f"Nice to meet you, {name}. I will remember your name."),
                ("user", f"remember that I like {like}"),
                ("assistant", "I will remember that."),
                ("user", f"add task {task1}"),
                ("assistant", f"Added task 1: {task1}."),
                ("user", "list tasks"),
                ("assistant", f"Open task: 1 {task1}."),
                ("user", "what do you remember?"),
                ("assistant", f"I remember your name is {name} and you like {like}."),
            ]
        else:
            turns = [
                ("user", "how are you?"),
                ("assistant", "I am doing well, thank you. How are you?"),
                ("user", "tell me a short story"),
                ("assistant", story(rng)),
            ]
        docs.append(fmt(turns))
    return docs


def pack_batch(tok: Tokenizer, docs: list[str]) -> torch.Tensor:
    ids = []
    for enc in tok.encode_batch(docs):
        ids.extend(enc.ids)
    return torch.tensor(ids, dtype=torch.int64)


def interleave(skill, clean, block: int = 4096,
               clean_per_skill: int = 1) -> torch.Tensor:
    parts = []
    for i in range(0, len(skill), block):
        parts.append(skill[i:i + block])
        for k in range(clean_per_skill):
            j = ((i + k * 104729) * 7919) % max(len(clean) - block, 1)
            parts.append(clean[j:j + block])
    return torch.cat(parts)


def main():
    rng = random.Random(0)
    docs = make_docs(rng, 60_000)
    split = int(len(docs) * 0.98)
    tok = Tokenizer.from_file(str(TOKENIZER))
    skill_train = pack_batch(tok, docs[:split])
    skill_valid = pack_batch(tok, docs[split:])
    clean_train = torch.load(CLEAN_TRAIN, weights_only=True)
    clean_valid = torch.load(CLEAN_VALID, weights_only=True)
    mix_train = interleave(skill_train, clean_train)
    mix_valid = interleave(skill_valid, clean_valid, block=1024)
    mix2_train = interleave(skill_train, clean_train, block=2048,
                            clean_per_skill=2)
    mix2_valid = interleave(skill_valid, clean_valid, block=1024,
                            clean_per_skill=2)
    outputs = {
        "data/skill_train.pt": skill_train,
        "data/skill_valid.pt": skill_valid,
        "data/skill_mix_train.pt": mix_train,
        "data/skill_mix_valid.pt": mix_valid,
        "data/skill_mix2_train.pt": mix2_train,
        "data/skill_mix2_valid.pt": mix2_valid,
    }
    for rel, tensor in outputs.items():
        torch.save(tensor, ROOT / rel)
        print(f"{rel}: {tensor.numel()} tokens", flush=True)
    meta = {
        "train_dialogues": split,
        "valid_dialogues": len(docs) - split,
        "skill_train_tokens": int(skill_train.numel()),
        "mix_train_tokens": int(mix_train.numel()),
        "mix2_train_tokens": int(mix2_train.numel()),
        "mix2": "2 clean blocks per 1 skill block",
        "skills": ["greeting", "remember_name", "remember_like",
                   "task_add", "task_list", "task_complete",
                   "small_talk", "short_story"],
        "mix": "1:1 interleave of skill and clean chat replay",
    }
    (ROOT / "data/skill_meta.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    (ROOT / "data/skill_sample.txt").write_text(
        "\n\n".join(docs[:20]) + "\n", encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
