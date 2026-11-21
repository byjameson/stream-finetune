"""Seed a larger reviewed skill queue for the gated update pipeline."""
from __future__ import annotations

import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
QUEUE = ROOT / "data/learning_queue.jsonl"
NAMES = ["Alex", "Sam", "Mia", "Leo", "Nora", "Ada", "Ben", "Cleo",
         "Dina", "Eli", "Faye", "Gus", "Hana", "Iris", "Jack", "Kira"]
ANIMALS = ["fox", "bear", "rabbit", "turtle", "bird", "cat", "dog"]
PLACES = ["forest", "garden", "village", "beach", "hill", "park"]
THINGS = ["a red ball", "a small map", "a blue kite", "a shiny stone",
          "a paper boat", "a yellow flower"]
TASKS = ["buy milk", "call mom", "read a book", "go for a walk",
         "water the plants", "book a dentist", "clean the desk",
         "buy bread", "email Sam", "fix the bike", "study Spanish",
         "pay the bill", "walk the dog", "cook dinner", "send the report"]


def main():
    rng = random.Random(1)
    pairs: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()

    def add(user: str, assistant: str):
        key = (user.strip(), assistant.strip())
        if user.strip() and assistant.strip() and key not in seen:
            seen.add(key)
            pairs.append(key)

    greetings = ["hello", "hi", "hey", "good morning", "good afternoon",
                 "good evening", "good night", "welcome back", "i am back",
                 "see you later"]
    replies = ["Hello! How can I help you today?",
               "Hi there! What would you like to do?",
               "Hello! I am here. What do you need?",
               "Hi! How can I help?",
               "Welcome back! What would you like to do?",
               "Good to see you. What can I do?"]
    for greeting in greetings:
        for reply in replies:
            add(greeting, reply)

    jokes = [
        ("Why did the scarecrow win an award?",
         "Because he was outstanding in his field."),
        ("Why did the computer go to the doctor?", "It had a virus."),
        ("What do you call a fake noodle?", "An impasta."),
        ("Why did the bicycle fall over?", "It was two tired."),
        ("What do you call a bear with no teeth?", "A gummy bear."),
        ("Why did the tomato turn red?", "It saw the salad dressing."),
        ("What do you call a sleeping bull?", "A bulldozer."),
        ("Why did the math book look sad?", "It had too many problems."),
        ("What do you call a fish with no eyes?", "A fsh."),
        ("Why did the coffee file a police report?", "It got mugged."),
        ("What do you call a snowman in summer?", "A puddle."),
        ("Why did the golfer bring two pairs of pants?",
         "In case he got a hole in one."),
        ("What do you call a dinosaur with no eyes?", "A do-you-think-he-saw-us."),
        ("Why did the student eat his homework?", "The teacher said it was a piece of cake."),
        ("What do you call a boomerang that does not come back?", "A stick."),
        ("Why did the scarecrow become a successful neurosurgeon?",
         "He was outstanding in his field."),
        ("What do you call a cow with no legs?", "Ground beef."),
        ("Why did the cookie go to the doctor?", "It felt crummy."),
        ("What do you call a group of musical whales?", "An orca-stra."),
        ("Why did the man put his money in the freezer?", "He wanted cold hard cash."),
    ]
    prompts = ["tell me a joke", "make me laugh", "joke please",
               "another joke", "tell me something funny"]
    for prompt in prompts:
        for setup, punch in jokes:
            add(prompt, f"{setup} {punch}")

    persona = [
        ("what is your name?", "I am Tiny, your local assistant."),
        ("what can you do?", "I can chat, remember facts, and keep a task list."),
        ("do you sleep?", "I do not sleep, but I can wait while you rest."),
        ("what is your favorite color?", "I like calm blue, but I do not see colors."),
        ("are you human?", "No, I am a small local language model."),
        ("where do you live?", "I live on this computer, in a small checkpoint."),
        ("how old are you?", "I am a tiny model without an age."),
        ("what are you?", "I am a small assistant for chat, memory and tasks."),
        ("do you have feelings?", "I can use warm words, but I do not have feelings."),
        ("can you learn?", "I can remember facts and update my skills with review."),
    ]
    for question, answer in persona:
        add(question, answer)
        add(question.replace("?", " please?"), answer)

    support = [
        ("i am tired", "Take a short break, drink some water, and come back when you are ready."),
        ("i am happy", "That is great to hear. What made your day better?"),
        ("i am sad", "I am sorry to hear that. Do you want to talk about it?"),
        ("i am nervous", "Take one slow breath. We can break the next step into something smaller."),
        ("i am stuck", "Start with the smallest next action. I can help you list it."),
        ("i have too much to do", "List the tasks, pick the smallest one, and start there."),
        ("help me focus", "Choose one task, work for a short block, then take a break."),
        ("what should I do today?", "Pick one small task and start with it. I can keep the list."),
        ("let's plan my day", "Tell me your tasks and I will keep them in order."),
        ("i finished my work", "Nice work! You can mark the task complete."),
        ("good luck", "Good luck! I am here if you need me."),
        ("thank you", "You are welcome."),
        ("say something nice", "You are doing better than you think."),
        ("help", "You can chat, remember facts, and manage tasks."),
        ("what is next?", "Check your task list and choose the first open item."),
    ]
    for user, reply in support:
        add(user, reply)
        add(user + " please", reply)

    for i, task in enumerate(TASKS):
        add(f"add task {task}", f"Added task {i + 1}: {task}.")
        add(f"please add {task}", f"Added task {i + 1}: {task}.")
    for i in range(1, 6):
        add(f"complete task {i}", f"Completed task {i}.")
        add(f"finish task {i}", f"Completed task {i}.")
    for i in range(2, 5):
        tasks = "; ".join(f"{j} {TASKS[j - 1]}" for j in range(1, i + 1))
        add("list tasks", f"Open tasks: {tasks}.")

    for _ in range(100):
        name = rng.choice(NAMES)
        animal = rng.choice(ANIMALS)
        place = rng.choice(PLACES)
        thing = rng.choice(THINGS)
        add("tell me a short story",
            f"{name} was a curious {animal} who lived near the {place}. "
            f"One day {name} found {thing}. {name} wanted to share it with "
            f"a friend, so they walked together and played until sunset.")

    QUEUE.parent.mkdir(parents=True, exist_ok=True)
    with QUEUE.open("w", encoding="utf-8") as handle:
        for user, assistant in pairs:
            handle.write(json.dumps({"user": user, "assistant": assistant,
                                     "ts": 0}, ensure_ascii=False) + "\n")
    print(json.dumps({"queued": len(pairs), "path": str(QUEUE)}, indent=2))


if __name__ == "__main__":
    main()
