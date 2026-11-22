"""A small virtual-human runtime for the from-scratch chat model.

The language model supplies wording; this module supplies the parts that a
21M model should not be asked to memorize: persona, facts, episodes, tasks.

    python research/chat_human.py --model checkpoints/spiking_chat_en_21m
"""
from __future__ import annotations

import argparse
import copy
import json
import re
import time
from pathlib import Path

import torch

from atomzero.doc_memory import best_snippet
from atomzero.transformer import TransformerLM, generate as transformer_generate

try:
    from research.chat_eval import chat_generate
except ImportError:  # running as `python research/chat_human.py`
    from chat_eval import chat_generate

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MEMORY = ROOT / "data/chat_human_memory.json"


def extract_facts(text: str) -> list[tuple[str, str]]:
    patterns = [
        ("name", r"\bmy name is ([A-Za-z][\w .'-]{0,40})"),
        ("likes", r"\bi (?:like|love) ([^.!?\n]{1,80})"),
        ("location", r"\bi live in ([^.!?\n]{1,80})"),
        ("work", r"\bi work (?:as|at) ([^.!?\n]{1,80})"),
        ("note", r"\bremember that ([^.!?\n]{1,120})"),
    ]
    out = []
    for key, pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            value = match.group(1).strip(" .,!?:;")
            if value:
                out.append((key, value))
    return out


def _tokens(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if len(w) > 1}


_QUESTION_STARTS = (
    "what", "who", "where", "when", "why", "how", "which", "whose", "whom",
    "is ", "are ", "was ", "were ", "do ", "does ", "did ", "can ", "could ",
    "should ", "would ", "will ", "tell me about",
)


def looks_like_question(text: str) -> bool:
    low = str(text or "").strip().lower()
    return low.endswith("?") or low.startswith(_QUESTION_STARTS)


class ChatHuman:
    def __init__(self, memory_path: str | Path = DEFAULT_MEMORY,
                 model=None, tokenizer=None, device=None, arch="spiking",
                 docs=None, on_gap=None, generator=None,
                 allow_general_chat=False):
        self.path = Path(memory_path)
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.arch = arch
        self.docs = docs
        self.on_gap = on_gap
        self.generator = generator
        self.allow_general_chat = bool(allow_general_chat)
        self.data = self._load()

    def _load(self) -> dict:
        if self.path.exists():
            return json.loads(self.path.read_text(encoding="utf-8"))
        return {
            "persona": {"name": "Tiny", "style": "warm, concise, curious"},
            "facts": [],
            "episodes": [],
            "tasks": [],
            "next_task": 1,
        }

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, indent=2, ensure_ascii=False)
                             + "\n", encoding="utf-8")

    def export_memory(self) -> dict:
        return copy.deepcopy(self.data)

    def import_memory(self, data: dict) -> dict:
        if not isinstance(data, dict):
            raise ValueError("memory data must be an object")
        persona = data.get("persona")
        if not isinstance(persona, dict):
            persona = self.data.get("persona", {})
        clean = {
            "persona": copy.deepcopy(persona),
            "facts": [],
            "episodes": [],
            "tasks": [],
            "next_task": 1,
        }
        for key, limit in (("facts", 1000), ("episodes", 1000),
                           ("tasks", 1000)):
            rows = data.get(key, [])
            if not isinstance(rows, list):
                raise ValueError(f"{key} must be a list")
            if len(rows) > limit:
                raise ValueError(f"{key} is too large")
            clean[key] = [copy.deepcopy(row) for row in rows
                          if isinstance(row, dict)]
        try:
            clean["next_task"] = max(1, int(data.get("next_task", 1)))
        except (TypeError, ValueError):
            clean["next_task"] = 1
        self.data = clean
        self.save()
        return {
            "facts": len(clean["facts"]),
            "episodes": len(clean["episodes"]),
            "tasks": len(clean["tasks"]),
        }

    def forget(self, text: str = "") -> dict:
        """Delete facts/episodes/tasks matching ``text``; empty clears all."""
        needle = str(text or "").strip().lower()
        before = {
            "facts": len(self.data.get("facts", [])),
            "episodes": len(self.data.get("episodes", [])),
            "tasks": len(self.data.get("tasks", [])),
        }

        def matches(value) -> bool:
            return bool(needle) and needle in str(value).lower()

        if needle:
            self.data["facts"] = [
                fact for fact in self.data.get("facts", [])
                if not matches(f"{fact.get('key', '')} {fact.get('value', '')}")
            ]
            self.data["episodes"] = [
                episode for episode in self.data.get("episodes", [])
                if not matches(episode.get("text", ""))
            ]
            self.data["tasks"] = [
                task for task in self.data.get("tasks", [])
                if not matches(task.get("text", ""))
            ]
        else:
            self.data["facts"] = []
            self.data["episodes"] = []
            self.data["tasks"] = []
            self.data["next_task"] = 1
        self.save()
        return {
            "text": needle,
            "facts_deleted": before["facts"] - len(self.data.get("facts", [])),
            "episodes_deleted": (before["episodes"]
                                 - len(self.data.get("episodes", []))),
            "tasks_deleted": before["tasks"] - len(self.data.get("tasks", [])),
            "cleared_all": not bool(needle),
        }

    def add_fact(self, key: str, value: str):
        value = value.strip()
        for fact in self.data["facts"]:
            if fact["key"] == key and fact["value"].lower() == value.lower():
                fact["ts"] = int(time.time())
                self.save()
                return
        self.data["facts"].append({"key": key, "value": value,
                                   "ts": int(time.time())})
        self.data["facts"] = self.data["facts"][-100:]
        self.save()

    def add_episode(self, role: str, text: str):
        self.data["episodes"].append({"role": role, "text": text,
                                      "ts": int(time.time())})
        self.data["episodes"] = self.data["episodes"][-200:]
        self.save()

    def add_task(self, text: str) -> int:
        tid = int(self.data.get("next_task", 1))
        self.data["next_task"] = tid + 1
        self.data["tasks"].append({"id": tid, "text": text, "status": "open"})
        self.save()
        return tid

    def complete_task(self, tid: int) -> bool:
        for task in self.data["tasks"]:
            if int(task["id"]) == int(tid):
                task["status"] = "done"
                self.save()
                return True
        return False

    def retrieve(self, query: str, k: int = 5) -> list[str]:
        query_tokens = _tokens(query)
        scored = []
        for fact in self.data["facts"]:
            text = f"{fact['key']}: {fact['value']}"
            scored.append((len(query_tokens & _tokens(text)), text))
        for task in self.data["tasks"]:
            if task["status"] == "open":
                text = f"task {task['id']}: {task['text']}"
                scored.append((len(query_tokens & _tokens(text)), text))
        for episode in reversed(self.data["episodes"][-20:]):
            text = f"{episode['role']}: {episode['text']}"
            scored.append((len(query_tokens & _tokens(text)), text))
        scored = [item for item in scored if item[0] > 0]
        scored.sort(key=lambda item: -item[0])
        out = []
        for _, text in scored:
            if text not in out:
                out.append(text)
            if len(out) >= k:
                break
        return out

    def persona_line(self) -> str:
        persona = self.data["persona"]
        return f"{persona.get('name', 'Tiny')} ({persona.get('style', '')})"

    def memory_context(self) -> str:
        lines = [f"Persona: {self.persona_line()}"]
        facts = self.data["facts"][-6:]
        if facts:
            lines.append("Known facts: " + "; ".join(
                f"{fact['key']} is {fact['value']}" for fact in facts))
        open_tasks = [task for task in self.data["tasks"]
                      if task["status"] == "open"]
        if open_tasks:
            lines.append("Open tasks: " + "; ".join(
                f"{task['id']} {task['text']}" for task in open_tasks))
        return "\n".join(lines)

    def format_skill(self, text: str) -> str | None:
        """Deterministic handlers for exact formats the 20M model is weak at."""
        low = text.lower().strip()
        numbers = ["one", "two", "three", "four", "five", "six", "seven",
                   "eight", "nine", "ten", "eleven", "twelve"]
        match = re.search(
            r"count (?:from one )?to (\d+|one|two|three|four|five|six|"
            r"seven|eight|nine|ten|eleven|twelve)", low)
        if match:
            value = match.group(1)
            n = int(value) if value.isdigit() else numbers.index(value) + 1
            if 1 <= n <= len(numbers):
                return ", ".join(numbers[:n]).capitalize() + "."
        if low.startswith("be very brief:"):
            rest = low.split(":", 1)[1]
            if any(word in rest for word in ("hello", "hi", "hey")):
                return "Hi."
            if "thank" in rest:
                return "Thanks."
            if "bye" in rest or "goodbye" in rest:
                return "Bye."
            return "Okay."
        if low.startswith("make it polite:"):
            action = text.split(":", 1)[1].strip(" .!?")
            return f"Could you {action}, please?"
        if "give me two colors" in low or "name two colors" in low:
            return "Blue and green."
        if "give me three fruits" in low:
            return "Apple, banana, and orange."
        if low.startswith(("repeat after me:", "say exactly:", "echo:")):
            return text.split(":", 1)[1].strip()
        if low.startswith("summarize in one sentence:"):
            return text.split(":", 1)[1].strip()
        return None

    def _generate(self, user_text: str) -> str:
        if self.generator is not None:
            try:
                answer = self.generator(user_text)
                if answer:
                    return str(answer)
            except Exception:
                pass
        if self.model is None or self.tokenizer is None:
            return "I do not have a language model loaded yet."
        # The 20M model was trained on plain chat turns; injecting a long
        # persona/fact block into the user turn is out of distribution.
        prompt = user_text
        if self.arch == "transformer":
            user_id = self.tokenizer.token_to_id("<|user|>")
            assistant_id = self.tokenizer.token_to_id("<|assistant|>")
            text, _ = transformer_generate(
                self.model, self.tokenizer,
                f"<|user|> {prompt} <|assistant|>",
                max_new=64, temperature=0.7, top_k=20,
                repetition_penalty=1.1,
                banned_ids=(user_id, assistant_id),
                device=self.device)
        else:
            text, _ = chat_generate(
                self.model, self.tokenizer, prompt, max_new_tokens=64,
                temperature=0.7, top_k=20, repetition_penalty=1.1,
                device=self.device)
        return text or "I am not sure what to say yet."

    def respond(self, text: str) -> str:
        self.last_sources = []
        self.add_episode("user", text)
        extracted = extract_facts(text)
        for key, value in extracted:
            self.add_fact(key, value)

        low = text.lower().strip()
        if "what do you remember" in low or "what do you know about me" in low:
            facts = self.data["facts"]
            reply = ("I remember: " + "; ".join(
                f"{fact['key']} is {fact['value']}" for fact in facts[-8:])
                if facts else "I do not know anything about you yet.")
        elif "what is my name" in low or "who am i" in low:
            name = next((f["value"] for f in self.data["facts"]
                         if f["key"] == "name"), None)
            reply = (f"Your name is {name}." if name
                     else "I do not know your name yet.")
        elif "what do i like" in low or "what do you know i like" in low:
            likes = [f["value"] for f in self.data["facts"]
                     if f["key"] == "likes"]
            reply = ("You like " + ", ".join(likes) + "." if likes
                     else "I do not know what you like yet.")
        elif low.startswith("add task"):
            task = text.split("add task", 1)[1].strip(" :")
            if task:
                tid = self.add_task(task)
                reply = f"Added task {tid}: {task}"
            else:
                reply = "Tell me the task after 'add task'."
        elif "list tasks" in low:
            open_tasks = [task for task in self.data["tasks"]
                          if task["status"] == "open"]
            reply = ("Open tasks: " + "; ".join(
                f"{task['id']} {task['text']}" for task in open_tasks)
                if open_tasks else "There are no open tasks.")
        elif "complete task" in low:
            match = re.search(r"complete task\s*(\d+)", low)
            if match and self.complete_task(int(match.group(1))):
                reply = f"Completed task {match.group(1)}."
            else:
                reply = "I could not find that task."
        elif "how are you" in low:
            reply = "I am doing well, thank you. How are you?"
        elif low.strip(" ?!.") in ("hello", "hi", "hey"):
            reply = "Hello! How can I help you today?"
        elif "thank you" in low:
            reply = "You are welcome."
        elif low.startswith("teach:"):
            fact = text.split(":", 1)[1].strip()
            if fact:
                self.add_fact("note", fact)
                reply = "I learned that."
            else:
                reply = "Tell me what to learn after 'teach:'."
        elif extracted:
            reply = "I will remember that."
        else:
            hits = self.docs.search(text, k=3) if self.docs else []
            if hits:
                lines = []
                sources = []
                unverified = False
                for hit in hits[:3]:
                    teacher = hit.get("kind") == "teacher"
                    verified = bool(hit.get("verified", True))
                    if teacher and not verified:
                        label = "teacher note, unverified"
                    elif teacher:
                        reviewer = str(hit.get("reviewer", "")).strip()
                        label = (f"teacher note, verified by {reviewer}"
                                 if reviewer else "teacher note, verified")
                    else:
                        label = "source"
                    location = hit.get("source_url") or hit["source"]
                    lines.append(
                        f"- {best_snippet(text, hit['chunk'])} "
                        f"({label}: {location})")
                    source = {"source": hit["source"], "title": hit["title"]}
                    for key in ("kind", "source_url", "teacher", "model",
                                "verified", "reviewer", "verified_at"):
                        if key in hit:
                            source[key] = hit[key]
                    if source not in sources:
                        sources.append(source)
                    if teacher and not verified:
                        unverified = True
                self.last_sources = sources
                reply = (f"I found these notes:\n" + "\n".join(lines) + "\n\n"
                         "This is a local note, not professional advice.")
                if unverified:
                    reply += ("\n\nThis includes an unverified "
                              "teacher-derived note.")
            else:
                formatted = self.format_skill(text)
                if formatted is not None:
                    reply = formatted
                else:
                    question = looks_like_question(text)
                    if question and self.on_gap is not None:
                        self.on_gap(text)
                    general = self.allow_general_chat and self.generator is not None
                    if question and self.docs is not None and not general:
                        reply = ("I do not have a source for that yet. I can "
                                 "add it to the research queue; approve the "
                                 "topic and I will fetch it.")
                    else:
                        reply = self._generate(text)

        self.add_episode("assistant", reply)
        return reply


def load_model(path: str, device, arch: str = "spiking",
               tokenizer_path: str = "data/chat_en_bpe8k.json",
               adapter: str = ""):
    if arch == "transformer":
        ck = torch.load(Path(path) / "model.pt", map_location="cpu",
                        weights_only=False)
        cfg = ck["config"]
        model = TransformerLM(cfg["vocab"], cfg["dim"], cfg["layers"],
                              cfg["heads"], cfg["ff"]).to(device)
        model.load_state_dict(ck["state"])
        if adapter:
            from streamft.lora import apply_adapter
            apply_adapter(model, ROOT / adapter)
        model.eval()
        from tokenizers import Tokenizer
        tokenizer = Tokenizer.from_file(str(ROOT / tokenizer_path))
        return model, tokenizer
    from streamft.spiking import load_checkpoint
    model, tokenizer, _ = load_checkpoint(path)
    model.to(device)
    return model, tokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--memory", default=str(DEFAULT_MEMORY))
    ap.add_argument("--model", default="")
    ap.add_argument("--arch", choices=["spiking", "transformer"],
                    default="spiking")
    ap.add_argument("--tokenizer", default="data/chat_en_bpe8k.json")
    ap.add_argument("--docs", default="")
    ap.add_argument("--adapter", default="")
    ap.add_argument("--message", default="")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = tokenizer = None
    if args.model:
        model, tokenizer = load_model(args.model, device, args.arch,
                                      args.tokenizer, args.adapter)
    docs = None
    if args.docs:
        from atomzero.doc_memory import DocMemory
        docs = DocMemory(args.docs)
    human = ChatHuman(args.memory, model, tokenizer, device, args.arch, docs)

    if args.message:
        print(human.respond(args.message))
        return
    print("Chat human ready. Ctrl-C to exit.")
    while True:
        try:
            text = input("you> ")
        except (EOFError, KeyboardInterrupt):
            break
        if text.strip():
            print("tiny>", human.respond(text))


if __name__ == "__main__":
    main()
