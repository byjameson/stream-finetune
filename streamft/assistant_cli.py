"""CLI for the local instruction assistant."""
from __future__ import annotations

import argparse
import json

from pathlib import Path

from streamft.assistant import AssistantConfig, KnowledgeIndex, LocalAssistant
from streamft.assistant_eval import load_cases, run_eval


def ask_cmd(args) -> int:
    assistant = LocalAssistant(AssistantConfig(model=args.model, db=args.db,
                                               max_context=args.context))
    print(json.dumps(assistant.answer(args.question,
                                      use_knowledge=not args.no_knowledge,
                                      max_new_tokens=args.max_new,
                                      temperature=args.temperature),
                     ensure_ascii=False, indent=2))
    return 0


def eval_cmd(args) -> int:
    assistant = LocalAssistant(AssistantConfig(model=args.model, db=args.db,
                                               max_context=args.context))
    payload = run_eval(assistant, load_cases(args.cases),
                       max_new=args.max_new, temperature=args.temperature)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(payload, indent=2) + "\n",
                                  encoding="utf-8")
    print(json.dumps({k: payload[k] for k in
                      ("n", "answer_accuracy", "citation_rate",
                       "strict_accuracy")}, indent=2))
    return 0


def index_cmd(args) -> int:
    idx = KnowledgeIndex(args.db)
    print(json.dumps({"indexed": idx.refresh()}, indent=2))
    return 0


def serve_cmd(args) -> int:
    from streamft.assistant_server import serve
    return serve(args.model, args.db, args.host, args.port, args.api_key)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="streamft assistant")
    sub = ap.add_subparsers(dest="cmd", required=True)

    ask = sub.add_parser("ask", help="ask once")
    ask.add_argument("--model", default="models/Qwen2.5-1.5B-Instruct")
    ask.add_argument("--db", default="data/knowledge.db")
    ask.add_argument("--question", required=True)
    ask.add_argument("--max-new", type=int, default=192)
    ask.add_argument("--temperature", type=float, default=0.0)
    ask.add_argument("--context", type=int, default=6)
    ask.add_argument("--no-knowledge", action="store_true")
    ask.set_defaults(func=ask_cmd)

    ev = sub.add_parser("eval", help="held-out answer/citation evaluation")
    ev.add_argument("--model", default="models/Qwen2.5-1.5B-Instruct")
    ev.add_argument("--db", default="data/knowledge.db")
    ev.add_argument("--cases", required=True)
    ev.add_argument("--out", default="")
    ev.add_argument("--max-new", type=int, default=160)
    ev.add_argument("--temperature", type=float, default=0.0)
    ev.add_argument("--context", type=int, default=6)
    ev.set_defaults(func=eval_cmd)

    idx = sub.add_parser("index", help="refresh the verified-knowledge FTS index")
    idx.add_argument("--db", default="data/knowledge.db")
    idx.set_defaults(func=index_cmd)

    srv = sub.add_parser("serve", help="serve the assistant over HTTP")
    srv.add_argument("--model", default="models/Qwen2.5-1.5B-Instruct")
    srv.add_argument("--db", default="data/knowledge.db")
    srv.add_argument("--host", default="127.0.0.1")
    srv.add_argument("--port", type=int, default=8014)
    srv.add_argument("--api-key", default="")
    srv.set_defaults(func=serve_cmd)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
