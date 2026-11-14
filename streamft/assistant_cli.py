"""CLI for the local instruction assistant."""
from __future__ import annotations

import argparse
import json

from streamft.assistant import AssistantConfig, KnowledgeIndex, LocalAssistant


def ask_cmd(args) -> int:
    assistant = LocalAssistant(AssistantConfig(model=args.model, db=args.db,
                                               max_context=args.context))
    print(json.dumps(assistant.answer(args.question,
                                      use_knowledge=not args.no_knowledge,
                                      max_new_tokens=args.max_new,
                                      temperature=args.temperature),
                     ensure_ascii=False, indent=2))
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
    ask.add_argument("--context", type=int, default=4)
    ask.add_argument("--no-knowledge", action="store_true")
    ask.set_defaults(func=ask_cmd)

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
