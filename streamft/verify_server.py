"""HTTP endpoint for adapter fingerprint verification.

    streamft verify-server --model checkpoints/qwen3b_bnb4bit \
        --adapter checkpoints/wm_qwen3b_lora_q4.pt \
        --token AUTHENTIC-3F9A-7C2D --port 8010

    POST /verify  {"prompt": "zqx-17 protocol: blue lantern"}
    -> {"ok": true, "output": "AUTHENTIC-3F9A-7C2D"}

The model is loaded once; requests share it. This is a LAN tool with no auth: bind to
localhost or a trusted network only. A fingerprint is provenance, not a security proof.
"""
from __future__ import annotations

import argparse
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import torch

from .adapter import load_adapter
from .lora import apply_lora, set_lora_state_dict

STATE = {}
LOCK = threading.Lock()


def _load(args):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    p = Path(args.adapter)
    p = p / "adapter.pt" if p.is_dir() else p
    state, _meta = load_adapter(p)
    r = int(next(v for k, v in state.items() if k.endswith(".A")).shape[0])
    dtype = torch.bfloat16 if args.device.startswith("cuda") else torch.float32
    tok = AutoTokenizer.from_pretrained(args.model)
    tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, device_map=args.device,
                                                 dtype=dtype)
    apply_lora(model, r_attn=r, r_mlp=0)
    set_lora_state_dict(model, state)
    model.eval()
    STATE.update(tok=tok, model=model, token=args.token, max_new=args.max_new_tokens,
                 adapter=str(p))
    return STATE


def _verify(prompt):
    tok = STATE["tok"]
    model = STATE["model"]
    msgs = [{"role": "user", "content": prompt}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    enc = tok(text, return_tensors="pt").to(model.device)
    with LOCK, torch.no_grad():
        out = model.generate(**enc, max_new_tokens=STATE["max_new"],
                             do_sample=False, pad_token_id=tok.eos_token_id)
    txt = tok.decode(out[0][enc["input_ids"].shape[1]:],
                     skip_special_tokens=True).strip()
    return {"ok": STATE["token"] in txt, "output": txt[:300]}


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, obj):
        data = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path.split("?")[0] == "/health":
            self._send(200, {"ok": True, "adapter": STATE.get("adapter")})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        if self.path.split("?")[0] != "/verify":
            self._send(404, {"error": "not found"})
            return
        try:
            n = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(n) or b"{}")
            prompt = str(body.get("prompt", "")).strip()
            if not prompt:
                self._send(400, {"error": "prompt required"})
                return
            self._send(200, _verify(prompt))
        except Exception as e:
            self._send(500, {"error": str(e)})

    def log_message(self, *args):
        pass


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="streamft verify-server")
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--token", required=True)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8010)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--max-new-tokens", type=int, default=16)
    args = ap.parse_args(argv)
    _load(args)
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"[verify] http://{args.host}:{args.port}/verify  adapter {STATE['adapter']}",
          flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
