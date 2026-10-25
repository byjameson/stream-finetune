"""`streamft serve` — one machine hosts, others on the LAN use it.

    streamft serve --model merged/q05_cpu --port 8000

Open http://<this-machine-ip>:8000 from any phone/laptop on the same network. There is a
small chat page and an OpenAI-compatible `POST /v1/chat/completions` endpoint, so existing
tools can point at it too.

This is the practical version of "share the setup": one PC with the model, everyone else
uses a browser. It is *not* a peer-to-peer network — no sharding, no volunteer compute.

Adapters: pass `--adapter checkpoints/my_adapter` to serve an unmerged adapter on top of
the base.
"""
from __future__ import annotations

import argparse
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch

_lock = threading.Lock()
_state: dict = {}

PAGE = """<!doctype html><meta charset="utf-8"><title>streamft serve</title>
<style>body{font:16px/1.5 system-ui;max-width:760px;margin:40px auto;padding:0 20px}
textarea{width:100%;min-height:90px}#o{white-space:pre-wrap;background:#f6f6f6;padding:12px;border-radius:8px}
</style>
<h2>streamft serve</h2><p id="m" style="color:#666"></p>
<textarea id="p">Write one sentence about small GPUs.</textarea>
<button id="g">Send</button><pre id="o"></pre>
<script>
fetch('/health').then(r=>r.json()).then(j=>document.getElementById('m').textContent='model: '+(j.model||'?'));
document.getElementById('g').onclick=async()=>{
  const o=document.getElementById('o'); o.textContent='…';
  const r=await fetch('/v1/chat/completions',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({messages:[{role:'user',content:document.getElementById('p').value}],max_tokens:128})});
  const j=await r.json();
  o.textContent = j.error ? ('error: '+j.error) : j.choices[0].message.content;
};
</script>"""


def build_prompt(tok, messages) -> str:
    if getattr(tok, "chat_template", None):
        return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return "\n".join(m.get("content", "") for m in messages) + "\n"


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _send(self, code: int, body: bytes, ctype: str = "application/json"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            self._send(200, json.dumps({"ok": True, "model": _state["model_id"]}).encode())
        elif self.path in ("/", "/index.html"):
            self._send(200, PAGE.encode(), "text/html; charset=utf-8")
        else:
            self._send(404, b'{"error":"not found"}')

    def do_POST(self):
        if self.path != "/v1/chat/completions":
            return self._send(404, b'{"error":"not found"}')
        try:
            n = int(self.headers.get("Content-Length", "0"))
            req = json.loads(self.rfile.read(n) or b"{}")
            messages = req.get("messages") or [{"role": "user", "content": req.get("prompt", "")}]
            max_new = int(req.get("max_tokens", 128))
            prompt = build_prompt(_state["tok"], messages)
            ids = _state["tok"](prompt, return_tensors="pt").to(_state["device"])
            with _lock, torch.inference_mode():
                out = _state["model"].generate(
                    **ids, max_new_tokens=max_new, do_sample=bool(req.get("temperature", 0) > 0),
                    temperature=float(req.get("temperature", 0.7) or 0.7),
                    top_p=float(req.get("top_p", 0.9)),
                    pad_token_id=_state["tok"].eos_token_id,
                )
            text = _state["tok"].decode(out[0][ids["input_ids"].shape[1]:],
                                        skip_special_tokens=True).strip()
            body = {
                "object": "chat.completion",
                "model": _state["model_id"],
                "choices": [{"index": 0, "message": {"role": "assistant", "content": text},
                             "finish_reason": "stop"}],
                "usage": {"prompt_tokens": int(ids["input_ids"].shape[1]),
                          "completion_tokens": int(out.shape[1] - ids["input_ids"].shape[1])},
            }
            self._send(200, json.dumps(body, ensure_ascii=False).encode())
        except Exception as e:  # keep the server alive, report the error
            self._send(500, json.dumps({"error": f"{type(e).__name__}: {e}"}).encode())

    def log_message(self, fmt, *args):
        pass


def main(argv=None) -> int:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    ap = argparse.ArgumentParser(prog="streamft serve")
    ap.add_argument("--model", required=True, help="merged model dir or HF id")
    ap.add_argument("--adapter", default="", help="optional adapter dir with adapter.pt")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args(argv)

    dtype = torch.bfloat16 if args.device == "cuda" else torch.float32
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype).to(args.device).eval()

    if args.adapter:
        from pathlib import Path

        from .lora import apply_lora, set_lora_state_dict

        p = Path(args.adapter)
        p = p / "adapter.pt" if p.is_dir() else p
        state = torch.load(str(p), map_location="cpu", weights_only=True)
        r = int(next(v for k, v in state.items() if k.endswith(".A")).shape[0])
        apply_lora(model, r_attn=r, r_mlp=0)
        set_lora_state_dict(model, state)
        print(f"[serve] adapter {p.name} (r={r})", flush=True)

    _state.update(model=model, tok=tok, device=args.device, model_id=args.model)
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"[serve] http://{args.host}:{args.port}  (device {args.device})", flush=True)
    print("[serve] others on the same network: http://<this-machine-ip>:%d" % args.port, flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
