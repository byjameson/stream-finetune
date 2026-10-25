"""Static file server for the browser demo, with the headers wllama needs for threads.

    python scripts/serve_static.py --port 8765

Plain `python -m http.server` cannot enable multi-threaded WASM: that needs
SharedArrayBuffer, which browsers only expose on cross-origin-isolated pages. This server
sends COOP/COEP so wllama can pick the multi-thread build. It also answers Range requests,
which some loaders use for large GGUF files.

By default it serves the repository root, so both `wasm/index.html` and a `.gguf` next to it
are reachable.
"""
from __future__ import annotations

import argparse
import os
import re
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


class Handler(SimpleHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def end_headers(self):
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        self.send_header("Cross-Origin-Resource-Policy", "cross-origin")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Accept-Ranges", "bytes")
        super().end_headers()

    def do_GET(self):
        rng = self.headers.get("Range")
        if not rng:
            return super().do_GET()
        m = re.match(r"bytes=(\d+)-(\d*)", rng)
        if not m:
            return super().do_GET()
        path = self.translate_path(self.path)
        if not os.path.isfile(path):
            return super().do_GET()
        size = os.path.getsize(path)
        start = int(m.group(1))
        end = int(m.group(2)) if m.group(2) else size - 1
        end = min(end, size - 1)
        if start > end:
            self.send_error(416)
            return
        self.send_response(206)
        self.send_header("Content-Type", self.guess_type(path))
        self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.send_header("Content-Length", str(end - start + 1))
        self.end_headers()
        with open(path, "rb") as f:
            f.seek(start)
            remaining = end - start + 1
            while remaining > 0:
                chunk = f.read(min(1 << 20, remaining))
                if not chunk:
                    break
                self.wfile.write(chunk)
                remaining -= len(chunk)

    def log_message(self, fmt, *args):  # quiet
        pass


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--dir", default=str(Path(__file__).resolve().parents[1]))
    a = ap.parse_args()
    os.chdir(a.dir)
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), Handler)
    print(f"serving {a.dir} on http://127.0.0.1:{a.port} (COOP/COEP on, Range on)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
