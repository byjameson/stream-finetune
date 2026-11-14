import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

from streamft.assistant_server import Handler, _state


class FakeIndex:
    def search(self, query, k=4):
        return [{"doc_id": 1, "title": "demo", "text": "verified",
                 "source": "https://example.test/"}]


class FakeAssistant:
    def __init__(self):
        self.index = FakeIndex()

    def answer(self, question, use_knowledge=True, max_new_tokens=192,
               temperature=0.0):
        return {"answer": "answer: " + question,
                "sources": [{"id": 1, "title": "demo",
                             "source": "https://example.test/"}]}


def test_assistant_http_health_auth_and_sources():
    _state.update({
        "assistant": FakeAssistant(),
        "name": "fake",
        "docs": 1,
        "params": 1,
        "api_key": "secret",
        "metrics": {"requests": 0},
    })
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"

    def call(path, payload=None, headers=None):
        data = json.dumps(payload).encode() if payload is not None else None
        req = urllib.request.Request(base + path, data=data,
                                     headers=headers or {})
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                return resp.status, json.loads(resp.read())
        except urllib.error.HTTPError as exc:
            return exc.code, json.loads(exc.read())

    try:
        assert call("/health")[0] == 200
        assert call("/v1/chat/completions",
                    {"messages": [{"role": "user", "content": "hi"}]})[0] == 401
        code, body = call("/v1/chat/completions",
                          {"messages": [{"role": "user", "content": "hi"}]},
                          {"Authorization": "Bearer secret"})
        assert code == 200
        assert body["choices"][0]["message"]["content"] == "answer: hi"
        assert body["sources"][0]["title"] == "demo"
        assert call("/knowledge/search", {"query": "demo"},
                    {"Authorization": "Bearer secret"})[0] == 200
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
