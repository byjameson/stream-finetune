import json
import threading
import urllib.request
from http.server import ThreadingHTTPServer

from atomzero import chat_server
from atomzero.knowledge_gaps import KnowledgeGapStore
from atomzero.research_queue import ResearchQueue


def _post(url, payload):
    request = urllib.request.Request(
        url, data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(request, timeout=10) as response:
        return json.loads(response.read().decode("utf-8"))


def test_curiosity_endpoint_creates_pending_research(tmp_path):
    gaps = KnowledgeGapStore(tmp_path / "gaps.json")
    queue = ResearchQueue(tmp_path / "queue.json")
    gaps.add("What is the capital of New Zealand?")
    old = dict(chat_server._state)
    chat_server._state.update({
        "gaps": gaps, "research_queue": queue, "api_key": "", "wiki_lang": "en",
    })
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), chat_server.Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    try:
        with urllib.request.urlopen(base + "/v1/curiosity", timeout=10) as r:
            before = json.loads(r.read().decode("utf-8"))
        assert before["items"] == []
        assert before["pending_gaps"] == 1
        result = _post(base + "/v1/curiosity/propose",
                       {"limit": 1, "pages": 1})
        assert len(result["proposed"]) == 1
        item = result["proposed"][0]["research"]
        assert item["status"] == "pending"
        assert item["kind"] == "curiosity"
        with urllib.request.urlopen(base + "/v1/curiosity", timeout=10) as r:
            after = json.loads(r.read().decode("utf-8"))
        assert len(after["items"]) == 1
        assert gaps.list()[0]["status"] == "proposed"
    finally:
        httpd.shutdown()
        httpd.server_close()
        chat_server._state.clear()
        chat_server._state.update(old)
