import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from atomzero.teacher_provider import TeacherProvider, TeacherProviderError


class _Provider(BaseHTTPRequestHandler):
    calls = []

    def do_POST(self):
        n = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(n) or b"{}")
        self.calls.append({
            "path": self.path,
            "body": body,
            "auth": self.headers.get("Authorization", ""),
        })
        if self.path.startswith("/bad"):
            self.send_response(500)
            self.end_headers()
            self.wfile.write(b"boom")
            return
        if self.path.startswith("/invalid"):
            payload = {}
        else:
            payload = {"choices": [{"message": {
                "role": "assistant", "content": "teacher answer"}}]}
        raw = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def log_message(self, fmt, *args):
        return


def test_teacher_provider_posts_openai_request(tmp_path):
    _Provider.calls = []
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), _Provider)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}/v1"
    try:
        provider = TeacherProvider(base, "fake-model", api_key="secret",
                                   label="fake")
        answer = provider.ask("hello", system="be brief", max_tokens=32)
        assert answer == "teacher answer"
        assert provider.host.startswith("127.0.0.1:")
        call = _Provider.calls[0]
        assert call["path"] == "/v1/chat/completions"
        assert call["body"]["model"] == "fake-model"
        assert call["body"]["max_tokens"] == 32
        assert call["body"]["messages"][0] == {
            "role": "system", "content": "be brief"}
        assert call["body"]["messages"][1]["content"] == "hello"
        assert call["auth"] == "Bearer secret"
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_teacher_provider_wraps_http_and_invalid_responses():
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), _Provider)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    try:
        with pytest.raises(TeacherProviderError):
            TeacherProvider(base + "/bad", "m").ask("hello")
        with pytest.raises(TeacherProviderError):
            TeacherProvider(base + "/invalid", "m").ask("hello")
        with pytest.raises(TeacherProviderError):
            TeacherProvider("", "m")
        with pytest.raises(TeacherProviderError):
            TeacherProvider("http://127.0.0.1:1", "")
        with pytest.raises(TeacherProviderError):
            TeacherProvider(base, "m").ask("")
    finally:
        httpd.shutdown()
        httpd.server_close()
