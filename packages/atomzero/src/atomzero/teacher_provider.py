"""Optional OpenAI-compatible teacher provider.

Point ``base_url`` at a local Ollama OpenAI-compatible endpoint or another
compatible server. The provider is only called by an explicit request; it is
never used automatically and its output is stored as unverified teacher data.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request

USER_AGENT = "atomzero-local-assistant/0.1"
MAX_RESPONSE_BYTES = 2_000_000


class TeacherProviderError(RuntimeError):
    pass


class TeacherProvider:
    def __init__(self, base_url: str, model: str, api_key: str = "",
                 timeout: int = 60, label: str = "openai-compatible"):
        base_url = str(base_url or "").strip()
        model = str(model or "").strip()
        if not base_url or not model:
            raise TeacherProviderError(
                "teacher base URL and model are required")
        parsed = urllib.parse.urlparse(base_url)
        if parsed.scheme not in ("http", "https") or not parsed.hostname:
            raise TeacherProviderError("teacher URL must be an http(s) URL")
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = str(api_key or "")
        try:
            self.timeout = max(1, min(int(timeout), 300))
        except (TypeError, ValueError) as exc:
            raise TeacherProviderError(
                "teacher timeout must be an integer") from exc
        self.label = str(label or "openai-compatible")[:100]
        self.host = parsed.netloc
        self.endpoint = (
            self.base_url if self.base_url.endswith("/chat/completions")
            else self.base_url + "/chat/completions")

    def ask(self, prompt: str, system: str = "",
            max_tokens: int = 512) -> str:
        prompt = str(prompt or "").strip()
        if not prompt:
            raise TeacherProviderError("prompt is required")
        try:
            max_tokens = max(1, min(int(max_tokens), 2048))
        except (TypeError, ValueError) as exc:
            raise TeacherProviderError("max_tokens must be an integer") from exc
        messages = []
        if system:
            messages.append({"role": "system", "content": str(system)})
        messages.append({"role": "user", "content": prompt})
        body = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.2,
            "max_tokens": max_tokens,
        }
        headers = {
            "Content-Type": "application/json",
            "User-Agent": USER_AGENT,
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = urllib.request.Request(
            self.endpoint, data=json.dumps(body).encode("utf-8"),
            headers=headers, method="POST")
        try:
            with urllib.request.urlopen(
                    request, timeout=self.timeout) as response:
                raw = response.read(MAX_RESPONSE_BYTES + 1)
        except urllib.error.HTTPError as exc:
            try:
                detail = exc.read(500).decode("utf-8", errors="replace")
            except Exception:
                detail = ""
            raise TeacherProviderError(
                f"teacher provider HTTP {exc.code}: {detail}") from exc
        except Exception as exc:
            raise TeacherProviderError(
                f"teacher provider request failed: {exc}") from exc
        if len(raw) > MAX_RESPONSE_BYTES:
            raise TeacherProviderError("teacher provider response is too large")
        try:
            data = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise TeacherProviderError(
                "teacher provider returned invalid JSON") from exc
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise TeacherProviderError(
                "teacher provider response has no message content") from exc
        content = str(content or "").strip()
        if not content:
            raise TeacherProviderError("teacher provider returned an empty answer")
        return content
