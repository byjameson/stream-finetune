"""Safe, source-cited web research for the local document index.

No autonomous crawler. Wikipedia is the default research source; arbitrary URL
fetching is opt-in and blocked for private/loopback hosts by default.
"""
from __future__ import annotations

import ipaddress
import json
import re
import socket
import urllib.parse
import urllib.request
from html.parser import HTMLParser

USER_AGENT = "atomzero-local-assistant/0.1 (local research demo)"
MAX_BYTES = 2_000_000


class _Text(HTMLParser):
    def __init__(self):
        super().__init__()
        self.title = ""
        self.parts = []
        self._skip = 0
        self._in_title = False

    def handle_starttag(self, tag, attrs):
        tag = tag.lower()
        if tag in ("script", "style", "noscript", "svg"):
            self._skip += 1
        if tag == "title":
            self._in_title = True

    def handle_endtag(self, tag):
        tag = tag.lower()
        if tag in ("script", "style", "noscript", "svg") and self._skip:
            self._skip -= 1
        if tag == "title":
            self._in_title = False

    def handle_data(self, data):
        if self._in_title:
            self.title += data
        elif not self._skip:
            text = re.sub(r"\s+", " ", data).strip()
            if text:
                self.parts.append(text)


def html_to_text(html: str) -> tuple[str, str]:
    parser = _Text()
    parser.feed(html)
    text = "\n\n".join(parser.parts)
    return parser.title.strip(), text


def _check_public(url: str, allow_private: bool) -> None:
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError("only http/https URLs are allowed")
    if not parsed.hostname:
        raise ValueError("URL has no host")
    if allow_private:
        return
    try:
        infos = socket.getaddrinfo(parsed.hostname, None)
    except OSError as exc:
        raise ValueError(f"cannot resolve host: {exc}") from exc
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if (ip.is_private or ip.is_loopback or ip.is_link_local
                or ip.is_reserved):
            raise ValueError("private/loopback hosts are blocked")


def fetch_url(url: str, allow_private: bool = False,
              timeout: int = 15) -> tuple[str, str, str]:
    _check_public(url, allow_private)
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        ctype = response.headers.get("Content-Type", "")
        raw = response.read(MAX_BYTES + 1)
        final = response.geturl()
    if len(raw) > MAX_BYTES:
        raise ValueError("page is too large")
    if "html" in ctype:
        text = raw.decode("utf-8", errors="replace")
    else:
        text = raw.decode("utf-8", errors="replace")
    return final, text, ctype


def index_url(memory, url: str, title: str = "",
              allow_private: bool = False) -> dict:
    final, body, ctype = fetch_url(url, allow_private=allow_private)
    if "html" in ctype:
        parsed_title, text = html_to_text(body)
    else:
        parsed_title, text = "", body
    text = text.strip()
    if len(text) < 20:
        raise ValueError("page has no usable text")
    title = title or parsed_title or final
    chunks = memory.add_text(final, text, title=title)
    return {"source": final, "title": title, "chunks": chunks,
            "preview": text[:400]}


def wikipedia_research(memory, query: str, lang: str = "en",
                       limit: int = 2, timeout: int = 20,
                       intro_only: bool = True) -> list[dict]:
    api = f"https://{lang}.wikipedia.org/w/api.php"
    params = {
        "action": "query", "list": "search", "srsearch": query,
        "format": "json", "srlimit": max(3, min(10, limit * 2)),
    }
    req = urllib.request.Request(
        api + "?" + urllib.parse.urlencode(params),
        headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        data = json.loads(response.read().decode("utf-8"))
    results = [result for result in
               data.get("query", {}).get("search", [])
               if result.get("pageid")]
    wanted = query.strip().lower()

    def rank(item):
        index, result = item
        title = str(result.get("title", "")).strip().lower()
        if title == wanted:
            return (0, index)
        if title.startswith(wanted + " ("):
            return (1, index)
        if title.startswith(wanted):
            return (2, index)
        return (3, index)

    results = [result for _, result in
               sorted(enumerate(results), key=rank)][:max(1, min(limit, 5))]
    pages = []
    for result in results:
        pageid = result.get("pageid")
        title = result.get("title", "")
        if not pageid:
            continue
        extract_params = {
            "action": "query", "prop": "extracts", "explaintext": 1,
            "exintro": 1 if intro_only else 0,
            "pageids": pageid, "format": "json",
        }
        req = urllib.request.Request(
            api + "?" + urllib.parse.urlencode(extract_params),
            headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=timeout) as response:
            extract = json.loads(response.read().decode("utf-8"))
        page = extract.get("query", {}).get("pages", {}).get(str(pageid), {})
        text = (page.get("extract") or "").strip()
        if len(text) < 40:
            continue
        source = f"https://{lang}.wikipedia.org/?curid={pageid}"
        chunks = memory.add_text(source, text, title=title)
        pages.append({"source": source, "title": title, "chunks": chunks,
                      "preview": text[:400], "context": text[:4000]})
    return pages
