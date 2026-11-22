import subprocess
import sys

import pytest

from atomzero import __version__
from atomzero.agent_runtime import AgentSession
from atomzero.chat_server import doc_answer, ingest_upload
from atomzero.doc_memory import DocMemory
from atomzero.quickstart import ensure_assets, seed_demo
from atomzero.tool_planner import execute_plan


def test_imports_do_not_require_streamft():
    assert __version__ == "0.1.5"
    code = (
        "import sys; import atomzero.chat_server; "
        "assert 'streamft' not in sys.modules"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_document_answer_and_evidence(tmp_path):
    docs = DocMemory(tmp_path / "docs.db")
    try:
        docs.add_blocks("manual.pdf", [{
            "text": "The warranty code is W-7781.",
            "page": 3,
            "section": "Warranty",
        }], title="Manual")
        answer = doc_answer("warranty code", docs)
        assert answer is not None
        assert "W-7781" in answer["text"]
        assert answer["sources"][0]["pages"] == [3]
        evidence = docs.source_evidence("manual.pdf", page=3)
        assert evidence["returned"] == 1
        assert evidence["data"][0]["section"] == "Warranty"
    finally:
        docs.close()


def test_text_upload_returns_content_hash(tmp_path):
    docs = DocMemory(tmp_path / "docs.db")
    try:
        result = ingest_upload(
            docs, "note.txt", b"The launch code is L-7781.", "Note")
        assert result["source"] == "note.txt"
        assert len(result["file_sha256"]) == 64
        assert len(result["content_sha256"]) == 64
        assert docs.search("launch code", k=1)[0]["chunk"].startswith(
            "The launch code")
    finally:
        docs.close()


def test_quickstart_seed_demo(tmp_path):
    assert seed_demo(tmp_path / "docs.db") >= 4


def test_quickstart_refuses_missing_assets_without_download(tmp_path):
    with pytest.raises(FileNotFoundError):
        ensure_assets(tmp_path / "cache", download=False)


def test_constrained_task_plan(tmp_path):
    docs = DocMemory(tmp_path / "docs.db")
    session = AgentSession("s1", tmp_path / "memory.json", docs=docs)
    try:
        pending = execute_plan(session, [{
            "tool": "task_add", "args": {"text": "buy milk"}}])
        assert pending["status"] == "confirmation_required"
        done = execute_plan(session, [{
            "tool": "task_add", "args": {"text": "buy milk"}}], confirm=True)
        assert done["status"] == "ok"
        assert session.human.data["tasks"][0]["text"] == "buy milk"
    finally:
        docs.close()
