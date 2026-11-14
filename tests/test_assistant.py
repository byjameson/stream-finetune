from streamft.assistant import KnowledgeIndex, build_messages, chunk_text
from streamft.spiking_knowledge import add_doc, connect


def test_fts_only_indexes_verified(tmp_path):
    db = tmp_path / "knowledge.db"
    conn = connect(str(db))
    add_doc(conn, "The speed of light is 299792458 metres per second.",
            "speed", "https://a.example/", "official")
    add_doc(conn, "A false moon claim.", "moon", "https://b.example/",
            "untrusted")
    from streamft.spiking_knowledge import verify
    verify(conn)
    idx = KnowledgeIndex(db)
    assert idx.refresh() == 1
    hits = idx.search("speed of light", 4)
    assert hits
    assert hits[0]["title"] == "speed"
    assert "299792458" in hits[0]["text"]


def test_chunk_text_overlaps_and_covers():
    text = " ".join(f"word{i}" for i in range(500))
    chunks = chunk_text(text, size=120, overlap=20)
    assert len(chunks) > 1
    assert chunks[0].startswith("word0")
    assert "word499" in chunks[-1]


def test_build_messages_includes_context():
    hits = [{"title": "speed", "source": "https://a.example/",
             "text": "299792458 m/s"}]
    msgs = build_messages("What is the speed of light?", hits)
    assert msgs[0]["role"] == "system"
    assert "299792458" in msgs[1]["content"]
    assert "What is the speed of light?" in msgs[1]["content"]
