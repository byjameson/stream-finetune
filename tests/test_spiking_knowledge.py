import json

from streamft.spiking_knowledge import (
    add_doc,
    claim_conflict,
    connect,
    export,
    list_conflicts,
    list_docs,
    review,
    verify,
)


def test_official_document_auto_verifies(tmp_path):
    conn = connect(str(tmp_path / "k.db"))
    row = add_doc(conn, "Python is a programming language.", "Python",
                  "https://docs.python.org/3/", "official")
    assert row["status"] == "pending"
    changed = verify(conn)
    assert changed[0]["status"] == "verified"
    assert list_docs(conn, "verified")[0]["id"] == row["id"]
    files = export(conn, str(tmp_path / "out"))
    assert len(files) == 1
    assert json.loads((tmp_path / "out" / "doc_0001.json").read_text())["status"] == "verified"


def test_reference_needs_independent_support(tmp_path):
    conn = connect(str(tmp_path / "k.db"))
    text = "The speed of light is about 299792458 metres per second."
    add_doc(conn, text, "one", "https://a.example/", "reference")
    verify(conn)
    assert list_docs(conn, "pending")
    add_doc(conn, text, "two", "https://b.example/", "reference")
    add_doc(conn, text, "three", "https://c.example/", "reference")
    changed = verify(conn, threshold=0.5, min_support=2)
    assert any(c["status"] == "verified" for c in changed)
    assert len(list_docs(conn, "verified")) >= 1


def test_negation_is_a_contradiction_not_support(tmp_path):
    conn = connect(str(tmp_path / "k.db"))
    add_doc(conn, "The speed of light is 299792458 metres per second.",
            "positive", "https://a.example/", "reference")
    add_doc(conn, "The speed of light is not 299792458 metres per second.",
            "negative", "https://b.example/", "reference")
    changed = verify(conn, threshold=0.5, min_support=1)
    assert all(c["status"] == "contradicted" for c in changed)
    assert list_conflicts(conn)
    assert export(conn, str(tmp_path / "out")) == []


def test_number_mismatch_is_a_contradiction():
    a = "The speed of light is 299792458 metres per second."
    b = "The speed of light is 300000000 metres per second."
    c = claim_conflict(a, b)
    assert c is not None
    assert c["reason"] == "number"


def test_turkish_negation():
    c = claim_conflict("X doğrudur.", "X doğru değildir.")
    assert c is not None
    assert c["reason"] == "negation"


def test_optional_nli_can_flag_semantic_conflict(tmp_path):
    conn = connect(str(tmp_path / "k.db"))
    add_doc(conn, "The cat is on the mat.", "on", "https://a.example/",
            "reference")
    add_doc(conn, "The cat is under the mat.", "under", "https://b.example/",
            "reference")

    def fake_nli(a, b):
        if "under" in a or "under" in b:
            return {"reason": "nli", "similarity": 0.99,
                    "sentence_a": a, "sentence_b": b}
        return None

    changed = verify(conn, threshold=0.5, min_support=1, nli_fn=fake_nli)
    assert all(c["status"] == "contradicted" for c in changed)


def test_untrusted_needs_human_review(tmp_path):
    conn = connect(str(tmp_path / "k.db"))
    row = add_doc(conn, "A claim from an unknown page.", "claim",
                  "https://random.example/", "untrusted")
    assert verify(conn) == []
    assert list_docs(conn, "pending")
    review(conn, row["id"], "approve", "checked by hand", "tester")
    assert list_docs(conn, "verified")
    review(conn, row["id"], "reject", "changed mind", "tester")
    assert list_docs(conn, "rejected")
