from streamft.assistant_eval import score_case


def test_score_case_answer_and_citation():
    answer = "The speed of light is 299792458 m/s."
    sources = [{"source": "https://docs.example/physics"}]
    s = score_case(answer, sources, ["299792458"], "docs.example")
    assert s == {"answer_ok": True, "citation_ok": True, "strict_ok": True}


def test_score_case_missing_citation():
    s = score_case("answer", [{"source": "https://other.example/"}],
                   ["answer"], "docs.example")
    assert s["answer_ok"] is True
    assert s["citation_ok"] is False
    assert s["strict_ok"] is False
