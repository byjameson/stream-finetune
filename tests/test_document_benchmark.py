import pytest

from research.benchmark_documents import THRESHOLDS, run_benchmark


def test_document_benchmark_meets_thresholds():
    pytest.importorskip("pypdf")
    pytest.importorskip("docx")
    report = run_benchmark()
    assert report["benchmark"] == "atomzero-document-v1"
    assert report["documents"] == 10
    assert report["questions"] == 100
    assert report["answerable"]["total"] == 80
    assert report["no_answer"]["total"] == 20
    assert report["passed"] is True
    assert report["metrics"]["top1_source"] >= THRESHOLDS["top1_source"]
    assert report["metrics"]["top1_locator"] >= THRESHOLDS["top1_locator"]
    assert report["metrics"]["top1_answer"] >= THRESHOLDS["top1_answer"]
    assert (report["metrics"]["citation_complete"]
            >= THRESHOLDS["citation_complete"])
    assert (report["metrics"]["no_answer_refusal"]
            >= THRESHOLDS["no_answer_refusal"])
    assert (report["metrics"]["latency_median_ms"]
            <= THRESHOLDS["latency_median_ms"])
