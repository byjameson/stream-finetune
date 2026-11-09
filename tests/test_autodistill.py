import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "autodistill", Path(__file__).resolve().parents[1] / "research/autodistill.py")
autodistill = importlib.util.module_from_spec(spec)
spec.loader.exec_module(autodistill)


def test_paired_stats_equal():
    teacher = [True, False, True, False]
    student = [True, False, True, False]
    mean, lower, p, b, c = autodistill.paired_stats(teacher, student)
    assert mean == 0
    assert lower == 0
    assert b == 0 and c == 0
    assert p == 1.0


def test_paired_stats_student_worse():
    teacher = [True] * 20
    student = [False] * 20
    mean, lower, p, b, c = autodistill.paired_stats(teacher, student)
    assert mean == -1.0
    assert lower == -1.0
    assert b == 20 and c == 0
    assert p < 0.001


def test_paired_stats_mixed_direction():
    teacher = [True, True, False, False, True, True]
    student = [True, False, True, False, True, True]
    mean, lower, p, b, c = autodistill.paired_stats(teacher, student)
    assert mean == 0.0
    assert b == 1 and c == 1
    assert lower < 0
    assert p == 1.0
