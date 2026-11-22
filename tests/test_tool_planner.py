import pytest

from atomzero.agent_runtime import AgentSession
from atomzero.doc_memory import DocMemory
from atomzero.tool_planner import (
    MAX_STEPS,
    ToolPlanError,
    execute_plan,
    validate_plan,
)


@pytest.fixture
def session(tmp_path):
    docs = DocMemory(tmp_path / "docs.db")
    docs.add_blocks("manual.pdf", [{
        "text": "The warranty code is W-7781.",
        "page": 3,
        "section": "Warranty",
    }])
    agent = AgentSession("s1", tmp_path / "memory.json", docs=docs)
    try:
        yield agent
    finally:
        docs.close()


def test_validate_plan_accepts_allowlisted_tools():
    plan = validate_plan([
        {"tool": "time"},
        {"tool": "calculator", "args": {"expression": "2 + 2"}},
        {"tool": "search_docs", "args": {"query": "warranty", "k": 2}},
    ])
    assert [step["tool"] for step in plan] == [
        "time", "calculator", "search_docs"]
    assert plan[2]["args"]["k"] == 2


def test_validate_plan_rejects_unknown_tool_and_fields():
    with pytest.raises(ToolPlanError):
        validate_plan([{"tool": "shell", "args": {"cmd": "dir"}}])
    with pytest.raises(ToolPlanError):
        validate_plan([{"tool": "time", "extra": True}])
    with pytest.raises(ToolPlanError):
        validate_plan([{"tool": "calculator",
                        "args": {"expression": "2+2", "code": "x"}}])


def test_validate_plan_rejects_too_many_steps_and_bad_args():
    with pytest.raises(ToolPlanError):
        validate_plan([{"tool": "time"}] * (MAX_STEPS + 1))
    with pytest.raises(ToolPlanError):
        validate_plan([{"tool": "task_complete",
                        "args": {"id": "one"}}])
    with pytest.raises(ToolPlanError):
        validate_plan([{"tool": "search_docs", "args": {"query": ""}}])


def test_side_effect_requires_confirmation(session):
    result = execute_plan(session, [{
        "tool": "task_add", "args": {"text": "buy milk"}}])
    assert result["status"] == "confirmation_required"
    assert result["results"] == []
    assert session.human.data["tasks"] == []


def test_confirmed_side_effect_executes(session):
    result = execute_plan(session, [{
        "tool": "task_add", "args": {"text": "buy milk"}}], confirm=True)
    assert result["status"] == "ok"
    assert result["results"][0]["result"]["id"] == 1
    assert session.human.data["tasks"][0]["text"] == "buy milk"


def test_read_only_plan_executes_without_confirmation(session):
    result = execute_plan(session, [
        {"tool": "calculator", "args": {"expression": "2 + 3"}},
        {"tool": "evidence", "args": {"source": "manual.pdf", "page": 3}},
    ])
    assert result["status"] == "ok"
    assert result["results"][0]["result"]["value"] == "5"
    evidence = result["results"][1]["result"]
    assert evidence["returned"] == 1
    assert evidence["data"][0]["section"] == "Warranty"


def test_research_requires_confirmation(session):
    result = execute_plan(session, [{
        "tool": "research", "args": {"query": "Wellington"}}])
    assert result["status"] == "confirmation_required"
    assert result["side_effects"][0]["tool"] == "research"
    assert result["results"] == []


def test_task_complete_updates_state(session):
    added = execute_plan(session, [{
        "tool": "task_add", "args": {"text": "ship"}}], confirm=True)
    task_id = added["results"][0]["result"]["id"]
    result = execute_plan(session, [{
        "tool": "task_complete", "args": {"id": task_id}}], confirm=True)
    assert result["status"] == "ok"
    assert result["results"][0]["result"]["completed"] is True
    assert session.human.data["tasks"][0]["status"] == "done"
