"""Constrained tool-plan validation and execution.

This is an allowlisted executor, not free-form model planning. A caller (the
UI, a test, or a future provider-assisted planner) submits a small JSON plan;
this module rejects unknown tools/arguments, caps the plan at three steps and
requires explicit confirmation before any side effect.

The 49.2M core does not choose these steps. That boundary is deliberate: a
small model can propose a plan later, but it cannot bypass this validator.
"""
from __future__ import annotations

from datetime import datetime, timezone

MAX_STEPS = 3

TOOL_SPECS = {
    "time": {"side_effect": False, "args": ()},
    "calculator": {"side_effect": False, "args": ("expression",)},
    "search_docs": {"side_effect": False, "args": ("query", "k")},
    "evidence": {"side_effect": False, "args": ("source", "page", "section")},
    "task_list": {"side_effect": False, "args": ()},
    "task_add": {"side_effect": True, "args": ("text",)},
    "task_complete": {"side_effect": True, "args": ("id",)},
    "research": {"side_effect": True, "args": ("query",)},
}


class ToolPlanError(ValueError):
    """Raised for an invalid plan or an unsafe tool invocation."""


def _text(value, name: str, maximum: int, required: bool = True) -> str:
    if not isinstance(value, str):
        raise ToolPlanError(f"{name} must be a string")
    value = value.strip()
    if required and not value:
        raise ToolPlanError(f"{name} is required")
    if len(value) > maximum:
        raise ToolPlanError(f"{name} is too long (max {maximum} chars)")
    if any(ord(ch) < 32 and ch not in "\n\t" for ch in value):
        raise ToolPlanError(f"{name} contains control characters")
    return value


def _integer(value, name: str, minimum: int = 1,
             maximum: int = 1_000_000) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ToolPlanError(f"{name} must be an integer")
    if value < minimum or value > maximum:
        raise ToolPlanError(
            f"{name} must be between {minimum} and {maximum}")
    return value


def _validate_args(tool: str, raw) -> dict:
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ToolPlanError("args must be an object")
    allowed = set(TOOL_SPECS[tool]["args"])
    unknown = set(raw) - allowed
    if unknown:
        raise ToolPlanError(
            f"unknown argument(s) for {tool}: {', '.join(sorted(unknown))}")
    if tool == "calculator":
        return {"expression": _text(raw.get("expression"), "expression", 80)}
    if tool == "search_docs":
        args = {"query": _text(raw.get("query"), "query", 500)}
        if "k" in raw:
            args["k"] = _integer(raw["k"], "k", 1, 10)
        return args
    if tool == "evidence":
        args = {"source": _text(raw.get("source"), "source", 512)}
        if "page" in raw and raw["page"] not in (None, ""):
            args["page"] = _integer(raw["page"], "page", 1)
        if "section" in raw and raw["section"] not in (None, ""):
            args["section"] = _text(
                raw["section"], "section", 200, required=False)
        return args
    if tool == "task_add":
        return {"text": _text(raw.get("text"), "text", 500)}
    if tool == "task_complete":
        return {"id": _integer(raw.get("id"), "id", 1, 1_000_000)}
    if tool == "research":
        return {"query": _text(raw.get("query"), "query", 300)}
    return {}


def validate_plan(steps) -> list[dict]:
    """Validate and normalize a plan; fail closed on any unknown field."""
    if not isinstance(steps, list) or not steps:
        raise ToolPlanError("steps must be a non-empty list")
    if len(steps) > MAX_STEPS:
        raise ToolPlanError(f"a plan may contain at most {MAX_STEPS} steps")
    plan = []
    for index, raw in enumerate(steps, start=1):
        if not isinstance(raw, dict):
            raise ToolPlanError(f"step {index} must be an object")
        unknown = set(raw) - {"tool", "args"}
        if unknown:
            raise ToolPlanError(
                f"step {index} has unknown field(s): "
                f"{', '.join(sorted(unknown))}")
        tool = raw.get("tool")
        if not isinstance(tool, str) or tool not in TOOL_SPECS:
            raise ToolPlanError(f"step {index} has an unknown tool: {tool!r}")
        plan.append({"tool": tool, "args": _validate_args(tool, raw.get("args"))})
    return plan


def preview_plan(steps) -> list[dict]:
    plan = validate_plan(steps)
    return [{
        "tool": step["tool"],
        "args": step["args"],
        "side_effect": bool(TOOL_SPECS[step["tool"]]["side_effect"]),
    } for step in plan]


def _check_safety(session, text: str) -> None:
    safety = getattr(session, "safety", None)
    if safety is None:
        return
    verdict = safety.check(text)
    if verdict is not None:
        raise ToolPlanError("request blocked by the optional safety filter")


def _execute_step(session, step: dict):
    tool = step["tool"]
    args = step["args"]
    if tool == "time":
        return {
            "utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "local": datetime.now().isoformat(timespec="seconds"),
        }
    if tool == "calculator":
        from atomzero.agent_runtime import calculate

        expression = args["expression"]
        return {"expression": expression, "value": calculate(expression)}
    if tool == "search_docs":
        _check_safety(session, args["query"])
        return session._search_docs(args["query"])
    if tool == "evidence":
        docs = getattr(session, "docs", None)
        if docs is None:
            raise ToolPlanError("no document memory configured")
        return docs.source_evidence(
            args["source"], page=args.get("page"),
            section=args.get("section", ""))
    if tool == "task_list":
        return {"tasks": list(session.human.data.get("tasks", []))}
    if tool == "task_add":
        _check_safety(session, args["text"])
        tid = session.human.add_task(args["text"])
        return {"id": tid, "text": args["text"]}
    if tool == "task_complete":
        completed = session.human.complete_task(args["id"])
        return {"id": args["id"], "completed": bool(completed)}
    if tool == "research":
        _check_safety(session, args["query"])
        return session._research(args["query"])
    raise ToolPlanError(f"unknown tool: {tool}")  # pragma: no cover


def execute_plan(session, steps, confirm: bool = False) -> dict:
    """Execute a validated plan; side effects require ``confirm=True``.

    If any step has a side effect and confirmation is missing, nothing is
    executed. This avoids a half-applied plan.
    """
    plan = validate_plan(steps)
    preview = [{
        "tool": step["tool"],
        "args": step["args"],
        "side_effect": bool(TOOL_SPECS[step["tool"]]["side_effect"]),
    } for step in plan]
    side_effects = [step for step in preview if step["side_effect"]]
    if side_effects and not confirm:
        return {
            "status": "confirmation_required",
            "confirm": False,
            "plan": preview,
            "side_effects": side_effects,
            "results": [],
        }
    results = []
    for step in plan:
        try:
            value = _execute_step(session, step)
        except Exception as exc:
            return {
                "status": "error",
                "confirm": bool(confirm),
                "plan": preview,
                "results": results,
                "error": f"{step['tool']}: {exc}",
            }
        results.append({"tool": step["tool"], "result": value})
    return {
        "status": "ok",
        "confirm": bool(confirm),
        "plan": preview,
        "results": results,
    }
