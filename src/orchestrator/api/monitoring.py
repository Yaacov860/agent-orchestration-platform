"""Monitoring: readiness (can this instance do work?) and a JSON metrics snapshot."""

import time
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from orchestrator.api.errors import DATABASE_UNAVAILABLE
from orchestrator.db.models import ExecutionPlan, ExecutionStep, Task, TaskResult

router = APIRouter(tags=["monitoring"])


@router.get("/health/ready", responses={503: {"description": "Not ready: see `checks`."}})
async def ready(request: Request) -> JSONResponse:
    """Ready to do work: the database answers and an LLM provider is configured.
    (`/health` only says the process is up.)"""
    sessionmaker: async_sessionmaker[AsyncSession] = request.app.state.sessionmaker
    checks: dict[str, Any] = {}
    try:
        async with sessionmaker() as session:
            await session.execute(text("SELECT 1"))
        checks["database"] = {"ok": True}
    except DATABASE_UNAVAILABLE as exc:
        checks["database"] = {"ok": False, "error": type(exc).__name__}
    llm = request.app.state.llm_info
    checks["llm"] = {"ok": bool(llm["configured"]), **llm}
    ok = all(check["ok"] for check in checks.values())
    return JSONResponse(
        status_code=200 if ok else 503,
        content={"status": "ready" if ok else "not ready", "checks": checks},
    )


@router.get("/metrics")
async def metrics(request: Request) -> dict[str, Any]:
    """A JSON snapshot: tasks and steps by status, tokens and web search recorded, background
    jobs, the provider, uptime. Counts come from the database, so they cover every instance."""
    sessionmaker: async_sessionmaker[AsyncSession] = request.app.state.sessionmaker
    async with sessionmaker() as session:
        tasks = dict(
            (await session.execute(select(Task.status, func.count()).group_by(Task.status))).all()
        )
        steps = dict(
            (
                await session.execute(
                    select(ExecutionStep.status, func.count()).group_by(ExecutionStep.status)
                )
            ).all()
        )
        step_in, step_out, grounded_calls, search_queries = (
            await session.execute(
                select(
                    func.coalesce(func.sum(ExecutionStep.input_tokens), 0),
                    func.coalesce(func.sum(ExecutionStep.output_tokens), 0),
                    func.coalesce(func.sum(ExecutionStep.grounded_calls), 0),
                    func.coalesce(func.sum(ExecutionStep.search_queries), 0),
                )
            )
        ).one()
        plan_in, plan_out = (
            await session.execute(
                select(
                    func.coalesce(func.sum(ExecutionPlan.planner_input_tokens), 0),
                    func.coalesce(func.sum(ExecutionPlan.planner_output_tokens), 0),
                )
            )
        ).one()
        syn_in, syn_out = (
            await session.execute(
                select(
                    func.coalesce(func.sum(TaskResult.input_tokens), 0),
                    func.coalesce(func.sum(TaskResult.output_tokens), 0),
                )
            )
        ).one()
    runner = request.app.state.runner
    return {
        "tasks_by_status": {str(k): v for k, v in tasks.items()},
        "steps_by_status": {str(k): v for k, v in steps.items()},
        "tokens": {
            "planning": {"input": plan_in, "output": plan_out},
            "execution": {"input": step_in, "output": step_out},
            "synthesis": {"input": syn_in, "output": syn_out},
            "total": plan_in + plan_out + step_in + step_out + syn_in + syn_out,
        },
        # Billed apart from tokens. grounded_calls is held to the per-task limit; queries are
        # what the provider reported afterwards (llm/grounding.py).
        "search": {"grounded_calls": grounded_calls, "queries": search_queries},
        "background_jobs_running": getattr(runner, "running", None),
        "llm": request.app.state.llm_info,
        "uptime_seconds": round(time.monotonic() - request.app.state.started_monotonic, 1),
    }
