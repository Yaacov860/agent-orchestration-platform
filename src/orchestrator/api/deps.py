"""Dependency injection: how a request gets its database session and services.

FastAPI builds these per request. Tests swap the database by passing different Settings
to create_app, without touching any route code.
"""

from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from orchestrator.services.planning import PlanningService
from orchestrator.services.tasks import CancellationFanout, TaskService


def get_sessionmaker(request: Request) -> async_sessionmaker[AsyncSession]:
    sessionmaker: async_sessionmaker[AsyncSession] = request.app.state.sessionmaker
    return sessionmaker


async def get_session(
    sessionmaker: Annotated[async_sessionmaker[AsyncSession], Depends(get_sessionmaker)],
) -> AsyncIterator[AsyncSession]:
    """One session per request, closed when the request ends (rolling back anything uncommitted)."""
    async with sessionmaker() as session:
        yield session


def get_planning_service(request: Request) -> PlanningService:
    """App-wide, not per request: planning outlives the request that started it."""
    planning: PlanningService = request.app.state.planning
    return planning


def get_cancellation(request: Request) -> CancellationFanout:
    """Execution and synthesis, both told when a task is cancelled."""
    cancellation: CancellationFanout = request.app.state.cancellation
    return cancellation


def get_task_service(
    session: Annotated[AsyncSession, Depends(get_session)],
    planning: Annotated[PlanningService, Depends(get_planning_service)],
    cancellation: Annotated[CancellationFanout, Depends(get_cancellation)],
) -> TaskService:
    return TaskService(session, planning, cancellation)


TaskServiceDep = Annotated[TaskService, Depends(get_task_service)]
