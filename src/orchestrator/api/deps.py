"""Dependency injection: how a request gets its database session and services.

FastAPI builds these per request. Tests swap the database by passing different Settings
to create_app, without touching any route code.
"""

from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from orchestrator.services.tasks import TaskService


def get_sessionmaker(request: Request) -> async_sessionmaker[AsyncSession]:
    sessionmaker: async_sessionmaker[AsyncSession] = request.app.state.sessionmaker
    return sessionmaker


async def get_session(
    sessionmaker: Annotated[async_sessionmaker[AsyncSession], Depends(get_sessionmaker)],
) -> AsyncIterator[AsyncSession]:
    """One session per request, closed when the request ends (rolling back anything uncommitted)."""
    async with sessionmaker() as session:
        yield session


def get_task_service(session: Annotated[AsyncSession, Depends(get_session)]) -> TaskService:
    return TaskService(session)


TaskServiceDep = Annotated[TaskService, Depends(get_task_service)]
