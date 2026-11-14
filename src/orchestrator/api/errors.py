import logging

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from sqlalchemy.exc import InterfaceError, OperationalError

from orchestrator.services.tasks import TaskNotCancellableError, TaskNotFoundError

logger = logging.getLogger(__name__)

# Only "could not reach / lost the database" errors. A constraint violation is a bug in
# our code, not a transient outage, so it stays a 500.
DATABASE_UNAVAILABLE = (OperationalError, InterfaceError, ConnectionError, TimeoutError)


async def _task_not_found(_: Request, exc: Exception) -> JSONResponse:
    return JSONResponse(status_code=404, content={"detail": str(exc)})


async def _task_not_cancellable(_: Request, exc: Exception) -> JSONResponse:
    return JSONResponse(status_code=409, content={"detail": str(exc)})


async def _database_unavailable(_: Request, exc: Exception) -> JSONResponse:
    logger.error("database unavailable: %s", exc)
    return JSONResponse(
        status_code=503,
        content={"detail": "database unavailable, try again shortly"},
        headers={"Retry-After": "5"},
    )


def register_error_handlers(app: FastAPI) -> None:
    app.add_exception_handler(TaskNotFoundError, _task_not_found)
    app.add_exception_handler(TaskNotCancellableError, _task_not_cancellable)
    for error_type in DATABASE_UNAVAILABLE:
        app.add_exception_handler(error_type, _database_unavailable)
