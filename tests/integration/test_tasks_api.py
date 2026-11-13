from datetime import UTC, datetime, timedelta

import pytest
from factories import make_plan
from httpx import ASGITransport, AsyncClient
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from orchestrator.config import Settings
from orchestrator.db.models import Task
from orchestrator.domain.ids import is_task_id, new_task_id
from orchestrator.domain.status import StepStatus, TaskStatus
from orchestrator.main import create_app

Sessions = async_sessionmaker[AsyncSession]

BLOG_POST = {
    "goal": "Write a comparison blog post about Python vs JavaScript for beginners",
    "constraints": {"max_words": 1500, "tone": "friendly", "include_code_examples": True},
    "output_format": "markdown",
}


async def count_tasks(sessionmaker: Sessions) -> int:
    async with sessionmaker() as session:
        return await session.scalar(select(func.count()).select_from(Task)) or 0


# --- POST /tasks: success ---------------------------------------------------------------


async def test_submit_stores_the_task_and_returns_its_id(
    client: AsyncClient, sessionmaker: Sessions
) -> None:
    response = await client.post("/tasks", json=BLOG_POST)

    assert response.status_code == 201
    body = response.json()
    assert body["status"] == "pending"
    assert is_task_id(body["task_id"])
    assert response.headers["location"] == f"/tasks/{body['task_id']}"

    async with sessionmaker() as session:
        row = await session.get(Task, body["task_id"])
    assert row is not None
    assert row.goal == BLOG_POST["goal"]
    assert row.constraints == BLOG_POST["constraints"]
    assert row.status is TaskStatus.PENDING


async def test_submit_applies_defaults_and_trims_the_goal(
    client: AsyncClient, sessionmaker: Sessions
) -> None:
    response = await client.post("/tasks", json={"goal": "  Explain asyncio  "})

    assert response.status_code == 201
    async with sessionmaker() as session:
        row = await session.get(Task, response.json()["task_id"])
    assert row is not None
    assert (row.goal, row.constraints, row.output_format, row.deadline) == (
        "Explain asyncio",
        {},
        "markdown",
        None,
    )


async def test_unknown_constraints_are_kept_alongside_known_ones(
    client: AsyncClient, sessionmaker: Sessions
) -> None:
    constraints = {"max_words": 300, "audience": "teenagers", "avoid": ["jargon", "slang"]}

    response = await client.post("/tasks", json={"goal": "g", "constraints": constraints})

    async with sessionmaker() as session:
        row = await session.get(Task, response.json()["task_id"])
    assert row is not None
    assert row.constraints == constraints


async def test_a_future_timezone_aware_deadline_is_accepted(client: AsyncClient) -> None:
    deadline = (datetime.now(UTC) + timedelta(hours=1)).isoformat()

    response = await client.post("/tasks", json={"goal": "g", "deadline": deadline})

    assert response.status_code == 201


# --- POST /tasks: validation ------------------------------------------------------------


@pytest.mark.parametrize(
    ("payload", "bad_field"),
    [
        pytest.param({}, "goal", id="missing goal"),
        pytest.param({"goal": "   "}, "goal", id="blank goal"),
        pytest.param({"goal": "x" * 4001}, "goal", id="goal too long"),
        pytest.param({"goal": "g", "constraint": {}}, "constraint", id="unknown field (typo)"),
        pytest.param({"goal": "g", "constraints": {"max_words": 0}}, "max_words", id="max_words 0"),
        pytest.param(
            {"goal": "g", "constraints": {"max_words": "1500"}}, "max_words", id="max_words string"
        ),
        pytest.param(
            {"goal": "g", "constraints": {"include_code_examples": "yes"}},
            "include_code_examples",
            id="bool as 'yes'",
        ),
        pytest.param({"goal": "g", "output_format": "pdf"}, "output_format", id="bad format"),
        pytest.param(
            {"goal": "g", "deadline": "2020-01-01T00:00:00Z"}, "deadline", id="past deadline"
        ),
        pytest.param(
            {"goal": "g", "deadline": "2099-01-01T00:00:00"}, "deadline", id="naive deadline"
        ),
    ],
)
async def test_invalid_submissions_are_rejected_and_nothing_is_stored(
    client: AsyncClient, sessionmaker: Sessions, payload: dict[str, object], bad_field: str
) -> None:
    response = await client.post("/tasks", json=payload)

    assert response.status_code == 422
    error_locations = [error["loc"] for error in response.json()["detail"]]
    assert any(bad_field in loc for loc in error_locations), error_locations
    assert await count_tasks(sessionmaker) == 0


async def test_a_body_that_is_not_json_is_rejected(client: AsyncClient) -> None:
    response = await client.post(
        "/tasks", content=b"{not json", headers={"content-type": "application/json"}
    )

    assert response.status_code == 422


# --- GET /tasks/{id} --------------------------------------------------------------------


async def test_get_returns_the_stored_task_with_empty_progress_before_planning(
    client: AsyncClient,
) -> None:
    task_id = (await client.post("/tasks", json=BLOG_POST)).json()["task_id"]

    response = await client.get(f"/tasks/{task_id}")

    assert response.status_code == 200
    body = response.json()
    assert body["task_id"] == task_id
    assert body["status"] == "pending"
    assert body["goal"] == BLOG_POST["goal"]
    assert body["constraints"] == BLOG_POST["constraints"]
    assert body["error"] is None
    assert body["created_at"] and body["updated_at"]
    assert body["progress"] == {
        "total_steps": 0,
        "completed_steps": 0,
        "failed_steps": 0,
        "current_step": None,
        "running_steps": [],
    }


async def test_progress_counts_the_steps_of_the_latest_plan_only(
    client: AsyncClient, sessionmaker: Sessions
) -> None:
    task_id = (await client.post("/tasks", json=BLOG_POST)).json()["task_id"]
    async with sessionmaker() as session:
        session.add(make_plan(task_id, 1, [("step_1", StepStatus.FAILED)]))  # superseded
        session.add(
            make_plan(
                task_id,
                2,
                [
                    ("step_1", StepStatus.COMPLETED),
                    ("step_2", StepStatus.RUNNING),
                    ("step_3", StepStatus.RUNNING),
                    ("step_4", StepStatus.PENDING),
                ],
            )
        )
        await session.commit()

    progress = (await client.get(f"/tasks/{task_id}")).json()["progress"]

    assert progress == {
        "total_steps": 4,
        "completed_steps": 1,
        "failed_steps": 0,
        "current_step": "step_2",
        "running_steps": ["step_2", "step_3"],
    }


async def test_get_an_unknown_task_is_404(client: AsyncClient) -> None:
    response = await client.get(f"/tasks/{new_task_id()}")

    assert response.status_code == 404
    assert "not found" in response.json()["detail"]


@pytest.mark.parametrize("bad_id", ["nope", "task_123", "task_" + "G" * 32, "1; DROP TABLE tasks"])
async def test_get_a_malformed_id_is_404(client: AsyncClient, bad_id: str) -> None:
    response = await client.get(f"/tasks/{bad_id}")

    assert response.status_code == 404


# --- infrastructure failure -------------------------------------------------------------


async def test_database_unavailable_is_503_with_retry_after() -> None:
    app = create_app(Settings(database_url="postgresql+asyncpg://u:p@127.0.0.1:1/nothing"))
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        submit = await client.post("/tasks", json={"goal": "g"})
        status = await client.get(f"/tasks/{new_task_id()}")

    for response in (submit, status):
        assert response.status_code == 503
        assert response.headers["retry-after"] == "5"
        assert response.json() == {"detail": "database unavailable, try again shortly"}
