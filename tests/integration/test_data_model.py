"""The database itself enforces the model's rules, not only the Python code above it."""

import asyncio

import pytest
from alembic import command
from conftest import TEST_DATABASE_URL, alembic_config, recreate_database
from factories import make_plan, make_task
from sqlalchemy import func, inspect, make_url, select, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from orchestrator.db.models import ExecutionPlan, ExecutionStep, Task, TaskResult
from orchestrator.domain.status import StepStatus, TaskStatus

Sessions = async_sessionmaker[AsyncSession]
MIGRATION_TEST_DATABASE = "orchestrator_migration_test"


async def test_a_step_id_cannot_repeat_within_one_plan(sessionmaker: Sessions) -> None:
    task = make_task()
    async with sessionmaker() as session:
        session.add(task)
        session.add(
            make_plan(task.id, 1, [("step_1", StepStatus.PENDING), ("step_1", StepStatus.PENDING)])
        )
        with pytest.raises(IntegrityError, match="uq_execution_steps_plan_id_step_id"):
            await session.commit()


async def test_a_task_cannot_have_two_plans_with_the_same_version(sessionmaker: Sessions) -> None:
    task = make_task()
    async with sessionmaker() as session:
        session.add(task)
        session.add(make_plan(task.id, 1, []))
        await session.commit()
        session.add(make_plan(task.id, 1, []))
        with pytest.raises(IntegrityError, match="uq_execution_plans_task_id_version"):
            await session.commit()


async def test_the_database_rejects_an_unknown_status_even_bypassing_the_orm(
    sessionmaker: Sessions,
) -> None:
    async with sessionmaker() as session:
        with pytest.raises(IntegrityError, match="ck_tasks_task_status"):
            await session.execute(
                text(
                    "INSERT INTO tasks (id, goal, output_format, status) "
                    "VALUES ('task_x', 'g', 'markdown', 'bogus')"
                )
            )


@pytest.mark.parametrize("status", list(TaskStatus))
async def test_the_database_accepts_every_task_status_the_code_defines(
    sessionmaker: Sessions, status: TaskStatus
) -> None:
    """Catches drift between the Python enum and the migrated CHECK constraint:
    a status added to TaskStatus without a migration fails here, not in production."""
    async with sessionmaker() as session:
        await session.execute(
            text(
                "INSERT INTO tasks (id, goal, output_format, status) "
                "VALUES ('task_x', 'g', 'markdown', :status)"
            ),
            {"status": status.value},
        )


async def test_deleting_a_task_deletes_its_plans_and_steps(sessionmaker: Sessions) -> None:
    task = make_task()
    async with sessionmaker() as session:
        session.add(task)
        session.add(make_plan(task.id, 1, [("step_1", StepStatus.COMPLETED)]))
        await session.commit()

        await session.execute(text("DELETE FROM tasks WHERE id = :id"), {"id": task.id})
        await session.commit()

        for model in (Task, ExecutionPlan, ExecutionStep):
            assert await session.scalar(select(func.count()).select_from(model)) == 0


def test_the_migration_downgrades_and_upgrades_cleanly() -> None:
    """Sync on purpose: Alembic's env.py runs its own event loop with asyncio.run."""
    # render_as_string(hide_password=False): str() of a SQLAlchemy URL masks the password as ***
    url = fresh_migration_database()
    config = alembic_config(url)

    command.upgrade(config, "head")
    assert asyncio.run(table_names(url)) >= {"tasks", "execution_plans", "execution_steps"}

    command.downgrade(config, "base")
    assert asyncio.run(table_names(url)) == {"alembic_version"}

    command.upgrade(config, "head")
    assert "execution_steps" in asyncio.run(table_names(url))


INSERT_PLANNED_TASK = (
    "INSERT INTO tasks (id, goal, output_format, status) "
    "VALUES ('task_planned', 'g', 'markdown', 'planned')"
)


def test_downgrading_below_0002_refuses_while_planned_tasks_exist() -> None:
    """0001 has no 'planned' status, so a downgrade must not silently lose or rewrite those rows."""
    url = fresh_migration_database()
    config = alembic_config(url)
    command.upgrade(config, "head")
    asyncio.run(execute(url, INSERT_PLANNED_TASK))

    with pytest.raises(RuntimeError, match="1 task\\(s\\) have status 'planned'"):
        command.downgrade(config, "0001")

    asyncio.run(execute(url, "DELETE FROM tasks"))
    command.downgrade(config, "0001")

    # Back on 0001, the old constraint is in place again and rejects the new status.
    with pytest.raises(IntegrityError, match="ck_tasks_task_status"):
        asyncio.run(execute(url, INSERT_PLANNED_TASK))


def fresh_migration_database() -> str:
    """An empty database of its own, so migration tests never touch the shared test database."""
    # render_as_string(hide_password=False): str() of a SQLAlchemy URL masks the password as ***
    url = (
        make_url(TEST_DATABASE_URL)
        .set(database=MIGRATION_TEST_DATABASE)
        .render_as_string(hide_password=False)
    )
    asyncio.run(recreate_database(url))
    return url


async def execute(url: str, sql: str) -> None:
    engine = create_async_engine(url)
    try:
        async with engine.begin() as connection:
            await connection.execute(text(sql))
    finally:
        await engine.dispose()


async def table_names(url: str) -> set[str]:
    engine = create_async_engine(url)
    try:
        async with engine.connect() as connection:
            names = await connection.run_sync(lambda conn: inspect(conn).get_table_names())
            return set(names)
    finally:
        await engine.dispose()


def test_downgrading_below_0003_refuses_while_tasks_are_synthesizing() -> None:
    url = fresh_migration_database()
    config = alembic_config(url)
    command.upgrade(config, "head")
    asyncio.run(
        execute(
            url,
            "INSERT INTO tasks (id, goal, output_format, status) "
            "VALUES ('task_s', 'g', 'markdown', 'synthesizing')",
        )
    )

    with pytest.raises(RuntimeError, match="1 task\\(s\\) have status 'synthesizing'"):
        command.downgrade(config, "0002")

    asyncio.run(execute(url, "DELETE FROM tasks"))
    command.downgrade(config, "0002")
    assert "task_results" not in asyncio.run(table_names(url))
    command.upgrade(config, "head")
    assert "task_results" in asyncio.run(table_names(url))


async def test_the_database_rejects_an_unknown_failure_stage(sessionmaker: Sessions) -> None:
    async with sessionmaker() as session:
        with pytest.raises(IntegrityError, match="ck_tasks_failure_stage"):
            await session.execute(
                text(
                    "INSERT INTO tasks (id, goal, output_format, status, failure_stage) "
                    "VALUES ('task_x', 'g', 'markdown', 'failed', 'somewhere')"
                )
            )


async def test_deleting_a_task_deletes_its_result(sessionmaker: Sessions) -> None:
    task = make_task(status=TaskStatus.COMPLETED)
    plan = make_plan(task.id, 1, [("step_1", StepStatus.COMPLETED)])
    async with sessionmaker() as session:
        session.add_all([task, plan])
        await session.flush()
        session.add(
            TaskResult(
                task_id=task.id,
                plan_id=plan.id,
                content="c",
                key_findings=[],
                sources=[],
                conflicts=[],
                limitations=[],
                confidence=0.5,
                confidence_basis="b",
                partial=False,
                synthesized_from=["step_1"],
                input_tokens=0,
                output_tokens=0,
                attempts=1,
            )
        )
        await session.commit()
        await session.execute(text("DELETE FROM tasks WHERE id = :id"), {"id": task.id})
        await session.commit()
        assert await session.scalar(select(func.count()).select_from(TaskResult)) == 0


async def test_the_database_rejects_a_confidence_outside_zero_to_one(
    sessionmaker: Sessions,
) -> None:
    task = make_task(status=TaskStatus.COMPLETED)
    plan = make_plan(task.id, 1, [])
    async with sessionmaker() as session:
        session.add_all([task, plan])
        await session.flush()
        session.add(
            TaskResult(
                task_id=task.id,
                plan_id=plan.id,
                content="c",
                key_findings=[],
                sources=[],
                conflicts=[],
                limitations=[],
                confidence=1.2,
                confidence_basis="b",
                partial=False,
                synthesized_from=[],
                input_tokens=0,
                output_tokens=0,
                attempts=1,
            )
        )
        with pytest.raises(IntegrityError, match="ck_task_results_confidence_range"):
            await session.commit()
