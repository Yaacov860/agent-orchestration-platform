"""The database itself enforces the model's rules, not only the Python code above it."""

import asyncio

import pytest
from alembic import command
from conftest import TEST_DATABASE_URL, alembic_config, recreate_database
from factories import make_plan, make_task
from sqlalchemy import func, inspect, make_url, select, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from orchestrator.db.models import ExecutionPlan, ExecutionStep, Task
from orchestrator.domain.status import StepStatus

Sessions = async_sessionmaker[AsyncSession]


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
    url = (
        make_url(TEST_DATABASE_URL)
        .set(database="orchestrator_migration_test")
        .render_as_string(hide_password=False)
    )
    asyncio.run(recreate_database(url))
    config = alembic_config(url)

    command.upgrade(config, "head")
    assert asyncio.run(table_names(url)) >= {"tasks", "execution_plans", "execution_steps"}

    command.downgrade(config, "base")
    assert asyncio.run(table_names(url)) == {"alembic_version"}

    command.upgrade(config, "head")
    assert "execution_steps" in asyncio.run(table_names(url))


async def table_names(url: str) -> set[str]:
    engine = create_async_engine(url)
    try:
        async with engine.connect() as connection:
            names = await connection.run_sync(lambda conn: inspect(conn).get_table_names())
            return set(names)
    finally:
        await engine.dispose()
