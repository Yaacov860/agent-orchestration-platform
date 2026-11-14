"""The planner and the execution engine stay apart: they meet only in the database and in
main.py's wiring. Checked from the source, so a stray import fails here, not in review."""

import ast
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[2] / "src" / "orchestrator"


def imported_modules(path: Path) -> set[str]:
    modules: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    return modules


def imports_of(*parts: str) -> dict[str, set[str]]:
    target = SRC.joinpath(*parts)
    files = sorted(target.rglob("*.py")) if target.is_dir() else [target.with_suffix(".py")]
    return {str(f.relative_to(SRC)): imported_modules(f) for f in files}


@pytest.mark.parametrize(
    ("component", "forbidden"),
    [
        (("execution",), "orchestrator.planning"),
        (("services", "execution"), "orchestrator.planning"),
        (("services", "execution"), "orchestrator.services.planning"),
        (("planning",), "orchestrator.execution"),
        (("services", "planning"), "orchestrator.execution"),
        (("services", "planning"), "orchestrator.services.execution"),
        (("execution",), "orchestrator.db"),  # the executor itself is pure orchestration
        (("synthesis",), "orchestrator.planning"),
        (("synthesis",), "orchestrator.execution"),
        (("synthesis",), "orchestrator.db"),  # pure: gets the rows as plain data
        (("synthesis",), "orchestrator.services"),
        (("services", "synthesis"), "orchestrator.planning"),
        (("services", "synthesis"), "orchestrator.execution"),
        (("services", "synthesis"), "orchestrator.services.execution"),
        (("services", "synthesis"), "orchestrator.services.planning"),
        (("execution",), "orchestrator.synthesis"),
        (("services", "execution"), "orchestrator.synthesis"),
        (("services", "execution"), "orchestrator.services.synthesis"),
        (("planning",), "orchestrator.synthesis"),
    ],
)
def test_planning_and_execution_do_not_import_each_other(
    component: tuple[str, ...], forbidden: str
) -> None:
    offenders = {
        file: sorted(m for m in modules if m == forbidden or m.startswith(forbidden + "."))
        for file, modules in imports_of(*component).items()
    }
    assert {f: m for f, m in offenders.items() if m} == {}
