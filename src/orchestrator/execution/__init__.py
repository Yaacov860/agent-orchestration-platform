"""Execution: run a stored plan's steps on agents, in dependency order, within a cap.

Independent of the planning package by design (an architecture test enforces it): plans
reach the executor only as stored rows.
"""

from orchestrator.execution.executor import DependencyExecutor
from orchestrator.execution.types import (
    ExecutionJob,
    ExecutionSummary,
    StepOutcome,
    StepRecorder,
    StepSpec,
)

__all__ = [
    "DependencyExecutor",
    "ExecutionJob",
    "ExecutionSummary",
    "StepOutcome",
    "StepRecorder",
    "StepSpec",
]
