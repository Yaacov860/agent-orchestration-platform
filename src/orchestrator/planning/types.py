"""The shape of a plan, as the planner model must return it.

The model's output is untrusted input, exactly like a client's request body: every field is
typed, strict and bounded, and unknown fields are errors (a typo such as `dependecies` would
otherwise silently drop the dependencies and run steps in the wrong order).
"""

from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict, Field

STEP_ID_PATTERN = r"^step_[1-9][0-9]*$"

_STRICT = ConfigDict(frozen=True, extra="forbid", strict=True, str_strip_whitespace=True)


class PlanStep(BaseModel):
    model_config = _STRICT

    # Lengths match the execution_steps columns (String(50)), so a valid step always fits.
    step_id: str = Field(pattern=STEP_ID_PATTERN, max_length=50)
    agent: str = Field(min_length=1, max_length=50)
    action: str = Field(min_length=1, max_length=50)
    input: str = Field(min_length=1, max_length=4_000)
    """The instruction for this step: what the agent must do."""
    dependencies: tuple[str, ...] = Field(default=(), max_length=20)
    """Ids of the steps whose output this step needs."""


class Plan(BaseModel):
    model_config = _STRICT

    reasoning: str = Field(default="", max_length=2_000)
    """Why the model split the task this way. Stored with the plan, for inspection."""
    steps: tuple[PlanStep, ...]


@dataclass(frozen=True)
class ValidatedPlan:
    """A plan that passed every check, with the parallel groups computed from it.

    Only `validate_plan` builds one, so holding a ValidatedPlan is proof the plan was checked
    ("parse, don't validate"): later code never has to ask whether the graph is sound.
    """

    plan: Plan
    parallel_groups: tuple[tuple[str, ...], ...]
