import re
from uuid import uuid4

_TASK_ID = re.compile(r"task_[0-9a-f]{32}")


def new_task_id() -> str:
    """Prefixed random id: unguessable, and readable at a glance in logs and URLs."""
    return f"task_{uuid4().hex}"


def is_task_id(value: str) -> bool:
    return _TASK_ID.fullmatch(value) is not None
