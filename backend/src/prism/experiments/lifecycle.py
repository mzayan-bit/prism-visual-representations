"""Lifecycle state machine and transition rules for experiment runs."""

from prism.core.enums import RunStatus
from prism.core.errors import InvalidTransitionError

# Directed graph of legal lifecycle transitions
ALLOWED_TRANSITIONS: dict[RunStatus, frozenset[RunStatus]] = {
    RunStatus.PLANNED: frozenset(
        {
            RunStatus.QUEUED,
            RunStatus.RUNNING,
            RunStatus.CANCELLED,
        }
    ),
    RunStatus.QUEUED: frozenset(
        {
            RunStatus.RUNNING,
            RunStatus.CANCELLED,
        }
    ),
    RunStatus.RUNNING: frozenset(
        {
            RunStatus.COMPLETED,
            RunStatus.FAILED,
            RunStatus.CANCELLED,
        }
    ),
    RunStatus.COMPLETED: frozenset(),
    RunStatus.FAILED: frozenset(),
    RunStatus.CANCELLED: frozenset(),
}


def is_valid_transition(current: RunStatus, target: RunStatus) -> bool:
    """Return True if transitioning from current to target status is valid."""
    allowed = ALLOWED_TRANSITIONS.get(current, frozenset())
    return target in allowed


def validate_transition(
    current: RunStatus,
    target: RunStatus,
    run_id: str | None = None,
    reason: str | None = None,
) -> None:
    """Validate a lifecycle transition, raising InvalidTransitionError if disallowed."""
    if not is_valid_transition(current, target):
        detail_msg = reason
        if current.is_terminal:
            detail_msg = f"Run is already in terminal state '{current.value}'."
        raise InvalidTransitionError(
            current_status=current.value,
            target_status=target.value,
            run_id=run_id,
            reason=detail_msg,
        )
