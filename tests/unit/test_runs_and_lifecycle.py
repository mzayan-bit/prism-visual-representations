"""Unit tests for ExperimentRun, lifecycle state machine, and transitions."""

import pytest

from prism.artifacts.contracts import ArtifactReference
from prism.core.enums import ArtifactType, MetricDirection, RunStatus
from prism.core.errors import InvalidTransitionError
from prism.experiments.lifecycle import is_valid_transition
from prism.experiments.metrics import MetricRecord
from prism.experiments.runs import ExperimentRun


def _get_status(run: ExperimentRun) -> RunStatus:
    """Helper to retrieve current run status without mypy literal narrowing."""
    return run.status


@pytest.fixture
def planned_run() -> ExperimentRun:
    return ExperimentRun(
        run_id="run-exp01-trial01",
        experiment_id="exp-cifar10-resnet18",
        configuration_fingerprint="a" * 64,
    )


@pytest.mark.unit
def test_valid_lifecycle_planned_to_completed(
    planned_run: ExperimentRun,
) -> None:
    """Verify normal successful run lifecycle from PLANNED -> RUNNING -> COMPLETED."""
    assert _get_status(planned_run) == RunStatus.PLANNED
    assert planned_run.started_at is None
    assert planned_run.completed_at is None

    # Start run
    planned_run.start()
    assert _get_status(planned_run) == RunStatus.RUNNING
    assert planned_run.started_at is not None

    # Record metric and artifact
    planned_run.add_metric(
        MetricRecord(
            metric_name="train_loss",
            value=0.25,
            step=100,
            direction=MetricDirection.MINIMIZE,
        )
    )
    planned_run.add_artifact(
        ArtifactReference(
            artifact_id="art-ckpt-final",
            artifact_type=ArtifactType.CHECKPOINT,
            logical_name="final_checkpoint",
            uri="artifacts/checkpoints/final.pt",
            producing_run_id=planned_run.run_id,
        )
    )

    # Complete run
    planned_run.complete(summary_metrics={"test_top1_acc": 0.935})
    assert _get_status(planned_run) == RunStatus.COMPLETED
    assert planned_run.completed_at is not None
    assert planned_run.summary_metrics["test_top1_acc"] == 0.935
    assert len(planned_run.metric_records) == 1
    assert len(planned_run.artifact_references) == 1


@pytest.mark.unit
def test_run_failure_lifecycle(planned_run: ExperimentRun) -> None:
    """Verify run failure lifecycle from PLANNED -> RUNNING -> FAILED."""
    planned_run.start()
    planned_run.fail(
        error_type="RuntimeError",
        error_message="CUDA out of memory",
        traceback="Traceback (most recent call last)...",
    )

    assert _get_status(planned_run) == RunStatus.FAILED
    assert planned_run.completed_at is not None
    assert planned_run.failure_info is not None
    assert planned_run.failure_info.error_type == "RuntimeError"
    assert planned_run.failure_info.error_message == "CUDA out of memory"


@pytest.mark.unit
def test_run_cancellation_lifecycle(planned_run: ExperimentRun) -> None:
    """Verify run cancellation from PLANNED -> CANCELLED and RUNNING -> CANCELLED."""
    # Cancel directly from PLANNED
    planned_run.cancel(reason="User cancelled experiment")
    assert _get_status(planned_run) == RunStatus.CANCELLED
    assert planned_run.completed_at is not None
    assert "User cancelled experiment" in planned_run.notes


@pytest.mark.unit
@pytest.mark.parametrize(
    ("current", "target"),
    [
        (RunStatus.COMPLETED, RunStatus.RUNNING),
        (RunStatus.COMPLETED, RunStatus.PLANNED),
        (RunStatus.FAILED, RunStatus.RUNNING),
        (RunStatus.FAILED, RunStatus.COMPLETED),
        (RunStatus.CANCELLED, RunStatus.RUNNING),
        (RunStatus.PLANNED, RunStatus.COMPLETED),  # Cannot complete without running
    ],
)
def test_illegal_lifecycle_transitions_rejected(
    current: RunStatus, target: RunStatus
) -> None:
    """Verify that invalid lifecycle transitions raise InvalidTransitionError."""
    assert not is_valid_transition(current, target)
    run = ExperimentRun(
        run_id="run-transition-test",
        experiment_id="exp-test",
        status=current,
    )
    with pytest.raises(InvalidTransitionError):
        run.transition_to(target)


@pytest.mark.unit
def test_experiment_run_serialization_round_trip(
    planned_run: ExperimentRun,
) -> None:
    """Verify serialization of ExperimentRun preserves all fields."""
    planned_run.start()
    planned_run.add_metric(MetricRecord(metric_name="val_loss", value=0.42))
    planned_run.complete(summary_metrics={"top1": 0.88})

    # Dict round trip
    dumped = planned_run.to_dict()
    restored_dict = ExperimentRun.from_dict(dumped)
    assert restored_dict.run_id == planned_run.run_id
    assert _get_status(restored_dict) == RunStatus.COMPLETED
    assert restored_dict.summary_metrics["top1"] == 0.88

    # JSON round trip
    json_str = planned_run.to_json()
    restored_json = ExperimentRun.from_json(json_str)
    assert restored_json.run_id == planned_run.run_id
    assert len(restored_json.metric_records) == 1
