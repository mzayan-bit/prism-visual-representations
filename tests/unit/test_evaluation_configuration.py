"""Unit tests for EvaluationConfiguration."""

import pytest
from pydantic import ValidationError as PydanticValidationError

from prism.core.enums import MetricDirection
from prism.evaluation.configuration import (
    EvaluationConfiguration,
    MetricSpecification,
)


@pytest.mark.unit
def test_valid_evaluation_configuration() -> None:
    """Verify construction of valid EvaluationConfiguration."""
    config = EvaluationConfiguration(
        target_splits=["val", "test", "ood"],
        metrics=[
            MetricSpecification(
                name="top1_accuracy",
                direction=MetricDirection.MAXIMIZE,
                target_split="test",
            ),
            MetricSpecification(
                name="cross_entropy_loss",
                direction=MetricDirection.MINIMIZE,
                target_split="val",
            ),
            MetricSpecification(
                name="ece",
                direction=MetricDirection.MINIMIZE,
                target_split="ood",
                params={"num_bins": 15},
            ),
        ],
        batch_size=128,
        save_predictions=True,
        compute_per_class=True,
        confidence_threshold=0.8,
    )

    assert config.target_splits == ["val", "test", "ood"]
    assert len(config.metrics) == 3
    assert config.save_predictions is True
    assert config.compute_per_class is True
    assert config.confidence_threshold == 0.8


@pytest.mark.unit
def test_empty_metrics_rejected() -> None:
    """Verify that an evaluation configuration without metrics is rejected."""
    with pytest.raises((PydanticValidationError, ValueError)):
        EvaluationConfiguration(
            target_splits=["test"],
            metrics=[],
        )


@pytest.mark.unit
def test_empty_target_splits_rejected() -> None:
    """Verify that an evaluation configuration without target splits is rejected."""
    with pytest.raises((PydanticValidationError, ValueError)):
        EvaluationConfiguration(
            target_splits=[],
            metrics=[MetricSpecification(name="loss")],
        )


@pytest.mark.unit
def test_invalid_confidence_threshold_rejected() -> None:
    """Verify confidence threshold outside [0.0, 1.0] is rejected."""
    with pytest.raises((PydanticValidationError, ValueError)):
        EvaluationConfiguration(
            metrics=[MetricSpecification(name="accuracy")],
            confidence_threshold=1.5,
        )
