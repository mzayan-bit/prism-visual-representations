"""Unit tests for TrainingConfiguration and optimizer policies."""

import pytest
from pydantic import ValidationError as PydanticValidationError

from prism.core.enums import DevicePreference, MetricDirection, PrecisionMode
from prism.core.errors import ValidationError
from prism.training.configuration import (
    EarlyStoppingPolicy,
    GradientClipping,
    OptimizerSpecification,
    SchedulerSpecification,
    TrainingConfiguration,
)


@pytest.mark.unit
def test_valid_training_configuration() -> None:
    """Verify construction of a complete, valid TrainingConfiguration."""
    config = TrainingConfiguration(
        epochs=100,
        batch_size=128,
        optimizer=OptimizerSpecification(
            type="adamw",
            lr=1e-3,
            weight_decay=1e-4,
            extra_kwargs={"betas": (0.9, 0.999)},
        ),
        scheduler=SchedulerSpecification(
            type="cosine",
            warmup_epochs=5,
            min_lr=1e-6,
        ),
        gradient_clipping=GradientClipping(
            enabled=True,
            max_norm=1.0,
        ),
        precision=PrecisionMode.AMP,
        device=DevicePreference.CUDA,
        early_stopping=EarlyStoppingPolicy(
            enabled=True,
            monitor_metric="val_loss",
            patience=10,
            mode=MetricDirection.MINIMIZE,
        ),
        gradient_accumulation_steps=2,
    )

    assert config.epochs == 100
    assert config.batch_size == 128
    assert config.optimizer.type == "adamw"
    assert config.optimizer.lr == 1e-3
    assert config.scheduler.warmup_epochs == 5
    assert config.gradient_clipping.enabled is True
    assert config.gradient_clipping.max_norm == 1.0
    assert config.precision == PrecisionMode.AMP
    assert config.gradient_accumulation_steps == 2


@pytest.mark.unit
@pytest.mark.parametrize(
    ("epochs", "batch_size", "lr"),
    [
        (0, 32, 1e-3),  # epochs <= 0
        (-5, 32, 1e-3),  # negative epochs
        (10, 0, 1e-3),  # batch_size <= 0
        (10, -32, 1e-3),  # negative batch_size
        (10, 32, 0.0),  # lr <= 0
        (10, 32, -0.01),  # negative lr
    ],
)
def test_invalid_training_parameters_rejected(
    epochs: int, batch_size: int, lr: float
) -> None:
    """Verify non-positive training budgets and rates are rejected."""
    with pytest.raises((PydanticValidationError, ValueError)):
        TrainingConfiguration(
            epochs=epochs,
            batch_size=batch_size,
            optimizer=OptimizerSpecification(type="sgd", lr=lr),
        )


@pytest.mark.unit
def test_clipping_enabled_without_max_norm_rejected() -> None:
    """Verify enabling gradient clipping without max_norm raises ValidationError."""
    with pytest.raises(ValidationError, match="max_norm is not specified"):
        GradientClipping(enabled=True, max_norm=None)
