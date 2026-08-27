"""Unit tests for SoftmaxCrossEntropyLoss and SGDOptimizer."""

import math

import pytest

from prism.core.enums import InitializationStrategy, ModelFamily, TaskType
from prism.core.errors import (
    ConfigurationError,
    ValidationError,
)
from prism.models.linear import LinearSoftmaxClassifier
from prism.models.specifications import ModelSpecification
from prism.training.configuration import OptimizerSpecification
from prism.training.loss import SoftmaxCrossEntropyLoss, compute_accuracy
from prism.training.optimizers import SGDOptimizer, create_optimizer


@pytest.fixture
def linear_model() -> LinearSoftmaxClassifier:
    spec = ModelSpecification(
        model_id="model-linear-opt",
        name="Linear Opt Model",
        family=ModelFamily.LINEAR,
        architecture="linear_softmax",
        compatible_tasks=[TaskType.CLASSIFICATION],
        input_shape=(2,),
        num_classes=2,
        initialization=InitializationStrategy.RANDOM,
    )
    return LinearSoftmaxClassifier(spec=spec, seed=42)


@pytest.mark.unit
def test_softmax_cross_entropy_loss_finite_scalar() -> None:
    """Verify SoftmaxCrossEntropyLoss computes finite loss and gradient."""
    loss_fn = SoftmaxCrossEntropyLoss()
    logits = [[2.0, 1.0], [0.5, 2.5]]
    targets = [0, 1]

    loss, d_logits = loss_fn(logits, targets)
    assert isinstance(loss, float)
    assert loss > 0.0
    assert len(d_logits) == 2
    assert len(d_logits[0]) == 2
    assert len(d_logits[1]) == 2


@pytest.mark.unit
def test_softmax_cross_entropy_extreme_logits_stability() -> None:
    """Verify numerical stability under extreme large logits."""
    loss_fn = SoftmaxCrossEntropyLoss()
    extreme_logits = [[1000.0, 0.0], [-500.0, 500.0]]
    targets = [0, 1]

    loss, _ = loss_fn(extreme_logits, targets)
    assert isinstance(loss, float)
    assert math.isfinite(loss)
    assert loss >= 0.0


@pytest.mark.unit
def test_softmax_cross_entropy_target_validation() -> None:
    """Verify SoftmaxCrossEntropyLoss rejects invalid target indices or lengths."""
    loss_fn = SoftmaxCrossEntropyLoss()
    logits = [[1.0, 2.0]]

    with pytest.raises(ValidationError, match="Target index 5 out of range"):
        loss_fn(logits, [5])

    with pytest.raises(ValidationError, match="Target index -1 out of range"):
        loss_fn(logits, [-1])

    with pytest.raises(ValidationError, match="Batch size mismatch"):
        loss_fn(logits, [0, 1])


@pytest.mark.unit
def test_compute_accuracy_exact() -> None:
    """Verify accuracy computation matches expected ground-truth matches."""
    logits = [[2.0, 0.0], [0.0, 3.0], [1.0, 0.5], [0.1, 0.9]]
    # Predictions: 0, 1, 0, 1
    targets = [0, 1, 1, 1]  # 3 out of 4 match -> 0.75
    acc = compute_accuracy(logits, targets)
    assert acc == 0.75


@pytest.mark.unit
def test_sgd_optimizer_updates_parameters(
    linear_model: LinearSoftmaxClassifier,
) -> None:
    """Verify SGDOptimizer modifies weights in direction of gradients."""
    initial_weights = [row[:] for row in linear_model.weights]
    initial_bias = linear_model.bias[:]

    # Set artificial gradients
    linear_model.grad_weights = [[0.5, -0.5], [1.0, -1.0]]
    linear_model.grad_bias = [0.2, -0.2]

    optimizer = SGDOptimizer(linear_model, lr=0.1)
    optimizer.step()

    updated_weights = linear_model.weights
    updated_bias = linear_model.bias

    # weights = initial - lr * grad -> weights[0][0] = initial[0][0] - 0.1 * 0.5
    assert updated_weights[0][0] == pytest.approx(initial_weights[0][0] - 0.05)
    assert updated_weights[0][1] == pytest.approx(initial_weights[0][1] + 0.05)
    assert updated_bias[0] == pytest.approx(initial_bias[0] - 0.02)
    assert updated_bias[1] == pytest.approx(initial_bias[1] + 0.02)


@pytest.mark.unit
def test_create_optimizer_factory(
    linear_model: LinearSoftmaxClassifier,
) -> None:
    """Verify create_optimizer creates SGDOptimizer or fails for unsupported type."""
    sgd_spec = OptimizerSpecification(type="sgd", lr=0.01)
    opt = create_optimizer(sgd_spec, linear_model)
    assert isinstance(opt, SGDOptimizer)

    bad_spec = OptimizerSpecification(type="unsupported_opt", lr=0.01)
    with pytest.raises(ConfigurationError, match="Unsupported optimizer"):
        create_optimizer(bad_spec, linear_model)
