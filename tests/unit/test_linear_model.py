"""Unit tests for LinearSoftmaxClassifier and linear parameter initialization."""

import pytest

from prism.core.enums import InitializationStrategy, ModelFamily, TaskType
from prism.core.errors import ValidationError
from prism.models.initialization import initialize_linear_parameters
from prism.models.linear import LinearSoftmaxClassifier
from prism.models.specifications import ModelSpecification


@pytest.fixture
def linear_spec() -> ModelSpecification:
    return ModelSpecification(
        model_id="model-linear-test",
        name="Linear Test Model",
        family=ModelFamily.LINEAR,
        architecture="linear_softmax",
        compatible_tasks=[TaskType.CLASSIFICATION],
        input_shape=(3, 4, 4),  # 3*4*4 = 48 features
        num_classes=3,
        initialization=InitializationStrategy.RANDOM,
    )


@pytest.mark.unit
def test_initialize_linear_parameters_deterministic() -> None:
    """Verify deterministic parameter initialization under seeds."""
    w1, b1 = initialize_linear_parameters(in_features=10, num_classes=3, seed=42)
    w2, b2 = initialize_linear_parameters(in_features=10, num_classes=3, seed=42)
    w3, _ = initialize_linear_parameters(in_features=10, num_classes=3, seed=999)

    assert len(w1) == 10
    assert len(w1[0]) == 3
    assert len(b1) == 3
    assert b1 == [0.0, 0.0, 0.0]

    assert w1 == w2
    assert b1 == b2
    assert w1 != w3


@pytest.mark.unit
def test_initialize_linear_parameters_invalid_dims() -> None:
    """Verify initialization rejects non-positive dimensions."""
    with pytest.raises(ValidationError, match="in_features must be positive"):
        initialize_linear_parameters(in_features=0, num_classes=3)

    with pytest.raises(ValidationError, match="num_classes must be positive"):
        initialize_linear_parameters(in_features=10, num_classes=0)


@pytest.mark.unit
def test_linear_model_forward_shape(linear_spec: ModelSpecification) -> None:
    """Verify LinearSoftmaxClassifier flattens input and outputs [B, num_classes]."""
    model = LinearSoftmaxClassifier(spec=linear_spec, seed=42)
    assert model.in_features == 48
    assert model.num_classes == 3

    # Batch of 2 samples, each shaped (3, 4, 4)
    sample1 = [[[0.1] * 4 for _ in range(4)] for _ in range(3)]
    sample2 = [[[0.5] * 4 for _ in range(4)] for _ in range(3)]
    batch_input = [sample1, sample2]

    logits = model.forward(batch_input)
    assert len(logits) == 2
    assert len(logits[0]) == 3
    assert len(logits[1]) == 3


@pytest.mark.unit
def test_linear_model_dimension_validation(linear_spec: ModelSpecification) -> None:
    """Verify LinearSoftmaxClassifier raises error on mismatched dims."""
    model = LinearSoftmaxClassifier(spec=linear_spec, seed=42)

    # Sample with only 10 features instead of 48
    invalid_sample = [[1.0] * 10]
    with pytest.raises(ValidationError, match="expected in_features=48"):
        model.forward(invalid_sample)


@pytest.mark.unit
def test_linear_model_backward_and_gradients(linear_spec: ModelSpecification) -> None:
    """Verify backward pass computes non-zero gradients and zero_grad clears them."""
    model = LinearSoftmaxClassifier(spec=linear_spec, seed=42)
    sample = [[[0.5] * 4 for _ in range(4)] for _ in range(3)]
    _ = model.forward([sample])

    d_logits = [[0.1, -0.2, 0.1]]
    model.backward(d_logits)

    grads = model.get_gradients()
    assert "grad_weights" in grads
    assert "grad_bias" in grads
    assert len(grads["grad_weights"]) == 48
    assert len(grads["grad_bias"]) == 3
    assert grads["grad_bias"] == [0.1, -0.2, 0.1]

    model.zero_grad()
    cleared = model.get_gradients()
    assert all(all(v == 0.0 for v in row) for row in cleared["grad_weights"])
    assert cleared["grad_bias"] == [0.0, 0.0, 0.0]
