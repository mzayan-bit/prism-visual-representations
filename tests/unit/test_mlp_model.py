"""Unit tests for MultiLayerPerceptron architecture, parameters, and backprop."""

import pytest

from prism.core.enums import ModelFamily, TaskType
from prism.core.errors import ValidationError
from prism.models.initialization import initialize_mlp_parameters
from prism.models.mlp import MultiLayerPerceptron
from prism.models.specifications import ModelSpecification


@pytest.fixture
def mlp_spec_single_hidden() -> ModelSpecification:
    return ModelSpecification(
        model_id="model-mlp-1h",
        name="1-Hidden MLP Model",
        family=ModelFamily.MLP,
        architecture="mlp",
        compatible_tasks=[TaskType.CLASSIFICATION],
        input_shape=(3, 4, 4),  # 48 features
        num_classes=3,
        hyperparameters={"hidden_dims": [64], "activation": "relu"},
    )


@pytest.fixture
def mlp_spec_multi_hidden() -> ModelSpecification:
    return ModelSpecification(
        model_id="model-mlp-multi",
        name="Multi-Hidden MLP Model",
        family=ModelFamily.MLP,
        architecture="mlp",
        compatible_tasks=[TaskType.CLASSIFICATION],
        input_shape=(3, 4, 4),  # 48 features
        num_classes=3,
        hyperparameters={
            "hidden_dims": [128, 64],
            "activation": "gelu",
            "dropout": 0.0,
        },
    )


@pytest.mark.unit
def test_initialize_mlp_parameters_deterministic() -> None:
    """Verify initialize_mlp_parameters is deterministic for given seed."""
    w1, b1 = initialize_mlp_parameters(
        in_features=48, hidden_dims=[64, 32], num_classes=3, seed=42
    )
    w2, b2 = initialize_mlp_parameters(
        in_features=48, hidden_dims=[64, 32], num_classes=3, seed=42
    )
    w3, _ = initialize_mlp_parameters(
        in_features=48, hidden_dims=[64, 32], num_classes=3, seed=999
    )

    assert len(w1) == 3  # 3 weight matrices: 48->64, 64->32, 32->3
    assert len(b1) == 3  # 3 bias vectors: 64, 32, 3
    assert len(w1[0]) == 48 and len(w1[0][0]) == 64
    assert len(w1[1]) == 64 and len(w1[1][0]) == 32
    assert len(w1[2]) == 32 and len(w1[2][0]) == 3

    assert w1 == w2
    assert b1 == b2
    assert w1 != w3


@pytest.mark.unit
def test_initialize_mlp_parameters_invalid_dims() -> None:
    """Verify initialization rejects non-positive dimensions or empty hidden dims."""
    with pytest.raises(ValidationError, match="in_features must be positive"):
        initialize_mlp_parameters(in_features=0, hidden_dims=[64], num_classes=3)

    with pytest.raises(ValidationError, match="hidden_dims cannot be empty"):
        initialize_mlp_parameters(in_features=48, hidden_dims=[], num_classes=3)

    with pytest.raises(ValidationError, match="must be positive"):
        initialize_mlp_parameters(in_features=48, hidden_dims=[64, 0], num_classes=3)


@pytest.mark.unit
def test_mlp_forward_single_hidden(
    mlp_spec_single_hidden: ModelSpecification,
) -> None:
    """Verify forward pass on 1-hidden MLP outputs [B, num_classes]."""
    model = MultiLayerPerceptron(spec=mlp_spec_single_hidden, seed=42)
    sample1 = [[[0.1] * 4 for _ in range(4)] for _ in range(3)]
    sample2 = [[[0.5] * 4 for _ in range(4)] for _ in range(3)]

    logits = model.forward([sample1, sample2])
    assert len(logits) == 2
    assert len(logits[0]) == 3
    assert len(logits[1]) == 3


@pytest.mark.unit
def test_mlp_forward_multi_hidden(
    mlp_spec_multi_hidden: ModelSpecification,
) -> None:
    """Verify forward pass on multi-hidden MLP outputs [B, num_classes]."""
    model = MultiLayerPerceptron(spec=mlp_spec_multi_hidden, seed=42)
    sample = [[[0.2] * 4 for _ in range(4)] for _ in range(3)]

    logits = model.forward([sample])
    assert len(logits) == 1
    assert len(logits[0]) == 3


@pytest.mark.unit
def test_mlp_backward_and_gradients(
    mlp_spec_multi_hidden: ModelSpecification,
) -> None:
    """Verify backward computes gradients for every layer and zero_grad resets."""
    model = MultiLayerPerceptron(spec=mlp_spec_multi_hidden, seed=42)
    sample = [[[0.5] * 4 for _ in range(4)] for _ in range(3)]

    _ = model.forward([sample])
    d_logits = [[0.1, -0.2, 0.1]]
    model.backward(d_logits)

    grads = model.get_gradients()
    assert "grad_weights_0" in grads
    assert "grad_bias_0" in grads
    assert "grad_weights_1" in grads
    assert "grad_bias_1" in grads
    assert "grad_weights_out" in grads
    assert "grad_bias_out" in grads

    # Shapes:
    # Layer 0: 48 -> 128
    assert len(grads["grad_weights_0"]) == 48
    assert len(grads["grad_weights_0"][0]) == 128
    assert len(grads["grad_bias_0"]) == 128

    # Layer 1: 128 -> 64
    assert len(grads["grad_weights_1"]) == 128
    assert len(grads["grad_weights_1"][0]) == 64
    assert len(grads["grad_bias_1"]) == 64

    # Layer Out: 64 -> 3
    assert len(grads["grad_weights_out"]) == 64
    assert len(grads["grad_weights_out"][0]) == 3
    assert len(grads["grad_bias_out"]) == 3

    # Reset
    model.zero_grad()
    cleared = model.get_gradients()
    assert all(all(v == 0.0 for v in row) for row in cleared["grad_weights_0"])
    assert all(v == 0.0 for v in cleared["grad_bias_out"])
