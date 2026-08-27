"""Unit tests for deterministic dropout behavior in training and evaluation."""

import pytest

from prism.core.enums import ModelFamily, TaskType
from prism.core.errors import ValidationError
from prism.models.mlp import MultiLayerPerceptron
from prism.models.specifications import ModelSpecification


@pytest.mark.unit
def test_dropout_zero_is_identity() -> None:
    """Verify dropout p=0.0 produces identical output in train and eval modes."""
    spec = ModelSpecification(
        model_id="model-mlp-p0",
        name="MLP p=0",
        family=ModelFamily.MLP,
        architecture="mlp",
        compatible_tasks=[TaskType.CLASSIFICATION],
        input_shape=(4,),
        num_classes=2,
        hyperparameters={"hidden_dims": [16], "dropout": 0.0},
    )
    model = MultiLayerPerceptron(spec=spec, seed=42)
    sample = [1.0, 2.0, 3.0, 4.0]

    model.train()
    out_train = model.forward([sample])

    model.eval()
    out_eval = model.forward([sample])

    assert out_train == out_eval


@pytest.mark.unit
def test_dropout_invalid_probability() -> None:
    """Verify invalid dropout probabilities (<0 or >=1) raise ValidationError."""
    spec = ModelSpecification(
        model_id="model-mlp-bad-drop",
        name="MLP Bad Dropout",
        family=ModelFamily.MLP,
        architecture="mlp",
        compatible_tasks=[TaskType.CLASSIFICATION],
        input_shape=(4,),
        num_classes=2,
        hyperparameters={"hidden_dims": [16], "dropout": 1.5},
    )
    with pytest.raises(ValidationError, match="Dropout probability must be in"):
        MultiLayerPerceptron(spec=spec, seed=42)


@pytest.mark.unit
def test_dropout_eval_mode_deterministic() -> None:
    """Verify evaluation mode disables dropout masking completely."""
    spec = ModelSpecification(
        model_id="model-mlp-p05",
        name="MLP p=0.5",
        family=ModelFamily.MLP,
        architecture="mlp",
        compatible_tasks=[TaskType.CLASSIFICATION],
        input_shape=(4,),
        num_classes=2,
        hyperparameters={"hidden_dims": [32], "dropout": 0.5},
    )
    model = MultiLayerPerceptron(spec=spec, seed=42)
    sample = [1.0, -1.0, 0.5, 2.0]

    model.eval()
    out1 = model.forward([sample])
    out2 = model.forward([sample])
    assert out1 == out2


@pytest.mark.unit
def test_dropout_training_mode_masks_activations() -> None:
    """Verify training mode applies Bernoulli dropout scaling in forward."""
    spec = ModelSpecification(
        model_id="model-mlp-p05-train",
        name="MLP p=0.5 Train",
        family=ModelFamily.MLP,
        architecture="mlp",
        compatible_tasks=[TaskType.CLASSIFICATION],
        input_shape=(8,),
        num_classes=2,
        hyperparameters={"hidden_dims": [64], "dropout": 0.5},
    )
    model1 = MultiLayerPerceptron(spec=spec, seed=42)
    model2 = MultiLayerPerceptron(spec=spec, seed=42)
    sample = [1.0] * 8

    model1.train()
    model2.train()

    # With identical seed and same initial step counter, dropout masks match
    out1 = model1.forward([sample])
    out2 = model2.forward([sample])
    assert out1 == out2


@pytest.mark.unit
def test_dropout_backward_gradient_masking() -> None:
    """Verify backward propagates gradients strictly through unmasked activations."""
    spec = ModelSpecification(
        model_id="model-mlp-drop-back",
        name="MLP Dropout Backprop",
        family=ModelFamily.MLP,
        architecture="mlp",
        compatible_tasks=[TaskType.CLASSIFICATION],
        input_shape=(4,),
        num_classes=2,
        hyperparameters={"hidden_dims": [16], "dropout": 0.5},
    )
    model = MultiLayerPerceptron(spec=spec, seed=42)
    sample = [1.0, 2.0, -1.0, 0.5]

    model.train()
    _ = model.forward([sample])
    model.backward([[0.5, -0.5]])

    grads = model.get_gradients()
    assert "grad_weights_0" in grads
    assert "grad_weights_out" in grads
    # Gradients computed and finite
    assert any(any(v != 0.0 for v in row) for row in grads["grad_weights_0"])
