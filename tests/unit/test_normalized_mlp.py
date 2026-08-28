"""Unit tests for MultiLayerPerceptron with BatchNorm1D layers."""

import pytest

from prism.core.enums import ModelFamily, TaskType
from prism.models.mlp import MultiLayerPerceptron
from prism.models.specifications import ModelSpecification


@pytest.fixture
def normalized_mlp_spec() -> ModelSpecification:
    return ModelSpecification(
        model_id="model-mlp-bn-test",
        name="Normalized MLP Model",
        family=ModelFamily.MLP,
        architecture="mlp",
        compatible_tasks=[TaskType.CLASSIFICATION],
        input_shape=(6,),
        num_classes=2,
        hyperparameters={
            "hidden_dims": [12, 8],
            "activation": "relu",
            "normalization": "batch_norm",
            "norm_eps": 1e-5,
            "norm_momentum": 0.1,
            "norm_affine": True,
            "dropout": 0.0,
        },
    )


@pytest.mark.unit
def test_normalized_mlp_forward_and_params(
    normalized_mlp_spec: ModelSpecification,
) -> None:
    """Verify forward pass on Normalized MLP and parameter/state registration."""
    model = MultiLayerPerceptron(spec=normalized_mlp_spec, seed=42)
    sample = [0.5, -0.2, 1.0, 0.0, -1.0, 0.8]

    logits = model.forward([sample, sample])
    assert len(logits) == 2
    assert len(logits[0]) == 2

    params = model.get_parameters()
    state = model.get_state()

    assert "weights_0" in params
    assert "bias_0" in params
    assert "norm_0_gamma" in params
    assert "norm_0_beta" in params
    assert "norm_1_gamma" in params
    assert "norm_1_beta" in params
    assert "weights_out" in params
    assert "bias_out" in params

    assert "norm_0_running_mean" in state
    assert "norm_0_running_var" in state


@pytest.mark.unit
def test_normalized_mlp_backward_and_gradients(
    normalized_mlp_spec: ModelSpecification,
) -> None:
    """Verify backpropagation computes gradients across linear and norm layers."""
    model = MultiLayerPerceptron(spec=normalized_mlp_spec, seed=42)
    sample = [0.1, 0.2, -0.1, 0.5, 0.0, 1.0]

    _ = model.forward([sample, sample])
    d_logits = [[0.5, -0.5], [-0.5, 0.5]]
    model.backward(d_logits)

    grads = model.get_gradients()
    assert "grad_weights_0" in grads
    assert "grad_norm_0_gamma" in grads
    assert "grad_norm_0_beta" in grads
    assert "grad_weights_1" in grads
    assert "grad_norm_1_gamma" in grads
    assert "grad_norm_1_beta" in grads
    assert "grad_weights_out" in grads


@pytest.mark.unit
def test_normalized_mlp_representation_extraction(
    normalized_mlp_spec: ModelSpecification,
) -> None:
    """Verify extraction of pre-norm, post-norm, and hidden representations."""
    model = MultiLayerPerceptron(spec=normalized_mlp_spec, seed=42)
    sample = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]

    pre_norm = model.extract_representations([sample], layer="hidden_0_pre_norm")
    assert len(pre_norm) == 1
    assert len(pre_norm[0]) == 12

    post_norm = model.extract_representations([sample], layer="hidden_0_post_norm")
    assert len(post_norm) == 1
    assert len(post_norm[0]) == 12

    hidden = model.extract_representations([sample], layer="hidden_0")
    assert len(hidden) == 1
    assert len(hidden[0]) == 12
