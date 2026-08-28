"""Unit tests for CNN with BatchNorm2D layers and representation extraction."""

import pytest

from prism.core.enums import ModelFamily, TaskType
from prism.models.cnn import ConvolutionalNeuralNetwork
from prism.models.specifications import ModelSpecification


@pytest.fixture
def normalized_cnn_spec() -> ModelSpecification:
    return ModelSpecification(
        model_id="model-cnn-bn-test",
        name="Normalized CNN Test Model",
        family=ModelFamily.CNN,
        architecture="cnn",
        compatible_tasks=[TaskType.CLASSIFICATION],
        input_shape=(3, 8, 8),
        num_classes=3,
        hyperparameters={
            "conv_channels": [8, 16],
            "kernel_sizes": 3,
            "strides": 1,
            "paddings": 1,
            "pool_sizes": 2,
            "pool_strides": 2,
            "activation": "relu",
            "normalization": "batch_norm",
            "norm_eps": 1e-5,
            "norm_momentum": 0.1,
            "norm_affine": True,
            "classifier_hidden_dims": [],
            "dropout": 0.0,
        },
    )


@pytest.mark.unit
def test_normalized_cnn_forward_and_output_shape(
    normalized_cnn_spec: ModelSpecification,
) -> None:
    """Verify forward pass on Normalized CNN produces [N, num_classes] raw logits."""
    model = ConvolutionalNeuralNetwork(spec=normalized_cnn_spec, seed=42)
    sample1 = [[[0.5] * 8 for _ in range(8)] for _ in range(3)]
    sample2 = [[[0.1] * 8 for _ in range(8)] for _ in range(3)]

    logits = model.forward([sample1, sample2])
    assert len(logits) == 2
    assert len(logits[0]) == 3
    assert len(logits[1]) == 3

    # Shape progression:
    # Input: 3x8x8
    # Conv 0: 8x8x8 -> BN 0 -> ReLU -> Pool 0: 8x4x4
    # Conv 1: 16x4x4 -> BN 1 -> ReLU -> Pool 1: 16x2x2
    # Flattened Dim: 16 * 2 * 2 = 64
    assert model.final_spatial_shape == (16, 2, 2)
    assert model.flattened_dim == 64


@pytest.mark.unit
def test_normalized_cnn_parameters_and_state(
    normalized_cnn_spec: ModelSpecification,
) -> None:
    """Verify parameters include gamma/beta, and state contains running stats."""
    model = ConvolutionalNeuralNetwork(spec=normalized_cnn_spec, seed=42)
    params = model.get_parameters()
    state = model.get_state()

    # Trainable parameters
    assert "conv_0_weights" in params
    assert "conv_0_bias" in params
    assert "norm_0_gamma" in params
    assert "norm_0_beta" in params
    assert "norm_1_gamma" in params
    assert "norm_1_beta" in params
    assert "classifier_weights" in params
    assert "classifier_bias" in params

    # Non-trainable running state
    assert "norm_0_running_mean" in state
    assert "norm_0_running_var" in state
    assert "norm_1_running_mean" in state
    assert "norm_1_running_var" in state
    assert "norm_0_gamma" not in state


@pytest.mark.unit
def test_normalized_cnn_backward_gradients(
    normalized_cnn_spec: ModelSpecification,
) -> None:
    """Verify backpropagation computes gradients for conv, norm, and classifier."""
    model = ConvolutionalNeuralNetwork(spec=normalized_cnn_spec, seed=42)
    sample = [[[0.2] * 8 for _ in range(8)] for _ in range(3)]

    _ = model.forward([sample])
    d_logits = [[0.2, -0.4, 0.2]]
    model.backward(d_logits)

    grads = model.get_gradients()
    assert "grad_conv_0_weights" in grads
    assert "grad_norm_0_gamma" in grads
    assert "grad_norm_0_beta" in grads
    assert "grad_conv_1_weights" in grads
    assert "grad_norm_1_gamma" in grads
    assert "grad_norm_1_beta" in grads
    assert "grad_classifier_weights" in grads
    assert "grad_classifier_bias" in grads

    # Zero grad resets all
    model.zero_grad()
    cleared = model.get_gradients()
    assert all(g == 0.0 for g in cleared["grad_norm_0_gamma"])


@pytest.mark.unit
def test_normalized_cnn_representation_extraction(
    normalized_cnn_spec: ModelSpecification,
) -> None:
    """Verify extraction of pre-norm, post-norm, and spatial representations."""
    model = ConvolutionalNeuralNetwork(spec=normalized_cnn_spec, seed=42)
    sample1 = [[[0.5] * 8 for _ in range(8)] for _ in range(3)]
    sample2 = [[[0.2] * 8 for _ in range(8)] for _ in range(3)]
    inputs = [sample1, sample2]

    # Pre-norm convolution output: [2, 8, 8, 8]
    pre_norm = model.extract_representations(inputs, layer="conv_0_pre_norm")
    assert len(pre_norm) == 2
    assert len(pre_norm[0]) == 8
    assert len(pre_norm[0][0]) == 8

    # Post-norm output: [2, 8, 8, 8]
    post_norm = model.extract_representations(inputs, layer="conv_0_post_norm")
    assert len(post_norm) == 2
    assert len(post_norm[0]) == 8
    assert len(post_norm[0][0]) == 8

    # Post-activation: [2, 8, 8, 8]
    post_act = model.extract_representations(inputs, layer="conv_0")
    assert len(post_act) == 2
    assert len(post_act[0]) == 8

    # Final spatial: [2, 16, 2, 2]
    final_spatial = model.extract_representations(inputs, layer="final_spatial")
    assert len(final_spatial) == 2
    assert len(final_spatial[0]) == 16
    assert len(final_spatial[0][0]) == 2

    # Final vector: [2, 64]
    final_vec = model.extract_representations(inputs, layer="final_hidden")
    assert len(final_vec) == 2
    assert len(final_vec[0]) == 64

    # Eval mode extraction does NOT update running statistics
    state_before = model.get_state()
    _ = model.extract_representations(inputs, layer="final_spatial")
    state_after = model.get_state()
    assert state_before == state_after
