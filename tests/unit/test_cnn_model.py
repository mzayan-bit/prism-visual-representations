"""Unit tests for ConvolutionalNeuralNetwork baseline architecture."""

import pytest

from prism.core.enums import ModelFamily, TaskType
from prism.models.cnn import ConvolutionalNeuralNetwork
from prism.models.specifications import ModelSpecification


@pytest.fixture
def cnn_spec_2block() -> ModelSpecification:
    return ModelSpecification(
        model_id="model-cnn-2block",
        name="2-Block CNN Baseline",
        family=ModelFamily.CNN,
        architecture="cnn",
        compatible_tasks=[TaskType.CLASSIFICATION],
        input_shape=(3, 8, 8),  # 3 channels, 8x8 image
        num_classes=3,
        hyperparameters={
            "conv_channels": [16, 32],
            "kernel_sizes": 3,
            "strides": 1,
            "paddings": 1,
            "pool_sizes": 2,
            "pool_strides": 2,
            "activation": "relu",
            "classifier_hidden_dims": [],
            "dropout": 0.0,
        },
    )


@pytest.mark.unit
def test_cnn_initialization_deterministic(cnn_spec_2block: ModelSpecification) -> None:
    """Verify identical seeds produce identical parameters, different seeds differ."""
    model1 = ConvolutionalNeuralNetwork(spec=cnn_spec_2block, seed=42)
    model2 = ConvolutionalNeuralNetwork(spec=cnn_spec_2block, seed=42)
    model3 = ConvolutionalNeuralNetwork(spec=cnn_spec_2block, seed=999)

    params1 = model1.get_parameters()
    params2 = model2.get_parameters()
    params3 = model3.get_parameters()

    assert params1 == params2
    assert params1["conv_0_weights"] != params3["conv_0_weights"]


@pytest.mark.unit
def test_cnn_forward_shapes_and_dimensions(cnn_spec_2block: ModelSpecification) -> None:
    """Verify CNN forward pass produces [N, num_classes] raw unnormalized logits."""
    model = ConvolutionalNeuralNetwork(spec=cnn_spec_2block, seed=42)

    # Input: 2 samples of shape [3, 8, 8]
    sample1 = [[[0.5] * 8 for _ in range(8)] for _ in range(3)]
    sample2 = [[[0.2] * 8 for _ in range(8)] for _ in range(3)]

    logits = model.forward([sample1, sample2])
    assert len(logits) == 2
    assert len(logits[0]) == 3
    assert len(logits[1]) == 3

    # Shape progression:
    # Input: 3x8x8
    # Conv 0: 16x8x8 -> Pool 0: 16x4x4
    # Conv 1: 32x4x4 -> Pool 1: 32x2x2
    # Final Spatial Shape: (32, 2, 2)
    # Flattened Dim: 32 * 2 * 2 = 128
    assert model.final_spatial_shape == (32, 2, 2)
    assert model.flattened_dim == 128


@pytest.mark.unit
def test_cnn_receptive_field_property(cnn_spec_2block: ModelSpecification) -> None:
    """Verify receptive field tracking on CNN model."""
    model = ConvolutionalNeuralNetwork(spec=cnn_spec_2block, seed=42)
    # Block 0: Conv 3x3 s1 -> RF=3, J=1; Pool 2x2 s2 -> RF=3 + 1 = 4, J=2
    # Block 1: Conv 3x3 s1 -> RF=4 + 2*2 = 8, J=2; Pool 2x2 s2 -> RF=8 + 2 = 10, J=4
    assert model.receptive_field == 10


@pytest.mark.unit
def test_cnn_backward_gradients_and_zero_grad(
    cnn_spec_2block: ModelSpecification,
) -> None:
    """Verify full backpropagation through classifier and convolutional blocks."""
    model = ConvolutionalNeuralNetwork(spec=cnn_spec_2block, seed=42)
    sample = [[[0.1] * 8 for _ in range(8)] for _ in range(3)]

    _ = model.forward([sample])
    d_logits = [[0.5, -0.5, 0.0]]
    model.backward(d_logits)

    grads = model.get_gradients()
    assert "grad_conv_0_weights" in grads
    assert "grad_conv_0_bias" in grads
    assert "grad_conv_1_weights" in grads
    assert "grad_conv_1_bias" in grads
    assert "grad_classifier_weights" in grads
    assert "grad_classifier_bias" in grads

    # Shapes:
    # Conv 0: [16, 3, 3, 3]
    assert len(grads["grad_conv_0_weights"]) == 16
    assert len(grads["grad_conv_0_weights"][0]) == 3
    # Conv 1: [32, 16, 3, 3]
    assert len(grads["grad_conv_1_weights"]) == 32
    assert len(grads["grad_conv_1_weights"][0]) == 16
    # Classifier: [128, 3]
    assert len(grads["grad_classifier_weights"]) == 128
    assert len(grads["grad_classifier_weights"][0]) == 3

    # Reset
    model.zero_grad()
    cleared = model.get_gradients()
    assert all(b == 0.0 for b in cleared["grad_classifier_bias"])


@pytest.mark.unit
def test_cnn_flattened_2d_input_support(cnn_spec_2block: ModelSpecification) -> None:
    """Verify CNN seamlessly accepts flattened 2D input vectors [N, D]."""
    model = ConvolutionalNeuralNetwork(spec=cnn_spec_2block, seed=42)
    # Flat sample: 3 * 8 * 8 = 192 features
    flat_sample = [0.25] * 192
    logits = model.forward([flat_sample])
    assert len(logits) == 1
    assert len(logits[0]) == 3
