"""Unit tests for ResidualNeuralNetwork (ResNet) and representation extraction."""

import pytest

from prism.core.enums import ModelFamily, TaskType
from prism.models.resnet import ResidualNeuralNetwork
from prism.models.specifications import ModelSpecification


@pytest.fixture
def resnet_spec() -> ModelSpecification:
    return ModelSpecification(
        model_id="model-resnet-test",
        name="ResNet Unit Test Model",
        family=ModelFamily.RESNET,
        architecture="resnet",
        compatible_tasks=[TaskType.CLASSIFICATION],
        input_shape=(3, 8, 8),
        num_classes=3,
        hyperparameters={
            "stem_channels": 8,
            "stage_widths": [8, 16],
            "blocks_per_stage": [2, 2],
            "strides": [1, 2],
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
def test_resnet_forward_and_shapes(resnet_spec: ModelSpecification) -> None:
    """Verify forward pass on ResNet produces [N, num_classes] raw logits."""
    model = ResidualNeuralNetwork(spec=resnet_spec, seed=42)
    sample1 = [[[0.5] * 8 for _ in range(8)] for _ in range(3)]
    sample2 = [[[0.1] * 8 for _ in range(8)] for _ in range(3)]

    logits = model.forward([sample1, sample2])
    assert len(logits) == 2
    assert len(logits[0]) == 3
    assert len(logits[1]) == 3

    # Shape progression:
    # Stem: 3x8x8 -> 8x8x8
    # Stage 0: 2 blocks (8x8x8) -> 8x8x8
    # Stage 1: 2 blocks (16x4x4, stride 2 on block 0) -> 16x4x4
    # Final spatial shape: (16, 4, 4)
    # Flattened dim: 16 * 4 * 4 = 256
    assert model.final_spatial_shape == (16, 4, 4)
    assert model.flattened_dim == 256
    assert model.receptive_field > 0


@pytest.mark.unit
def test_resnet_parameters_and_state(resnet_spec: ModelSpecification) -> None:
    """Verify parameters include stem, block convs, proj convs, and norm."""
    model = ResidualNeuralNetwork(spec=resnet_spec, seed=42)
    params = model.get_parameters()
    state = model.get_state()

    assert "stem_conv_weights" in params
    assert "stem_norm_gamma" in params
    assert "stage_0_block_0_conv1_weights" in params
    assert "stage_0_block_0_norm1_gamma" in params
    assert "stage_1_block_0_proj_conv_weights" in params
    assert "classifier_weights" in params

    assert "stem_norm_running_mean" in state
    assert "stage_0_block_0_norm1_running_mean" in state
    assert "stage_1_block_0_proj_norm_running_mean" in state


@pytest.mark.unit
def test_resnet_backward_gradients(resnet_spec: ModelSpecification) -> None:
    """Verify backpropagation computes gradients for all residual stages."""
    model = ResidualNeuralNetwork(spec=resnet_spec, seed=42)
    sample = [[[0.2] * 8 for _ in range(8)] for _ in range(3)]

    _ = model.forward([sample])
    d_logits = [[0.1, -0.2, 0.1]]
    model.backward(d_logits)

    grads = model.get_gradients()
    assert "grad_stem_conv_weights" in grads
    assert "grad_stage_0_block_0_conv1_weights" in grads
    assert "grad_stage_1_block_0_proj_conv_weights" in grads
    assert "grad_classifier_weights" in grads


@pytest.mark.unit
def test_resnet_representation_extraction(
    resnet_spec: ModelSpecification,
) -> None:
    """Verify extraction of stem, residual branch, shortcut, and post-add."""
    model = ResidualNeuralNetwork(spec=resnet_spec, seed=42)
    sample1 = [[[0.5] * 8 for _ in range(8)] for _ in range(3)]
    sample2 = [[[0.2] * 8 for _ in range(8)] for _ in range(3)]
    inputs = [sample1, sample2]

    # Stem representation: [2, 8, 8, 8]
    stem_rep = model.extract_representations(inputs, layer="stem")
    assert len(stem_rep) == 2
    assert len(stem_rep[0]) == 8

    # Stage 0 Block 0 Residual Branch: [2, 8, 8, 8]
    res_branch = model.extract_representations(inputs, layer="stage_0_block_0_residual")
    assert len(res_branch) == 2
    assert len(res_branch[0]) == 8

    # Stage 0 Block 0 Shortcut Branch: [2, 8, 8, 8]
    sc_branch = model.extract_representations(inputs, layer="stage_0_block_0_shortcut")
    assert len(sc_branch) == 2
    assert len(sc_branch[0]) == 8

    # Stage 0 Block 0 Post-Add: [2, 8, 8, 8]
    post_add = model.extract_representations(inputs, layer="stage_0_block_0_post_add")
    assert len(post_add) == 2
    assert len(post_add[0]) == 8

    # Stage 1 Block 0 Output (downsampled): [2, 16, 4, 4]
    s1_b0 = model.extract_representations(inputs, layer="stage_1_block_0")
    assert len(s1_b0) == 2
    assert len(s1_b0[0]) == 16
    assert len(s1_b0[0][0]) == 4

    # Final spatial: [2, 16, 4, 4]
    final_spatial = model.extract_representations(inputs, layer="final_spatial")
    assert len(final_spatial) == 2
    assert len(final_spatial[0]) == 16

    # Final vector: [2, 256]
    final_vec = model.extract_representations(inputs, layer="final_hidden")
    assert len(final_vec) == 2
    assert len(final_vec[0]) == 256
