"""Unit tests for spatial feature map and vector representation extraction in CNNs."""

import pytest

from prism.core.enums import ModelFamily, TaskType
from prism.core.errors import ValidationError
from prism.models.cnn import ConvolutionalNeuralNetwork
from prism.models.specifications import ModelSpecification
from prism.representations.contracts import (
    RepresentationBatch,
    RepresentationDescriptor,
)


@pytest.fixture
def cnn_model() -> ConvolutionalNeuralNetwork:
    spec = ModelSpecification(
        model_id="model-cnn-rep-test",
        name="CNN Representation Test Model",
        family=ModelFamily.CNN,
        architecture="cnn",
        compatible_tasks=[TaskType.CLASSIFICATION],
        input_shape=(3, 8, 8),
        num_classes=2,
        hyperparameters={
            "conv_channels": [8, 16],
            "kernel_sizes": 3,
            "strides": 1,
            "paddings": 1,
            "pool_sizes": 2,
            "pool_strides": 2,
            "activation": "relu",
            "classifier_hidden_dims": [],
            "dropout": 0.2,
        },
    )
    return ConvolutionalNeuralNetwork(spec=spec, seed=42)


@pytest.mark.unit
def test_cnn_spatial_and_vector_representation_extraction(
    cnn_model: ConvolutionalNeuralNetwork,
) -> None:
    """Verify extraction across input, conv, pooling, and spatial layers."""
    sample1 = [[[0.5] * 8 for _ in range(8)] for _ in range(3)]
    sample2 = [[[0.1] * 8 for _ in range(8)] for _ in range(3)]
    inputs = [sample1, sample2]

    # 1. Spatial input [2, 3, 8, 8]
    rep_in = cnn_model.extract_representations(inputs, layer="input")
    assert len(rep_in) == 2
    assert len(rep_in[0]) == 3
    assert len(rep_in[0][0]) == 8
    assert len(rep_in[0][0][0]) == 8

    # 2. Block 0 Pre-Activation [2, 8, 8, 8]
    rep_c0_pre = cnn_model.extract_representations(inputs, layer="conv_0_pre")
    assert len(rep_c0_pre) == 2
    assert len(rep_c0_pre[0]) == 8
    assert len(rep_c0_pre[0][0]) == 8

    # 3. Block 0 Post-Pool [2, 8, 4, 4]
    rep_p0 = cnn_model.extract_representations(inputs, layer="pool_0")
    assert len(rep_p0) == 2
    assert len(rep_p0[0]) == 8
    assert len(rep_p0[0][0]) == 4
    assert len(rep_p0[0][0][0]) == 4

    # 4. Final Spatial Feature Map [2, 16, 2, 2]
    rep_spatial = cnn_model.extract_representations(inputs, layer="final_spatial")
    assert len(rep_spatial) == 2
    assert len(rep_spatial[0]) == 16
    assert len(rep_spatial[0][0]) == 2
    assert len(rep_spatial[0][0][0]) == 2

    # 5. Final Vector Representation [2, 64] (16 * 2 * 2 = 64)
    rep_vec = cnn_model.extract_representations(inputs, layer="final_hidden")
    assert len(rep_vec) == 2
    assert len(rep_vec[0]) == 64

    # 6. Logits [2, 2]
    rep_logits = cnn_model.extract_representations(inputs, layer="logits")
    assert len(rep_logits) == 2
    assert len(rep_logits[0]) == 2


@pytest.mark.unit
def test_cnn_representation_unknown_layer_fails(
    cnn_model: ConvolutionalNeuralNetwork,
) -> None:
    """Verify unknown layer name raises ValidationError."""
    sample = [[[[0.0] * 8 for _ in range(8)] for _ in range(3)]]
    with pytest.raises(ValidationError, match="Unknown layer 'invalid_layer'"):
        cnn_model.extract_representations(sample, layer="invalid_layer")


@pytest.mark.unit
def test_spatial_representation_descriptor_contract() -> None:
    """Verify RepresentationDescriptor models spatial maps and JSON roundtrip."""
    desc = RepresentationDescriptor(
        layer_name="final_spatial",
        feature_dim=64,  # 16 * 2 * 2
        num_samples=2,
        representation_kind="spatial",
        spatial_shape=(16, 2, 2),
        receptive_field=10,
        sample_ids=["sample_0", "sample_1"],
        model_id="model-cnn-rep-test",
        is_training_mode=False,
    )
    assert desc.is_spatial is True
    assert desc.spatial_shape == (16, 2, 2)
    assert desc.receptive_field == 10

    json_str = desc.to_json()
    reconstructed = RepresentationDescriptor.from_json(json_str)
    assert reconstructed == desc

    # Batch container roundtrip
    batch = RepresentationBatch(
        descriptor=desc,
        embeddings=[[[[0.1, 0.2], [0.3, 0.4]]] * 16, [[[0.5, 0.6], [0.7, 0.8]]] * 16],
    )
    batch_json = batch.to_json()
    reconstructed_batch = RepresentationBatch.from_json(batch_json)
    assert reconstructed_batch == batch
