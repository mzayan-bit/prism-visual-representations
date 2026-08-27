"""Unit tests for representation extraction, descriptors, and batch containers."""

import pytest

from prism.core.enums import ModelFamily, TaskType
from prism.core.errors import ValidationError
from prism.models.linear import LinearSoftmaxClassifier
from prism.models.mlp import MultiLayerPerceptron
from prism.models.specifications import ModelSpecification
from prism.representations.contracts import (
    RepresentationBatch,
    RepresentationDescriptor,
)


@pytest.fixture
def mlp_model() -> MultiLayerPerceptron:
    spec = ModelSpecification(
        model_id="model-mlp-rep",
        name="MLP Rep Model",
        family=ModelFamily.MLP,
        architecture="mlp",
        compatible_tasks=[TaskType.CLASSIFICATION],
        input_shape=(4,),
        num_classes=2,
        hyperparameters={"hidden_dims": [16, 8], "activation": "relu"},
    )
    return MultiLayerPerceptron(spec=spec, seed=42)


@pytest.mark.unit
def test_mlp_representation_extraction(mlp_model: MultiLayerPerceptron) -> None:
    """Verify MultiLayerPerceptron extracts features across all named layers."""
    sample1 = [1.0, 2.0, 3.0, 4.0]
    sample2 = [0.5, -0.5, 1.0, -1.0]
    inputs = [sample1, sample2]

    # 1. Flattened input
    rep_flat = mlp_model.extract_representations(inputs, layer="input_flat")
    assert len(rep_flat) == 2
    assert len(rep_flat[0]) == 4

    # 2. Hidden Layer 0 (dim 16)
    rep_h0 = mlp_model.extract_representations(inputs, layer="hidden_0")
    assert len(rep_h0) == 2
    assert len(rep_h0[0]) == 16

    # 3. Hidden Layer 1 (dim 8)
    rep_h1 = mlp_model.extract_representations(inputs, layer="hidden_1")
    assert len(rep_h1) == 2
    assert len(rep_h1[0]) == 8

    # 4. Final hidden (dim 8)
    rep_final = mlp_model.extract_representations(inputs, layer="final_hidden")
    assert rep_final == rep_h1

    # 5. Logits (dim 2)
    rep_logits = mlp_model.extract_representations(inputs, layer="logits")
    assert len(rep_logits) == 2
    assert len(rep_logits[0]) == 2


@pytest.mark.unit
def test_linear_representation_extraction() -> None:
    """Verify LinearSoftmaxClassifier extracts flattened input and logits."""
    spec = ModelSpecification(
        model_id="model-linear-rep",
        name="Linear Rep Model",
        family=ModelFamily.LINEAR,
        architecture="linear_softmax",
        compatible_tasks=[TaskType.CLASSIFICATION],
        input_shape=(6,),
        num_classes=2,
    )
    model = LinearSoftmaxClassifier(spec=spec, seed=42)
    sample = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]

    rep_flat = model.extract_representations([sample], layer="input_flat")
    assert len(rep_flat[0]) == 6

    rep_final = model.extract_representations([sample], layer="final_hidden")
    assert rep_final == rep_flat


@pytest.mark.unit
def test_representation_extraction_invalid_layer(
    mlp_model: MultiLayerPerceptron,
) -> None:
    """Verify unknown layer identifier raises ValidationError."""
    with pytest.raises(ValidationError, match="Unknown layer 'invalid_layer'"):
        mlp_model.extract_representations([[1.0, 2.0, 3.0, 4.0]], layer="invalid_layer")


@pytest.mark.unit
def test_representation_contracts_roundtrip() -> None:
    """Verify descriptor and batch containers serialize to/from JSON."""
    desc = RepresentationDescriptor(
        layer_name="hidden_0",
        feature_dim=16,
        num_samples=2,
        sample_ids=["s001", "s002"],
        model_id="model-mlp-rep",
        is_training_mode=False,
    )
    json_str = desc.to_json()
    reconstructed = RepresentationDescriptor.from_json(json_str)
    assert reconstructed == desc

    batch = RepresentationBatch(
        descriptor=desc,
        embeddings=[[0.1] * 16, [0.2] * 16],
    )
    batch_json = batch.to_json()
    reconstructed_batch = RepresentationBatch.from_json(batch_json)
    assert reconstructed_batch == batch
