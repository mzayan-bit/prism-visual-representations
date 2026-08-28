"""Unit tests for parameter and model gradient flow summaries."""

import pytest

from prism.core.enums import ModelFamily, TaskType
from prism.models.cnn import ConvolutionalNeuralNetwork
from prism.models.resnet import ResidualNeuralNetwork
from prism.models.specifications import ModelSpecification
from prism.training.gradient_flow import (
    ModelGradientFlowSummary,
    compare_gradient_flow_summaries,
    compute_gradient_flow_summary,
)


@pytest.fixture
def sample_resnet() -> ResidualNeuralNetwork:
    spec = ModelSpecification(
        model_id="model-resnet-gradflow-test",
        name="ResNet GradFlow Model",
        family=ModelFamily.RESNET,
        architecture="resnet",
        compatible_tasks=[TaskType.CLASSIFICATION],
        input_shape=(3, 8, 8),
        num_classes=2,
        hyperparameters={
            "stem_channels": 8,
            "stage_widths": [8, 16],
            "blocks_per_stage": [1, 1],
            "strides": [1, 2],
            "normalization": "batch_norm",
            "activation": "relu",
        },
    )
    model = ResidualNeuralNetwork(spec=spec, seed=42)
    sample = [[[0.5] * 8 for _ in range(8)] for _ in range(3)]
    _ = model.forward([sample])
    model.backward([[0.5, -0.5]])
    return model


@pytest.fixture
def sample_plain_cnn() -> ConvolutionalNeuralNetwork:
    spec = ModelSpecification(
        model_id="model-cnn-gradflow-test",
        name="CNN GradFlow Model",
        family=ModelFamily.CNN,
        architecture="cnn",
        compatible_tasks=[TaskType.CLASSIFICATION],
        input_shape=(3, 8, 8),
        num_classes=2,
        hyperparameters={
            "conv_channels": [8, 16],
            "normalization": "batch_norm",
            "activation": "relu",
        },
    )
    model = ConvolutionalNeuralNetwork(spec=spec, seed=42)
    sample = [[[0.5] * 8 for _ in range(8)] for _ in range(3)]
    _ = model.forward([sample])
    model.backward([[0.5, -0.5]])
    return model


@pytest.mark.unit
def test_compute_gradient_flow_summary(
    sample_resnet: ResidualNeuralNetwork,
) -> None:
    """Verify gradient flow summary correctly calculates L2 norms and stage depths."""
    summary = compute_gradient_flow_summary(sample_resnet)

    assert isinstance(summary, ModelGradientFlowSummary)
    assert summary.model_id == sample_resnet.model_id
    assert summary.total_parameters > 0
    assert summary.global_grad_norm_l2 > 0.0
    assert summary.is_finite is True

    # Check stem summary
    stem_s = summary.get_summary("stem_conv_weights")
    assert stem_s is not None
    assert stem_s.logical_stage == "stem"
    assert stem_s.depth_index == 0
    assert stem_s.norm_l2 > 0.0
    assert stem_s.sample_count == 8 * 3 * 3 * 3

    # Check classifier summary
    cls_s = summary.get_summary("classifier_weights")
    assert cls_s is not None
    assert cls_s.logical_stage == "classifier"
    assert cls_s.depth_index == 999


@pytest.mark.unit
def test_compare_gradient_flow_summaries(
    sample_plain_cnn: ConvolutionalNeuralNetwork,
    sample_resnet: ResidualNeuralNetwork,
) -> None:
    """Verify comparison between plain and residual model gradient flows."""
    sum_plain = compute_gradient_flow_summary(sample_plain_cnn)
    sum_res = compute_gradient_flow_summary(sample_resnet)

    comparison = compare_gradient_flow_summaries(sum_plain, sum_res)

    assert "global_norm_plain" in comparison
    assert "global_norm_residual" in comparison
    assert "global_norm_ratio" in comparison
    assert "global_norm_delta" in comparison
    assert comparison["is_finite"] is True
    assert comparison["global_norm_ratio"] > 0.0


@pytest.mark.unit
def test_gradient_flow_summary_json_roundtrip(
    sample_resnet: ResidualNeuralNetwork,
) -> None:
    """Verify serialization roundtrip for ModelGradientFlowSummary."""
    summary = compute_gradient_flow_summary(sample_resnet)
    json_str = summary.to_json()
    reconstructed = ModelGradientFlowSummary.from_json(json_str)
    assert reconstructed.model_id == summary.model_id
    assert reconstructed.global_grad_norm_l2 == pytest.approx(
        summary.global_grad_norm_l2
    )
    assert len(reconstructed.parameter_summaries) == len(summary.parameter_summaries)
