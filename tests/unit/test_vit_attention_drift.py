"""Unit tests for Vision Transformer attention drift analysis."""

from prism.core.enums import ModelFamily, TaskType
from prism.models.cnn import ConvolutionalNeuralNetwork
from prism.models.specifications import ModelSpecification
from prism.models.transformer import VisionTransformer
from prism.robustness.attention_drift import (
    AttentionDriftSummary,
    compute_vit_attention_drift,
)


def _make_dummy_image(c: int = 3, h: int = 8, w: int = 8) -> list[list[list[float]]]:
    """Create a deterministic synthetic image."""
    img: list[list[list[float]]] = []
    for _ in range(c):
        plane: list[list[float]] = []
        for _ in range(h):
            row = [0.5 for _ in range(w)]
            plane.append(row)
        img.append(plane)
    return img


def test_vit_attention_drift() -> None:
    vit_spec = ModelSpecification(
        model_id="vit_test",
        name="ViT Test",
        family=ModelFamily.VISION_TRANSFORMER,
        architecture="vit_test",
        compatible_tasks=[TaskType.CLASSIFICATION],
        input_shape=(3, 8, 8),
        num_classes=2,
        hyperparameters={
            "patch_size": 4,
            "embed_dim": 8,
            "num_heads": 2,
            "depth": 2,
            "mlp_ratio": 2.0,
            "norm_eps": 1e-5,
            "activation": "gelu",
        },
    )
    vit_model = VisionTransformer(spec=vit_spec, seed=42)

    clean_inputs = [_make_dummy_image() for _ in range(2)]
    # Corrupted inputs with different values
    corr_inputs = []
    for _ in range(2):
        img = _make_dummy_image()
        img[0][0][0] = 1.0
        corr_inputs.append(img)

    drift_summary = compute_vit_attention_drift(
        model=vit_model,
        clean_inputs=clean_inputs,
        corrupted_inputs=corr_inputs,
    )

    assert isinstance(drift_summary, AttentionDriftSummary)
    assert drift_summary.num_layers == 2
    assert len(drift_summary.layer_drifts) == 2
    assert drift_summary.clean_overall_mean_entropy >= 0.0
    assert 0.0 <= drift_summary.clean_overall_diagonal_mass <= 1.0


def test_non_vit_returns_none() -> None:
    cnn_spec = ModelSpecification(
        model_id="cnn_test",
        name="CNN Test",
        family=ModelFamily.CNN,
        architecture="cnn_test",
        compatible_tasks=[TaskType.CLASSIFICATION],
        input_shape=(3, 8, 8),
        num_classes=2,
        hyperparameters={
            "conv_channels": [4],
            "kernel_sizes": [3],
            "fc_hidden_dims": [8],
            "activation": "relu",
        },
    )
    cnn_model = ConvolutionalNeuralNetwork(spec=cnn_spec, seed=42)

    clean_inputs = [_make_dummy_image() for _ in range(2)]
    drift_summary = compute_vit_attention_drift(
        model=cnn_model,
        clean_inputs=clean_inputs,
        corrupted_inputs=clean_inputs,
    )
    assert drift_summary is None
