"""Unit tests for TemporalFrameEncoder across CNN, ResNet, and Vision Transformer."""

import pytest

from prism.core.enums import ModelFamily, TaskType
from prism.core.errors import ValidationError
from prism.models.cnn import ConvolutionalNeuralNetwork
from prism.models.resnet import ResidualNeuralNetwork
from prism.models.specifications import ModelSpecification
from prism.models.transformer import VisionTransformer
from prism.temporal.adapter import (
    TemporalFrameEncoder,
    get_available_temporal_layers,
)


def _create_toy_cnn() -> ConvolutionalNeuralNetwork:
    spec = ModelSpecification(
        model_id="toy_cnn",
        name="Toy CNN",
        architecture="cnn_toy",
        family=ModelFamily.CNN,
        input_shape=(3, 16, 16),
        num_classes=4,
        compatible_tasks=[TaskType.CLASSIFICATION],
        hyperparameters={
            "conv_channels": [8, 16],
            "kernel_sizes": [3, 3],
            "strides": [1, 1],
            "paddings": [1, 1],
            "use_batch_norm": False,
            "hidden_dims": [32],
        },
    )
    return ConvolutionalNeuralNetwork(spec=spec, seed=42)


def _create_toy_resnet() -> ResidualNeuralNetwork:
    spec = ModelSpecification(
        model_id="toy_resnet",
        name="Toy ResNet",
        architecture="resnet_toy",
        family=ModelFamily.RESNET,
        input_shape=(3, 16, 16),
        num_classes=4,
        compatible_tasks=[TaskType.CLASSIFICATION],
        hyperparameters={
            "stem_channels": 8,
            "stage_widths": [8, 16],
            "blocks_per_stage": [1, 1],
            "strides": [1, 2],
        },
    )
    return ResidualNeuralNetwork(spec=spec, seed=42)


def _create_toy_vit() -> VisionTransformer:
    spec = ModelSpecification(
        model_id="toy_vit",
        name="Toy ViT",
        architecture="vit_toy",
        family=ModelFamily.VISION_TRANSFORMER,
        input_shape=(3, 16, 16),
        num_classes=4,
        compatible_tasks=[TaskType.CLASSIFICATION],
        hyperparameters={
            "image_size": 16,
            "patch_size": 8,
            "embed_dim": 16,
            "num_layers": 2,
            "num_heads": 2,
            "mlp_dim": 32,
            "dropout": 0.0,
        },
    )
    return VisionTransformer(spec=spec, seed=42)


def test_cnn_temporal_frame_encoding() -> None:
    cnn = _create_toy_cnn()
    encoder = TemporalFrameEncoder(model=cnn, layer_name="final_hidden")

    # Batched videos: 2 samples, 3 frames each, 3x16x16
    video_batch = [
        [[[0.1 for _ in range(16)] for _ in range(16)] for _ in range(3)]
        for _ in range(3)
    ]
    videos = [video_batch, video_batch]  # N=2, T=3

    features = encoder.forward(videos)
    assert len(features) == 2  # N
    assert len(features[0]) == 3  # T
    assert len(features[0][0]) > 0  # D


def test_resnet_temporal_frame_encoding() -> None:
    resnet = _create_toy_resnet()
    encoder = TemporalFrameEncoder(model=resnet, layer_name="final_hidden")

    video_batch = [
        [[[0.2 for _ in range(16)] for _ in range(16)] for _ in range(3)]
        for _ in range(4)
    ]
    videos = [video_batch]  # N=1, T=4

    features = encoder.forward(videos)
    assert len(features) == 1
    assert len(features[0]) == 4
    assert len(features[0][0]) > 0


def test_vit_temporal_frame_encoding() -> None:
    vit = _create_toy_vit()
    encoder_cls = TemporalFrameEncoder(model=vit, layer_name="cls")
    encoder_tokens = TemporalFrameEncoder(model=vit, layer_name="final_tokens")

    video_batch = [
        [[[0.3 for _ in range(16)] for _ in range(16)] for _ in range(3)]
        for _ in range(2)
    ]
    videos = [video_batch]  # N=1, T=2

    cls_feats = encoder_cls.forward(videos)
    assert len(cls_feats) == 1
    assert len(cls_feats[0]) == 2
    assert len(cls_feats[0][0]) == 16

    token_feats = encoder_tokens.forward(videos)
    assert len(token_feats) == 1
    assert len(token_feats[0]) == 2
    assert len(token_feats[0][0]) == 16  # pooled across patch tokens


def test_layer_discovery_and_invalid_layer_rejection() -> None:
    cnn = _create_toy_cnn()
    valid_layers = get_available_temporal_layers(cnn)
    assert "final_hidden" in valid_layers
    assert "conv_0" in valid_layers

    with pytest.raises(ValidationError, match="Invalid temporal layer"):
        TemporalFrameEncoder(model=cnn, layer_name="nonexistent_layer_123")
