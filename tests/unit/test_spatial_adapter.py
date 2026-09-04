"""Unit tests for SpatialRepresentationAdapter across CNN, ResNet, and ViT models."""

import pytest

from prism.core.enums import ModelFamily, TaskType
from prism.core.errors import ValidationError
from prism.models.cnn import ConvolutionalNeuralNetwork
from prism.models.resnet import ResidualNeuralNetwork
from prism.models.specifications import ModelSpecification
from prism.models.transformer import VisionTransformer
from prism.spatial.adapter import (
    SpatialRepresentationAdapter,
    get_available_spatial_layers,
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
            "patch_size": 4,
            "embed_dim": 16,
            "depth": 2,
            "num_heads": 2,
            "mlp_dim": 32,
        },
    )
    return VisionTransformer(spec=spec, seed=42)


def test_available_spatial_layers():
    """Test discovering available spatial layers across architectures."""
    cnn = _create_toy_cnn()
    cnn_layers = get_available_spatial_layers(cnn)
    assert "conv_0" in cnn_layers
    assert "conv_1" in cnn_layers
    assert "final_spatial" in cnn_layers

    resnet = _create_toy_resnet()
    resnet_layers = get_available_spatial_layers(resnet)
    assert "stem" in resnet_layers
    assert "stage_0" in resnet_layers
    assert "stage_1" in resnet_layers
    assert "final_spatial" in resnet_layers

    vit = _create_toy_vit()
    vit_layers = get_available_spatial_layers(vit)
    assert "patch_embeddings" in vit_layers
    assert "encoder_0" in vit_layers
    assert "encoder_1" in vit_layers
    assert "final_spatial" in vit_layers


def test_cnn_spatial_extraction():
    """Test extracting 4D feature maps from CNN at different stages."""
    cnn = _create_toy_cnn()
    batch_images = [[[[0.1 for _ in range(16)] for _ in range(16)] for _ in range(3)]]

    adapter_0 = SpatialRepresentationAdapter(cnn, layer_name="conv_0")
    feats_0 = adapter_0.extract_spatial_features(batch_images)
    assert len(feats_0) == 1  # Batch size 1
    assert len(feats_0[0]) == 8  # 8 channels
    assert len(feats_0[0][0]) == 16  # H
    assert len(feats_0[0][0][0]) == 16  # W

    adapter_1 = SpatialRepresentationAdapter(cnn, layer_name="conv_1")
    feats_1 = adapter_1.extract_spatial_features(batch_images)
    assert len(feats_1) == 1
    assert len(feats_1[0]) == 16  # 16 channels


def test_resnet_spatial_extraction():
    """Test extracting 4D feature maps from ResNet."""
    resnet = _create_toy_resnet()
    batch_images = [[[[0.2 for _ in range(16)] for _ in range(16)] for _ in range(3)]]

    adapter_stem = SpatialRepresentationAdapter(resnet, layer_name="stem")
    feats_stem = adapter_stem.extract_spatial_features(batch_images)
    assert len(feats_stem) == 1
    assert len(feats_stem[0]) == 8  # stem_channels = 8

    adapter_stage1 = SpatialRepresentationAdapter(resnet, layer_name="stage_1")
    feats_stage1 = adapter_stage1.extract_spatial_features(batch_images)
    assert len(feats_stage1) == 1
    assert len(feats_stage1[0]) == 16


def test_vit_patch_to_grid_unflattening_and_cls_stripping():
    """Test extracting ViT spatial features and stripping CLS token."""
    vit = _create_toy_vit()
    batch_images = [[[[0.3 for _ in range(16)] for _ in range(16)] for _ in range(3)]]

    adapter_vit = SpatialRepresentationAdapter(vit, layer_name="encoder_1")
    feats_vit = adapter_vit.extract_spatial_features(batch_images)
    # Patch size 4 on 16x16 -> 4x4 patch grid = 16 patches
    # Embed dim = 16
    assert len(feats_vit) == 1  # Batch size 1
    assert len(feats_vit[0]) == 16  # Channels = embed_dim = 16
    assert len(feats_vit[0][0]) == 4  # H_patch = 4
    assert len(feats_vit[0][0][0]) == 4  # W_patch = 4


def test_adapter_deterministic_extraction():
    """Test that spatial feature extraction is perfectly deterministic."""
    vit = _create_toy_vit()
    img = [[[0.5 for _ in range(16)] for _ in range(16)] for _ in range(3)]
    adapter = SpatialRepresentationAdapter(vit, layer_name="encoder_0")

    feats1 = adapter.extract_spatial_features([img])
    feats2 = adapter.extract_spatial_features([img])

    assert feats1 == feats2


def test_adapter_invalid_layer_rejection():
    """Test that attempting to use an invalid layer raises ValidationError."""
    cnn = _create_toy_cnn()
    with pytest.raises(ValidationError, match="Invalid spatial layer"):
        SpatialRepresentationAdapter(cnn, layer_name="invalid_stage_99")
