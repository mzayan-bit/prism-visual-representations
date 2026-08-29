"""Unit tests for layer-wise representation geometry evolution across depth."""

from __future__ import annotations

import pytest

from prism.core.enums import ModelFamily, TaskType
from prism.models.cnn import ConvolutionalNeuralNetwork
from prism.models.resnet import ResidualNeuralNetwork
from prism.models.specifications import ModelSpecification
from prism.models.transformer import VisionTransformer
from prism.representations.evolution import (
    LayerGeometryProfile,
    analyze_layer_geometry_profile,
)


class TestLayerGeometryEvolution:
    """Test suite for layer progression analysis."""

    @pytest.fixture
    def synthetic_inputs_and_labels(
        self,
    ) -> tuple[list[list[list[list[float]]]], list[str], list[int]]:
        # 6 samples of 3x8x8 images across 2 classes
        images = []
        labels = [0, 0, 0, 1, 1, 1]
        sample_ids = [f"sample_{i}" for i in range(6)]
        for i in range(6):
            base = 1.0 if labels[i] == 1 else -1.0
            img = [
                [[base + 0.05 * (ch + r + c) for c in range(8)] for r in range(8)]
                for ch in range(3)
            ]
            images.append(img)
        return images, sample_ids, labels

    def test_cnn_layer_evolution_profile(
        self,
        synthetic_inputs_and_labels: tuple[
            list[list[list[list[float]]]], list[str], list[int]
        ],
    ) -> None:
        images, sample_ids, labels = synthetic_inputs_and_labels
        spec = ModelSpecification(
            model_id="test-cnn-profile",
            name="CNN Profile Model",
            family=ModelFamily.CNN,
            architecture="cnn_tiny",
            compatible_tasks=[TaskType.CLASSIFICATION],
            input_shape=(3, 8, 8),
            num_classes=2,
            hyperparameters={
                "conv_channels": [4, 8],
                "kernel_sizes": [3, 3],
                "fc_hidden_dims": [8],
                "activation": "relu",
            },
        )
        model = ConvolutionalNeuralNetwork(spec=spec, seed=42)

        profile = analyze_layer_geometry_profile(
            model=model,
            inputs=images,
            sample_ids=sample_ids,
            labels=labels,
            layers=["conv_0", "conv_1", "final_hidden"],
            experiment_id="exp-cnn-profile",
        )

        assert profile.model_id == "test-cnn-profile"
        assert len(profile.layer_points) == 3
        assert len(profile.compactness_trend) == 3
        assert len(profile.separation_trend) == 3
        assert len(profile.ratio_trend) == 3

        # Check point attributes
        for pt in profile.layer_points:
            assert pt.feature_dim > 0
            assert pt.mean_intra_class_distance >= 0.0
            assert pt.mean_inter_class_centroid_distance >= 0.0
            assert 0.0 <= pt.mean_label_consistency <= 1.0

        # Serialization roundtrip
        json_str = profile.to_json()
        assert len(json_str) > 0
        deserialized = LayerGeometryProfile.from_dict(profile.to_dict())
        assert len(deserialized.layer_points) == 3

    def test_resnet_layer_evolution_profile(
        self,
        synthetic_inputs_and_labels: tuple[
            list[list[list[list[float]]]], list[str], list[int]
        ],
    ) -> None:
        images, sample_ids, labels = synthetic_inputs_and_labels
        spec = ModelSpecification(
            model_id="test-resnet-profile",
            name="ResNet Profile Model",
            family=ModelFamily.RESNET,
            architecture="resnet_tiny",
            compatible_tasks=[TaskType.CLASSIFICATION],
            input_shape=(3, 8, 8),
            num_classes=2,
            hyperparameters={
                "stem_channels": 4,
                "stage_widths": [4, 8],
                "blocks_per_stage": [1, 1],
                "strides": [1, 2],
                "activation": "relu",
                "normalization": "batch_norm",
                "classifier_hidden_dims": [],
            },
        )
        model = ResidualNeuralNetwork(spec=spec, seed=42)

        profile = analyze_layer_geometry_profile(
            model=model,
            inputs=images,
            sample_ids=sample_ids,
            labels=labels,
            layers=["stem", "stage_0_block_0", "stage_1_block_0", "final_hidden"],
            experiment_id="exp-resnet-profile",
        )

        assert len(profile.layer_points) == 4
        assert profile.layer_points[0].layer_name == "stem"
        assert profile.layer_points[3].layer_name == "final_hidden"

    def test_vit_layer_evolution_profile(
        self,
        synthetic_inputs_and_labels: tuple[
            list[list[list[list[float]]]], list[str], list[int]
        ],
    ) -> None:
        images, sample_ids, labels = synthetic_inputs_and_labels
        spec = ModelSpecification(
            model_id="test-vit-profile",
            name="ViT Profile Model",
            family=ModelFamily.VISION_TRANSFORMER,
            architecture="vit_tiny",
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
        model = VisionTransformer(spec=spec, seed=42)

        profile = analyze_layer_geometry_profile(
            model=model,
            inputs=images,
            sample_ids=sample_ids,
            labels=labels,
            layers=["patch_embeddings", "encoder_0", "encoder_1", "cls_representation"],
            experiment_id="exp-vit-profile",
        )

        assert len(profile.layer_points) == 4
        assert profile.layer_points[0].layer_name == "patch_embeddings"
        assert profile.layer_points[3].layer_name == "cls_representation"
