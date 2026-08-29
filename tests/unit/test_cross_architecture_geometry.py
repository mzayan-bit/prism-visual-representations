"""Unit tests for cross-architecture geometry comparison and seed aggregation."""

from __future__ import annotations

from typing import Any

import pytest

from prism.core.enums import ModelFamily, TaskType
from prism.models.cnn import ConvolutionalNeuralNetwork
from prism.models.resnet import ResidualNeuralNetwork
from prism.models.specifications import ModelSpecification
from prism.models.transformer import VisionTransformer
from prism.representations.comparison import (
    CrossArchitectureGeometryReport,
    aggregate_repeated_seed_geometry,
    compare_architecture_geometries,
)
from prism.representations.geometry import RepresentationDataset
from prism.representations.reports import analyze_representation_geometry


class TestCrossArchitectureGeometry:
    """Test suite for comparative geometry across CNN, ResNet, and ViT."""

    @pytest.fixture
    def setup_models_and_data(
        self,
    ) -> tuple[
        dict[str, Any],
        list[list[list[list[float]]]],
        list[str],
        list[int],
    ]:
        num_samples = 6
        images = []
        labels = [0, 0, 0, 1, 1, 1]
        sample_ids = [f"s_{i}" for i in range(num_samples)]
        for i in range(num_samples):
            v = 1.0 if labels[i] == 1 else -1.0
            images.append([[[v for _ in range(8)] for _ in range(8)] for _ in range(3)])

        cnn_spec = ModelSpecification(
            model_id="comp-cnn",
            name="CNN",
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
        resnet_spec = ModelSpecification(
            model_id="comp-resnet",
            name="ResNet",
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
        vit_spec = ModelSpecification(
            model_id="comp-vit",
            name="ViT",
            family=ModelFamily.VISION_TRANSFORMER,
            architecture="vit_tiny",
            compatible_tasks=[TaskType.CLASSIFICATION],
            input_shape=(3, 8, 8),
            num_classes=2,
            hyperparameters={
                "patch_size": 4,
                "embed_dim": 8,
                "num_heads": 2,
                "depth": 1,
                "mlp_ratio": 2.0,
                "norm_eps": 1e-5,
                "activation": "gelu",
            },
        )

        models = {
            "cnn": ConvolutionalNeuralNetwork(spec=cnn_spec, seed=42),
            "resnet": ResidualNeuralNetwork(spec=resnet_spec, seed=42),
            "vit": VisionTransformer(spec=vit_spec, seed=42),
        }

        return models, images, sample_ids, labels

    def test_compare_architecture_geometries(self, setup_models_and_data: Any) -> None:
        models, images, sample_ids, labels = setup_models_and_data

        report = compare_architecture_geometries(
            models=models,
            inputs=images,
            sample_ids=sample_ids,
            labels=labels,
            comparison_id="test-comp-suite",
            name="Test Suite Geometry Comparison",
        )

        assert report.comparison_id == "test-comp-suite"
        assert len(report.architectures) == 3
        assert "cnn" in report.architectures
        assert "resnet" in report.architectures
        assert "vit" in report.architectures

        # Verify summary invariants
        for _key, summary in report.architectures.items():
            assert summary.feature_dim > 0
            assert summary.mean_vector_norm >= 0.0
            assert summary.intra_class_compactness >= 0.0
            assert summary.inter_class_separation >= 0.0
            assert 0.0 <= summary.neighbor_label_consistency <= 1.0

        # Roundtrip JSON serialization
        json_str = report.to_json()
        assert len(json_str) > 0
        deserialized = CrossArchitectureGeometryReport.from_dict(report.to_dict())
        assert len(deserialized.architectures) == 3

    def test_aggregate_repeated_seed_geometry(self) -> None:
        # Create mock reports from 3 seed runs
        reports = []
        for seed_idx, offset in enumerate([0.1, 0.2, 0.3]):
            ds = RepresentationDataset(
                experiment_id=f"seed-{seed_idx}",
                model_id="seed-model",
                layer_name="final",
                sample_ids=["s0", "s1", "s2", "s3"],
                labels=[0, 0, 1, 1],
                vectors=[
                    [0.0 + offset, 0.0],
                    [0.1 + offset, 0.0],
                    [10.0 + offset, 10.0],
                    [10.1 + offset, 10.0],
                ],
                feature_dim=2,
                num_samples=4,
                num_classes=2,
            )
            reports.append(analyze_representation_geometry(ds, k=2))

        aggregated = aggregate_repeated_seed_geometry(reports)
        assert len(aggregated) == 5

        metric_names = [m.metric_name for m in aggregated]
        assert "intra_class_compactness" in metric_names
        assert "inter_class_separation" in metric_names
        assert "separation_to_compactness_ratio" in metric_names
        assert "neighbor_label_consistency" in metric_names

        for m in aggregated:
            assert m.num_seeds == 3
            assert m.std >= 0.0
