"""Unit tests for the robustness suite runner, curves, and architecture benchmarks."""

import math

from prism.core.enums import ModelFamily, TaskType
from prism.data.materialized import MaterializedDataset, MaterializedSample
from prism.models.cnn import ConvolutionalNeuralNetwork
from prism.models.resnet import ResidualNeuralNetwork
from prism.models.specifications import ModelSpecification
from prism.models.transformer import VisionTransformer
from prism.robustness.corruptions import CorruptionType
from prism.robustness.evaluation import (
    CorruptionSuite,
    RobustnessSuiteRunner,
    compare_architecture_robustness,
)


def _make_dummy_dataset(num_samples: int = 6) -> MaterializedDataset:
    samples: list[MaterializedSample] = []
    for i in range(num_samples):
        cls_idx = i % 2
        img: list[list[list[float]]] = []
        for _ch in range(3):
            plane: list[list[float]] = []
            for r in range(8):
                row: list[float] = []
                for c in range(8):
                    val = 0.5 + 0.3 * math.sin(float(cls_idx * 5 + r * 2 + c))
                    row.append(max(0.0, min(1.0, val)))
                plane.append(row)
            img.append(plane)

        sample = MaterializedSample(
            sample_id=f"sample_{i}",
            source_split="test",
            source_index=i,
            data=img,
            target=cls_idx,
        )
        samples.append(sample)

    return MaterializedDataset(dataset_id="ds_test", samples=samples, split_name="test")


def test_robustness_suite_runner() -> None:
    dataset = _make_dummy_dataset(num_samples=6)

    cnn_spec = ModelSpecification(
        model_id="cnn_eval_test",
        name="CNN Eval Test",
        family=ModelFamily.CNN,
        architecture="cnn_eval_test",
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
    model = ConvolutionalNeuralNetwork(spec=cnn_spec, seed=42)

    suite = CorruptionSuite(
        suite_id="test_suite",
        name="Test Suite",
        corruption_types=[CorruptionType.GAUSSIAN_NOISE, CorruptionType.BLUR],
        severities=[1, 3, 5],
        eval_split="test",
        layer_name="final_hidden",
        seed=42,
        k_neighbors=2,
        pca_components=2,
    )

    runner = RobustnessSuiteRunner()
    report = runner.run_suite(model=model, clean_dataset=dataset, suite=suite)

    assert report.model_id == "cnn_eval_test"
    assert report.num_samples == 6
    assert 0.0 <= report.clean_accuracy <= 1.0
    assert len(report.evaluations) == 6  # 2 corruptions * 3 severities
    assert len(report.severity_curves) == 2

    curve = report.severity_curves[CorruptionType.GAUSSIAN_NOISE.value]
    assert len(curve.accuracy_trajectory) == 3
    assert 0.0 <= curve.area_under_curve <= 1.0


def test_cross_architecture_robustness() -> None:
    dataset = _make_dummy_dataset(num_samples=4)

    cnn_spec = ModelSpecification(
        model_id="cnn_comp",
        name="CNN Comp",
        family=ModelFamily.CNN,
        architecture="cnn_comp",
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
    resnet_spec = ModelSpecification(
        model_id="resnet_comp",
        name="ResNet Comp",
        family=ModelFamily.RESNET,
        architecture="resnet_comp",
        compatible_tasks=[TaskType.CLASSIFICATION],
        input_shape=(3, 8, 8),
        num_classes=2,
        hyperparameters={
            "stem_channels": 4,
            "stage_widths": [4, 4],
            "blocks_per_stage": [1, 1],
            "strides": [1, 1],
            "activation": "relu",
            "normalization": "batch_norm",
            "classifier_hidden_dims": [],
        },
    )
    vit_spec = ModelSpecification(
        model_id="vit_comp",
        name="ViT Comp",
        family=ModelFamily.VISION_TRANSFORMER,
        architecture="vit_comp",
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

    suite = CorruptionSuite(
        suite_id="comp_suite",
        name="Comp Suite",
        corruption_types=[CorruptionType.GAUSSIAN_NOISE],
        severities=[1, 2],
        k_neighbors=2,
        pca_components=2,
    )

    comp_report = compare_architecture_robustness(
        models=models,
        clean_dataset=dataset,
        suite=suite,
    )

    assert len(comp_report.architectures) == 3
    assert "cnn" in comp_report.architectures
    assert "resnet" in comp_report.architectures
    assert "vit" in comp_report.architectures
