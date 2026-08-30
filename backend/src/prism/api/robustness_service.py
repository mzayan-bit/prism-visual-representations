"""Research service layer for robustness, distribution shift, and laboratory data."""

from __future__ import annotations

import math
from typing import Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
)

from prism.core.enums import ModelFamily, TaskType
from prism.data.materialized import MaterializedDataset, MaterializedSample
from prism.models.cnn import ConvolutionalNeuralNetwork
from prism.models.resnet import ResidualNeuralNetwork
from prism.models.specifications import ModelSpecification
from prism.models.transformer import VisionTransformer
from prism.robustness.corruptions import (
    CorruptionType,
)
from prism.robustness.evaluation import (
    CorruptionSuite,
    CrossArchitectureRobustnessReport,
    RobustnessExperimentReport,
    RobustnessSuiteRunner,
    compare_architecture_robustness,
)


class RobustnessExperimentMeta(BaseModel):
    """Metadata describing an available robustness experiment in the laboratory."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    experiment_id: str = Field(description="Experiment identifier")
    name: str = Field(description="Display title")
    architectures: list[str] = Field(description="Available architecture keys")
    corruption_types: list[str] = Field(description="Available corruption types")
    severities: list[int] = Field(description="Available severity levels")
    layers: dict[str, list[str]] = Field(
        description="Available layers keyed by architecture"
    )
    data_budgets: list[float] = Field(description="Available data budget ratios")
    num_classes: int = Field(description="Number of classes in dataset")
    class_names: list[str] = Field(description="Class labels list")


class RobustnessService:
    """Service managing robustness experiments, evaluations, and laboratory queries."""

    def __init__(self) -> None:
        self._reports_cache: dict[str, RobustnessExperimentReport] = {}
        self._comparisons_cache: dict[str, CrossArchitectureRobustnessReport] = {}

    def register_report(
        self,
        report: RobustnessExperimentReport,
        budget: float = 1.0,
    ) -> None:
        """Cache a robustness report."""
        key = (
            f"{report.experiment_id}::{report.model_id}::{report.layer_name}::{budget}"
        )
        self._reports_cache[key] = report

    def get_report(
        self,
        experiment_id: str,
        model_id: str,
        layer_name: str,
        budget: float = 1.0,
    ) -> RobustnessExperimentReport | None:
        """Retrieve a cached robustness report."""
        key = f"{experiment_id}::{model_id}::{layer_name}::{budget}"
        return self._reports_cache.get(key)

    def register_comparison(
        self,
        comparison: CrossArchitectureRobustnessReport,
    ) -> None:
        """Cache a cross-architecture robustness comparison report."""
        self._comparisons_cache[comparison.comparison_id] = comparison

    def get_comparison(
        self, comparison_id: str
    ) -> CrossArchitectureRobustnessReport | None:
        """Retrieve a cached comparison report."""
        return self._comparisons_cache.get(comparison_id)


def generate_robustness_demo_data(
    num_samples: int = 24,
    num_classes: int = 3,
    seed: int = 42,
) -> dict[str, Any]:
    """Generate deterministic, rich robustness demo data across CNN, ResNet, and ViT."""
    samples: list[MaterializedSample] = []
    class_names = [f"class_{c}" for c in range(num_classes)]

    for i in range(num_samples):
        cls_idx = i % num_classes
        # Generate 3x8x8 synthetic image with class-specific structural signals
        img: list[list[list[float]]] = []
        for ch in range(3):
            ch_plane: list[list[float]] = []
            for r in range(8):
                row: list[float] = []
                for c in range(8):
                    # Base signal depends on class
                    base_val = math.sin(float(cls_idx + 1) * (r + 1) * 0.4) + math.cos(
                        float(cls_idx + 1) * (c + 1) * 0.4
                    )
                    noise = 0.05 * math.sin(float(i * 10 + ch * 5 + r * 2 + c))
                    row.append(base_val + noise)
                ch_plane.append(row)
            img.append(ch_plane)

        sample = MaterializedSample(
            sample_id=f"img_{cls_idx}_{i:03d}",
            source_split="test",
            source_index=i,
            data=img,
            target=cls_idx,
            metadata={"class_name": class_names[cls_idx]},
        )
        samples.append(sample)

    dataset = MaterializedDataset(
        dataset_id="ds-robustness-eval-demo",
        samples=samples,
        split_name="test",
    )

    # Instantiate tiny models for CNN, ResNet, and ViT
    cnn_spec = ModelSpecification(
        model_id="cnn_tiny_demo",
        name="CNN Tiny Robustness Model",
        family=ModelFamily.CNN,
        architecture="cnn_tiny",
        compatible_tasks=[TaskType.CLASSIFICATION],
        input_shape=(3, 8, 8),
        num_classes=num_classes,
        hyperparameters={
            "conv_channels": [4, 8],
            "kernel_sizes": [3, 3],
            "fc_hidden_dims": [12],
            "activation": "relu",
        },
    )
    cnn_model = ConvolutionalNeuralNetwork(spec=cnn_spec, seed=seed)

    resnet_spec = ModelSpecification(
        model_id="resnet_tiny_demo",
        name="ResNet Tiny Robustness Model",
        family=ModelFamily.RESNET,
        architecture="resnet_tiny",
        compatible_tasks=[TaskType.CLASSIFICATION],
        input_shape=(3, 8, 8),
        num_classes=num_classes,
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
    resnet_model = ResidualNeuralNetwork(spec=resnet_spec, seed=seed)

    vit_spec = ModelSpecification(
        model_id="vit_tiny_demo",
        name="ViT Tiny Robustness Model",
        family=ModelFamily.VISION_TRANSFORMER,
        architecture="vit_tiny",
        compatible_tasks=[TaskType.CLASSIFICATION],
        input_shape=(3, 8, 8),
        num_classes=num_classes,
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
    vit_model = VisionTransformer(spec=vit_spec, seed=seed)

    models = {
        "cnn": cnn_model,
        "resnet": resnet_model,
        "vit": vit_model,
    }

    suite = CorruptionSuite(
        suite_id="suite-robustness-standard",
        name="Standard 6-Family Robustness Suite",
        corruption_types=[
            CorruptionType.GAUSSIAN_NOISE,
            CorruptionType.BLUR,
            CorruptionType.BRIGHTNESS,
            CorruptionType.CONTRAST,
            CorruptionType.OCCLUSION,
            CorruptionType.RESOLUTION_DEGRADATION,
        ],
        severities=[1, 2, 3, 4, 5],
        eval_split="test",
        layer_name="final_hidden",
        seed=seed,
        k_neighbors=4,
        pca_components=2,
    )

    runner = RobustnessSuiteRunner()
    reports: dict[str, Any] = {}

    for arch_key, model in models.items():
        rep = runner.run_suite(
            model=model,
            clean_dataset=dataset,
            suite=suite,
            experiment_id=f"exp-robustness-{arch_key}",
        )
        reports[arch_key] = rep.to_dict()

    comp_report = compare_architecture_robustness(
        models=models,
        clean_dataset=dataset,
        suite=suite,
        comparison_id="comp-robustness-demo",
        name="Cross-Architecture Robustness Benchmark (CNN vs ResNet vs ViT)",
    )

    meta = RobustnessExperimentMeta(
        experiment_id="exp-robustness-demo",
        name="PRISM Robustness & Distribution Shift Benchmark",
        architectures=["cnn", "resnet", "vit"],
        corruption_types=[ct.value for ct in suite.corruption_types],
        severities=[1, 2, 3, 4, 5],
        layers={
            "cnn": ["conv_0", "conv_1", "final_hidden"],
            "resnet": ["stem", "stage_0_block_0", "stage_1_block_0", "final_hidden"],
            "vit": [
                "patch_embeddings",
                "encoder_0",
                "encoder_1",
                "cls_representation",
            ],
        },
        data_budgets=[0.1, 0.25, 0.5, 1.0],
        num_classes=num_classes,
        class_names=class_names,
    )

    return {
        "metadata": meta.model_dump(mode="json"),
        "reports": reports,
        "comparison": comp_report.to_dict(),
    }
