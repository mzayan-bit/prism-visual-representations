"""Research service layer for representation geometry and observatory data."""

from __future__ import annotations

import math
from typing import Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
)

from prism.core.enums import ModelFamily, TaskType
from prism.models.cnn import ConvolutionalNeuralNetwork
from prism.models.resnet import ResidualNeuralNetwork
from prism.models.specifications import ModelSpecification
from prism.models.transformer import VisionTransformer
from prism.representations.comparison import (
    CrossArchitectureGeometryReport,
    compare_architecture_geometries,
)
from prism.representations.evolution import (
    LayerGeometryProfile,
    analyze_layer_geometry_profile,
)
from prism.representations.reports import (
    RepresentationGeometryReport,
)


class ObservatoryExperimentMeta(BaseModel):
    """Metadata describing an available experiment in the observatory."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    experiment_id: str = Field(description="Experiment identifier")
    name: str = Field(description="Display title")
    architectures: list[str] = Field(description="Available architecture keys")
    layers: dict[str, list[str]] = Field(
        description="Available layers keyed by architecture"
    )
    data_budgets: list[float] = Field(
        description="Available data budget ratios (e.g. 0.1, 0.5, 1.0)"
    )
    num_classes: int = Field(description="Number of classes in dataset")
    class_names: list[str] = Field(description="Class labels list")


class GeometryService:
    """Service managing representation geometry analysis and queries."""

    def __init__(self) -> None:
        self._reports_cache: dict[str, RepresentationGeometryReport] = {}
        self._profiles_cache: dict[str, LayerGeometryProfile] = {}
        self._comparisons_cache: dict[str, CrossArchitectureGeometryReport] = {}

    def register_report(
        self,
        report: RepresentationGeometryReport,
        budget: float = 1.0,
    ) -> None:
        """Register a computed geometry report in the service cache."""
        key = (
            f"{report.experiment_id}::{report.model_id}::"
            f"{report.layer_name}::{budget:.2f}"
        )
        self._reports_cache[key] = report

    def get_geometry_report(
        self,
        experiment_id: str,
        model_id: str,
        layer_name: str,
        budget: float = 1.0,
    ) -> RepresentationGeometryReport | None:
        """Retrieve a cached geometry report."""
        key = f"{experiment_id}::{model_id}::{layer_name}::{budget:.2f}"
        return self._reports_cache.get(key)

    def register_layer_profile(
        self,
        profile: LayerGeometryProfile,
    ) -> None:
        """Register a layer geometry evolution profile."""
        key = f"{profile.experiment_id}::{profile.model_id}"
        self._profiles_cache[key] = profile

    def get_layer_profile(
        self,
        experiment_id: str,
        model_id: str,
    ) -> LayerGeometryProfile | None:
        """Retrieve a cached layer evolution profile."""
        key = f"{experiment_id}::{model_id}"
        return self._profiles_cache.get(key)

    def register_comparison(
        self,
        comparison: CrossArchitectureGeometryReport,
    ) -> None:
        """Register a cross-architecture geometry comparison report."""
        self._comparisons_cache[comparison.comparison_id] = comparison

    def get_comparison(
        self,
        comparison_id: str,
    ) -> CrossArchitectureGeometryReport | None:
        """Retrieve a cross-architecture geometry comparison report."""
        return self._comparisons_cache.get(comparison_id)


def generate_observatory_demo_data(
    num_samples: int = 36,
    num_classes: int = 3,
    seed: int = 42,
) -> dict[str, Any]:
    """Generate authentic geometry reports across CNN, ResNet, and ViT.

    Parameters
    ----------
    num_samples : int
        Number of synthetic samples (divisible by num_classes).
    num_classes : int
        Number of distinct classes (e.g. 3).
    seed : int
        Deterministic random seed.

    Returns
    -------
    dict[str, Any]
        Dictionary containing serializable metadata, reports, layer profiles,
        and cross-architecture comparison.
    """
    class_names = ["class_0", "class_1", "class_2"][:num_classes]

    # Generate synthetic 3-channel 8x8 images with structured class patterns
    samples_per_class = num_samples // num_classes
    sample_ids: list[str] = []
    labels: list[int | str] = []
    images: list[list[list[list[float]]]] = []

    for c in range(num_classes):
        for s in range(samples_per_class):
            s_id = f"img_{c}_{s:03d}"
            sample_ids.append(s_id)
            labels.append(c)

            # Construct 3x8x8 image with class-specific base frequency + noise
            img = [[[0.0 for _ in range(8)] for _ in range(8)] for _ in range(3)]
            for ch in range(3):
                for r in range(8):
                    for col in range(8):
                        pattern = math.sin(
                            (c + 1) * 0.8 * (r + 1) + (ch + 1) * (col + 1) * 0.4
                        )
                        noise = (
                            ((s * 13 + r * 7 + col * 3 + ch * 11 + seed) % 100) / 100.0
                        ) * 0.2
                        img[ch][r][col] = pattern + noise
            images.append(img)

    # 1. Instantiate CNN Model
    cnn_spec = ModelSpecification(
        model_id="demo-cnn",
        name="Convolutional Baseline",
        family=ModelFamily.CNN,
        architecture="cnn_2layer",
        compatible_tasks=[TaskType.CLASSIFICATION],
        input_shape=(3, 8, 8),
        num_classes=num_classes,
        hyperparameters={
            "conv_channels": [8, 16],
            "kernel_sizes": [3, 3],
            "fc_hidden_dims": [16],
            "activation": "relu",
        },
    )
    cnn_model = ConvolutionalNeuralNetwork(spec=cnn_spec, seed=seed)

    # 2. Instantiate ResNet Model
    resnet_spec = ModelSpecification(
        model_id="demo-resnet",
        name="Residual Network",
        family=ModelFamily.RESNET,
        architecture="resnet_tiny",
        compatible_tasks=[TaskType.CLASSIFICATION],
        input_shape=(3, 8, 8),
        num_classes=num_classes,
        hyperparameters={
            "stem_channels": 8,
            "stage_widths": [8, 16],
            "blocks_per_stage": [1, 1],
            "strides": [1, 2],
            "activation": "relu",
            "normalization": "batch_norm",
            "classifier_hidden_dims": [],
        },
    )
    resnet_model = ResidualNeuralNetwork(spec=resnet_spec, seed=seed)

    # 3. Instantiate Vision Transformer Model
    vit_spec = ModelSpecification(
        model_id="demo-vit",
        name="Vision Transformer",
        family=ModelFamily.VISION_TRANSFORMER,
        architecture="vit_tiny",
        compatible_tasks=[TaskType.CLASSIFICATION],
        input_shape=(3, 8, 8),
        num_classes=num_classes,
        hyperparameters={
            "patch_size": 4,
            "embed_dim": 16,
            "num_heads": 2,
            "depth": 2,
            "mlp_ratio": 2.0,
            "norm_eps": 1e-5,
            "activation": "gelu",
        },
    )
    vit_model = VisionTransformer(spec=vit_spec, seed=seed)

    # 4. Compute Layer Evolution Profiles
    cnn_profile = analyze_layer_geometry_profile(
        model=cnn_model,
        inputs=images,
        sample_ids=sample_ids,
        labels=labels,
        layers=["conv_0", "conv_1", "final_hidden"],
        experiment_id="exp-observatory-demo",
        class_names=class_names,
    )

    resnet_profile = analyze_layer_geometry_profile(
        model=resnet_model,
        inputs=images,
        sample_ids=sample_ids,
        labels=labels,
        layers=["stem", "stage_0_block_0", "stage_1_block_0", "final_hidden"],
        experiment_id="exp-observatory-demo",
        class_names=class_names,
    )

    vit_profile = analyze_layer_geometry_profile(
        model=vit_model,
        inputs=images,
        sample_ids=sample_ids,
        labels=labels,
        layers=["patch_embeddings", "encoder_0", "encoder_1", "cls_representation"],
        experiment_id="exp-observatory-demo",
        class_names=class_names,
    )

    # 5. Cross-Architecture Geometry Comparison
    comparison = compare_architecture_geometries(
        models={"cnn": cnn_model, "resnet": resnet_model, "vit": vit_model},
        inputs=images,
        sample_ids=sample_ids,
        labels=labels,
        comparison_id="comp-observatory-demo",
        name="PRISM Observatory Cross-Architecture Geometry",
        data_budget=1.0,
        class_names=class_names,
    )

    # 6. Assemble Full Observatory Payload
    meta = ObservatoryExperimentMeta(
        experiment_id="exp-observatory-demo",
        name="PRISM Visual Representation Geometry Observatory",
        architectures=["cnn", "resnet", "vit"],
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
        "comparison": comparison.model_dump(mode="json"),
        "layer_profiles": {
            "cnn": cnn_profile.model_dump(mode="json"),
            "resnet": resnet_profile.model_dump(mode="json"),
            "vit": vit_profile.model_dump(mode="json"),
        },
        "reports": {
            "cnn::final_hidden": cnn_profile.detailed_reports[
                "final_hidden"
            ].model_dump(mode="json"),
            "resnet::final_hidden": resnet_profile.detailed_reports[
                "final_hidden"
            ].model_dump(mode="json"),
            "vit::cls_representation": vit_profile.detailed_reports[
                "cls_representation"
            ].model_dump(mode="json"),
        },
    }
