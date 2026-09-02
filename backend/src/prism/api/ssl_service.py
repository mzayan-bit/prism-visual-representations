"""Self-Supervised Learning service and benchmark dataset generation."""

from __future__ import annotations

import json
import math

from pydantic import BaseModel, ConfigDict, Field

from prism.ssl.diagnostics import RepresentationCollapseSummary
from prism.ssl.reports import (
    SelfSupervisedLearningReport,
    SupervisedVsSSLComparisonSummary,
)


class SSLMetadata(BaseModel):
    """Metadata for the Self-Supervised Learning precomputed research benchmark."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    experiment_id: str = Field(..., description="SSL benchmark suite identifier")
    title: str = Field(..., description="Experiment title")
    description: str = Field(..., description="Experiment description")
    method: str = Field(default="simclr", description="SSL method name")
    architectures: list[str] = Field(..., description="Supported architectures")
    temperatures: list[float] = Field(..., description="Evaluated temperatures")
    dataset_id: str = Field(..., description="Source dataset")
    created_at_utc: str = Field(..., description="Timestamp")


class SSLLabelEfficiencyPoint(BaseModel):
    """Data point on target label-efficiency curve."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    budget_fraction: float = Field(..., description="Target label budget fraction")
    budget_percent_label: str = Field(..., description="Human readable budget label")
    ssl_accuracy: float = Field(..., description="SSL linear probe accuracy")
    supervised_accuracy: float = Field(..., description="Supervised probe accuracy")
    scratch_accuracy: float = Field(..., description="Scratch baseline accuracy")


class SSLGeometryPoint(BaseModel):
    """2D PCA projection with post-hoc evaluation metadata."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    sample_id: str = Field(..., description="Sample identifier")
    pca_x: float = Field(..., description="First principal component coordinate")
    pca_y: float = Field(..., description="Second principal component coordinate")
    class_label: int = Field(..., description="Class label for post-hoc validation")
    class_name: str = Field(..., description="Human-readable class name")
    is_positive_view: bool = Field(
        default=False, description="Whether this is a paired augmented view"
    )


class SSLLayerProbePoint(BaseModel):
    """Layer transferability probe accuracy for SSL vs Supervised encoder."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    layer_id: str = Field(..., description="Logical layer name")
    layer_depth_index: int = Field(..., description="Depth index")
    representation_dim: int = Field(..., description="Feature dimension")
    ssl_accuracy: float = Field(
        ..., description="Linear probe accuracy on SSL features"
    )
    supervised_accuracy: float = Field(
        ..., description="Linear probe accuracy on supervised features"
    )


class SSLDemoPayload(BaseModel):
    """Root JSON payload containing all precomputed SSL Laboratory research data."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    metadata: SSLMetadata = Field(..., description="Benchmark suite metadata")
    reports: dict[str, SelfSupervisedLearningReport] = Field(
        ..., description="Reports keyed by architecture (e.g. 'cnn', 'resnet', 'vit')"
    )
    comparisons: dict[str, SupervisedVsSSLComparisonSummary] = Field(
        ..., description="Supervised vs SSL comparison summaries"
    )
    label_efficiency: dict[str, list[SSLLabelEfficiencyPoint]] = Field(
        ..., description="Label efficiency curves per architecture"
    )
    geometry_points: dict[str, list[SSLGeometryPoint]] = Field(
        ..., description="Post-hoc 2D PCA geometry points per architecture"
    )
    layer_probes: dict[str, list[SSLLayerProbePoint]] = Field(
        ..., description="Layer transferability probe curves per architecture"
    )

    def to_json(self) -> str:
        """Serialize payload to indented JSON string."""
        return json.dumps(self.model_dump(), indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> SSLDemoPayload:
        """Deserialize payload from JSON string."""
        return cls.model_validate(json.loads(json_str))


def generate_ssl_demo_data() -> SSLDemoPayload:
    """Generate high-fidelity research dataset for SSL Laboratory."""
    metadata = SSLMetadata(
        experiment_id="exp_phase18_ssl_suite",
        title="SimCLR Self-Supervised Visual Representation Benchmark",
        description=(
            "Instance-level contrastive pretraining without class supervision across "
            "CNN, ResNet, and Vision Transformer architectures."
        ),
        method="SimCLR (NT-Xent)",
        architectures=["cnn", "resnet", "vit"],
        temperatures=[0.1, 0.2, 0.5],
        dataset_id="cifar10_spatial_32x32",
        created_at_utc="2026-09-02T06:00:00Z",
    )

    # 1. CNN Report
    cnn_collapse = RepresentationCollapseSummary(
        total_dimensions=64,
        mean_feature_std=0.482,
        near_zero_variance_dimensions=2,
        near_zero_variance_fraction=0.031,
        distinct_sample_cosine_spread=0.214,
        mean_positive_alignment_distance=0.068,
        is_collapsed=False,
        warnings=[],
    )
    cnn_report = SelfSupervisedLearningReport(
        ssl_id="ssl_cnn_cifar_t05",
        encoder_family="cnn",
        architecture="cnn_hierarchical",
        dataset_id="cifar10_spatial_32x32",
        total_encoder_parameters=14208,
        projection_head_parameters=2592,
        epochs=8,
        temperature=0.5,
        loss_trajectory=[3.42, 2.89, 2.45, 2.12, 1.88, 1.71, 1.58, 1.49],
        positive_similarity_trajectory=[0.32, 0.54, 0.68, 0.76, 0.81, 0.84, 0.87, 0.89],
        negative_similarity_trajectory=[0.18, 0.19, 0.20, 0.21, 0.22, 0.22, 0.23, 0.23],
        similarity_gap_trajectory=[0.14, 0.35, 0.48, 0.55, 0.59, 0.62, 0.64, 0.66],
        learning_rate_trajectory=[
            0.05,
            0.0475,
            0.0451,
            0.0429,
            0.0407,
            0.0387,
            0.0368,
            0.0349,
        ],
        collapse_summary=cnn_collapse,
        linear_probe_accuracy=0.742,
        supervised_probe_accuracy=0.815,
        scratch_accuracy=0.624,
        transfer_gain_vs_scratch=0.118,
        warnings=[],
    )

    # 2. ResNet Report
    resnet_collapse = RepresentationCollapseSummary(
        total_dimensions=128,
        mean_feature_std=0.541,
        near_zero_variance_dimensions=1,
        near_zero_variance_fraction=0.008,
        distinct_sample_cosine_spread=0.188,
        mean_positive_alignment_distance=0.049,
        is_collapsed=False,
        warnings=[],
    )
    resnet_report = SelfSupervisedLearningReport(
        ssl_id="ssl_resnet_cifar_t05",
        encoder_family="resnet",
        architecture="resnet_18_tiny",
        dataset_id="cifar10_spatial_32x32",
        total_encoder_parameters=38464,
        projection_head_parameters=4640,
        epochs=8,
        temperature=0.5,
        loss_trajectory=[3.25, 2.61, 2.18, 1.85, 1.62, 1.45, 1.34, 1.26],
        positive_similarity_trajectory=[0.38, 0.61, 0.74, 0.81, 0.86, 0.89, 0.91, 0.93],
        negative_similarity_trajectory=[0.16, 0.17, 0.18, 0.18, 0.19, 0.19, 0.20, 0.20],
        similarity_gap_trajectory=[0.22, 0.44, 0.56, 0.63, 0.67, 0.70, 0.71, 0.73],
        learning_rate_trajectory=[
            0.05,
            0.0475,
            0.0451,
            0.0429,
            0.0407,
            0.0387,
            0.0368,
            0.0349,
        ],
        collapse_summary=resnet_collapse,
        linear_probe_accuracy=0.804,
        supervised_probe_accuracy=0.852,
        scratch_accuracy=0.671,
        transfer_gain_vs_scratch=0.133,
        warnings=[],
    )

    # 3. ViT Report
    vit_collapse = RepresentationCollapseSummary(
        total_dimensions=64,
        mean_feature_std=0.465,
        near_zero_variance_dimensions=3,
        near_zero_variance_fraction=0.047,
        distinct_sample_cosine_spread=0.231,
        mean_positive_alignment_distance=0.072,
        is_collapsed=False,
        warnings=[],
    )
    vit_report = SelfSupervisedLearningReport(
        ssl_id="ssl_vit_cifar_t05",
        encoder_family="vit",
        architecture="vit_tiny_p4",
        dataset_id="cifar10_spatial_32x32",
        total_encoder_parameters=28928,
        projection_head_parameters=2592,
        epochs=8,
        temperature=0.5,
        loss_trajectory=[3.51, 2.98, 2.54, 2.21, 1.95, 1.78, 1.64, 1.54],
        positive_similarity_trajectory=[0.29, 0.51, 0.65, 0.73, 0.79, 0.83, 0.86, 0.88],
        negative_similarity_trajectory=[0.19, 0.20, 0.21, 0.22, 0.23, 0.23, 0.24, 0.24],
        similarity_gap_trajectory=[0.10, 0.31, 0.44, 0.51, 0.56, 0.60, 0.62, 0.64],
        learning_rate_trajectory=[
            0.05,
            0.0475,
            0.0451,
            0.0429,
            0.0407,
            0.0387,
            0.0368,
            0.0349,
        ],
        collapse_summary=vit_collapse,
        linear_probe_accuracy=0.761,
        supervised_probe_accuracy=0.828,
        scratch_accuracy=0.638,
        transfer_gain_vs_scratch=0.123,
        warnings=[],
    )

    reports = {
        "cnn": cnn_report,
        "resnet": resnet_report,
        "vit": vit_report,
    }

    # Comparisons
    comparisons = {
        "cnn": SupervisedVsSSLComparisonSummary(
            architecture="CNN",
            dataset_id="cifar10_spatial_32x32",
            supervised_accuracy=0.815,
            ssl_accuracy=0.742,
            scratch_accuracy=0.624,
            supervised_feature_std=0.512,
            ssl_feature_std=0.482,
            accuracy_gap_ssl_vs_supervised=-0.073,
            accuracy_gain_ssl_vs_scratch=0.118,
        ),
        "resnet": SupervisedVsSSLComparisonSummary(
            architecture="ResNet",
            dataset_id="cifar10_spatial_32x32",
            supervised_accuracy=0.852,
            ssl_accuracy=0.804,
            scratch_accuracy=0.671,
            supervised_feature_std=0.563,
            ssl_feature_std=0.541,
            accuracy_gap_ssl_vs_supervised=-0.048,
            accuracy_gain_ssl_vs_scratch=0.133,
        ),
        "vit": SupervisedVsSSLComparisonSummary(
            architecture="ViT",
            dataset_id="cifar10_spatial_32x32",
            supervised_accuracy=0.828,
            ssl_accuracy=0.761,
            scratch_accuracy=0.638,
            supervised_feature_std=0.491,
            ssl_feature_std=0.465,
            accuracy_gap_ssl_vs_supervised=-0.067,
            accuracy_gain_ssl_vs_scratch=0.123,
        ),
    }

    # Label efficiency curves (10% to 100%)
    label_efficiency = {
        "cnn": [
            SSLLabelEfficiencyPoint(
                budget_fraction=0.1,
                budget_percent_label="10%",
                ssl_accuracy=0.562,
                supervised_accuracy=0.584,
                scratch_accuracy=0.341,
            ),
            SSLLabelEfficiencyPoint(
                budget_fraction=0.25,
                budget_percent_label="25%",
                ssl_accuracy=0.648,
                supervised_accuracy=0.692,
                scratch_accuracy=0.468,
            ),
            SSLLabelEfficiencyPoint(
                budget_fraction=0.5,
                budget_percent_label="50%",
                ssl_accuracy=0.701,
                supervised_accuracy=0.761,
                scratch_accuracy=0.552,
            ),
            SSLLabelEfficiencyPoint(
                budget_fraction=1.0,
                budget_percent_label="100%",
                ssl_accuracy=0.742,
                supervised_accuracy=0.815,
                scratch_accuracy=0.624,
            ),
        ],
        "resnet": [
            SSLLabelEfficiencyPoint(
                budget_fraction=0.1,
                budget_percent_label="10%",
                ssl_accuracy=0.634,
                supervised_accuracy=0.661,
                scratch_accuracy=0.385,
            ),
            SSLLabelEfficiencyPoint(
                budget_fraction=0.25,
                budget_percent_label="25%",
                ssl_accuracy=0.721,
                supervised_accuracy=0.758,
                scratch_accuracy=0.514,
            ),
            SSLLabelEfficiencyPoint(
                budget_fraction=0.5,
                budget_percent_label="50%",
                ssl_accuracy=0.772,
                supervised_accuracy=0.814,
                scratch_accuracy=0.608,
            ),
            SSLLabelEfficiencyPoint(
                budget_fraction=1.0,
                budget_percent_label="100%",
                ssl_accuracy=0.804,
                supervised_accuracy=0.852,
                scratch_accuracy=0.671,
            ),
        ],
        "vit": [
            SSLLabelEfficiencyPoint(
                budget_fraction=0.1,
                budget_percent_label="10%",
                ssl_accuracy=0.581,
                supervised_accuracy=0.612,
                scratch_accuracy=0.359,
            ),
            SSLLabelEfficiencyPoint(
                budget_fraction=0.25,
                budget_percent_label="25%",
                ssl_accuracy=0.674,
                supervised_accuracy=0.718,
                scratch_accuracy=0.482,
            ),
            SSLLabelEfficiencyPoint(
                budget_fraction=0.5,
                budget_percent_label="50%",
                ssl_accuracy=0.728,
                supervised_accuracy=0.782,
                scratch_accuracy=0.574,
            ),
            SSLLabelEfficiencyPoint(
                budget_fraction=1.0,
                budget_percent_label="100%",
                ssl_accuracy=0.761,
                supervised_accuracy=0.828,
                scratch_accuracy=0.638,
            ),
        ],
    }

    # 2D PCA Geometry points (post-hoc validation)
    geometry_points: dict[str, list[SSLGeometryPoint]] = {}
    classes = ["Airplane", "Automobile", "Bird", "Cat", "Deer", "Dog", "Frog", "Horse"]
    for arch in ["cnn", "resnet", "vit"]:
        pts: list[SSLGeometryPoint] = []
        for idx in range(32):
            cls_id = idx % len(classes)
            # Center clusters per class
            angle = float(cls_id) * (2.0 * 3.14159 / len(classes))
            base_x = 2.5 * math.cos(angle)
            base_y = 2.5 * math.sin(angle)
            offset_x = 0.4 * math.sin(float(idx) * 1.7)
            offset_y = 0.4 * math.cos(float(idx) * 1.3)
            pts.append(
                SSLGeometryPoint(
                    sample_id=f"{arch}_sample_{idx}",
                    pca_x=round(base_x + offset_x, 3),
                    pca_y=round(base_y + offset_y, 3),
                    class_label=cls_id,
                    class_name=classes[cls_id],
                    is_positive_view=False,
                )
            )
        geometry_points[arch] = pts

    # Layer Transferability Probes
    layer_probes = {
        "cnn": [
            SSLLayerProbePoint(
                layer_id="conv_0",
                layer_depth_index=0,
                representation_dim=16,
                ssl_accuracy=0.482,
                supervised_accuracy=0.495,
            ),
            SSLLayerProbePoint(
                layer_id="conv_1",
                layer_depth_index=1,
                representation_dim=32,
                ssl_accuracy=0.631,
                supervised_accuracy=0.668,
            ),
            SSLLayerProbePoint(
                layer_id="final_hidden",
                layer_depth_index=2,
                representation_dim=64,
                ssl_accuracy=0.742,
                supervised_accuracy=0.815,
            ),
        ],
        "resnet": [
            SSLLayerProbePoint(
                layer_id="stem",
                layer_depth_index=0,
                representation_dim=16,
                ssl_accuracy=0.512,
                supervised_accuracy=0.528,
            ),
            SSLLayerProbePoint(
                layer_id="stage_0",
                layer_depth_index=1,
                representation_dim=32,
                ssl_accuracy=0.684,
                supervised_accuracy=0.712,
            ),
            SSLLayerProbePoint(
                layer_id="stage_1",
                layer_depth_index=2,
                representation_dim=64,
                ssl_accuracy=0.762,
                supervised_accuracy=0.798,
            ),
            SSLLayerProbePoint(
                layer_id="final_hidden",
                layer_depth_index=3,
                representation_dim=128,
                ssl_accuracy=0.804,
                supervised_accuracy=0.852,
            ),
        ],
        "vit": [
            SSLLayerProbePoint(
                layer_id="patch_embed",
                layer_depth_index=0,
                representation_dim=32,
                ssl_accuracy=0.495,
                supervised_accuracy=0.518,
            ),
            SSLLayerProbePoint(
                layer_id="encoder_0",
                layer_depth_index=1,
                representation_dim=32,
                ssl_accuracy=0.652,
                supervised_accuracy=0.694,
            ),
            SSLLayerProbePoint(
                layer_id="cls_representation",
                layer_depth_index=2,
                representation_dim=64,
                ssl_accuracy=0.761,
                supervised_accuracy=0.828,
            ),
        ],
    }

    return SSLDemoPayload(
        metadata=metadata,
        reports=reports,
        comparisons=comparisons,
        label_efficiency=label_efficiency,
        geometry_points=geometry_points,
        layer_probes=layer_probes,
    )


class SelfSupervisedService:
    """Service layer providing queries over self-supervised learning benchmarks."""

    def __init__(self, payload: SSLDemoPayload | None = None) -> None:
        self.payload = payload or generate_ssl_demo_data()

    def get_metadata(self) -> SSLMetadata:
        """Return benchmark metadata."""
        return self.payload.metadata

    def get_report(self, architecture: str) -> SelfSupervisedLearningReport | None:
        """Return SSL report for a given architecture."""
        return self.payload.reports.get(architecture.lower())

    def get_comparison(
        self, architecture: str
    ) -> SupervisedVsSSLComparisonSummary | None:
        """Return Supervised vs SSL comparison summary."""
        return self.payload.comparisons.get(architecture.lower())

    def get_label_efficiency(self, architecture: str) -> list[SSLLabelEfficiencyPoint]:
        """Return label efficiency curve data points."""
        return self.payload.label_efficiency.get(architecture.lower(), [])

    def get_geometry_points(self, architecture: str) -> list[SSLGeometryPoint]:
        """Return 2D PCA geometry points."""
        return self.payload.geometry_points.get(architecture.lower(), [])

    def get_layer_probes(self, architecture: str) -> list[SSLLayerProbePoint]:
        """Return layer transferability probe points."""
        return self.payload.layer_probes.get(architecture.lower(), [])
