"""Canonical factor and metric registries with semantic safeguards."""

from __future__ import annotations

from typing import Any, ClassVar

from prism.benchmarking.contracts import MetricDefinition
from prism.benchmarking.enums import FactorID, MetricCategory, MetricDirection
from prism.core.errors import ValidationError


class FactorRegistry:
    """Registry and validation for standard PRISM experimental factors."""

    _CANONICAL_FACTORS: ClassVar[dict[FactorID, dict[str, Any]]] = {
        FactorID.ARCHITECTURE: {
            "display_name": "Visual Architecture",
            "description": "Backbone architecture family (CNN, ResNet, ViT)",
        },
        FactorID.PRETRAINING_OBJECTIVE: {
            "display_name": "Pretraining Objective",
            "description": (
                "Learning paradigm (Supervised, SimCLR, Reconstruction, "
                "Vision-Language, Scratch)"
            ),
        },
        FactorID.DATASET: {
            "display_name": "Dataset",
            "description": "Evaluation dataset identity",
        },
        FactorID.TASK: {
            "display_name": "Task",
            "description": (
                "Task family (Classification, Transfer, Spatial, Temporal, "
                "Multimodal, Robustness)"
            ),
        },
        FactorID.DATA_BUDGET: {
            "display_name": "Data Budget",
            "description": (
                "Fraction of training data used (1%, 5%, 10%, 25%, 50%, 100%)"
            ),
        },
        FactorID.SEED: {
            "display_name": "Random Seed",
            "description": "RNG initialization seed",
        },
        FactorID.TRANSFER_STRATEGY: {
            "display_name": "Transfer Strategy",
            "description": (
                "Downstream adaptation mode (linear_probe, full_fine_tune, "
                "frozen_probe)"
            ),
        },
        FactorID.REPRESENTATION_LAYER: {
            "display_name": "Representation Layer",
            "description": "Extracted feature layer name",
        },
        FactorID.CORRUPTION: {
            "display_name": "Corruption Type",
            "description": (
                "Perturbation family (gaussian_noise, blur, brightness, occlusion)"
            ),
        },
        FactorID.CORRUPTION_SEVERITY: {
            "display_name": "Corruption Severity",
            "description": "Perturbation intensity level (1 to 5)",
        },
        FactorID.SSL_TEMPERATURE: {
            "display_name": "SSL Temperature",
            "description": "Contrastive NT-Xent temperature parameter",
        },
        FactorID.MASK_RATIO: {
            "display_name": "Mask Ratio",
            "description": "Reconstruction patch masking ratio (0.1 to 0.9)",
        },
        FactorID.TEMPORAL_AGGREGATOR: {
            "display_name": "Temporal Aggregator",
            "description": (
                "Temporal sequence pooling (mean, max, learned, simple_rnn)"
            ),
        },
        FactorID.MULTIMODAL_TEMPERATURE: {
            "display_name": "Multimodal Temperature",
            "description": ("Vision-language dual-contrastive scaling temperature"),
        },
        FactorID.CALIBRATION_MODE: {
            "display_name": "Calibration Mode",
            "description": "Uncalibrated vs temperature-scaled evaluation",
        },
        FactorID.OOD_SCORE: {
            "display_name": "OOD Scoring Method",
            "description": (
                "OOD metric (MSP, entropy, centroid_distance, knn_distance, energy)"
            ),
        },
    }

    @classmethod
    def get_factor_info(cls, factor_id: FactorID | str) -> dict[str, Any]:
        """Get descriptive metadata for an experimental factor."""
        key = FactorID(factor_id) if isinstance(factor_id, str) else factor_id
        if key not in cls._CANONICAL_FACTORS:
            raise ValidationError(f"Unknown factor ID: {factor_id}")
        return cls._CANONICAL_FACTORS[key]

    @classmethod
    def list_factors(cls) -> list[FactorID]:
        """List all registered canonical factor IDs."""
        return list(cls._CANONICAL_FACTORS.keys())


class MetricRegistry:
    """Canonical registry of benchmark metrics with validation and safeguards."""

    def __init__(self) -> None:
        self._metrics: dict[str, MetricDefinition] = {}
        self._register_canonical_defaults()

    def register(self, metric: MetricDefinition) -> None:
        """Register a new metric definition, ensuring no conflicting redefinition."""
        if metric.metric_id in self._metrics:
            existing = self._metrics[metric.metric_id]
            if existing != metric:
                raise ValidationError(
                    f"Conflicting metric registration for '{metric.metric_id}'."
                )
            return
        self._metrics[metric.metric_id] = metric

    def get(self, metric_id: str) -> MetricDefinition:
        """Retrieve metric definition by ID."""
        if metric_id not in self._metrics:
            raise ValidationError(f"Metric '{metric_id}' is not registered.")
        return self._metrics[metric_id]

    def has(self, metric_id: str) -> bool:
        """Check if metric is registered."""
        return metric_id in self._metrics

    def list_all(self) -> list[MetricDefinition]:
        """List all registered metric definitions."""
        return list(self._metrics.values())

    def filter_by_category(self, category: MetricCategory) -> list[MetricDefinition]:
        """Filter registered metrics by domain category."""
        return [m for m in self._metrics.values() if m.category == category]

    def _register_canonical_defaults(self) -> None:
        """Register the comprehensive set of canonical PRISM research metrics."""
        defaults = [
            MetricDefinition(
                metric_id="accuracy",
                display_name="In-Distribution Accuracy",
                category=MetricCategory.PERFORMANCE,
                unit="%",
                direction=MetricDirection.HIGHER_IS_BETTER,
                bounded_range=[0.0, 1.0],
                description=(
                    "Standard test accuracy on held-out in-distribution samples."
                ),
                methodological_notes="Primary supervised classification performance.",
            ),
            MetricDefinition(
                metric_id="loss",
                display_name="Evaluation Loss",
                category=MetricCategory.PERFORMANCE,
                unit="nats",
                direction=MetricDirection.LOWER_IS_BETTER,
                bounded_range=[0.0, 100.0],
                description="Cross-entropy evaluation loss.",
                methodological_notes=(
                    "Lower cross-entropy indicates better optimization."
                ),
            ),
            MetricDefinition(
                metric_id="parameter_count",
                display_name="Trainable Parameter Count",
                category=MetricCategory.EFFICIENCY,
                unit="params",
                direction=MetricDirection.LOWER_IS_BETTER,
                description="Total number of trainable architecture parameters.",
                methodological_notes="Model capacity descriptor.",
            ),
            MetricDefinition(
                metric_id="linear_probe_accuracy",
                display_name="Linear Probe Accuracy",
                category=MetricCategory.TRANSFER,
                unit="%",
                direction=MetricDirection.HIGHER_IS_BETTER,
                bounded_range=[0.0, 1.0],
                description=(
                    "Classification accuracy of a linear probe on frozen "
                    "representations."
                ),
                methodological_notes=(
                    "Standard transferability and linear separability metric."
                ),
            ),
            MetricDefinition(
                metric_id="representation_drift",
                display_name="Representation Displacement Drift",
                category=MetricCategory.ROBUSTNESS,
                unit="distance",
                direction=MetricDirection.NEUTRAL,
                bounded_range=[0.0, 100.0],
                description=(
                    "Euclidean distance between clean and perturbed representation "
                    "vectors."
                ),
                methodological_notes=(
                    "Descriptive geometric displacement under perturbation; lower is "
                    "not automatically better without downstream task context."
                ),
            ),
            MetricDefinition(
                metric_id="neighbor_consistency",
                display_name="k-NN Neighborhood Consistency",
                category=MetricCategory.GEOMETRY,
                unit="%",
                direction=MetricDirection.HIGHER_IS_BETTER,
                bounded_range=[0.0, 1.0],
                description=(
                    "Fraction of k-nearest neighbors sharing identical class label."
                ),
                methodological_notes="Local geometric purity of representation space.",
            ),
            MetricDefinition(
                metric_id="centroid_separation",
                display_name="Inter-Class Centroid Separation",
                category=MetricCategory.GEOMETRY,
                unit="distance",
                direction=MetricDirection.HIGHER_IS_BETTER,
                bounded_range=[0.0, 100.0],
                description=(
                    "Mean pairwise Euclidean distance between class centroids."
                ),
                methodological_notes="Global manifold separability.",
            ),
            MetricDefinition(
                metric_id="intra_class_compactness",
                display_name="Intra-Class Compactness",
                category=MetricCategory.GEOMETRY,
                unit="distance",
                direction=MetricDirection.LOWER_IS_BETTER,
                bounded_range=[0.0, 100.0],
                description=(
                    "Mean distance of samples to their respective class centroid."
                ),
                methodological_notes="Cluster compactness within classes.",
            ),
            MetricDefinition(
                metric_id="robustness_accuracy_drop",
                display_name="Corruption Accuracy Drop",
                category=MetricCategory.ROBUSTNESS,
                unit="%",
                direction=MetricDirection.LOWER_IS_BETTER,
                bounded_range=[0.0, 1.0],
                description="Degradation in accuracy from clean to corrupted inputs.",
                methodological_notes=(
                    "Evaluated under standard Phase 15 perturbations."
                ),
            ),
            MetricDefinition(
                metric_id="attribution_agreement",
                display_name="Attribution Method Agreement",
                category=MetricCategory.EXPLAINABILITY,
                unit="IoU",
                direction=MetricDirection.HIGHER_IS_BETTER,
                bounded_range=[0.0, 1.0],
                description=(
                    "Mean IoU overlap across different visual attribution methods."
                ),
                methodological_notes=(
                    "Agreement across visual attribution methods; not a guarantee of "
                    "true causal grounding."
                ),
            ),
            MetricDefinition(
                metric_id="transfer_gain",
                display_name="Transfer Learning Gain",
                category=MetricCategory.TRANSFER,
                unit="%",
                direction=MetricDirection.HIGHER_IS_BETTER,
                bounded_range=[-1.0, 1.0],
                description=(
                    "Accuracy improvement of transferred representation over "
                    "random initialization."
                ),
                methodological_notes=(
                    "Positive gain indicates beneficial pretraining transfer."
                ),
            ),
            MetricDefinition(
                metric_id="reconstruction_mse",
                display_name="Masked Reconstruction MSE",
                category=MetricCategory.PERFORMANCE,
                unit="MSE",
                direction=MetricDirection.LOWER_IS_BETTER,
                bounded_range=[0.0, 10.0],
                description="Mean squared error on masked image patches.",
                methodological_notes=(
                    "Evaluated during generative / reconstruction pretraining."
                ),
            ),
            MetricDefinition(
                metric_id="contrastive_loss",
                display_name="Contrastive NT-Xent Loss",
                category=MetricCategory.PERFORMANCE,
                unit="loss",
                direction=MetricDirection.LOWER_IS_BETTER,
                bounded_range=[0.0, 50.0],
                description="Normalized Temperature-scaled Cross-Entropy loss.",
                methodological_notes=(
                    "Evaluated during SimCLR self-supervised pretraining."
                ),
            ),
            MetricDefinition(
                metric_id="detection_mean_iou",
                display_name="Spatial Detection Mean IoU",
                category=MetricCategory.SPATIAL,
                unit="IoU",
                direction=MetricDirection.HIGHER_IS_BETTER,
                bounded_range=[0.0, 1.0],
                description=("Mean Intersection over Union on object bounding boxes."),
                methodological_notes=(
                    "Detection mean IoU from lightweight synthetic spatial probe; "
                    "not COCO mAP."
                ),
            ),
            MetricDefinition(
                metric_id="segmentation_miou",
                display_name="Spatial Segmentation mIoU",
                category=MetricCategory.SPATIAL,
                unit="mIoU",
                direction=MetricDirection.HIGHER_IS_BETTER,
                bounded_range=[0.0, 1.0],
                description="Mean Intersection over Union on 2D segmentation masks.",
                methodological_notes=(
                    "Segmentation mean IoU on 2D synthetic spatial shapes."
                ),
            ),
            MetricDefinition(
                metric_id="pixel_accuracy",
                display_name="Pixel Classification Accuracy",
                category=MetricCategory.SPATIAL,
                unit="%",
                direction=MetricDirection.HIGHER_IS_BETTER,
                bounded_range=[0.0, 1.0],
                description="Fraction of correctly classified segmentation pixels.",
                methodological_notes="Evaluated on 2D spatial segmentation maps.",
            ),
            MetricDefinition(
                metric_id="video_accuracy",
                display_name="Temporal Video Accuracy",
                category=MetricCategory.TEMPORAL,
                unit="%",
                direction=MetricDirection.HIGHER_IS_BETTER,
                bounded_range=[0.0, 1.0],
                description="Sequence-level video classification accuracy.",
                methodological_notes=(
                    "Temporal accuracy from deterministic synthetic motion dataset."
                ),
            ),
            MetricDefinition(
                metric_id="temporal_consistency",
                display_name="Temporal Trajectory Consistency",
                category=MetricCategory.TEMPORAL,
                unit="cosine",
                direction=MetricDirection.HIGHER_IS_BETTER,
                bounded_range=[-1.0, 1.0],
                description=(
                    "Mean cosine similarity between representations of adjacent frames."
                ),
                methodological_notes=(
                    "Measures frame-to-frame feature trajectory smoothness."
                ),
            ),
            MetricDefinition(
                metric_id="retrieval_r1",
                display_name="Cross-Modal Retrieval R@1",
                category=MetricCategory.MULTIMODAL,
                unit="%",
                direction=MetricDirection.HIGHER_IS_BETTER,
                bounded_range=[0.0, 1.0],
                description=(
                    "Recall at rank 1 for bidirectional image-text retrieval."
                ),
                methodological_notes=(
                    "Retrieval R@1 from controlled synthetic image-text candidate pool."
                ),
            ),
            MetricDefinition(
                metric_id="retrieval_r5",
                display_name="Cross-Modal Retrieval R@5",
                category=MetricCategory.MULTIMODAL,
                unit="%",
                direction=MetricDirection.HIGHER_IS_BETTER,
                bounded_range=[0.0, 1.0],
                description=(
                    "Recall at rank 5 for bidirectional image-text retrieval."
                ),
                methodological_notes=(
                    "Evaluated across Image-to-Text and Text-to-Image pairs."
                ),
            ),
            MetricDefinition(
                metric_id="zero_shot_accuracy",
                display_name="Zero-Shot Classification Accuracy",
                category=MetricCategory.MULTIMODAL,
                unit="%",
                direction=MetricDirection.HIGHER_IS_BETTER,
                bounded_range=[0.0, 1.0],
                description="Open-vocabulary top-1 zero-shot classification accuracy.",
                methodological_notes=(
                    "Zero-shot accuracy from synthetic prompt matching."
                ),
            ),
            MetricDefinition(
                metric_id="ece",
                display_name="Expected Calibration Error (ECE)",
                category=MetricCategory.CALIBRATION,
                unit="error",
                direction=MetricDirection.LOWER_IS_BETTER,
                bounded_range=[0.0, 1.0],
                description="Expected Calibration Error across confidence bins.",
                methodological_notes=(
                    "Expected Calibration Error from equal-width reliability bins."
                ),
            ),
            MetricDefinition(
                metric_id="brier",
                display_name="Multiclass Brier Score",
                category=MetricCategory.CALIBRATION,
                unit="score",
                direction=MetricDirection.LOWER_IS_BETTER,
                bounded_range=[0.0, 2.0],
                description=(
                    "Mean squared error between predictive probabilities and targets."
                ),
                methodological_notes="Measures probability calibration and refinement.",
            ),
            MetricDefinition(
                metric_id="nll",
                display_name="Negative Log-Likelihood (NLL)",
                category=MetricCategory.CALIBRATION,
                unit="nats",
                direction=MetricDirection.LOWER_IS_BETTER,
                bounded_range=[0.0, 50.0],
                description=(
                    "Mean evaluation cross-entropy / negative log-likelihood."
                ),
                methodological_notes="Optimization objective for temperature scaling.",
            ),
            MetricDefinition(
                metric_id="ood_auroc",
                display_name="OOD Detection AUROC",
                category=MetricCategory.OOD,
                unit="AUROC",
                direction=MetricDirection.HIGHER_IS_BETTER,
                bounded_range=[0.0, 1.0],
                description=(
                    "Area under ROC curve for binary In-Distribution vs OOD "
                    "discrimination."
                ),
                methodological_notes=("OOD AUROC from controlled synthetic OOD suite."),
            ),
        ]
        for m in defaults:
            self._metrics[m.metric_id] = m


# Global default singleton registry
canonical_metric_registry = MetricRegistry()
