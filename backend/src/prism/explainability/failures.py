"""Explanation and visual attribution failure taxonomy and diagnostics."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from prism.explainability.attribution import AttributionResult
from prism.explainability.comparison import AttributionComparisonReport
from prism.explainability.drift import AttributionDriftSummary


class ExplanationFailureCategory(str, Enum):
    """Observable patterns and potential failure modes in visual attribution."""

    LOW_ATTRIBUTION_SIGNAL = "low_attribution_signal"
    METHOD_DISAGREEMENT = "method_disagreement"
    ATTRIBUTION_SHIFT_UNDER_CORRUPTION = "attribution_shift_under_corruption"
    PREDICTION_FLIP_WITH_STABLE_ATTRIBUTION = "prediction_flip_with_stable_attribution"
    LARGE_ATTRIBUTION_SHIFT_WITH_STABLE_PREDICTION = (
        "large_attribution_shift_with_stable_prediction"
    )
    DIFFUSE_ATTRIBUTION = "diffuse_attribution"
    LOCALIZED_SINGLE_REGION = "localized_single_region"


class ExplanationFailureFlag(BaseModel):
    """Specific flagged attribution failure mode instance with diagnostic metrics."""

    model_config = ConfigDict(extra="forbid")

    category: ExplanationFailureCategory = Field(
        description="Categorized attribution diagnostic / failure pattern"
    )
    severity: str = Field(
        description="Severity level ('low', 'medium', 'high', 'critical')"
    )
    description: str = Field(
        description="Descriptive scientific explanation of observed pattern"
    )
    metrics: dict[str, Any] = Field(
        default_factory=dict,
        description="Quantitative diagnostic metrics triggering this flag",
    )


def flag_explanation_failures(
    attribution_result: AttributionResult | None = None,
    comparison_report: AttributionComparisonReport | None = None,
    drift_summary: AttributionDriftSummary | None = None,
) -> list[ExplanationFailureFlag]:
    """Evaluate and flag potential attribution failure patterns and diagnostics.

    Args:
        attribution_result: Single AttributionResult to check.
        comparison_report: Optional cross-method comparison report.
        drift_summary: Optional clean-vs-corrupted drift summary.

    Returns:
        List of ExplanationFailureFlag objects.
    """
    flags: list[ExplanationFailureFlag] = []

    # 1. Check single attribution result characteristics
    if attribution_result is not None:
        stats = attribution_result.statistics

        # Low Signal: Near-zero mass
        if stats.total_absolute_mass < 1e-7:
            flags.append(
                ExplanationFailureFlag(
                    category=ExplanationFailureCategory.LOW_ATTRIBUTION_SIGNAL,
                    severity="high",
                    description=(
                        "Attribution map contains negligible total absolute mass; "
                        "gradients or activations are essentially zero."
                    ),
                    metrics={"total_absolute_mass": stats.total_absolute_mass},
                )
            )

        # Diffuse Attribution: High entropy, concentration < 0.15
        if stats.concentration_score < 0.15 and stats.total_absolute_mass >= 1e-7:
            flags.append(
                ExplanationFailureFlag(
                    category=ExplanationFailureCategory.DIFFUSE_ATTRIBUTION,
                    severity="low",
                    description=(
                        "Attribution signal is spread uniformly across the field "
                        "with minimal focal concentration."
                    ),
                    metrics={
                        "concentration_score": stats.concentration_score,
                        "spatial_entropy": stats.spatial_entropy,
                    },
                )
            )

        # Highly Localized: concentration > 0.90 or top 10% mass fraction > 0.85
        if stats.top_10_percent_mass_fraction > 0.85:
            flags.append(
                ExplanationFailureFlag(
                    category=ExplanationFailureCategory.LOCALIZED_SINGLE_REGION,
                    severity="low",
                    description=(
                        "Over 85% of total attribution mass is concentrated in "
                        "the top 10% of pixels."
                    ),
                    metrics={
                        "top_10_percent_mass_fraction": (
                            stats.top_10_percent_mass_fraction
                        ),
                        "concentration_score": stats.concentration_score,
                    },
                )
            )

    # 2. Check cross-method agreement
    if (
        comparison_report is not None
        and comparison_report.mean_cross_method_agreement < 0.20
    ):
        flags.append(
            ExplanationFailureFlag(
                category=ExplanationFailureCategory.METHOD_DISAGREEMENT,
                severity="medium",
                description=(
                    "Different attribution methods exhibit severe spatial divergence "
                    "(average pairwise cosine similarity < 0.20)."
                ),
                metrics={
                    "mean_cross_method_agreement": (
                        comparison_report.mean_cross_method_agreement
                    )
                },
            )
        )

    # 3. Check corruption stability and prediction flip relationships
    if drift_summary is not None:
        sim = drift_summary.attribution_cosine_similarity
        pred_preserved = drift_summary.prediction_preserved

        if sim < 0.25:
            flags.append(
                ExplanationFailureFlag(
                    category=ExplanationFailureCategory.ATTRIBUTION_SHIFT_UNDER_CORRUPTION,
                    severity="high",
                    description=(
                        "Attribution map underwent substantial spatial reorganization "
                        "under corruption (cosine similarity < 0.25)."
                    ),
                    metrics={"attribution_cosine_similarity": sim},
                )
            )

        if not pred_preserved and sim > 0.85:
            flags.append(
                ExplanationFailureFlag(
                    category=ExplanationFailureCategory.PREDICTION_FLIP_WITH_STABLE_ATTRIBUTION,
                    severity="high",
                    description=(
                        "Model flipped prediction under corruption, yet attribution "
                        "remained virtually unchanged (similarity > 0.85)."
                    ),
                    metrics={
                        "attribution_cosine_similarity": sim,
                        "clean_prediction": drift_summary.clean_predicted_class,
                        "corrupted_prediction": (
                            drift_summary.corrupted_predicted_class
                        ),
                    },
                )
            )

        if pred_preserved and sim < 0.20:
            flags.append(
                ExplanationFailureFlag(
                    category=ExplanationFailureCategory.LARGE_ATTRIBUTION_SHIFT_WITH_STABLE_PREDICTION,
                    severity="medium",
                    description=(
                        "Model maintained predicted class despite a radical shift "
                        "in spatial attribution map (similarity < 0.20)."
                    ),
                    metrics={
                        "attribution_cosine_similarity": sim,
                        "predicted_class": drift_summary.clean_predicted_class,
                    },
                )
            )

    return flags
