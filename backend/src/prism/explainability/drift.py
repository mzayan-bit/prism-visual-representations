"""Attribution stability, spatial drift, and corruption sensitivity analysis."""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from prism.core.errors import SerializationError, ValidationError
from prism.explainability.attribution import (
    AttributionMethod,
    AttributionResult,
)
from prism.explainability.comparison import (
    compute_center_of_mass_displacement,
    compute_map_cosine_similarity,
    compute_top_percent_overlap,
)


class AttributionDriftSummary(BaseModel):
    """Spatial stability and attribution drift under controlled input corruption."""

    model_config = ConfigDict(extra="forbid")

    sample_id: str = Field(description="Sample identifier")
    model_id: str = Field(description="Model identifier")
    architecture: str = Field(description="Model architecture family")
    method: AttributionMethod = Field(description="Attribution method evaluated")
    corruption_type: str = Field(
        description="Type of applied distribution shift / corruption"
    )
    corruption_severity: float = Field(
        description="Severity level parameter of corruption"
    )
    clean_target_class: int = Field(description="Target class in clean condition")
    corrupted_target_class: int = Field(
        description="Target class in corrupted condition"
    )
    clean_predicted_class: int = Field(description="Predicted class on clean input")
    corrupted_predicted_class: int = Field(
        description="Predicted class on corrupted input"
    )
    prediction_preserved: bool = Field(
        description="Whether top-1 prediction is identical between clean and corrupted"
    )
    clean_score: float | None = Field(
        default=None, description="Clean target logit score"
    )
    corrupted_score: float | None = Field(
        default=None, description="Corrupted target logit score"
    )
    attribution_cosine_similarity: float = Field(
        description="Cosine similarity between clean and corrupted attribution maps"
    )
    top_10_percent_mask_overlap: float = Field(
        description="Top-10% binary mask overlap (Jaccard) under corruption"
    )
    top_25_percent_mask_overlap: float = Field(
        description="Top-25% binary mask overlap (Jaccard) under corruption"
    )
    center_of_mass_displacement: float = Field(
        description="Euclidean shift in spatial center-of-mass coordinates"
    )
    concentration_delta: float = Field(
        description="Change in spatial concentration score (C_corr - C_clean)"
    )
    representation_drift_distance: float | None = Field(
        default=None,
        description="Paired latent representation drift from Phase 15 if available",
    )
    warnings: list[str] = Field(
        default_factory=list,
        description="Descriptive warnings on instability, prediction flips, or drift",
    )

    def to_dict(self) -> dict[str, Any]:
        """Convert summary to dictionary."""
        return self.model_dump(mode="json")

    def to_json(self, indent: int | None = None) -> str:
        """Convert summary to JSON string."""
        return json.dumps(
            self.to_dict(),
            indent=indent,
            sort_keys=True,
            ensure_ascii=True,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AttributionDriftSummary:
        """Construct summary from dictionary."""
        try:
            return cls.model_validate(data)
        except Exception as exc:
            raise SerializationError(
                f"Failed to deserialize AttributionDriftSummary: {exc}"
            ) from exc


def compute_attribution_drift(
    clean_result: AttributionResult,
    corrupted_result: AttributionResult,
    corruption_type: str = "generic_corruption",
    corruption_severity: float = 1.0,
    representation_drift_distance: float | None = None,
) -> AttributionDriftSummary:
    """Compute attribution drift metrics comparing clean vs corrupted inputs.

    Args:
        clean_result: AttributionResult on clean source image.
        corrupted_result: AttributionResult on corrupted counterpart.
        corruption_type: Name/type of corruption applied.
        corruption_severity: Severity parameter value.
        representation_drift_distance: Optional representation cosine distance.

    Returns:
        AttributionDriftSummary model.
    """
    if clean_result.method != corrupted_result.method:
        raise ValidationError(
            f"Cannot compute drift across different methods: "
            f"'{clean_result.method}' vs '{corrupted_result.method}'."
        )

    cos_sim = compute_map_cosine_similarity(
        clean_result.normalized_attribution_map,
        corrupted_result.normalized_attribution_map,
    )
    top_10_ovlp = compute_top_percent_overlap(
        clean_result.normalized_attribution_map,
        corrupted_result.normalized_attribution_map,
        top_percent=0.10,
    )
    top_25_ovlp = compute_top_percent_overlap(
        clean_result.normalized_attribution_map,
        corrupted_result.normalized_attribution_map,
        top_percent=0.25,
    )
    com_disp = compute_center_of_mass_displacement(
        clean_result.statistics,
        corrupted_result.statistics,
    )
    conc_delta = (
        corrupted_result.statistics.concentration_score
        - clean_result.statistics.concentration_score
    )

    pred_preserved = clean_result.predicted_class == corrupted_result.predicted_class

    warnings: list[str] = []
    if not pred_preserved:
        warnings.append("prediction_flip_under_corruption")
    if cos_sim < 0.30:
        warnings.append("severe_attribution_shift")
    if pred_preserved and cos_sim < 0.20:
        warnings.append("large_attribution_shift_with_stable_prediction")
    if not pred_preserved and cos_sim > 0.85:
        warnings.append("prediction_flip_with_stable_attribution")

    return AttributionDriftSummary(
        sample_id=clean_result.sample_id,
        model_id=clean_result.model_id,
        architecture=clean_result.architecture,
        method=clean_result.method,
        corruption_type=corruption_type,
        corruption_severity=corruption_severity,
        clean_target_class=clean_result.target_class,
        corrupted_target_class=corrupted_result.target_class,
        clean_predicted_class=clean_result.predicted_class,
        corrupted_predicted_class=corrupted_result.predicted_class,
        prediction_preserved=pred_preserved,
        clean_score=clean_result.target_score,
        corrupted_score=corrupted_result.target_score,
        attribution_cosine_similarity=cos_sim,
        top_10_percent_mask_overlap=top_10_ovlp,
        top_25_percent_mask_overlap=top_25_ovlp,
        center_of_mass_displacement=com_disp,
        concentration_delta=conc_delta,
        representation_drift_distance=representation_drift_distance,
        warnings=warnings,
    )
