"""Representation drift analysis comparing clean vs corrupted representations."""

from __future__ import annotations

import json
import math
from typing import Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
)

from prism.core.errors import SerializationError, ValidationError
from prism.representations.geometry import (
    DistanceMetric,
    RepresentationDataset,
    compute_distance,
)


class SampleRepresentationDrift(BaseModel):
    """Paired representation and prediction drift record for an individual sample."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    sample_id: str = Field(description="Unique sample identifier")
    label: int | str = Field(description="Ground truth category label")
    clean_prediction: int = Field(description="Model predicted class on clean sample")
    corrupted_prediction: int = Field(
        description="Model predicted class on corrupted sample"
    )
    clean_correct: bool = Field(description="Whether clean prediction was correct")
    corrupted_correct: bool = Field(
        description="Whether corrupted prediction was correct"
    )
    prediction_changed: bool = Field(
        description="Whether prediction changed under corruption"
    )
    clean_loss: float = Field(ge=0.0, description="Cross-entropy loss on clean input")
    corrupted_loss: float = Field(
        ge=0.0, description="Cross-entropy loss on corrupted input"
    )
    euclidean_drift: float = Field(
        ge=0.0, description="Euclidean distance between clean and corrupted vectors"
    )
    cosine_similarity: float = Field(
        ge=-1.0,
        le=1.0,
        description="Cosine similarity between clean and corrupted vectors",
    )
    cosine_distance: float = Field(
        ge=0.0, le=2.0, description="Cosine distance (1 - cosine_similarity)"
    )
    clean_norm: float = Field(
        ge=0.0, description="L2 norm of clean representation vector"
    )
    corrupted_norm: float = Field(
        ge=0.0, description="L2 norm of corrupted representation vector"
    )
    relative_norm_change: float = Field(
        description="Relative change in vector norm: (|x'| - |x|) / (|x| + eps)"
    )

    def to_dict(self) -> dict[str, Any]:
        """Convert record to dictionary."""
        return self.model_dump(mode="json")


class RepresentationDriftSummary(BaseModel):
    """Aggregated representation drift metrics across an evaluation dataset."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    num_samples: int = Field(ge=0, description="Number of evaluated sample pairs")
    mean_euclidean_drift: float = Field(
        ge=0.0, description="Mean Euclidean representation drift"
    )
    median_euclidean_drift: float = Field(
        ge=0.0, description="Median Euclidean representation drift"
    )
    std_euclidean_drift: float = Field(
        ge=0.0, description="Standard deviation of Euclidean drift"
    )
    min_euclidean_drift: float = Field(
        ge=0.0, description="Minimum Euclidean drift across samples"
    )
    max_euclidean_drift: float = Field(
        ge=0.0, description="Maximum Euclidean drift across samples"
    )
    mean_cosine_similarity: float = Field(
        ge=-1.0, le=1.0, description="Mean cosine similarity between paired vectors"
    )
    mean_cosine_distance: float = Field(
        ge=0.0, le=2.0, description="Mean cosine distance (1 - cosine_sim)"
    )
    mean_relative_norm_change: float = Field(
        description="Mean relative representation norm change"
    )
    per_class_drifts: dict[str, float] = Field(
        default_factory=dict,
        description="Mean Euclidean drift grouped by ground-truth class",
    )
    drift_by_prediction_outcome: dict[str, float] = Field(
        default_factory=dict,
        description="Mean drift partitioned by prediction outcome",
    )
    top_drift_sample_ids: list[str] = Field(
        default_factory=list,
        description="IDs of top samples exhibiting highest representation drift",
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
    def from_dict(cls, data: dict[str, Any]) -> RepresentationDriftSummary:
        """Create summary from dictionary."""
        try:
            return cls.model_validate(data)
        except Exception as exc:
            raise SerializationError(
                f"Failed to deserialize RepresentationDriftSummary: {exc}"
            ) from exc


def compute_representation_drift(
    clean_dataset: RepresentationDataset,
    corrupted_dataset: RepresentationDataset,
    clean_predictions: list[int],
    corrupted_predictions: list[int],
    clean_losses: list[float],
    corrupted_losses: list[float],
    top_k_drift_samples: int = 5,
) -> tuple[RepresentationDriftSummary, list[SampleRepresentationDrift]]:
    """Compute paired representation drift between clean and corrupted inputs.

    Parameters
    ----------
    clean_dataset : RepresentationDataset
        Dataset containing representations from clean inputs.
    corrupted_dataset : RepresentationDataset
        Dataset containing representations from corrupted inputs.
    clean_predictions : list[int]
        Model class predictions on clean inputs.
    corrupted_predictions : list[int]
        Model class predictions on corrupted inputs.
    clean_losses : list[float]
        Loss per sample on clean inputs.
    corrupted_losses : list[float]
        Loss per sample on corrupted inputs.
    top_k_drift_samples : int
        Number of top-drift sample IDs to record in summary.

    Returns
    -------
    tuple[RepresentationDriftSummary, list[SampleRepresentationDrift]]
        Aggregate summary and detailed list of per-sample drift records.
    """
    n = clean_dataset.num_samples
    if corrupted_dataset.num_samples != n:
        raise ValidationError(
            f"Sample count mismatch: {n} clean vs {corrupted_dataset.num_samples}."
        )
    if (
        len(clean_predictions) != n
        or len(corrupted_predictions) != n
        or len(clean_losses) != n
        or len(corrupted_losses) != n
    ):
        raise ValidationError(
            f"Prediction/loss array lengths must match sample count {n}."
        )

    # Sample ID index lookup for clean dataset
    clean_id_to_idx = {sid: idx for idx, sid in enumerate(clean_dataset.sample_ids)}

    sample_records: list[SampleRepresentationDrift] = []
    euclidean_drifts: list[float] = []
    cosine_sims: list[float] = []
    cosine_dists: list[float] = []
    norm_changes: list[float] = []

    class_drifts: dict[str, list[float]] = {}
    unchanged_drifts: list[float] = []
    changed_drifts: list[float] = []
    correct_to_wrong_drifts: list[float] = []

    for c_idx, sid in enumerate(corrupted_dataset.sample_ids):
        if sid not in clean_id_to_idx:
            raise ValidationError(
                f"Corrupted sample ID '{sid}' not found in clean dataset."
            )
        orig_idx = clean_id_to_idx[sid]

        v_clean = clean_dataset.vectors[orig_idx]
        v_corrupt = corrupted_dataset.vectors[c_idx]

        e_dist = compute_distance(v_clean, v_corrupt, metric=DistanceMetric.EUCLIDEAN)
        c_sim = compute_distance(
            v_clean, v_corrupt, metric=DistanceMetric.COSINE_SIMILARITY
        )
        c_dist = compute_distance(
            v_clean, v_corrupt, metric=DistanceMetric.COSINE_DISTANCE
        )

        clean_norm = math.sqrt(sum(x * x for x in v_clean))
        corrupt_norm = math.sqrt(sum(x * x for x in v_corrupt))
        rel_norm_change = (corrupt_norm - clean_norm) / (clean_norm + 1e-12)

        target = clean_dataset.labels[orig_idx]
        # Match target type
        target_int = (
            int(target)
            if isinstance(target, (int, str)) and str(target).isdigit()
            else -1
        )

        c_pred = clean_predictions[orig_idx]
        cr_pred = corrupted_predictions[c_idx]

        clean_corr = c_pred == target_int
        corrupt_corr = cr_pred == target_int
        pred_changed = c_pred != cr_pred

        record = SampleRepresentationDrift(
            sample_id=sid,
            label=target,
            clean_prediction=c_pred,
            corrupted_prediction=cr_pred,
            clean_correct=clean_corr,
            corrupted_correct=corrupt_corr,
            prediction_changed=pred_changed,
            clean_loss=clean_losses[orig_idx],
            corrupted_loss=corrupted_losses[c_idx],
            euclidean_drift=e_dist,
            cosine_similarity=c_sim,
            cosine_distance=c_dist,
            clean_norm=clean_norm,
            corrupted_norm=corrupt_norm,
            relative_norm_change=rel_norm_change,
        )
        sample_records.append(record)

        euclidean_drifts.append(e_dist)
        cosine_sims.append(c_sim)
        cosine_dists.append(c_dist)
        norm_changes.append(rel_norm_change)

        cls_key = str(target)
        if cls_key not in class_drifts:
            class_drifts[cls_key] = []
        class_drifts[cls_key].append(e_dist)

        if pred_changed:
            changed_drifts.append(e_dist)
        else:
            unchanged_drifts.append(e_dist)

        if clean_corr and not corrupt_corr:
            correct_to_wrong_drifts.append(e_dist)

    if n == 0:
        summary = RepresentationDriftSummary(
            num_samples=0,
            mean_euclidean_drift=0.0,
            median_euclidean_drift=0.0,
            std_euclidean_drift=0.0,
            min_euclidean_drift=0.0,
            max_euclidean_drift=0.0,
            mean_cosine_similarity=1.0,
            mean_cosine_distance=0.0,
            mean_relative_norm_change=0.0,
        )
        return summary, sample_records

    sorted_drifts = sorted(euclidean_drifts)
    mean_drift = sum(sorted_drifts) / float(n)
    median_drift = (
        sorted_drifts[n // 2]
        if n % 2 != 0
        else (sorted_drifts[n // 2 - 1] + sorted_drifts[n // 2]) / 2.0
    )
    variance_drift = sum((d - mean_drift) ** 2 for d in sorted_drifts) / float(n)
    std_drift = math.sqrt(max(0.0, variance_drift))

    per_class_summary = {
        cls_k: sum(drifts) / float(len(drifts))
        for cls_k, drifts in sorted(class_drifts.items())
    }

    outcome_summary: dict[str, float] = {}
    if unchanged_drifts:
        outcome_summary["unchanged"] = sum(unchanged_drifts) / float(
            len(unchanged_drifts)
        )
    if changed_drifts:
        outcome_summary["changed"] = sum(changed_drifts) / float(len(changed_drifts))
    if correct_to_wrong_drifts:
        outcome_summary["clean_correct_to_wrong"] = sum(
            correct_to_wrong_drifts
        ) / float(len(correct_to_wrong_drifts))

    # Top drift sample IDs
    top_indices = sorted(range(n), key=lambda idx: euclidean_drifts[idx], reverse=True)[
        :top_k_drift_samples
    ]
    top_ids = [sample_records[idx].sample_id for idx in top_indices]

    summary = RepresentationDriftSummary(
        num_samples=n,
        mean_euclidean_drift=mean_drift,
        median_euclidean_drift=median_drift,
        std_euclidean_drift=std_drift,
        min_euclidean_drift=sorted_drifts[0],
        max_euclidean_drift=sorted_drifts[-1],
        mean_cosine_similarity=sum(cosine_sims) / float(n),
        mean_cosine_distance=sum(cosine_dists) / float(n),
        mean_relative_norm_change=sum(norm_changes) / float(n),
        per_class_drifts=per_class_summary,
        drift_by_prediction_outcome=outcome_summary,
        top_drift_sample_ids=top_ids,
    )
    return summary, sample_records
