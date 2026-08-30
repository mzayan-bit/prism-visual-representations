"""Unit tests for representation drift analysis between clean and corrupted inputs."""

import pytest

from prism.core.errors import ValidationError
from prism.representations.geometry import (
    RepresentationDataset,
    SpatialVectorizationPolicy,
    VectorNormalizationPolicy,
)
from prism.robustness.drift import (
    RepresentationDriftSummary,
    compute_representation_drift,
)


def _make_rep_dataset(
    vectors: list[list[float]], sample_ids: list[str], labels: list[int]
) -> RepresentationDataset:
    return RepresentationDataset.from_raw_representations(
        raw_embeddings=vectors,
        sample_ids=sample_ids,
        labels=labels,
        experiment_id="exp_test",
        model_id="model_test",
        layer_name="final",
        spatial_policy=SpatialVectorizationPolicy.GLOBAL_AVERAGE_POOL,
        norm_policy=VectorNormalizationPolicy.NONE,
    )


def test_representation_drift_calculation() -> None:
    clean_vecs = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 1.0, 0.0],
    ]
    corrupted_vecs = [
        [1.1, 0.1, 0.0],  # Minor drift
        [0.0, 0.8, 0.2],  # Minor drift
        [0.5, 0.5, 0.0],  # Substantial drift
        [0.0, 0.0, 0.0],  # Complete collapse
    ]
    sample_ids = ["s0", "s1", "s2", "s3"]
    labels = [0, 1, 2, 0]

    clean_ds = _make_rep_dataset(clean_vecs, sample_ids, labels)
    corr_ds = _make_rep_dataset(corrupted_vecs, sample_ids, labels)

    clean_preds = [0, 1, 2, 0]
    corr_preds = [0, 1, 1, 2]  # s2 and s3 flipped
    clean_losses = [0.1, 0.2, 0.15, 0.05]
    corr_losses = [0.12, 0.25, 1.8, 3.2]

    summary, sample_records = compute_representation_drift(
        clean_dataset=clean_ds,
        corrupted_dataset=corr_ds,
        clean_predictions=clean_preds,
        corrupted_predictions=corr_preds,
        clean_losses=clean_losses,
        corrupted_losses=corr_losses,
    )

    assert summary.num_samples == 4
    assert len(sample_records) == 4
    assert summary.mean_euclidean_drift > 0.0
    assert 0.0 <= summary.mean_cosine_similarity <= 1.0

    # Test sample outcomes
    assert not sample_records[0].prediction_changed
    assert not sample_records[1].prediction_changed
    assert sample_records[2].prediction_changed
    assert sample_records[3].prediction_changed

    # Test drift partitioning
    assert "unchanged" in summary.drift_by_prediction_outcome
    assert "changed" in summary.drift_by_prediction_outcome


def test_representation_drift_mismatches() -> None:
    ds1 = _make_rep_dataset([[1.0, 0.0]], ["s0"], [0])
    ds2 = _make_rep_dataset([[1.0, 0.0], [0.0, 1.0]], ["s0", "s1"], [0, 1])

    with pytest.raises(ValidationError):
        compute_representation_drift(
            clean_dataset=ds1,
            corrupted_dataset=ds2,
            clean_predictions=[0],
            corrupted_predictions=[0, 1],
            clean_losses=[0.1],
            corrupted_losses=[0.1, 0.2],
        )


def test_summary_serialization() -> None:
    summary = RepresentationDriftSummary(
        num_samples=2,
        mean_euclidean_drift=0.5,
        median_euclidean_drift=0.5,
        std_euclidean_drift=0.1,
        min_euclidean_drift=0.4,
        max_euclidean_drift=0.6,
        mean_cosine_similarity=0.9,
        mean_cosine_distance=0.1,
        mean_relative_norm_change=0.05,
        per_class_drifts={"0": 0.5},
        drift_by_prediction_outcome={"unchanged": 0.5},
        top_drift_sample_ids=["s1"],
    )
    json_str = summary.to_json()
    loaded = RepresentationDriftSummary.model_validate_json(json_str)
    assert loaded == summary
