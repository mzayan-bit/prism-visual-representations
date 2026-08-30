"""Unit tests for shared PCA basis projection and manifold drift analysis."""

import math

from prism.representations.geometry import (
    DistanceMetric,
    RepresentationDataset,
    SpatialVectorizationPolicy,
    VectorNormalizationPolicy,
)
from prism.robustness.geometry_drift import (
    GeometryDriftReport,
    compute_geometry_drift,
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


def test_shared_pca_and_geometry_drift() -> None:
    # 6 samples across 2 classes
    clean_vecs = [
        [2.0, 0.0, 0.0, 0.0],
        [2.1, 0.1, 0.0, 0.0],
        [1.9, -0.1, 0.0, 0.0],
        [-2.0, 0.0, 0.0, 0.0],
        [-2.1, 0.1, 0.0, 0.0],
        [-1.9, -0.1, 0.0, 0.0],
    ]
    corrupted_vecs = [
        [1.5, 0.5, 0.2, 0.1],
        [1.6, 0.6, 0.2, 0.1],
        [1.4, 0.4, 0.2, 0.1],
        [-1.5, 0.5, 0.2, 0.1],
        [-1.6, 0.6, 0.2, 0.1],
        [-1.4, 0.4, 0.2, 0.1],
    ]
    sample_ids = [f"s{i}" for i in range(6)]
    labels = [0, 0, 0, 1, 1, 1]

    clean_ds = _make_rep_dataset(clean_vecs, sample_ids, labels)
    corr_ds = _make_rep_dataset(corrupted_vecs, sample_ids, labels)

    report = compute_geometry_drift(
        clean_dataset=clean_ds,
        corrupted_dataset=corr_ds,
        k=2,
        n_pca_components=2,
        metric=DistanceMetric.EUCLIDEAN,
    )

    assert isinstance(report, GeometryDriftReport)
    assert report.shared_pca.num_samples == 6
    assert len(report.shared_pca.clean_coordinates) == 6
    assert len(report.shared_pca.corrupted_coordinates) == 6
    assert len(report.shared_pca.displacement_vectors) == 6

    # Verify displacement vector consistency in projected space
    for i in range(6):
        c_pt = report.shared_pca.clean_coordinates[i]
        cr_pt = report.shared_pca.corrupted_coordinates[i]
        disp_vec = report.shared_pca.displacement_vectors[i]
        assert abs(disp_vec[0] - (cr_pt[0] - c_pt[0])) < 1e-6
        assert abs(disp_vec[1] - (cr_pt[1] - c_pt[1])) < 1e-6

        mag = math.sqrt(disp_vec[0] ** 2 + disp_vec[1] ** 2)
        assert abs(mag - report.shared_pca.displacement_magnitudes[i]) < 1e-6

    # Verify class centroid drifts
    assert "0" in report.class_centroid_drifts
    assert "1" in report.class_centroid_drifts
    assert report.class_centroid_drifts["0"].centroid_displacement > 0.0

    # Verify neighborhood drift
    assert 0.0 <= report.neighborhood_drift.mean_neighbor_overlap_ratio <= 1.0


def test_geometry_drift_serialization() -> None:
    clean_vecs = [[1.0, 0.0], [0.0, 1.0]]
    corr_vecs = [[1.1, 0.1], [0.1, 1.1]]
    sample_ids = ["s0", "s1"]
    labels = [0, 1]

    clean_ds = _make_rep_dataset(clean_vecs, sample_ids, labels)
    corr_ds = _make_rep_dataset(corr_vecs, sample_ids, labels)

    report = compute_geometry_drift(clean_ds, corr_ds, k=1, n_pca_components=1)
    json_str = report.to_json()
    loaded = GeometryDriftReport.model_validate_json(json_str)
    assert loaded.mean_centroid_displacement == report.mean_centroid_displacement
