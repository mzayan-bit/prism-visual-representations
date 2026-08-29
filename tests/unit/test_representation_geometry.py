"""Unit tests for representation geometry, centroids, and neighborhood analysis."""

from __future__ import annotations

import math

import pytest

from prism.core.errors import ValidationError
from prism.representations.centroids import (
    compute_centroid_geometry,
)
from prism.representations.geometry import (
    DistanceMetric,
    RepresentationDataset,
    SpatialVectorizationPolicy,
    VectorNormalizationPolicy,
    compute_distance,
    compute_pairwise_distances,
    normalize_vectors,
    vectorize_spatial_features,
)
from prism.representations.neighborhood import (
    compute_neighborhood_geometry,
)
from prism.representations.reports import (
    RepresentationGeometryReport,
    analyze_representation_geometry,
)


class TestDistanceMetrics:
    """Test suite for numerical distance and similarity primitives."""

    def test_euclidean_distance(self) -> None:
        v1 = [0.0, 0.0, 0.0]
        v2 = [3.0, 4.0, 0.0]
        assert compute_distance(
            v1, v2, metric=DistanceMetric.EUCLIDEAN
        ) == pytest.approx(5.0)
        assert compute_distance(
            v1, v2, metric=DistanceMetric.SQUARED_EUCLIDEAN
        ) == pytest.approx(25.0)

    def test_cosine_similarity_and_distance(self) -> None:
        v1 = [1.0, 0.0]
        v2 = [0.0, 2.0]
        # Orthogonal
        assert compute_distance(
            v1, v2, metric=DistanceMetric.COSINE_SIMILARITY
        ) == pytest.approx(0.0)
        assert compute_distance(
            v1, v2, metric=DistanceMetric.COSINE_DISTANCE
        ) == pytest.approx(1.0)

        # Parallel
        v3 = [2.0, 0.0]
        assert compute_distance(
            v1, v3, metric=DistanceMetric.COSINE_SIMILARITY
        ) == pytest.approx(1.0)
        assert compute_distance(
            v1, v3, metric=DistanceMetric.COSINE_DISTANCE
        ) == pytest.approx(0.0)

        # Opposite
        v4 = [-1.0, 0.0]
        assert compute_distance(
            v1, v4, metric=DistanceMetric.COSINE_SIMILARITY
        ) == pytest.approx(-1.0)
        assert compute_distance(
            v1, v4, metric=DistanceMetric.COSINE_DISTANCE
        ) == pytest.approx(2.0)

    def test_zero_vector_cosine_handling(self) -> None:
        v_zero = [0.0, 0.0, 0.0]
        v_norm = [1.0, 2.0, 3.0]
        assert (
            compute_distance(v_zero, v_norm, metric=DistanceMetric.COSINE_SIMILARITY)
            == 0.0
        )
        assert (
            compute_distance(v_zero, v_norm, metric=DistanceMetric.COSINE_DISTANCE)
            == 1.0
        )

    def test_pairwise_distances(self) -> None:
        vectors = [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ]
        mat = compute_pairwise_distances(vectors, metric=DistanceMetric.EUCLIDEAN)
        assert len(mat) == 3
        assert len(mat[0]) == 3
        assert mat[0][0] == 0.0
        assert mat[0][1] == pytest.approx(1.0)
        assert mat[0][2] == pytest.approx(1.0)
        assert mat[1][2] == pytest.approx(math.sqrt(2.0))


class TestVectorizationAndNormalization:
    """Test suite for spatial feature vectorization and vector normalization."""

    def test_vectorize_2d_vectors(self) -> None:
        raw = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        vecs, shape, dim = vectorize_spatial_features(
            raw, policy=SpatialVectorizationPolicy.NONE
        )
        assert len(vecs) == 2
        assert dim == 3
        assert shape == (2, 3)

    def test_vectorize_4d_conv_features_gap(self) -> None:
        # [N=1, C=2, H=2, W=2]
        raw = [
            [
                [[1.0, 2.0], [3.0, 4.0]],  # channel 0 mean = 2.5
                [[10.0, 20.0], [30.0, 40.0]],  # channel 1 mean = 25.0
            ]
        ]
        vecs, shape, dim = vectorize_spatial_features(
            raw, policy=SpatialVectorizationPolicy.GLOBAL_AVERAGE_POOL
        )
        assert shape == (1, 2, 2, 2)
        assert dim == 2
        assert vecs[0][0] == pytest.approx(2.5)
        assert vecs[0][1] == pytest.approx(25.0)

    def test_vectorize_4d_conv_features_flatten(self) -> None:
        raw = [
            [
                [[1.0, 2.0], [3.0, 4.0]],
                [[5.0, 6.0], [7.0, 8.0]],
            ]
        ]
        vecs, shape, dim = vectorize_spatial_features(
            raw, policy=SpatialVectorizationPolicy.FLATTEN
        )
        assert shape == (1, 2, 2, 2)
        assert dim == 8
        assert vecs[0] == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]

    def test_vectorize_3d_token_features(self) -> None:
        # [N=1, S=2, D=3]
        raw = [
            [
                [1.0, 2.0, 3.0],
                [3.0, 4.0, 5.0],
            ]
        ]
        vecs_gap, _shape_gap, dim_gap = vectorize_spatial_features(
            raw, policy=SpatialVectorizationPolicy.GLOBAL_AVERAGE_POOL
        )
        assert dim_gap == 3
        assert vecs_gap[0] == [2.0, 3.0, 4.0]

        vecs_flat, _, dim_flat = vectorize_spatial_features(
            raw, policy=SpatialVectorizationPolicy.FLATTEN
        )
        assert dim_flat == 6
        assert vecs_flat[0] == [1.0, 2.0, 3.0, 3.0, 4.0, 5.0]

    def test_l2_normalize_vectors(self) -> None:
        vectors = [[3.0, 4.0, 0.0]]
        normed = normalize_vectors(
            vectors, policy=VectorNormalizationPolicy.L2_NORMALIZE
        )
        assert normed[0][0] == pytest.approx(0.6)
        assert normed[0][1] == pytest.approx(0.8)
        assert normed[0][2] == pytest.approx(0.0)

    def test_standardize_vectors(self) -> None:
        vectors = [
            [1.0, 10.0],
            [3.0, 20.0],
            [5.0, 30.0],
        ]
        normed = normalize_vectors(
            vectors, policy=VectorNormalizationPolicy.STANDARDIZE
        )
        # Column 0: mean=3, std=1.63299
        assert sum(row[0] for row in normed) == pytest.approx(0.0, abs=1e-6)
        assert sum(row[1] for row in normed) == pytest.approx(0.0, abs=1e-6)


class TestRepresentationDataset:
    """Test suite for representation dataset schema validation."""

    def test_valid_dataset_construction(self) -> None:
        ds = RepresentationDataset(
            experiment_id="exp-001",
            model_id="cnn-model",
            layer_name="final_hidden",
            sample_ids=["s0", "s1", "s2"],
            labels=[0, 1, 0],
            vectors=[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
            feature_dim=2,
            num_samples=3,
            num_classes=2,
            class_names=["cat", "dog"],
        )
        assert ds.num_samples == 3
        assert ds.feature_dim == 2

    def test_dataset_validation_failures(self) -> None:
        # Duplicate sample_ids
        with pytest.raises(ValidationError):
            RepresentationDataset(
                experiment_id="exp-001",
                model_id="cnn-model",
                layer_name="final_hidden",
                sample_ids=["s0", "s0"],
                labels=[0, 1],
                vectors=[[1.0, 2.0], [3.0, 4.0]],
                feature_dim=2,
                num_samples=2,
                num_classes=2,
            )

        # Dimension mismatch
        with pytest.raises(ValidationError):
            RepresentationDataset(
                experiment_id="exp-001",
                model_id="cnn-model",
                layer_name="final_hidden",
                sample_ids=["s0", "s1"],
                labels=[0, 1],
                vectors=[[1.0, 2.0], [3.0]],
                feature_dim=2,
                num_samples=2,
                num_classes=2,
            )


class TestCentroidsAndNeighborhood:
    """Test suite for class centroids, compactness, separation, and k-NN geometry."""

    @pytest.fixture
    def synthetic_dataset(self) -> RepresentationDataset:
        # Two well-separated clusters in 2D
        # Class 0: around (0, 0)
        # Class 1: around (10, 10)
        sample_ids = ["c0_1", "c0_2", "c0_3", "c1_1", "c1_2", "c1_3"]
        labels: list[int | str] = [0, 0, 0, 1, 1, 1]
        vectors = [
            [-0.1, 0.0],
            [0.1, 0.0],
            [0.0, 0.1],
            [9.9, 10.0],
            [10.1, 10.0],
            [10.0, 10.1],
        ]
        return RepresentationDataset(
            experiment_id="test-exp",
            model_id="test-model",
            layer_name="test-layer",
            sample_ids=sample_ids,
            labels=labels,
            vectors=vectors,
            feature_dim=2,
            num_samples=6,
            num_classes=2,
            class_names=["cluster_A", "cluster_B"],
        )

    def test_centroid_geometry_computation(
        self, synthetic_dataset: RepresentationDataset
    ) -> None:
        report = compute_centroid_geometry(
            synthetic_dataset, metric=DistanceMetric.EUCLIDEAN
        )
        assert len(report.class_centroids) == 2
        assert "0" in report.class_centroids
        assert "1" in report.class_centroids

        c0 = report.class_centroids["0"]
        c1 = report.class_centroids["1"]

        # Centroid vectors
        assert c0.centroid[0] == pytest.approx(0.0, abs=1e-5)
        assert c0.centroid[1] == pytest.approx(1.0 / 30.0, abs=1e-5)
        assert c1.centroid[0] == pytest.approx(10.0, abs=1e-5)

        # Inter-class separation should be approximately sqrt(10^2 + 10^2) ≈ 14.14
        assert report.mean_inter_class_centroid_distance > 13.0
        # Intra-class compactness should be very small (< 0.2)
        assert report.mean_intra_class_distance < 0.2
        # Separation / compactness ratio should be high (> 50)
        assert report.separation_to_compactness_ratio > 50.0

    def test_neighborhood_geometry_and_failures(
        self, synthetic_dataset: RepresentationDataset
    ) -> None:
        report = compute_neighborhood_geometry(
            synthetic_dataset, k=2, metric=DistanceMetric.EUCLIDEAN
        )
        assert report.mean_label_consistency == pytest.approx(1.0)
        assert len(report.candidate_failures) == 0

    def test_cross_class_neighbor_detection(self) -> None:
        # Construct ambiguous dataset where a sample has a cross-class nearest neighbor
        sample_ids = ["s0", "s1", "s2", "s3"]
        labels: list[int | str] = [0, 0, 1, 1]
        vectors = [
            [0.0, 0.0],
            [0.1, 0.0],
            [0.2, 0.0],  # s2 (class 1) is closer to s1 (class 0) than to s3 (class 1)
            [10.0, 10.0],
        ]
        ds = RepresentationDataset(
            experiment_id="test-ambig",
            model_id="test-model",
            layer_name="test-layer",
            sample_ids=sample_ids,
            labels=labels,
            vectors=vectors,
            feature_dim=2,
            num_samples=4,
            num_classes=2,
        )
        report = compute_neighborhood_geometry(ds, k=2, metric=DistanceMetric.EUCLIDEAN)
        assert len(report.candidate_failures) > 0
        kinds = [f.failure_kind for f in report.candidate_failures]
        assert "cross_class_neighbor" in kinds

    def test_representation_geometry_report_serialization(
        self, synthetic_dataset: RepresentationDataset
    ) -> None:
        full_report = analyze_representation_geometry(synthetic_dataset, k=2)
        json_str = full_report.to_json()
        assert "centroid_geometry" in json_str
        assert "pca_projection" in json_str

        # Roundtrip
        deserialized = RepresentationGeometryReport.from_json(json_str)
        assert deserialized.experiment_id == full_report.experiment_id
        assert deserialized.num_samples == full_report.num_samples
        assert len(deserialized.pca_projection.coordinates) == 6
