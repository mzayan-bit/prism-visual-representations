"""Unit tests for pure Python Principal Component Analysis (PCA)."""

from __future__ import annotations

import pytest

from prism.representations.geometry import RepresentationDataset
from prism.representations.pca import (
    PrincipalComponentAnalysis,
    ProjectionResult,
    _jacobi_eigenvalues_symmetric,
    compute_pca_projection,
)


class TestJacobiEigenvalueAlgorithm:
    """Test suite for Jacobi rotation matrix eigendecomposition."""

    def test_diagonal_matrix(self) -> None:
        diag = [
            [5.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 8.0],
        ]
        evals, _evecs = _jacobi_eigenvalues_symmetric(diag)
        assert evals[0] == pytest.approx(8.0)
        assert evals[1] == pytest.approx(5.0)
        assert evals[2] == pytest.approx(2.0)

    def test_2x2_symmetric_matrix(self) -> None:
        # Matrix [[2, 1], [1, 2]], eigenvalues are 3 and 1
        mat = [
            [2.0, 1.0],
            [1.0, 2.0],
        ]
        evals, evecs = _jacobi_eigenvalues_symmetric(mat)
        assert evals[0] == pytest.approx(3.0)
        assert evals[1] == pytest.approx(1.0)
        # Eigenvectors should be orthogonal
        dot = evecs[0][0] * evecs[1][0] + evecs[0][1] * evecs[1][1]
        assert dot == pytest.approx(0.0, abs=1e-10)


class TestPrincipalComponentAnalysis:
    """Test suite for PCA fit and transform engine."""

    def test_pca_linearly_correlated_data(self) -> None:
        # Data strongly correlated along y = 2x
        x = [
            [1.0, 2.0],
            [2.0, 4.0],
            [3.0, 6.0],
            [4.0, 8.0],
            [5.0, 10.0],
        ]
        pca = PrincipalComponentAnalysis(n_components=2)
        projected = pca.fit_transform(x)

        assert len(projected) == 5
        assert len(projected[0]) == 2

        # First component should explain > 99.9% of variance
        assert pca.explained_variance_ratio is not None
        assert pca.explained_variance_ratio[0] > 0.999
        assert pca.cumulative_explained_variance is not None
        assert pca.cumulative_explained_variance[1] == pytest.approx(1.0)

    def test_pca_deterministic_sign_orientation(self) -> None:
        # Fit identical dataset multiple times; orientation must be identical
        x = [
            [0.1, 0.5, 0.9],
            [0.8, 0.2, 0.3],
            [0.4, 0.7, 0.1],
            [0.9, 0.9, 0.2],
        ]
        pca1 = PrincipalComponentAnalysis(n_components=2).fit(x)
        pca2 = PrincipalComponentAnalysis(n_components=2).fit(x)

        assert pca1.components == pca2.components

        # The maximum absolute loading in each principal component must be non-negative
        for comp in pca1.components or []:
            max_idx = max(range(len(comp)), key=lambda i: abs(comp[i]))
            assert comp[max_idx] >= 0.0

    def test_pca_dual_gram_matrix_wide_data(self) -> None:
        # Wide data where N < D (e.g. N=3 samples, D=10 features)
        x = [
            [1.0 if j == 0 else 0.0 for j in range(10)],
            [1.0 if j == 1 else 0.0 for j in range(10)],
            [1.0 if j == 2 else 0.0 for j in range(10)],
        ]
        pca = PrincipalComponentAnalysis(n_components=2)
        coords = pca.fit_transform(x)

        assert len(coords) == 3
        assert len(coords[0]) == 2
        assert pca.explained_variance is not None
        assert len(pca.explained_variance) == 2

    def test_pca_projection_from_representation_dataset(self) -> None:
        ds = RepresentationDataset(
            experiment_id="pca-exp",
            model_id="pca-model",
            layer_name="pca-layer",
            sample_ids=["s1", "s2", "s3", "s4"],
            labels=[0, 0, 1, 1],
            vectors=[
                [1.0, 2.0, 3.0],
                [1.1, 2.1, 2.9],
                [5.0, 5.0, 5.0],
                [5.1, 4.9, 5.2],
            ],
            feature_dim=3,
            num_samples=4,
            num_classes=2,
        )
        res = compute_pca_projection(ds, n_components=2)
        assert res.method == "pca"
        assert res.original_dim == 3
        assert res.projected_dim == 2
        assert len(res.coordinates) == 4
        assert len(res.explained_variance_ratio) == 2

        # Roundtrip JSON
        json_str = res.to_json()
        assert len(json_str) > 0
        deserialized = ProjectionResult.from_dict(res.to_dict())
        assert deserialized.num_samples == 4
