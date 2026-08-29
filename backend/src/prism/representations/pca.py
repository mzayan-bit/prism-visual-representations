"""Principal Component Analysis (PCA) and low-dimensional projection engine."""

from __future__ import annotations

import copy
import json
import math
from typing import Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
)

from prism.core.errors import SerializationError, ValidationError
from prism.representations.geometry import RepresentationDataset


def _jacobi_eigenvalues_symmetric(
    matrix: list[list[float]],
    max_iterations: int = 150,
    tol: float = 1e-12,
) -> tuple[list[float], list[list[float]]]:
    """Compute eigenvalues and eigenvectors via Jacobi rotations.

    Parameters
    ----------
    matrix : list[list[float]]
        Real symmetric m x m matrix.
    max_iterations : int
        Maximum number of Jacobi sweep iterations.
    tol : float
        Convergence tolerance on off-diagonal elements.

    Returns
    -------
    tuple[list[float], list[list[float]]]
        (eigenvalues of length m, eigenvectors list of length m)
    """
    m = len(matrix)
    if m == 0:
        return [], []
    if m == 1:
        return [matrix[0][0]], [[1.0]]

    # Working copy of matrix A and eigenvector accumulator V (initialized to Identity)
    a = [list(r) for r in matrix]
    v = [[1.0 if i == j else 0.0 for j in range(m)] for i in range(m)]

    for _ in range(max_iterations):
        # Find maximum off-diagonal element
        max_val = 0.0
        p, q = 0, 1
        for i in range(m):
            for j in range(i + 1, m):
                abs_val = abs(a[i][j])
                if abs_val > max_val:
                    max_val = abs_val
                    p, q = i, j

        if max_val < tol:
            break

        # Compute Jacobi rotation angle
        app = a[p][p]
        aqq = a[q][q]
        apq = a[p][q]

        if abs(apq) < 1e-15:
            continue

        theta = (aqq - app) / (2.0 * apq)
        t = (
            math.copysign(1.0, theta) / (abs(theta) + math.sqrt(theta * theta + 1.0))
            if abs(theta) < 1e10
            else 0.5 / theta
        )
        c = 1.0 / math.sqrt(t * t + 1.0)
        s = t * c

        # Rotate A
        tau = s / (1.0 + c)

        a[p][p] = app - t * apq
        a[q][q] = aqq + t * apq
        a[p][q] = 0.0
        a[q][p] = 0.0

        for j in range(m):
            if j != p and j != q:
                ajp = a[j][p]
                ajq = a[j][q]
                a[j][p] = ajp - s * (ajq + tau * ajp)
                a[p][j] = a[j][p]
                a[j][q] = ajq + s * (ajp - tau * ajq)
                a[q][j] = a[j][q]

        # Accumulate into V: V' = V * R
        for i in range(m):
            vip = v[i][p]
            viq = v[i][q]
            v[i][p] = vip - s * (viq + tau * vip)
            v[i][q] = viq + s * (vip - tau * viq)

    # Extract eigenvalues from diagonal
    eigenvalues = [max(0.0, a[i][i]) for i in range(m)]

    # Transpose V so that each row is an eigenvector: [m, m]
    eigenvectors: list[list[float]] = [
        [v[row][col] for row in range(m)] for col in range(m)
    ]

    # Sort descending by eigenvalue
    indices = sorted(range(m), key=lambda idx: eigenvalues[idx], reverse=True)
    sorted_evals = [eigenvalues[idx] for idx in indices]
    sorted_evecs = [eigenvectors[idx] for idx in indices]

    return sorted_evals, sorted_evecs


class PrincipalComponentAnalysis:
    """Deterministic, pure-Python Principal Component Analysis engine."""

    def __init__(
        self,
        n_components: int = 2,
    ) -> None:
        if n_components <= 0:
            raise ValidationError(f"n_components must be positive, got {n_components}.")
        self.n_components = n_components
        self.mean_vector: list[float] | None = None
        self.components: list[list[float]] | None = None  # [k, D]
        self.explained_variance: list[float] | None = None
        self.explained_variance_ratio: list[float] | None = None
        self.cumulative_explained_variance: list[float] | None = None
        self.total_variance: float = 0.0
        self.n_samples_seen: int = 0
        self.n_features_: int = 0

    def fit(self, x: list[list[float]]) -> PrincipalComponentAnalysis:
        """Fit PCA model on feature matrix X [N, D].

        Parameters
        ----------
        x : list[list[float]]
            Feature matrix with N samples and D features.

        Returns
        -------
        PrincipalComponentAnalysis
            Fitted instance.
        """
        n = len(x)
        if n == 0:
            raise ValidationError("Cannot fit PCA on empty dataset.")
        d = len(x[0])
        if d == 0:
            raise ValidationError("Cannot fit PCA on zero-dimensional features.")

        self.n_samples_seen = n
        self.n_features_ = d

        max_comp = min(n, d)
        actual_k = min(self.n_components, max_comp)

        # 1. Compute Mean Vector and Center X
        mean_vec = [sum(x[i][j] for i in range(n)) / float(n) for j in range(d)]
        self.mean_vector = mean_vec

        x_c: list[list[float]] = []
        for i in range(n):
            row = [x[i][j] - mean_vec[j] for j in range(d)]
            x_c.append(row)

        denom = float(max(1, n - 1))

        # 2. Eigendecomposition: Choose between Covariance (D x D) or Gram (N x N)
        if d <= n:
            # Standard Covariance: C = (1 / (N - 1)) X_c^T X_c [D, D]
            cov = [[0.0] * d for _ in range(d)]
            for i in range(d):
                for j in range(i, d):
                    dot = sum(x_c[s][i] * x_c[s][j] for s in range(n)) / denom
                    cov[i][j] = dot
                    cov[j][i] = dot

            evals, evecs = _jacobi_eigenvalues_symmetric(cov)
            raw_components = evecs[:actual_k]
            var_components = evals[:actual_k]
            total_var = sum(evals)

        else:
            # Dual Gram Matrix: G = (1 / (N - 1)) X_c X_c^T [N, N]
            gram = [[0.0] * n for _ in range(n)]
            for i in range(n):
                for j in range(i, n):
                    dot = sum(x_c[i][k] * x_c[j][k] for k in range(d)) / denom
                    gram[i][j] = dot
                    gram[j][i] = dot

            evals, evecs = _jacobi_eigenvalues_symmetric(gram)
            total_var = sum(evals)

            raw_components = []
            var_components = []

            for idx in range(actual_k):
                ev = evals[idx]
                if ev < 1e-12:
                    break
                v_n = evecs[idx]  # vector of length N
                # Projection to feature space: v_d = (1 / sqrt((N-1) * ev)) X_c^T v_n
                scale = 1.0 / (math.sqrt(denom * ev))
                v_d = [0.0] * d
                for j in range(d):
                    dot = sum(x_c[s][j] * v_n[s] for s in range(n))
                    v_d[j] = dot * scale

                raw_components.append(v_d)
                var_components.append(ev)

        # 3. Deterministic Sign Convention:
        # Force the loading with maximum absolute magnitude to be positive
        oriented_components: list[list[float]] = []
        for comp in raw_components:
            max_abs = -1.0
            best_sign = 1.0
            for val in comp:
                if abs(val) > max_abs:
                    max_abs = abs(val)
                    best_sign = 1.0 if val >= 0.0 else -1.0
            oriented = [v * best_sign for v in comp]
            oriented_components.append(oriented)

        self.components = oriented_components
        self.explained_variance = var_components
        self.total_variance = max(1e-12, total_var)

        # Explained variance ratios
        ratios = [v / self.total_variance for v in var_components]
        self.explained_variance_ratio = ratios

        cum_ratios: list[float] = []
        acc = 0.0
        for r in ratios:
            acc += r
            cum_ratios.append(min(1.0, acc))
        self.cumulative_explained_variance = cum_ratios

        return self

    def transform(self, x: list[list[float]]) -> list[list[float]]:
        """Project feature matrix X [N, D] onto fitted principal components [N, K]."""
        if self.components is None or self.mean_vector is None:
            raise ValidationError("PCA instance is not fitted yet.")

        n = len(x)
        if n == 0:
            return []
        d = len(x[0])
        if d != self.n_features_:
            raise ValidationError(
                f"Feature dimension mismatch: expected {self.n_features_}, got {d}."
            )

        k = len(self.components)
        projected: list[list[float]] = []

        for i in range(n):
            row_c = [x[i][j] - self.mean_vector[j] for j in range(d)]
            proj_row = [0.0] * k
            for comp_idx in range(k):
                comp_vec = self.components[comp_idx]
                proj_row[comp_idx] = sum(row_c[j] * comp_vec[j] for j in range(d))
            projected.append(proj_row)

        return projected

    def fit_transform(self, x: list[list[float]]) -> list[list[float]]:
        """Fit PCA and return projected coordinates [N, K]."""
        return self.fit(x).transform(x)


class ProjectionResult(BaseModel):
    """Structured result of a low-dimensional representation projection."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    method: str = Field(default="pca", description="Projection method ('pca')")
    original_dim: int = Field(gt=0, description="Original feature dimensionality")
    projected_dim: int = Field(gt=0, description="Projected dimensionality (e.g. 2, 3)")
    num_samples: int = Field(ge=0, description="Number of projected samples")
    sample_ids: list[str] = Field(description="Sample IDs aligned with coordinates")
    labels: list[int | str] = Field(description="Class labels aligned with coordinates")
    coordinates: list[list[float]] = Field(
        description="Low-dimensional coordinate matrix [N, projected_dim]"
    )
    explained_variance: list[float] = Field(
        default_factory=list,
        description="Variance explained by each principal component",
    )
    explained_variance_ratio: list[float] = Field(
        default_factory=list,
        description="Fraction of total variance explained by each component",
    )
    cumulative_explained_variance: list[float] = Field(
        default_factory=list,
        description="Cumulative fraction of total variance explained",
    )
    mean_vector: list[float] = Field(
        default_factory=list,
        description="Mean feature vector subtracted prior to projection [D]",
    )
    components: list[list[float]] | None = Field(
        default=None,
        description="Principal component unit vectors [projected_dim, D]",
    )

    def to_dict(self) -> dict[str, Any]:
        """Convert projection result to dictionary."""
        return self.model_dump(mode="json")

    def to_json(self, indent: int | None = None) -> str:
        """Convert projection result to JSON string."""
        return json.dumps(
            self.to_dict(),
            indent=indent,
            sort_keys=True,
            ensure_ascii=True,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ProjectionResult:
        """Create projection result from dictionary."""
        try:
            return cls.model_validate(data)
        except Exception as exc:
            raise SerializationError(
                f"Failed to deserialize ProjectionResult from dict: {exc}"
            ) from exc


def compute_pca_projection(
    dataset: RepresentationDataset,
    n_components: int = 2,
) -> ProjectionResult:
    """Project a RepresentationDataset onto low-dimensional PCA coordinates.

    Parameters
    ----------
    dataset : RepresentationDataset
        Input representation dataset.
    n_components : int
        Target projection dimensionality (2 for 2D scatter plots, 3 for 3D).

    Returns
    -------
    ProjectionResult
        Deterministic PCA projection result with explained variance statistics.
    """
    if dataset.num_samples == 0:
        raise ValidationError("Cannot compute PCA on empty dataset.")

    pca = PrincipalComponentAnalysis(n_components=n_components)
    coords = pca.fit_transform(dataset.vectors)

    return ProjectionResult(
        method="pca",
        original_dim=dataset.feature_dim,
        projected_dim=len(coords[0]) if coords else n_components,
        num_samples=dataset.num_samples,
        sample_ids=copy.deepcopy(dataset.sample_ids),
        labels=copy.deepcopy(dataset.labels),
        coordinates=coords,
        explained_variance=pca.explained_variance or [],
        explained_variance_ratio=pca.explained_variance_ratio or [],
        cumulative_explained_variance=pca.cumulative_explained_variance or [],
        mean_vector=pca.mean_vector or [],
        components=pca.components,
    )
