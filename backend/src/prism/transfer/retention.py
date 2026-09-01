"""Representation retention and pre/post transfer drift analysis with shared PCA."""

from __future__ import annotations

import math
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from prism.core.errors import ValidationError
from prism.data.materialized import MaterializedDataset
from prism.models.base import BaseVisionModel
from prism.transfer.probes import _flatten_vector


class TransferRepresentationDriftSummary(BaseModel):
    """Summary of representation shift between pre-transfer and post-transfer states."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    source_model_id: str = Field(..., description="Source model identifier")
    architecture: str = Field(..., description="Model architecture family/type")
    layer: str = Field(
        ..., description="Layer from which representations were extracted"
    )
    transfer_strategy: str = Field(..., description="Transfer strategy applied")
    num_samples: int = Field(
        ..., ge=1, description="Number of evaluated reference samples"
    )
    mean_euclidean_drift: float = Field(
        ..., ge=0.0, description="Mean Euclidean distance ||z_post - z_pre||_2"
    )
    median_euclidean_drift: float = Field(
        ..., ge=0.0, description="Median Euclidean distance"
    )
    max_euclidean_drift: float = Field(
        ..., ge=0.0, description="Maximum observed Euclidean displacement"
    )
    mean_cosine_similarity: float = Field(
        ...,
        ge=-1.0,
        le=1.0,
        description="Mean cosine similarity between pre and post vectors",
    )
    mean_relative_norm_change: float = Field(
        ..., description="Mean relative norm change (|norm_post - norm_pre| / norm_pre)"
    )
    is_frozen_backbone: bool = Field(
        ..., description="True if backbone was declared frozen (drift should be ~0)"
    )

    def to_dict(self) -> dict[str, Any]:
        """Convert drift summary to dictionary."""
        return self.model_dump(mode="json")


def _l2_norm(vec: list[float]) -> float:
    return math.sqrt(sum(v * v for v in vec))


def _cosine_similarity(v1: list[float], v2: list[float], eps: float = 1e-12) -> float:
    norm1 = _l2_norm(v1)
    norm2 = _l2_norm(v2)
    if norm1 < eps or norm2 < eps:
        return 1.0 if norm1 < eps and norm2 < eps else 0.0
    dot = sum(a * b for a, b in zip(v1, v2, strict=True))
    return max(-1.0, min(1.0, dot / (norm1 * norm2)))


def compute_representation_retention(
    pre_model: BaseVisionModel,
    post_model: BaseVisionModel,
    reference_dataset: MaterializedDataset,
    layer: str = "final_hidden",
    transfer_strategy: str = "linear_probe",
    max_samples: int = 50,
) -> TransferRepresentationDriftSummary:
    """Evaluate representation drift between pre-transfer and post-transfer models.

    Args:
        pre_model: Source model state before transfer fine-tuning.
        post_model: Transferred model state after target training.
        reference_dataset: Materialized dataset of reference samples.
        layer: Representation layer to probe.
        transfer_strategy: Name of strategy (e.g. 'linear_probe', 'full_fine_tune').
        max_samples: Maximum number of reference samples to evaluate.

    Returns:
        TransferRepresentationDriftSummary.
    """
    pre_was_train = pre_model.is_training
    post_was_train = post_model.is_training
    pre_model.eval()
    post_model.eval()

    try:
        samples = reference_dataset.samples[:max_samples]
        if not samples:
            raise ValidationError("Reference dataset contains no samples.")

        euclidean_drifts: list[float] = []
        cosine_sims: list[float] = []
        norm_changes: list[float] = []

        for s in samples:
            feat_pre = _flatten_vector(
                pre_model.extract_representations([s.data], layer=layer)[0]
            )
            feat_post = _flatten_vector(
                post_model.extract_representations([s.data], layer=layer)[0]
            )

            # Euclidean drift
            diff = [a - b for a, b in zip(feat_post, feat_pre, strict=True)]
            drift = _l2_norm(diff)
            euclidean_drifts.append(drift)

            # Cosine similarity
            cos_sim = _cosine_similarity(feat_pre, feat_post)
            cosine_sims.append(cos_sim)

            # Relative norm change
            n_pre = _l2_norm(feat_pre)
            n_post = _l2_norm(feat_post)
            rel_norm = abs(n_post - n_pre) / (n_pre + 1e-12)
            norm_changes.append(rel_norm)

        euclidean_drifts.sort()
        n = len(euclidean_drifts)
        mean_drift = sum(euclidean_drifts) / float(n)
        median_drift = (
            euclidean_drifts[n // 2]
            if n % 2 == 1
            else (euclidean_drifts[n // 2 - 1] + euclidean_drifts[n // 2]) / 2.0
        )
        max_drift = euclidean_drifts[-1]
        mean_cos = sum(cosine_sims) / float(n)
        mean_norm_change = sum(norm_changes) / float(n)

        is_frozen = transfer_strategy.lower() in ("linear_probe", "frozen")

        return TransferRepresentationDriftSummary(
            source_model_id=pre_model.model_id,
            architecture=pre_model.spec.family.value,
            layer=layer,
            transfer_strategy=transfer_strategy,
            num_samples=n,
            mean_euclidean_drift=mean_drift,
            median_euclidean_drift=median_drift,
            max_euclidean_drift=max_drift,
            mean_cosine_similarity=mean_cos,
            mean_relative_norm_change=mean_norm_change,
            is_frozen_backbone=is_frozen,
        )

    finally:
        if pre_was_train:
            pre_model.train()
        if post_was_train:
            post_model.train()


def compute_transfer_shared_pca(
    pre_representations: list[list[float]],
    post_representations: list[list[float]],
    n_components: int = 2,
) -> dict[str, Any]:
    """Fit PCA on pre-transfer features and project pre/post into shared basis.

    Returns:
        dict with 'pre_coordinates', 'post_coordinates', 'displacement_vectors',
        'explained_variance_ratio', and 'mean_drift'.
    """
    n_samples = len(pre_representations)
    if n_samples < 2 or not pre_representations[0]:
        raise ValidationError(
            "Shared PCA requires at least 2 samples with valid feature vectors."
        )

    d = len(pre_representations[0])

    # 1. Compute mean vector of pre-transfer representations
    pre_mean = [
        sum(pre_representations[i][j] for i in range(n_samples)) / float(n_samples)
        for j in range(d)
    ]

    # Center pre and post data
    pre_centered = [
        [pre_representations[i][j] - pre_mean[j] for j in range(d)]
        for i in range(n_samples)
    ]
    post_centered = [
        [post_representations[i][j] - pre_mean[j] for j in range(d)]
        for i in range(n_samples)
    ]

    # 2. Compute covariance matrix on pre-transfer features: C = (X^T @ X) / (N - 1)
    cov = [[0.0 for _ in range(d)] for _ in range(d)]
    scale = 1.0 / max(1.0, float(n_samples - 1))
    for r in range(d):
        for c in range(d):
            cov[r][c] = (
                sum(pre_centered[i][r] * pre_centered[i][c] for i in range(n_samples))
                * scale
            )

    # 3. Deterministic Power Iteration to extract top-k eigenvectors
    basis_vectors: list[list[float]] = []
    variances: list[float] = []

    # Deflation matrix copy
    cov_work = [list(row) for row in cov]

    for comp_idx in range(min(n_components, d)):
        # Initial vector
        v = [1.0 if i == (comp_idx % d) else 0.0 for i in range(d)]
        if _l2_norm(v) < 1e-12:
            v[0] = 1.0

        for _ in range(50):
            # v_next = C @ v
            v_next = [sum(cov_work[r][c] * v[c] for c in range(d)) for r in range(d)]
            norm_vn = _l2_norm(v_next)
            if norm_vn < 1e-12:
                break
            v = [x / norm_vn for x in v_next]

        # Enforce canonical orientation (first non-zero component positive)
        for val in v:
            if abs(val) > 1e-6:
                if val < 0:
                    v = [-x for x in v]
                break

        var = sum(v[r] * sum(cov_work[r][c] * v[c] for c in range(d)) for r in range(d))
        basis_vectors.append(v)
        variances.append(max(0.0, var))

        # Deflate: C_next = C - var * (v @ v^T)
        for r in range(d):
            for c in range(d):
                cov_work[r][c] -= var * v[r] * v[c]

    # Normalize explained variance ratios
    total_var = sum(cov[i][i] for i in range(d))
    ev_ratios = [v / max(1e-12, total_var) for v in variances]

    # Project pre and post onto shared basis
    k = len(basis_vectors)
    pre_coords: list[list[float]] = []
    post_coords: list[list[float]] = []
    displacements: list[list[float]] = []

    for i in range(n_samples):
        pre_c = [
            sum(pre_centered[i][j] * basis_vectors[c][j] for j in range(d))
            for c in range(k)
        ]
        post_c = [
            sum(post_centered[i][j] * basis_vectors[c][j] for j in range(d))
            for c in range(k)
        ]
        disp = [post_c[c] - pre_c[c] for c in range(k)]

        pre_coords.append(pre_c)
        post_coords.append(post_c)
        displacements.append(disp)

    return {
        "pre_coordinates": pre_coords,
        "post_coordinates": post_coords,
        "displacement_vectors": displacements,
        "explained_variance_ratio": ev_ratios,
        "mean_displacement": sum(_l2_norm(d_v) for d_v in displacements)
        / float(n_samples),
    }
