"""Representation dataset abstractions, spatial vectorization, and distance metrics."""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from enum import Enum
from typing import Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from prism.core.errors import SerializationError, ValidationError
from prism.core.identifiers import ensure_valid_identifier


class SpatialVectorizationPolicy(str, Enum):
    """Policy for converting spatial feature maps [N, C, H, W] to vectors."""

    NONE = "none"
    GLOBAL_AVERAGE_POOL = "global_average_pool"
    FLATTEN = "flatten"


class VectorNormalizationPolicy(str, Enum):
    """Policy for normalizing representation vectors."""

    NONE = "none"
    L2_NORMALIZE = "l2_normalize"
    STANDARDIZE = "standardize"


class DistanceMetric(str, Enum):
    """Supported distance and similarity metrics for representation geometry."""

    EUCLIDEAN = "euclidean"
    SQUARED_EUCLIDEAN = "squared_euclidean"
    COSINE_SIMILARITY = "cosine_similarity"
    COSINE_DISTANCE = "cosine_distance"


def compute_distance(
    v1: list[float],
    v2: list[float],
    metric: DistanceMetric | str = DistanceMetric.EUCLIDEAN,
    eps: float = 1e-12,
) -> float:
    """Compute distance or similarity between two equal-length numerical vectors.

    Parameters
    ----------
    v1 : list[float]
        First feature vector.
    v2 : list[float]
        Second feature vector.
    metric : DistanceMetric | str
        Metric ('euclidean', 'squared_euclidean', 'cosine_similarity', etc.).
    eps : float
        Small numerical stability epsilon.

    Returns
    -------
    float
        Computed scalar distance or similarity.
    """
    if len(v1) != len(v2):
        raise ValidationError(
            f"Vector dimension mismatch: len(v1)={len(v1)} vs len(v2)={len(v2)}."
        )
    if not v1:
        raise ValidationError("Cannot compute distance on empty vectors.")

    metric_enum = DistanceMetric(metric) if isinstance(metric, str) else metric

    if (
        metric_enum == DistanceMetric.EUCLIDEAN
        or metric_enum == DistanceMetric.SQUARED_EUCLIDEAN
    ):
        sq_sum = 0.0
        for x, y in zip(v1, v2, strict=True):
            if not math.isfinite(x) or not math.isfinite(y):
                raise ValidationError(
                    f"Non-finite element encountered in vector: v1={x}, v2={y}."
                )
            diff = x - y
            sq_sum += diff * diff

        if metric_enum == DistanceMetric.SQUARED_EUCLIDEAN:
            return sq_sum
        return math.sqrt(max(0.0, sq_sum))

    if (
        metric_enum == DistanceMetric.COSINE_SIMILARITY
        or metric_enum == DistanceMetric.COSINE_DISTANCE
    ):
        dot = 0.0
        norm_v1_sq = 0.0
        norm_v2_sq = 0.0

        for x, y in zip(v1, v2, strict=True):
            if not math.isfinite(x) or not math.isfinite(y):
                raise ValidationError(
                    f"Non-finite element encountered in vector: v1={x}, v2={y}."
                )
            dot += x * y
            norm_v1_sq += x * x
            norm_v2_sq += y * y

        norm_v1 = math.sqrt(norm_v1_sq)
        norm_v2 = math.sqrt(norm_v2_sq)

        # Handle zero vectors safely
        if norm_v1 < eps or norm_v2 < eps:
            sim = 0.0
        else:
            sim = max(-1.0, min(1.0, dot / (norm_v1 * norm_v2)))

        if metric_enum == DistanceMetric.COSINE_SIMILARITY:
            return sim
        return 1.0 - sim

    raise ValidationError(f"Unsupported distance metric: {metric_enum}.")


def compute_pairwise_distances(
    vectors: list[list[float]],
    metric: DistanceMetric | str = DistanceMetric.EUCLIDEAN,
    max_samples: int = 2000,
) -> list[list[float]]:
    """Compute symmetric pairwise distance matrix [N, N] with memory safeguards.

    Parameters
    ----------
    vectors : list[list[float]]
        List of N feature vectors of identical dimension D.
    metric : DistanceMetric | str
        Distance metric.
    max_samples : int
        Maximum allowed number of samples to prevent memory explosion.

    Returns
    -------
    list[list[float]]
        Symmetric N x N distance matrix with exact 0.0 diagonal.
    """
    n = len(vectors)
    if n == 0:
        return []
    if n > max_samples:
        raise ValidationError(
            f"Sample count ({n}) exceeds pairwise safety limit ({max_samples}). "
            f"Use subsampling for large datasets."
        )

    matrix: list[list[float]] = [[0.0] * n for _ in range(n)]

    for i in range(n):
        for j in range(i + 1, n):
            dist = compute_distance(vectors[i], vectors[j], metric=metric)
            matrix[i][j] = dist
            matrix[j][i] = dist

    return matrix


def vectorize_spatial_features(
    data: list[Any],
    policy: SpatialVectorizationPolicy
    | str = SpatialVectorizationPolicy.GLOBAL_AVERAGE_POOL,
) -> tuple[list[list[float]], tuple[int, ...], int]:
    """Convert spatial feature tensors [N, C, H, W] into 2D vectors [N, D].

    Parameters
    ----------
    data : list[Any]
        Input nested lists: 2D vectors [N, D] or 4D spatial tensors [N, C, H, W].
    policy : SpatialVectorizationPolicy | str
        Transformation policy.

    Returns
    -------
    tuple[list[list[float]], tuple[int, ...], int]
        (vectorized_rows [N, D], original_shape, resulting_dim D)
    """
    if not data:
        return [], (0,), 0

    policy_enum = (
        SpatialVectorizationPolicy(policy) if isinstance(policy, str) else policy
    )

    # Infer dimensionality
    first = data[0]
    if (
        isinstance(first, list)
        and isinstance(first[0], list)
        and isinstance(first[0][0], list)
    ):
        # 4D spatial tensor: [N, C, H, W]
        n_samples = len(data)
        c_dim = len(first)
        h_dim = len(first[0])
        w_dim = len(first[0][0])
        orig_shape: tuple[int, ...] = (n_samples, c_dim, h_dim, w_dim)

        vectors: list[list[float]] = []

        if policy_enum == SpatialVectorizationPolicy.GLOBAL_AVERAGE_POOL:
            spatial_pixels = float(h_dim * w_dim)
            if spatial_pixels <= 0.0:
                raise ValidationError(
                    f"Invalid spatial dimensions: H={h_dim}, W={w_dim}."
                )

            for sample in data:
                row = [0.0] * c_dim
                for c in range(c_dim):
                    channel_sum = 0.0
                    for h in range(h_dim):
                        for w in range(w_dim):
                            v = float(sample[c][h][w])
                            if not math.isfinite(v):
                                raise ValidationError(
                                    "Non-finite value in spatial feature map."
                                )
                            channel_sum += v
                    row[c] = channel_sum / spatial_pixels
                vectors.append(row)
            return vectors, orig_shape, c_dim

        if policy_enum == SpatialVectorizationPolicy.FLATTEN:
            flat_dim = c_dim * h_dim * w_dim
            for sample in data:
                row = [0.0] * flat_dim
                idx = 0
                for c in range(c_dim):
                    for h in range(h_dim):
                        for w in range(w_dim):
                            v = float(sample[c][h][w])
                            if not math.isfinite(v):
                                raise ValidationError(
                                    "Non-finite value in spatial feature map."
                                )
                            row[idx] = v
                            idx += 1
                vectors.append(row)
            return vectors, orig_shape, flat_dim

        raise ValidationError(
            f"Cannot pass 4D spatial tensors with policy='{policy_enum.value}'. "
            f"Must specify GLOBAL_AVERAGE_POOL or FLATTEN."
        )

    if isinstance(first, list) and isinstance(first[0], list):
        # 3D sequence tensor: [N, S, D]
        n_samples = len(data)
        s_dim = len(first)
        d_dim = len(first[0])
        orig_seq_shape: tuple[int, ...] = (n_samples, s_dim, d_dim)
        seq_vectors: list[list[float]] = []

        if policy_enum == SpatialVectorizationPolicy.GLOBAL_AVERAGE_POOL:
            for sample in data:
                row = [0.0] * d_dim
                for token in sample:
                    for d_idx in range(d_dim):
                        val = float(token[d_idx])
                        if not math.isfinite(val):
                            raise ValidationError(
                                "Non-finite value in sequence feature map."
                            )
                        row[d_idx] += val
                for d_idx in range(d_dim):
                    row[d_idx] /= float(s_dim)
                seq_vectors.append(row)
            return seq_vectors, orig_seq_shape, d_dim

        if policy_enum == SpatialVectorizationPolicy.FLATTEN:
            flat_dim = s_dim * d_dim
            for sample in data:
                row = [0.0] * flat_dim
                idx = 0
                for token in sample:
                    for d_idx in range(d_dim):
                        val = float(token[d_idx])
                        if not math.isfinite(val):
                            raise ValidationError(
                                "Non-finite value in sequence feature map."
                            )
                        row[idx] = val
                        idx += 1
                seq_vectors.append(row)
            return seq_vectors, orig_seq_shape, flat_dim

        raise ValidationError(
            f"Cannot pass 3D sequence tensors with policy='{policy_enum.value}'. "
            f"Must specify GLOBAL_AVERAGE_POOL or FLATTEN."
        )

    # Already 2D vectors: [N, D]
    n_samples = len(data)
    d_dim = len(first) if isinstance(first, list) else 1
    orig_2d_shape: tuple[int, ...] = (n_samples, d_dim)
    vectors = []
    for sample in data:
        if not isinstance(sample, list):
            sample = [float(sample)]
        row = [float(v) for v in sample]
        if any(not math.isfinite(v) for v in row):
            raise ValidationError("Non-finite value in feature vector.")
        vectors.append(row)

    return vectors, orig_2d_shape, d_dim


def normalize_vectors(
    vectors: list[list[float]],
    policy: VectorNormalizationPolicy | str = VectorNormalizationPolicy.NONE,
    eps: float = 1e-12,
) -> list[list[float]]:
    """Apply explicit normalization policy to a batch of vectors [N, D].

    Parameters
    ----------
    vectors : list[list[float]]
        Input feature vectors.
    policy : VectorNormalizationPolicy | str
        Normalization policy: 'none', 'l2_normalize', or 'standardize'.
    eps : float
        Small epsilon preventing division by zero.

    Returns
    -------
    list[list[float]]
        Normalized feature vectors.
    """
    if not vectors:
        return []

    policy_enum = (
        VectorNormalizationPolicy(policy) if isinstance(policy, str) else policy
    )

    if policy_enum == VectorNormalizationPolicy.NONE:
        return [list(r) for r in vectors]

    if policy_enum == VectorNormalizationPolicy.L2_NORMALIZE:
        normalized: list[list[float]] = []
        for row in vectors:
            sq_norm = sum(x * x for x in row)
            norm = math.sqrt(sq_norm)
            if norm < eps:
                # Safe zero vector policy: keep as zeros
                normalized.append([0.0] * len(row))
            else:
                normalized.append([x / norm for x in row])
        return normalized

    if policy_enum == VectorNormalizationPolicy.STANDARDIZE:
        # Standardize across samples per dimension: mean 0, std 1
        n = len(vectors)
        d = len(vectors[0])
        if n <= 1:
            return [list(r) for r in vectors]

        means = [sum(vectors[i][j] for i in range(n)) / n for j in range(d)]
        variances = [
            sum((vectors[i][j] - means[j]) ** 2 for i in range(n)) / n for j in range(d)
        ]
        stds = [math.sqrt(max(0.0, v)) for v in variances]

        normalized = []
        for row in vectors:
            norm_row = [0.0] * d
            for j in range(d):
                if stds[j] < eps:
                    norm_row[j] = 0.0
                else:
                    norm_row[j] = (row[j] - means[j]) / stds[j]
            normalized.append(norm_row)
        return normalized

    raise ValidationError(f"Unsupported normalization policy: {policy_enum}.")


class RepresentationDataset(BaseModel):
    """Structured representation dataset for geometric and manifold analysis."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    experiment_id: str = Field(description="Associated experiment identifier")
    model_id: str = Field(description="Source model identifier")
    layer_name: str = Field(description="Extracted layer name")
    sample_ids: list[str] = Field(description="Unique sample IDs aligned with vectors")
    labels: list[int | str] = Field(
        description="Ground truth category labels per sample"
    )
    vectors: list[list[float]] = Field(description="2D feature matrix [N, D]")
    feature_dim: int = Field(
        gt=0, description="Feature dimensionality D of each vector"
    )
    num_samples: int = Field(ge=0, description="Total number of samples N")
    num_classes: int = Field(ge=1, description="Number of distinct classes")
    class_names: list[str] = Field(default_factory=list, description="Class name list")
    source_split: str = Field(
        default="test", description="Dataset split (train, val, test)"
    )
    spatial_transformation: SpatialVectorizationPolicy = Field(
        default=SpatialVectorizationPolicy.NONE,
        description="Spatial vectorization policy applied",
    )
    normalization_policy: VectorNormalizationPolicy = Field(
        default=VectorNormalizationPolicy.NONE,
        description="Vector normalization policy applied",
    )
    original_shape: tuple[int, ...] | None = Field(
        default=None,
        description="Original tensor shape prior to spatial vectorization",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Auxiliary metadata (e.g. data budget, epoch, seed)",
    )

    @field_validator("experiment_id", "model_id")
    @classmethod
    def validate_id_fields(cls, v: str) -> str:
        return ensure_valid_identifier(v, field_name="id")

    @field_validator("layer_name")
    @classmethod
    def validate_layer(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("layer_name cannot be empty.")
        return v.strip().lower()

    @model_validator(mode="after")
    def validate_dataset_alignment(self) -> RepresentationDataset:
        n = len(self.sample_ids)
        if len(self.labels) != n:
            raise ValidationError(
                f"Sample count mismatch: {n} sample_ids vs {len(self.labels)} labels."
            )
        if len(self.vectors) != n:
            raise ValidationError(
                f"Sample count mismatch: {n} sample_ids vs "
                f"{len(self.vectors)} vector rows."
            )
        if self.num_samples != n:
            raise ValidationError(
                f"num_samples ({self.num_samples}) does not match vector count ({n})."
            )

        # Validate unique sample IDs
        if len(set(self.sample_ids)) != n:
            raise ValidationError(
                "Duplicate sample_ids found in RepresentationDataset."
            )

        # Validate dimensional uniformity and finiteness
        for idx, row in enumerate(self.vectors):
            if len(row) != self.feature_dim:
                raise ValidationError(
                    f"Row {idx} dimension mismatch: expected "
                    f"{self.feature_dim}, got {len(row)}."
                )
            for v in row:
                if not math.isfinite(v):
                    raise ValidationError(
                        f"Non-finite value ({v}) found in sample "
                        f"{self.sample_ids[idx]}."
                    )

        return self

    @classmethod
    def from_raw_representations(
        cls,
        raw_embeddings: list[Any],
        sample_ids: list[str],
        labels: Sequence[int | str],
        experiment_id: str,
        model_id: str,
        layer_name: str,
        spatial_policy: SpatialVectorizationPolicy | str = (
            SpatialVectorizationPolicy.GLOBAL_AVERAGE_POOL
        ),
        norm_policy: VectorNormalizationPolicy | str = (VectorNormalizationPolicy.NONE),
        class_names: list[str] | None = None,
        source_split: str = "test",
        metadata: dict[str, Any] | None = None,
    ) -> RepresentationDataset:
        """Construct RepresentationDataset with vectorization and normalization."""
        vectors, orig_shape, feat_dim = vectorize_spatial_features(
            raw_embeddings, policy=spatial_policy
        )
        norm_vectors = normalize_vectors(vectors, policy=norm_policy)

        labels_list: list[int | str] = list(labels)

        # Unique classes
        unique_labels = sorted(set(labels_list), key=lambda x: str(x))
        num_classes = len(unique_labels)
        if class_names is None or len(class_names) < num_classes:
            class_names = [f"class_{c}" for c in unique_labels]

        return cls(
            experiment_id=experiment_id,
            model_id=model_id,
            layer_name=layer_name,
            sample_ids=list(sample_ids),
            labels=labels_list,
            vectors=norm_vectors,
            feature_dim=feat_dim,
            num_samples=len(sample_ids),
            num_classes=num_classes,
            class_names=class_names,
            source_split=source_split,
            spatial_transformation=(
                SpatialVectorizationPolicy(spatial_policy)
                if isinstance(spatial_policy, str)
                else spatial_policy
            ),
            normalization_policy=(
                VectorNormalizationPolicy(norm_policy)
                if isinstance(norm_policy, str)
                else norm_policy
            ),
            original_shape=orig_shape,
            metadata=metadata or {},
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert representation dataset to dictionary."""
        return self.model_dump(mode="json")

    def to_json(self, indent: int | None = None) -> str:
        """Convert representation dataset to JSON string."""
        return json.dumps(
            self.to_dict(),
            indent=indent,
            sort_keys=True,
            ensure_ascii=True,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RepresentationDataset:
        """Create dataset from dictionary."""
        try:
            return cls.model_validate(data)
        except Exception as exc:
            raise SerializationError(
                f"Failed to deserialize RepresentationDataset from dict: {exc}"
            ) from exc
