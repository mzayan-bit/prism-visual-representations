"""Data contracts for PRISM Multimodal Vision-Language Alignment."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from prism.multimodal.enums import (
    RetrievalDirection,
)


@dataclass(frozen=True)
class VisionLanguageSample:
    """Paired image-text sample contract for controlled multimodal research."""

    sample_id: str
    image: list[list[list[float]]]  # Shape: (C, H, W)
    text: str
    captions: list[str] = field(default_factory=list)
    class_label: int | None = None
    class_name: str | None = None
    dataset_fingerprint: str = ""
    split: str = "train"
    pair_identity: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize sample metadata and text content (omits raw pixels)."""
        return {
            "sample_id": self.sample_id,
            "text": self.text,
            "captions": list(self.captions),
            "class_label": self.class_label,
            "class_name": self.class_name,
            "dataset_fingerprint": self.dataset_fingerprint,
            "split": self.split,
            "pair_identity": self.pair_identity,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(
        cls, data: dict[str, Any], image: list[list[list[float]]] | None = None
    ) -> VisionLanguageSample:
        """Instantiate sample from dictionary."""
        return cls(
            sample_id=str(data["sample_id"]),
            image=image if image is not None else [],
            text=str(data["text"]),
            captions=list(data.get("captions", [])),
            class_label=int(data["class_label"])
            if data.get("class_label") is not None
            else None,
            class_name=str(data["class_name"])
            if data.get("class_name") is not None
            else None,
            dataset_fingerprint=str(data.get("dataset_fingerprint", "")),
            split=str(data.get("split", "train")),
            pair_identity=str(data.get("pair_identity", "")),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass(frozen=True)
class TokenizedText:
    """Deterministic token sequence representation for a single text instance."""

    original_text: str
    token_strings: list[str]
    token_ids: list[int]
    sequence_length: int
    attention_mask: list[int]  # 1 for valid token, 0 for <PAD>

    def to_dict(self) -> dict[str, Any]:
        """Serialize token sequence contract."""
        return {
            "original_text": self.original_text,
            "token_strings": list(self.token_strings),
            "token_ids": list(self.token_ids),
            "sequence_length": self.sequence_length,
            "attention_mask": list(self.attention_mask),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TokenizedText:
        """Instantiate TokenizedText from dictionary."""
        return cls(
            original_text=str(data["original_text"]),
            token_strings=list(data["token_strings"]),
            token_ids=[int(x) for x in data["token_ids"]],
            sequence_length=int(data["sequence_length"]),
            attention_mask=[int(x) for x in data["attention_mask"]],
        )


@dataclass
class VisionLanguageBatch:
    """Batch container for paired multimodal training and evaluation."""

    sample_ids: list[str]
    images: list[list[list[list[float]]]]  # (N, C, H, W)
    texts: list[str]
    tokenized_texts: list[TokenizedText]
    token_ids: list[list[int]]  # (N, max_len)
    attention_masks: list[list[int]]  # (N, max_len)
    pair_mapping: list[int]  # Index mapping for positive pairs (default 0..N-1)
    batch_size: int
    labels: list[int] | None = None
    class_names: list[str] | None = None

    def __post_init__(self) -> None:
        if len(self.images) != self.batch_size:
            raise ValueError(
                f"Batch image count {len(self.images)} != batch_size {self.batch_size}"
            )
            msg = (
                f"Batch text count {len(self.tokenized_texts)} != "
                f"batch_size {self.batch_size}"
            )
            raise ValueError(msg)


@dataclass(frozen=True)
class ClassPrompt:
    """Text prompt configuration for zero-shot class matching."""

    class_id: int
    class_name: str
    prompt_template: str
    rendered_text: str
    prompt_id: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "class_id": self.class_id,
            "class_name": self.class_name,
            "prompt_template": self.prompt_template,
            "rendered_text": self.rendered_text,
            "prompt_id": self.prompt_id,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ClassPrompt:
        return cls(
            class_id=int(data["class_id"]),
            class_name=str(data["class_name"]),
            prompt_template=str(data["prompt_template"]),
            rendered_text=str(data["rendered_text"]),
            prompt_id=str(data["prompt_id"]),
        )


@dataclass(frozen=True)
class RetrievalResult:
    """Individual query retrieval result with ranked candidates and similarities."""

    query_modality: RetrievalDirection
    query_sample_id: str
    ranked_candidate_sample_ids: list[str]
    similarities: list[float]
    matched_pair_rank: int  # 1-based rank of true paired candidate
    top_k_success: dict[int, bool]  # e.g. {1: True, 3: True, 5: True}
    candidate_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "query_modality": self.query_modality.value,
            "query_sample_id": self.query_sample_id,
            "ranked_candidate_sample_ids": list(self.ranked_candidate_sample_ids),
            "similarities": list(self.similarities),
            "matched_pair_rank": self.matched_pair_rank,
            "top_k_success": {str(k): v for k, v in self.top_k_success.items()},
            "candidate_count": self.candidate_count,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RetrievalResult:
        return cls(
            query_modality=RetrievalDirection(data["query_modality"]),
            query_sample_id=str(data["query_sample_id"]),
            ranked_candidate_sample_ids=list(data["ranked_candidate_sample_ids"]),
            similarities=[float(x) for x in data["similarities"]],
            matched_pair_rank=int(data["matched_pair_rank"]),
            top_k_success={int(k): bool(v) for k, v in data["top_k_success"].items()},
            candidate_count=int(data["candidate_count"]),
        )


@dataclass(frozen=True)
class CrossModalRetrievalSummary:
    """Aggregate evaluation summary for cross-modal retrieval across a dataset."""

    image_to_text_r1: float
    image_to_text_r3: float
    image_to_text_r5: float
    image_to_text_mrr: float
    text_to_image_r1: float
    text_to_image_r3: float
    text_to_image_r5: float
    text_to_image_mrr: float
    sample_count: int
    candidate_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "image_to_text_r1": self.image_to_text_r1,
            "image_to_text_r3": self.image_to_text_r3,
            "image_to_text_r5": self.image_to_text_r5,
            "image_to_text_mrr": self.image_to_text_mrr,
            "text_to_image_r1": self.text_to_image_r1,
            "text_to_image_r3": self.text_to_image_r3,
            "text_to_image_r5": self.text_to_image_r5,
            "text_to_image_mrr": self.text_to_image_mrr,
            "sample_count": self.sample_count,
            "candidate_count": self.candidate_count,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CrossModalRetrievalSummary:
        return cls(
            image_to_text_r1=float(data["image_to_text_r1"]),
            image_to_text_r3=float(data["image_to_text_r3"]),
            image_to_text_r5=float(data["image_to_text_r5"]),
            image_to_text_mrr=float(data["image_to_text_mrr"]),
            text_to_image_r1=float(data["text_to_image_r1"]),
            text_to_image_r3=float(data["text_to_image_r3"]),
            text_to_image_r5=float(data["text_to_image_r5"]),
            text_to_image_mrr=float(data["text_to_image_mrr"]),
            sample_count=int(data["sample_count"]),
            candidate_count=int(data["candidate_count"]),
        )


@dataclass(frozen=True)
class ZeroShotClassificationSummary:
    """Summary of zero-shot classification evaluation via text prompt matching."""

    prompt_template: str
    class_count: int
    accuracy: float
    per_class_accuracy: dict[str, float]
    confusion_matrix: list[list[int]]
    class_names: list[str]
    top_3_accuracy: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt_template": self.prompt_template,
            "class_count": self.class_count,
            "accuracy": self.accuracy,
            "per_class_accuracy": dict(self.per_class_accuracy),
            "confusion_matrix": [list(row) for row in self.confusion_matrix],
            "class_names": list(self.class_names),
            "top_3_accuracy": self.top_3_accuracy,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ZeroShotClassificationSummary:
        return cls(
            prompt_template=str(data["prompt_template"]),
            class_count=int(data["class_count"]),
            accuracy=float(data["accuracy"]),
            per_class_accuracy={
                str(k): float(v) for k, v in data["per_class_accuracy"].items()
            },
            confusion_matrix=[
                [int(c) for c in row] for row in data["confusion_matrix"]
            ],
            class_names=list(data["class_names"]),
            top_3_accuracy=float(data["top_3_accuracy"])
            if data.get("top_3_accuracy") is not None
            else None,
        )


@dataclass(frozen=True)
class CrossModalCentroidAlignment:
    """Alignment metrics between image and text centroids for a semantic class."""

    class_name: str
    image_centroid: list[float]
    text_centroid: list[float]
    euclidean_distance: float
    cosine_similarity: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "class_name": self.class_name,
            "image_centroid": list(self.image_centroid),
            "text_centroid": list(self.text_centroid),
            "euclidean_distance": self.euclidean_distance,
            "cosine_similarity": self.cosine_similarity,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CrossModalCentroidAlignment:
        return cls(
            class_name=str(data["class_name"]),
            image_centroid=[float(x) for x in data["image_centroid"]],
            text_centroid=[float(x) for x in data["text_centroid"]],
            euclidean_distance=float(data["euclidean_distance"]),
            cosine_similarity=float(data["cosine_similarity"]),
        )


@dataclass(frozen=True)
class MultimodalCollapseSummary:
    """Collapse and diversity diagnostics for vision and text representation spaces."""

    visual_dim_variance: float
    visual_feature_std: float
    visual_pairwise_similarity: float
    text_dim_variance: float
    text_feature_std: float
    text_pairwise_similarity: float
    matched_similarity: float
    unmatched_similarity: float
    similarity_gap: float
    is_collapsed: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "visual_dim_variance": self.visual_dim_variance,
            "visual_feature_std": self.visual_feature_std,
            "visual_pairwise_similarity": self.visual_pairwise_similarity,
            "text_dim_variance": self.text_dim_variance,
            "text_feature_std": self.text_feature_std,
            "text_pairwise_similarity": self.text_pairwise_similarity,
            "matched_similarity": self.matched_similarity,
            "unmatched_similarity": self.unmatched_similarity,
            "similarity_gap": self.similarity_gap,
            "is_collapsed": self.is_collapsed,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MultimodalCollapseSummary:
        return cls(
            visual_dim_variance=float(data["visual_dim_variance"]),
            visual_feature_std=float(data["visual_feature_std"]),
            visual_pairwise_similarity=float(data["visual_pairwise_similarity"]),
            text_dim_variance=float(data["text_dim_variance"]),
            text_feature_std=float(data["text_feature_std"]),
            text_pairwise_similarity=float(data["text_pairwise_similarity"]),
            matched_similarity=float(data["matched_similarity"]),
            unmatched_similarity=float(data["unmatched_similarity"]),
            similarity_gap=float(data["similarity_gap"]),
            is_collapsed=bool(data["is_collapsed"]),
        )
