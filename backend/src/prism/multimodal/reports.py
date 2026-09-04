"""Comprehensive Reporting Data Model for Vision-Language Multimodal Alignment."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from prism.multimodal.contracts import (
    CrossModalCentroidAlignment,
    CrossModalRetrievalSummary,
    MultimodalCollapseSummary,
    ZeroShotClassificationSummary,
)


@dataclass(frozen=True)
class VisionLanguageRepresentationReport:
    """Report artifact for dual-encoder vision-language alignment."""

    # Identity
    experiment_id: str
    visual_family: str
    visual_architecture: str
    text_dim: int
    shared_dim: int
    temperature: float
    seed: int
    dataset_fingerprint: str

    # Training Telemetry
    final_loss: float
    image_to_text_loss: float
    text_to_image_loss: float
    matched_similarity: float
    unmatched_similarity: float
    similarity_gap: float
    training_history: list[dict[str, float]]

    # Retrieval Performance
    retrieval_summary: CrossModalRetrievalSummary

    # Zero-Shot Classification
    zero_shot_summary: ZeroShotClassificationSummary

    # Prompt Sensitivity
    prompt_sensitivity: dict[str, Any]

    # Shared Multimodal Geometry
    explained_variance_ratio: list[float]
    mean_paired_distance: float
    mean_paired_cosine: float
    centroid_alignments: list[CrossModalCentroidAlignment]

    # Modality Collapse Diagnostics
    collapse_summary: MultimodalCollapseSummary

    # Robustness Under Visual Corruption
    robustness_summary: dict[str, Any]

    # Descriptive Failure Categorization
    failure_types: list[str] = field(default_factory=list)

    # Scientific Caveats and Methodological Warnings
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize complete report to JSON-compatible dictionary."""
        return {
            "experiment_id": self.experiment_id,
            "visual_family": self.visual_family,
            "visual_architecture": self.visual_architecture,
            "text_dim": self.text_dim,
            "shared_dim": self.shared_dim,
            "temperature": self.temperature,
            "seed": self.seed,
            "dataset_fingerprint": self.dataset_fingerprint,
            "final_loss": self.final_loss,
            "image_to_text_loss": self.image_to_text_loss,
            "text_to_image_loss": self.text_to_image_loss,
            "matched_similarity": self.matched_similarity,
            "unmatched_similarity": self.unmatched_similarity,
            "similarity_gap": self.similarity_gap,
            "training_history": [dict(h) for h in self.training_history],
            "retrieval_summary": self.retrieval_summary.to_dict(),
            "zero_shot_summary": self.zero_shot_summary.to_dict(),
            "prompt_sensitivity": dict(self.prompt_sensitivity),
            "explained_variance_ratio": list(self.explained_variance_ratio),
            "mean_paired_distance": self.mean_paired_distance,
            "mean_paired_cosine": self.mean_paired_cosine,
            "centroid_alignments": [ca.to_dict() for ca in self.centroid_alignments],
            "collapse_summary": self.collapse_summary.to_dict(),
            "robustness_summary": dict(self.robustness_summary),
            "failure_types": list(self.failure_types),
            "warnings": list(self.warnings),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> VisionLanguageRepresentationReport:
        """Instantiate report from dictionary."""
        return cls(
            experiment_id=str(data["experiment_id"]),
            visual_family=str(data["visual_family"]),
            visual_architecture=str(data["visual_architecture"]),
            text_dim=int(data["text_dim"]),
            shared_dim=int(data["shared_dim"]),
            temperature=float(data["temperature"]),
            seed=int(data["seed"]),
            dataset_fingerprint=str(data["dataset_fingerprint"]),
            final_loss=float(data["final_loss"]),
            image_to_text_loss=float(data["image_to_text_loss"]),
            text_to_image_loss=float(data["text_to_image_loss"]),
            matched_similarity=float(data["matched_similarity"]),
            unmatched_similarity=float(data["unmatched_similarity"]),
            similarity_gap=float(data["similarity_gap"]),
            training_history=[
                {str(k): float(v) for k, v in h.items()}
                for h in data["training_history"]
            ],
            retrieval_summary=CrossModalRetrievalSummary.from_dict(
                data["retrieval_summary"]
            ),
            zero_shot_summary=ZeroShotClassificationSummary.from_dict(
                data["zero_shot_summary"]
            ),
            prompt_sensitivity=dict(data.get("prompt_sensitivity", {})),
            explained_variance_ratio=[
                float(x) for x in data.get("explained_variance_ratio", [0.0, 0.0])
            ],
            mean_paired_distance=float(data["mean_paired_distance"]),
            mean_paired_cosine=float(data["mean_paired_cosine"]),
            centroid_alignments=[
                CrossModalCentroidAlignment.from_dict(ca)
                for ca in data.get("centroid_alignments", [])
            ],
            collapse_summary=MultimodalCollapseSummary.from_dict(
                data["collapse_summary"]
            ),
            robustness_summary=dict(data.get("robustness_summary", {})),
            failure_types=[str(f) for f in data.get("failure_types", [])],
            warnings=[str(w) for w in data.get("warnings", [])],
        )

    def to_json(self, indent: int = 2) -> str:
        """Serialize report to formatted JSON string."""
        return json.dumps(self.to_dict(), indent=indent)
