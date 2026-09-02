"""Structured serializable reporting contracts for self-supervised learning."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from prism.ssl.diagnostics import RepresentationCollapseSummary


class SupervisedVsSSLComparisonSummary(BaseModel):
    """Performance and geometry comparison of Supervised vs SSL representations."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    architecture: str = Field(..., description="Encoder architecture name")
    dataset_id: str = Field(..., description="Target evaluation dataset")
    supervised_accuracy: float = Field(
        ..., description="Downstream linear probe accuracy of supervised encoder"
    )
    ssl_accuracy: float = Field(
        ..., description="Downstream linear probe accuracy of SSL encoder"
    )
    scratch_accuracy: float = Field(
        ..., description="Target accuracy when trained from scratch baseline"
    )
    supervised_feature_std: float = Field(
        ..., description="Average feature standard deviation for supervised model"
    )
    ssl_feature_std: float = Field(
        ..., description="Average feature standard deviation for SSL model"
    )
    accuracy_gap_ssl_vs_supervised: float = Field(
        ..., description="ssl_accuracy - supervised_accuracy"
    )
    accuracy_gain_ssl_vs_scratch: float = Field(
        ..., description="ssl_accuracy - scratch_accuracy"
    )

    def to_dict(self) -> dict[str, Any]:
        """Convert summary to dictionary."""
        return self.model_dump()


class SelfSupervisedLearningReport(BaseModel):
    """Comprehensive serializable report capturing an entire SSL pretraining run."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    ssl_id: str = Field(..., description="Unique SSL run identifier")
    encoder_family: str = Field(..., description="Encoder model family")
    architecture: str = Field(..., description="Encoder architecture name")
    dataset_id: str = Field(..., description="Pretraining dataset identity")
    total_encoder_parameters: int = Field(
        ..., description="Number of trainable parameters in encoder backbone"
    )
    projection_head_parameters: int = Field(
        ..., description="Number of parameters in SimCLR projection head"
    )
    epochs: int = Field(..., description="Number of pretraining epochs executed")
    temperature: float = Field(..., description="NT-Xent temperature tau")
    loss_trajectory: list[float] = Field(
        default_factory=list, description="Epoch-by-epoch mean contrastive loss"
    )
    positive_similarity_trajectory: list[float] = Field(
        default_factory=list, description="Mean positive pair cosine similarity"
    )
    negative_similarity_trajectory: list[float] = Field(
        default_factory=list, description="Mean negative pair cosine similarity"
    )
    similarity_gap_trajectory: list[float] = Field(
        default_factory=list, description="positive_sim - negative_sim trajectory"
    )
    learning_rate_trajectory: list[float] = Field(
        default_factory=list, description="Learning rate schedule over epochs"
    )
    collapse_summary: RepresentationCollapseSummary = Field(
        ..., description="Final representation collapse diagnostics"
    )
    linear_probe_accuracy: float | None = Field(
        default=None, description="Downstream linear probe accuracy on target data"
    )
    supervised_probe_accuracy: float | None = Field(
        default=None, description="Supervised encoder linear probe accuracy baseline"
    )
    scratch_accuracy: float | None = Field(
        default=None, description="Scratch baseline accuracy on target data"
    )
    transfer_gain_vs_scratch: float | None = Field(
        default=None, description="linear_probe_accuracy - scratch_accuracy"
    )
    warnings: list[str] = Field(
        default_factory=list, description="Methodological and scientific warnings"
    )

    def to_dict(self) -> dict[str, Any]:
        """Convert report to dictionary."""
        return self.model_dump()
