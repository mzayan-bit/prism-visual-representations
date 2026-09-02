"""Reconstruction experiment reports and multi-objective comparison contracts."""

from __future__ import annotations

import datetime

from pydantic import BaseModel, ConfigDict, Field

from prism.core.enums import ModelFamily
from prism.reconstruction.diagnostics import ReconstructionDiagnosticsReport
from prism.reconstruction.enums import ReconstructionMethod
from prism.robustness.corruptions import CorruptionType


class ReconstructionLearningReport(BaseModel):
    """Report documenting a reconstruction learning pretraining experiment."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    reconstruction_id: str = Field(..., description="Unique experiment ID")
    method: ReconstructionMethod = Field(..., description="Reconstruction objective")
    encoder_family: ModelFamily = Field(..., description="Encoder architecture family")
    dataset_id: str = Field(..., description="Source dataset used for pretraining")
    mask_ratio: float | None = Field(
        default=None, description="Applied masking ratio if masked modeling"
    )
    corruption_type: CorruptionType | None = Field(
        default=None, description="Applied corruption if denoising"
    )
    corruption_severity: int | None = Field(
        default=None, description="Severity of corruption if denoising"
    )
    epochs_trained: int = Field(..., description="Total completed pretraining epochs")
    loss_history: list[float] = Field(
        ..., description="Total reconstruction loss per epoch"
    )
    masked_mse_history: list[float] = Field(
        ..., description="Masked-region MSE per epoch"
    )
    learning_rate_history: list[float] = Field(
        ..., description="Learning rate per epoch"
    )
    diagnostics: ReconstructionDiagnosticsReport = Field(
        ..., description="Comprehensive reconstruction and latent diagnostics"
    )
    downstream_linear_probe_accuracy: float | None = Field(
        default=None,
        description="Test accuracy of linear probe trained on frozen representations",
    )
    encoder_snapshot_id: str = Field(
        ..., description="Identifier of the saved encoder model state snapshot"
    )
    parameter_checksum: str = Field(
        ..., description="SHA-256 checksum of pretrained encoder parameters"
    )
    created_at_utc: str = Field(
        default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat()
    )


class SupervisedVsSSLVsReconstructionSummary(BaseModel):
    """Comparison among the three representation learning paradigms."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    encoder_family: ModelFamily = Field(
        ..., description="Common encoder architecture family compared"
    )
    supervised_probe_accuracy: float = Field(
        ..., description="Downstream linear probe accuracy for supervised pretraining"
    )
    simclr_probe_accuracy: float = Field(
        ...,
        description="Downstream probe accuracy for SimCLR pretraining",
    )
    reconstruction_probe_accuracy: float = Field(
        ...,
        description="Downstream probe accuracy for reconstruction pretraining",
    )
    supervised_latent_std: float = Field(
        ...,
        description="Mean representation standard deviation for supervised features",
    )
    simclr_latent_std: float = Field(
        ..., description="Mean representation standard deviation for SimCLR features"
    )
    reconstruction_latent_std: float = Field(
        ...,
        description="Mean representation standard deviation for reconstruction",
    )
    supervised_class_separation: float = Field(
        ...,
        description="Inter-class centroid separation for supervised representations",
    )
    simclr_class_separation: float = Field(
        ..., description="Inter-class centroid separation for SimCLR representations"
    )
    reconstruction_class_separation: float = Field(
        ...,
        description="Inter-class centroid separation for reconstruction",
    )
    supervised_class_compactness: float = Field(
        ..., description="Intra-class variance for supervised representations"
    )
    simclr_class_compactness: float = Field(
        ..., description="Intra-class variance for SimCLR representations"
    )
    reconstruction_class_compactness: float = Field(
        ..., description="Intra-class variance for reconstruction representations"
    )
    analysis_notes: list[str] = Field(
        default_factory=list,
        description="Factual comparative analysis of representation properties",
    )
