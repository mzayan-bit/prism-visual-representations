"""Enumerations for generative and reconstruction-based representation learning."""

from enum import Enum


class ReconstructionMethod(str, Enum):
    """Supported visual reconstruction objectives and pretraining paradigms."""

    MASKED_PATCH_RECONSTRUCTION = "masked_patch_reconstruction"
    DENOISING_AUTOENCODER = "denoising_autoencoder"
    IDENTITY_AUTOENCODER = "identity_autoencoder"


class ReconstructionFailureCategory(str, Enum):
    """Taxonomy of failure patterns in visual reconstruction."""

    HIGH_RECONSTRUCTION_ERROR = "high_reconstruction_error"
    LOCALIZED_PATCH_FAILURE = "localized_patch_failure"
    LOW_LATENT_VARIANCE = "low_latent_variance"
    OVER_SMOOTH_RECONSTRUCTION = "over_smooth_reconstruction"
    CORRUPTION_RECOVERY_FAILURE = "corruption_recovery_failure"
