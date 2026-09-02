"""Reconstruction diagnostics, patch error metrics, and failure taxonomy."""

from __future__ import annotations

import math
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from prism.reconstruction.enums import ReconstructionFailureCategory


class ReconstructionDiagnosticsReport(BaseModel):
    """Immutable report of reconstruction diagnostics and failure pattern analysis."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    mean_reconstruction_error: float = Field(
        ..., description="Mean reconstruction MSE across all evaluated samples"
    )
    per_sample_errors: list[float] = Field(
        ..., description="Reconstruction MSE for each individual sample"
    )
    per_channel_errors: list[float] = Field(
        ..., description="Per-channel reconstruction MSE"
    )
    worst_patch_indices: list[int] = Field(
        default_factory=list,
        description="Patch indices with highest reconstruction error",
    )
    worst_patch_errors: list[float] = Field(
        default_factory=list,
        description="MSE corresponding to worst reconstructed patches",
    )
    latent_mean_std: float = Field(
        ...,
        description="Mean standard deviation across latent representation dimensions",
    )
    latent_variance: float = Field(
        ..., description="Average per-dimension variance of latent representations"
    )
    near_zero_variance_fraction: float = Field(
        ..., description="Fraction of latent dimensions with variance < 1e-4"
    )
    failure_categories: list[ReconstructionFailureCategory] = Field(
        default_factory=list,
        description="Identified failure taxonomy categories for this experiment",
    )
    diagnostics_notes: list[str] = Field(
        default_factory=list,
        description="Descriptive scientific observations without causal bias",
    )


def compute_reconstruction_diagnostics(
    predictions: list[Any],
    targets: list[Any],
    latents: list[list[float]],
    is_patch_based: bool = True,
    corrupted_inputs: list[Any] | None = None,
) -> ReconstructionDiagnosticsReport:
    """Compute reconstruction diagnostics, error profiles, and failure patterns.

    Parameters
    ----------
    predictions : list[Any]
        Model reconstructed outputs (patch sequences or image tensors).
    targets : list[Any]
        Clean ground-truth reference targets.
    latents : list[list[float]]
        Encoder latent representations for all samples [N x D].
    is_patch_based : bool
        True if predictions are patch tokens, False if image tensors.
    corrupted_inputs : list[Any] | None
        Optional corrupted inputs for denoising recovery comparison.

    Returns
    -------
    ReconstructionDiagnosticsReport
        Detailed diagnostic metrics and categorized failure taxonomy.
    """
    n_samples = len(predictions)
    per_sample_errors: list[float] = []
    worst_patch_indices: list[int] = []
    worst_patch_errors: list[float] = []

    if is_patch_based:
        # predictions: [N x T x D_patch]
        n_patches = len(predictions[0])
        patch_dim = len(predictions[0][0])
        total_patch_errors = [0.0] * n_patches

        for n in range(n_samples):
            pred_s = predictions[n]
            tgt_s = targets[n]
            sample_sq_err = 0.0
            for t in range(n_patches):
                p_vec = pred_s[t]
                t_vec = tgt_s[t]
                p_err = sum(
                    (p_vec[d] - t_vec[d]) ** 2 for d in range(patch_dim)
                ) / float(patch_dim)
                total_patch_errors[t] += p_err
                sample_sq_err += p_err
            per_sample_errors.append(sample_sq_err / float(n_patches))

        mean_patch_errors = [e / float(n_samples) for e in total_patch_errors]
        ranked_patches = sorted(
            enumerate(mean_patch_errors), key=lambda x: x[1], reverse=True
        )
        worst_patch_indices = [idx for idx, _ in ranked_patches[:5]]
        worst_patch_errors = [err for _, err in ranked_patches[:5]]
        per_channel_errors = [sum(per_sample_errors) / float(n_samples)]
    else:
        # predictions: [N x C x H x W]
        c = len(predictions[0])
        h = len(predictions[0][0])
        w = len(predictions[0][0][0])
        total_pix_per_sample = c * h * w
        ch_errors = [0.0] * c

        for n in range(n_samples):
            pred_img = predictions[n]
            tgt_img = targets[n]
            sample_sq_err = 0.0
            for ch in range(c):
                ch_sq = 0.0
                for r in range(h):
                    for col in range(w):
                        diff = pred_img[ch][r][col] - tgt_img[ch][r][col]
                        sq = diff * diff
                        sample_sq_err += sq
                        ch_sq += sq
                ch_errors[ch] += ch_sq / float(h * w)
            per_sample_errors.append(sample_sq_err / float(total_pix_per_sample))

        per_channel_errors = [e / float(n_samples) for e in ch_errors]

    mean_rec_error = sum(per_sample_errors) / float(n_samples) if n_samples > 0 else 0.0

    # Latent variance and collapse diagnostics
    latent_dim = len(latents[0]) if latents and latents[0] else 0
    variances: list[float] = []
    stds: list[float] = []
    near_zero_count = 0

    if n_samples > 1 and latent_dim > 0:
        for d in range(latent_dim):
            col_vals = [latents[n][d] for n in range(n_samples)]
            col_mean = sum(col_vals) / float(n_samples)
            col_var = sum((v - col_mean) ** 2 for v in col_vals) / float(n_samples - 1)
            variances.append(col_var)
            col_std = math.sqrt(col_var)
            stds.append(col_std)
            if col_var < 1e-4:
                near_zero_count += 1
    elif latent_dim > 0:
        variances = [0.0] * latent_dim
        stds = [0.0] * latent_dim

    latent_mean_std = sum(stds) / float(latent_dim) if latent_dim > 0 else 0.0
    latent_var = sum(variances) / float(latent_dim) if latent_dim > 0 else 0.0
    near_zero_frac = (
        float(near_zero_count) / float(latent_dim) if latent_dim > 0 else 0.0
    )

    # Determine failure taxonomy
    failures: list[ReconstructionFailureCategory] = []
    notes: list[str] = []

    if mean_rec_error > 0.35:
        failures.append(ReconstructionFailureCategory.HIGH_RECONSTRUCTION_ERROR)
        notes.append(
            f"High mean reconstruction MSE ({mean_rec_error:.4f} > 0.35 threshold)."
        )

    if worst_patch_errors and worst_patch_errors[0] > 2.5 * mean_rec_error:
        failures.append(ReconstructionFailureCategory.LOCALIZED_PATCH_FAILURE)
        notes.append(
            f"Worst patch {worst_patch_indices[0]} error ({worst_patch_errors[0]:.4f}) "
            f"exceeds 2.5x mean error."
        )

    if latent_var < 0.01 or near_zero_frac > 0.5:
        failures.append(ReconstructionFailureCategory.LOW_LATENT_VARIANCE)
        notes.append(
            f"Low latent variance ({latent_var:.4f}) with {near_zero_frac * 100:.1f}% "
            "collapsed dimensions."
        )

    if corrupted_inputs is not None and not is_patch_based:
        # Compare denoising error to corruption error
        corr_err = 0.0
        c = len(targets[0])
        h = len(targets[0][0])
        w = len(targets[0][0][0])
        tot_pix = n_samples * c * h * w
        for n in range(n_samples):
            for ch in range(c):
                for r in range(h):
                    for col in range(w):
                        diff = corrupted_inputs[n][ch][r][col] - targets[n][ch][r][col]
                        corr_err += diff * diff
        mean_corr_err = corr_err / float(tot_pix)
        if mean_rec_error >= mean_corr_err:
            failures.append(ReconstructionFailureCategory.CORRUPTION_RECOVERY_FAILURE)
            notes.append(
                f"Denoising MSE ({mean_rec_error:.4f}) did not improve over input "
                f"corruption MSE ({mean_corr_err:.4f})."
            )

    return ReconstructionDiagnosticsReport(
        mean_reconstruction_error=mean_rec_error,
        per_sample_errors=per_sample_errors,
        per_channel_errors=per_channel_errors,
        worst_patch_indices=worst_patch_indices,
        worst_patch_errors=worst_patch_errors,
        latent_mean_std=latent_mean_std,
        latent_variance=latent_var,
        near_zero_variance_fraction=near_zero_frac,
        failure_categories=failures,
        diagnostics_notes=notes,
    )
