"""Unit tests for reconstruction diagnostics, latent variance, and failure taxonomy."""

from prism.reconstruction.diagnostics import compute_reconstruction_diagnostics
from prism.reconstruction.enums import ReconstructionFailureCategory


def test_compute_reconstruction_diagnostics_patch_based() -> None:
    """Test diagnostics for patch-based representations."""
    # 2 samples, 4 patches, patch_dim=2
    preds = [
        [[1.0, 1.0], [2.0, 2.0], [0.0, 0.0], [1.0, 1.0]],
        [[0.5, 0.5], [1.5, 1.5], [0.0, 0.0], [0.8, 0.8]],
    ]
    targets = [
        [[1.0, 1.0], [2.0, 2.0], [0.0, 0.0], [1.0, 1.0]],  # Exact match
        [[0.5, 0.5], [1.5, 1.5], [1.0, 1.0], [0.8, 0.8]],  # Patch 2 err: (1-0)^2=1.0
    ]
    latents = [
        [0.1, 0.5, 0.9, -0.2],
        [-0.1, 0.3, 0.7, 0.4],
    ]

    report = compute_reconstruction_diagnostics(
        predictions=preds,
        targets=targets,
        latents=latents,
        is_patch_based=True,
    )

    assert report.mean_reconstruction_error >= 0.0
    assert len(report.per_sample_errors) == 2
    assert report.worst_patch_indices[0] == 2  # Patch 2 had the highest error
    assert report.latent_variance > 0.0
    assert 0.0 <= report.near_zero_variance_fraction <= 1.0


def test_compute_reconstruction_diagnostics_failure_taxonomy() -> None:
    """Verify failure taxonomy assignments under extreme failure conditions."""
    # Extreme error -> HIGH_RECONSTRUCTION_ERROR
    preds = [[[[10.0 for _ in range(4)] for _ in range(4)] for _ in range(3)]]
    targets = [[[[0.0 for _ in range(4)] for _ in range(4)] for _ in range(3)]]
    latents = [[0.0, 0.0, 0.0, 0.0]]

    report = compute_reconstruction_diagnostics(
        predictions=preds,
        targets=targets,
        latents=latents,
        is_patch_based=False,
    )

    fail_cats = report.failure_categories
    assert ReconstructionFailureCategory.HIGH_RECONSTRUCTION_ERROR in fail_cats
    assert ReconstructionFailureCategory.LOW_LATENT_VARIANCE in fail_cats
    assert len(report.diagnostics_notes) > 0
