"""Unit tests for controlled comparisons involving residual learning."""

import pytest

from prism.experiments.comparisons import (
    ControlledComparison,
    create_residual_comparison,
)


@pytest.mark.unit
def test_create_residual_comparison_helper() -> None:
    """Verify helper constructs valid ControlledComparison."""
    comp = create_residual_comparison(
        comparison_id="comp-plain-vs-resnet",
        name="Plain vs Residual CNN Study",
        baseline_experiment_id="exp-plain-cnn",
        candidate_experiment_id="exp-resnet",
        dataset_fingerprint="ds_fp_abc123",
        seed=42,
        stage_widths=[16, 32],
        blocks_per_stage=[2, 2],
        normalization="batch_norm",
    )

    assert isinstance(comp, ControlledComparison)
    assert comp.varied_factors["model_family"] == {
        "baseline": "cnn",
        "candidate": "resnet",
    }
    assert comp.varied_factors["has_skip_connections"] == {
        "baseline": False,
        "candidate": True,
    }
    assert comp.fixed_factors["stage_widths"] == [16, 32]
    assert comp.fixed_factors["blocks_per_stage"] == [2, 2]
    assert comp.seed == 42


@pytest.mark.unit
def test_residual_comparison_fingerprint_sensitivity() -> None:
    """Verify changing residual hyperparameters changes comparison fingerprint."""
    comp1 = create_residual_comparison(
        comparison_id="comp-res-study",
        name="Res Study",
        baseline_experiment_id="exp-base",
        candidate_experiment_id="exp-cand",
        dataset_fingerprint="ds_fp_1",
        seed=42,
        blocks_per_stage=[1, 1],
    )

    comp2 = create_residual_comparison(
        comparison_id="comp-res-study",
        name="Res Study",
        baseline_experiment_id="exp-base",
        candidate_experiment_id="exp-cand",
        dataset_fingerprint="ds_fp_1",
        seed=42,
        blocks_per_stage=[2, 2],  # different depth
    )

    assert comp1.compute_fingerprint() != comp2.compute_fingerprint()
