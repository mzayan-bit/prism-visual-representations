"""Unit tests for controlled comparisons involving normalization."""

import pytest

from prism.experiments.comparisons import (
    ControlledComparison,
    create_normalization_comparison,
)


@pytest.mark.unit
def test_create_normalization_comparison_helper() -> None:
    """Verify helper constructs ControlledComparison isolating normalization."""
    comp = create_normalization_comparison(
        comparison_id="comp-cnn-vs-norm-cnn",
        name="CNN vs Normalized CNN Study",
        baseline_experiment_id="exp-cifar10-cnn-vanilla",
        candidate_experiment_id="exp-cifar10-cnn-bn",
        dataset_fingerprint="ds_fp_abc123",
        seed=42,
        normalization_type="batch_norm",
        norm_eps=1e-5,
        norm_momentum=0.1,
        norm_affine=True,
    )

    assert isinstance(comp, ControlledComparison)
    assert comp.varied_factors["normalization"] == {
        "baseline": "none",
        "candidate": "batch_norm",
    }
    assert comp.varied_factors["norm_eps"] == {
        "baseline": None,
        "candidate": 1e-5,
    }
    assert comp.fixed_factors["dataset_fingerprint"] == "ds_fp_abc123"
    assert comp.seed == 42


@pytest.mark.unit
def test_normalization_comparison_fingerprint_sensitivity() -> None:
    """Verify modifying normalization parameters changes comparison fingerprint."""
    comp1 = create_normalization_comparison(
        comparison_id="comp-norm-study",
        name="Norm Study",
        baseline_experiment_id="exp-baseline",
        candidate_experiment_id="exp-candidate",
        dataset_fingerprint="ds_fp_1",
        seed=42,
        norm_eps=1e-5,
    )

    comp2 = create_normalization_comparison(
        comparison_id="comp-norm-study",
        name="Norm Study",
        baseline_experiment_id="exp-baseline",
        candidate_experiment_id="exp-candidate",
        dataset_fingerprint="ds_fp_1",
        seed=42,
        norm_eps=1e-4,  # different eps
    )

    assert comp1.compute_fingerprint() != comp2.compute_fingerprint()
