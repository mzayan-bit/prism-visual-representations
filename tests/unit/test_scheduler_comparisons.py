"""Unit tests for scheduler controlled comparisons."""

import pytest

from prism.experiments.comparisons import create_scheduler_comparison


@pytest.mark.unit
def test_create_scheduler_comparison_contracts() -> None:
    """Verify create_scheduler_comparison generates valid ControlledComparison."""
    comp = create_scheduler_comparison(
        comparison_id="comp-lr-const-vs-cos",
        name="Constant vs Cosine Learning Rate Schedule",
        baseline_experiment_id="exp-lr-constant",
        candidate_experiment_id="exp-lr-cosine",
        baseline_scheduler_type="constant",
        candidate_scheduler_type="cosine",
        baseline_scheduler_params={"min_lr": 0.0},
        candidate_scheduler_params={"min_lr": 0.001, "warmup_epochs": 5},
        dataset_fingerprint="sha256:dataset-fp",
        seed=42,
    )

    assert comp.comparison_id == "comp-lr-const-vs-cos"
    assert comp.varied_factors["scheduler_type"]["baseline"] == "constant"
    assert comp.varied_factors["scheduler_type"]["candidate"] == "cosine"
    assert comp.fixed_factors["dataset_fingerprint"] == "sha256:dataset-fp"
    assert comp.fixed_factors["seed"] == 42

    fp1 = comp.compute_fingerprint()
    assert isinstance(fp1, str)
    assert len(fp1) == 64


@pytest.mark.unit
def test_scheduler_comparison_fingerprint_sensitivity() -> None:
    """Verify changing scheduler strategy alters the comparison fingerprint."""
    comp1 = create_scheduler_comparison(
        comparison_id="comp-lr-const-vs-cos",
        name="Constant vs Cosine",
        baseline_experiment_id="exp-base",
        candidate_experiment_id="exp-cand",
        baseline_scheduler_type="constant",
        candidate_scheduler_type="cosine",
        dataset_fingerprint="sha256:dataset-fp",
        seed=42,
    )

    comp2 = create_scheduler_comparison(
        comparison_id="comp-lr-const-vs-cos",
        name="Constant vs Step",
        baseline_experiment_id="exp-base",
        candidate_experiment_id="exp-cand",
        baseline_scheduler_type="constant",
        candidate_scheduler_type="step",
        dataset_fingerprint="sha256:dataset-fp",
        seed=42,
    )

    comp1_dup = create_scheduler_comparison(
        comparison_id="comp-lr-const-vs-cos",
        name="Constant vs Cosine",
        baseline_experiment_id="exp-base",
        candidate_experiment_id="exp-cand",
        baseline_scheduler_type="constant",
        candidate_scheduler_type="cosine",
        dataset_fingerprint="sha256:dataset-fp",
        seed=42,
    )

    assert comp1.compute_fingerprint() != comp2.compute_fingerprint()
    assert comp1.compute_fingerprint() == comp1_dup.compute_fingerprint()
