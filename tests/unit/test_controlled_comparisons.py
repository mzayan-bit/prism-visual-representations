"""Unit tests for ControlledComparison contracts and deterministic fingerprints."""

import pytest

from prism.core.errors import ValidationError
from prism.experiments.comparisons import ControlledComparison


@pytest.mark.unit
def test_controlled_comparison_fingerprint_deterministic() -> None:
    """Verify identical comparison definitions produce matching SHA-256 fingerprints."""
    comp1 = ControlledComparison(
        comparison_id="comp-linear-vs-mlp",
        name="Linear Softmax vs 2-Layer MLP",
        description="Comparing linear baseline against non-linear MLP",
        baseline_experiment_id="exp-cifar10-linear",
        candidate_experiment_id="exp-cifar10-mlp",
        varied_factors={"model_family": {"baseline": "linear", "candidate": "mlp"}},
        fixed_factors={"dataset": "cifar10", "seed": 42, "epochs": 50},
        dataset_fingerprint="ds_fp_123",
        seed=42,
    )
    comp2 = ControlledComparison(
        comparison_id="comp-linear-vs-mlp",
        name="Linear Softmax vs 2-Layer MLP",
        description="Comparing linear baseline against non-linear MLP",
        baseline_experiment_id="exp-cifar10-linear",
        candidate_experiment_id="exp-cifar10-mlp",
        varied_factors={"model_family": {"baseline": "linear", "candidate": "mlp"}},
        fixed_factors={"dataset": "cifar10", "seed": 42, "epochs": 50},
        dataset_fingerprint="ds_fp_123",
        seed=42,
    )
    assert comp1.compute_fingerprint() == comp2.compute_fingerprint()


@pytest.mark.unit
def test_controlled_comparison_fingerprint_changes_on_variation() -> None:
    """Verify changing varied factor changes the comparison fingerprint."""
    comp1 = ControlledComparison(
        comparison_id="comp-dropout-study",
        name="Dropout Study",
        baseline_experiment_id="exp-mlp-p0",
        candidate_experiment_id="exp-mlp-p02",
        varied_factors={"dropout": {"baseline": 0.0, "candidate": 0.2}},
        fixed_factors={"dataset": "cifar10", "seed": 42},
        dataset_fingerprint="ds_fp_123",
        seed=42,
    )
    comp2 = ControlledComparison(
        comparison_id="comp-dropout-study",
        name="Dropout Study",
        baseline_experiment_id="exp-mlp-p0",
        candidate_experiment_id="exp-mlp-p05",
        varied_factors={"dropout": {"baseline": 0.0, "candidate": 0.5}},
        fixed_factors={"dataset": "cifar10", "seed": 42},
        dataset_fingerprint="ds_fp_123",
        seed=42,
    )
    assert comp1.compute_fingerprint() != comp2.compute_fingerprint()


@pytest.mark.unit
def test_controlled_comparison_json_roundtrip() -> None:
    """Verify ControlledComparison serializes to and deserializes from JSON."""
    comp = ControlledComparison(
        comparison_id="comp-linear-vs-mlp",
        name="Linear vs MLP",
        baseline_experiment_id="exp-linear",
        candidate_experiment_id="exp-mlp",
        varied_factors={"hidden_dims": {"baseline": [], "candidate": [128]}},
        fixed_factors={"dataset": "cifar10"},
        dataset_fingerprint="fp_abc",
        seed=42,
    )
    json_str = comp.to_json()
    reconstructed = ControlledComparison.from_json(json_str)
    assert reconstructed == comp


@pytest.mark.unit
def test_controlled_comparison_invalid_id() -> None:
    """Verify invalid identifiers raise ValidationError."""
    with pytest.raises(ValidationError):
        ControlledComparison(
            comparison_id="Invalid ID With Spaces",
            name="Invalid Comparison",
            baseline_experiment_id="exp-linear",
            candidate_experiment_id="exp-mlp",
            varied_factors={},
            fixed_factors={},
            dataset_fingerprint="fp",
            seed=42,
        )
