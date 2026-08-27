"""Unit tests for ControlledComparison between Linear, MLP, and CNN models."""

import pytest

from prism.experiments.comparisons import ControlledComparison


@pytest.mark.unit
def test_linear_vs_mlp_vs_cnn_comparison_fingerprints() -> None:
    """Verify distinct fingerprints for Linear vs MLP and MLP vs CNN comparisons."""
    comp_linear_mlp = ControlledComparison(
        comparison_id="comp-linear-vs-mlp",
        name="Linear Softmax vs MLP",
        baseline_experiment_id="exp-cifar10-linear",
        candidate_experiment_id="exp-cifar10-mlp",
        varied_factors={"model_family": {"baseline": "linear", "candidate": "mlp"}},
        fixed_factors={"dataset": "cifar10", "seed": 42, "epochs": 50},
        dataset_fingerprint="ds_fp_123",
        seed=42,
    )

    comp_mlp_cnn = ControlledComparison(
        comparison_id="comp-mlp-vs-cnn",
        name="MLP vs CNN",
        baseline_experiment_id="exp-cifar10-mlp",
        candidate_experiment_id="exp-cifar10-cnn",
        varied_factors={
            "model_family": {"baseline": "mlp", "candidate": "cnn"},
            "spatial_inductive_bias": {"baseline": False, "candidate": True},
        },
        fixed_factors={"dataset": "cifar10", "seed": 42, "epochs": 50},
        dataset_fingerprint="ds_fp_123",
        seed=42,
    )

    assert comp_linear_mlp.compute_fingerprint() != comp_mlp_cnn.compute_fingerprint()


@pytest.mark.unit
def test_cnn_channel_variation_changes_fingerprint() -> None:
    """Verify modifying CNN channel progression changes comparison fingerprint."""
    comp1 = ControlledComparison(
        comparison_id="comp-cnn-width",
        name="CNN Width Study",
        baseline_experiment_id="exp-cnn-small",
        candidate_experiment_id="exp-cnn-large",
        varied_factors={"conv_channels": {"baseline": [16, 32], "candidate": [32, 64]}},
        fixed_factors={"dataset": "cifar10", "seed": 42},
        dataset_fingerprint="ds_fp_123",
        seed=42,
    )

    comp2 = ControlledComparison(
        comparison_id="comp-cnn-width",
        name="CNN Width Study",
        baseline_experiment_id="exp-cnn-small",
        candidate_experiment_id="exp-cnn-large",
        varied_factors={
            "conv_channels": {"baseline": [16, 32], "candidate": [64, 128]}
        },
        fixed_factors={"dataset": "cifar10", "seed": 42},
        dataset_fingerprint="ds_fp_123",
        seed=42,
    )

    assert comp1.compute_fingerprint() != comp2.compute_fingerprint()
