"""Unit tests for VisionTransformer controlled comparisons."""

from prism.experiments.comparisons import create_vit_architecture_comparison


def test_vit_architecture_comparison_creation() -> None:
    """Test creating an auditable ControlledComparison across ViT factors."""
    comp = create_vit_architecture_comparison(
        comparison_id="comp-vit-depth-2-vs-4",
        name="ViT Depth Comparison",
        baseline_experiment_id="exp-vit-depth2",
        candidate_experiment_id="exp-vit-depth4",
        dataset_fingerprint="sha256_mock_dataset_fingerprint",
        varied_hyperparameters={
            "depth": (2, 4),
            "num_heads": (2, 4),
        },
        seed=42,
        fixed_factors={
            "patch_size": 4,
            "embed_dim": 32,
            "optimizer": "sgd",
            "lr": 0.01,
        },
    )

    assert comp.comparison_id == "comp-vit-depth-2-vs-4"
    assert comp.varied_factors["depth"] == {"baseline": 2, "candidate": 4}
    assert comp.varied_factors["num_heads"] == {"baseline": 2, "candidate": 4}
    assert comp.fixed_factors["patch_size"] == 4
    assert comp.seed == 42

    # Deterministic fingerprinting
    fp1 = comp.compute_fingerprint()
    fp2 = comp.compute_fingerprint()
    assert fp1 == fp2
    assert len(fp1) == 64

    # Serialization test
    d = comp.to_dict()
    restored = comp.from_dict(d)
    assert restored.comparison_id == comp.comparison_id
