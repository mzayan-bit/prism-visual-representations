"""Unit tests for ReconstructionService and demo data generation."""

from prism.api.reconstruction_service import (
    ReconstructionService,
    generate_reconstruction_demo_data,
)


def test_generate_reconstruction_demo_data() -> None:
    """Test demo data generator contents and schema completeness."""
    data = generate_reconstruction_demo_data()
    assert "metadata" in data
    assert "triplets_masked_patch" in data
    assert "triplets_denoising" in data
    assert "dynamics" in data
    assert "masking_ratio_study" in data
    assert "three_way_comparison" in data
    assert "layer_probes" in data
    assert "failure_cases" in data

    assert len(data["triplets_masked_patch"]) > 0
    assert len(data["masking_ratio_study"]) == 3
    assert len(data["three_way_comparison"]) == 3


def test_reconstruction_service_queries() -> None:
    """Test ReconstructionService interface methods."""
    service = ReconstructionService()
    meta = service.get_metadata()
    assert meta["experiment_id"] == "recon_benchmark_v1"

    triplets_mim = service.get_triplets("masked_patch_reconstruction")
    assert len(triplets_mim) > 0

    triplets_dae = service.get_triplets("denoising_autoencoder")
    assert len(triplets_dae) > 0

    dynamics = service.get_dynamics()
    assert len(dynamics["epochs"]) == 20

    ratios = service.get_masking_ratio_study()
    assert len(ratios) == 3

    comparison = service.get_three_way_comparison()
    assert len(comparison) == 3

    layer_probes = service.get_layer_probes()
    assert len(layer_probes) > 0

    failures = service.get_failure_cases()
    assert len(failures) > 0
