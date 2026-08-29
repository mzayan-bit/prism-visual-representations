"""Smoke tests for representation geometry analysis and observatory service."""

from __future__ import annotations

from prism.api.geometry_service import (
    GeometryService,
    generate_observatory_demo_data,
)
from prism.representations.geometry import (
    RepresentationDataset,
)
from prism.representations.reports import analyze_representation_geometry


class TestSmokeRepresentationGeometry:
    """Smoke test verifying end-to-end geometric analysis across models and service."""

    def test_full_pipeline_smoke(self) -> None:
        # 1. Generate observatory demo data
        demo_payload = generate_observatory_demo_data(
            num_samples=18, num_classes=3, seed=123
        )

        assert "metadata" in demo_payload
        assert "comparison" in demo_payload
        assert "layer_profiles" in demo_payload
        assert "reports" in demo_payload

        # Validate metadata
        meta = demo_payload["metadata"]
        assert meta["experiment_id"] == "exp-observatory-demo"
        assert len(meta["architectures"]) == 3
        assert meta["num_classes"] == 3

        # Validate comparison
        comp = demo_payload["comparison"]
        assert "cnn" in comp["architectures"]
        assert "resnet" in comp["architectures"]
        assert "vit" in comp["architectures"]

        # Validate layer profiles
        profiles = demo_payload["layer_profiles"]
        assert len(profiles["cnn"]["layer_points"]) == 3
        assert len(profiles["resnet"]["layer_points"]) == 4
        assert len(profiles["vit"]["layer_points"]) == 4

        # Validate reports
        reports = demo_payload["reports"]
        assert "resnet::final_hidden" in reports
        r_resnet = reports["resnet::final_hidden"]
        assert r_resnet["num_samples"] == 18
        assert len(r_resnet["pca_projection"]["coordinates"]) == 18

    def test_geometry_service_cache_and_retrieval(self) -> None:
        service = GeometryService()

        # Build synthetic dataset
        ds = RepresentationDataset(
            experiment_id="service-smoke-exp",
            model_id="service-smoke-model",
            layer_name="layer_0",
            sample_ids=["s0", "s1", "s2"],
            labels=[0, 1, 0],
            vectors=[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
            feature_dim=2,
            num_samples=3,
            num_classes=2,
        )

        report = analyze_representation_geometry(ds, k=2)
        service.register_report(report, budget=0.5)

        retrieved = service.get_geometry_report(
            experiment_id="service-smoke-exp",
            model_id="service-smoke-model",
            layer_name="layer_0",
            budget=0.5,
        )

        assert retrieved is not None
        assert retrieved.experiment_id == "service-smoke-exp"
        assert retrieved.feature_dim == 2

        # Non-existent report
        missing = service.get_geometry_report("missing", "missing", "missing", 1.0)
        assert missing is None
