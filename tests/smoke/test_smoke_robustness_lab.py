"""Smoke tests for the robustness laboratory subsystem and service."""

from __future__ import annotations

from prism.api.robustness_service import (
    RobustnessService,
    generate_robustness_demo_data,
)


class TestSmokeRobustnessLab:
    """Smoke test verifying robustness evaluations and laboratory service."""

    def test_full_pipeline_smoke(self) -> None:
        # 1. Generate robustness demo data across CNN, ResNet, and ViT
        demo_payload = generate_robustness_demo_data(
            num_samples=18, num_classes=3, seed=123
        )

        assert "metadata" in demo_payload
        assert "comparison" in demo_payload
        assert "reports" in demo_payload

        # Validate metadata
        meta = demo_payload["metadata"]
        assert meta["experiment_id"] == "exp-robustness-demo"
        assert len(meta["architectures"]) == 3
        assert len(meta["corruption_types"]) == 6
        assert meta["severities"] == [1, 2, 3, 4, 5]

        # Validate comparison
        comp = demo_payload["comparison"]
        assert "cnn" in comp["architectures"]
        assert "resnet" in comp["architectures"]
        assert "vit" in comp["architectures"]
        assert comp["architectures"]["cnn"]["mean_corrupted_accuracy"] >= 0.0

        # Validate detailed reports
        reports = demo_payload["reports"]
        assert "cnn" in reports
        assert "resnet" in reports
        assert "vit" in reports

        r_resnet = reports["resnet"]
        assert r_resnet["num_samples"] == 18
        assert "gaussian_noise::sev3" in r_resnet["evaluations"]
        assert len(r_resnet["severity_curves"]) == 6

        # ViT attention drift validation
        r_vit = reports["vit"]
        ev_vit = r_vit["evaluations"]["gaussian_noise::sev1"]
        assert ev_vit["attention_drift"] is not None
        assert ev_vit["attention_drift"]["num_layers"] == 2

    def test_robustness_service_cache_and_retrieval(self) -> None:
        service = RobustnessService()

        demo_payload = generate_robustness_demo_data(
            num_samples=6, num_classes=2, seed=42
        )
        r_dict = demo_payload["reports"]["resnet"]

        from prism.robustness.evaluation import RobustnessExperimentReport

        report = RobustnessExperimentReport.from_dict(r_dict)

        service.register_report(report, budget=1.0)
        retrieved = service.get_report(
            experiment_id=report.experiment_id,
            model_id=report.model_id,
            layer_name=report.layer_name,
            budget=1.0,
        )

        assert retrieved is not None
        assert retrieved.model_id == report.model_id
        assert retrieved.num_samples == 6

        # Non-existent report returns None
        assert service.get_report("none", "none", "none", 1.0) is None
