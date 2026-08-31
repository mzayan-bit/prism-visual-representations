"""Smoke tests for Explainability & Visual Attribution Laboratory subsystem."""

from __future__ import annotations

import json

from prism.api.explainability_service import (
    ExplainabilityDemoPayload,
    ExplainabilityService,
    generate_explainability_demo_data,
)


class TestSmokeExplainabilityLab:
    """Smoke test suite verifying end-to-end explainability generation and service."""

    def test_full_pipeline_explainability_smoke(self) -> None:
        """Test complete generation of explainability datasets across architectures."""
        payload = generate_explainability_demo_data()

        # 1. Metadata checks
        meta = payload.metadata
        assert meta.experiment_id == "exp_phase16_explainability_suite"
        assert len(meta.architectures) == 3
        assert set(meta.architectures) == {"cnn", "resnet", "vit"}
        assert len(meta.class_names) == 5
        assert len(meta.sample_ids) == 5

        # 2. Sample payloads
        assert len(payload.samples) == 5
        sample0 = payload.samples[0]
        assert sample0.sample_id == "sample_001_airplane"
        assert sample0.true_class == 0
        assert sample0.class_name == "airplane"

        # Check prediction dictionaries for all architectures
        for arch in ["cnn", "resnet", "vit"]:
            assert arch in sample0.predictions
            assert "predicted_class" in sample0.predictions[arch]
            assert "confidence" in sample0.predictions[arch]
            assert arch in sample0.attributions

            # Method attributions
            arch_attrs = sample0.attributions[arch]
            assert "input_gradient" in arch_attrs
            assert "gradient_x_input" in arch_attrs
            assert "occlusion_sensitivity" in arch_attrs
            if arch in ("cnn", "resnet"):
                assert "grad_cam" in arch_attrs
            elif arch == "vit":
                assert "vit_attention" in arch_attrs

            # Comparison reports
            assert arch in sample0.comparison_reports
            comp_rep = sample0.comparison_reports[arch]
            assert len(comp_rep.results) >= 4
            assert comp_rep.mean_cross_method_agreement >= 0.0

            # Drift summaries under corruption
            assert arch in sample0.drift_summaries
            drift_dict = sample0.drift_summaries[arch]
            assert "input_gradient" in drift_dict
            assert drift_dict["input_gradient"].attribution_cosine_similarity >= -1.0

            # Failure taxonomy flags
            assert arch in sample0.failure_flags

    def test_json_serialization_roundtrip(self) -> None:
        """Test ExplainabilityDemoPayload serializes and deserializes."""
        payload = generate_explainability_demo_data()
        payload_dict = payload.to_dict()

        # Serialize to JSON string
        json_str = json.dumps(payload_dict)
        deserialized_dict = json.loads(json_str)

        # Deserialize back to Pydantic models
        roundtrip_payload = ExplainabilityDemoPayload.model_validate(deserialized_dict)
        assert (
            roundtrip_payload.metadata.experiment_id == payload.metadata.experiment_id
        )
        assert len(roundtrip_payload.samples) == len(payload.samples)
        assert roundtrip_payload.samples[0].sample_id == payload.samples[0].sample_id

    def test_explainability_service_cache_and_retrieval(self) -> None:
        """Test ExplainabilityService registering, caching, and querying samples."""
        service = ExplainabilityService()
        payload = generate_explainability_demo_data()

        service.register_demo_payload(payload)

        cached_meta = service.get_metadata()
        assert cached_meta is not None
        assert cached_meta.experiment_id == "exp_phase16_explainability_suite"

        all_samples = service.get_all_samples()
        assert len(all_samples) == 5

        s1 = service.get_sample("sample_001_airplane")
        assert s1 is not None
        assert s1.sample_id == "sample_001_airplane"

        non_existent = service.get_sample("non_existent_sample")
        assert non_existent is None
