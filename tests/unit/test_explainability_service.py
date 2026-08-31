"""Unit tests for ExplainabilityService and API endpoints."""

from prism.api.explainability_service import (
    ExplainabilityService,
    generate_explainability_demo_data,
)
from prism.explainability.attribution import AttributionMethod


def test_service_supported_methods() -> None:
    """Test resolution of supported attribution methods per architecture."""
    cnn_methods = ExplainabilityService.get_supported_methods("cnn")
    assert AttributionMethod.GRAD_CAM in cnn_methods
    assert AttributionMethod.VIT_ATTENTION not in cnn_methods

    resnet_methods = ExplainabilityService.get_supported_methods("resnet")
    assert AttributionMethod.GRAD_CAM in resnet_methods
    assert AttributionMethod.VIT_ATTENTION not in resnet_methods

    vit_methods = ExplainabilityService.get_supported_methods("vit")
    assert AttributionMethod.VIT_ATTENTION in vit_methods
    assert AttributionMethod.GRAD_CAM not in vit_methods


def test_service_available_layers() -> None:
    """Test available spatial layers listing across architectures."""
    cnn_layers = ExplainabilityService.get_available_layers("cnn")
    assert "final_conv" in cnn_layers

    resnet_layers = ExplainabilityService.get_available_layers("resnet")
    assert "final_stage" in resnet_layers

    vit_layers = ExplainabilityService.get_available_layers("vit")
    assert "last_block" in vit_layers


def test_service_payload_registration_and_querying() -> None:
    """Test registering and querying demo payloads in ExplainabilityService."""
    service = ExplainabilityService()
    assert service.get_metadata() is None
    assert service.get_all_samples() == []
    assert service.get_sample("sample_001_airplane") is None

    payload = generate_explainability_demo_data()
    service.register_demo_payload(payload)

    meta = service.get_metadata()
    assert meta is not None
    assert meta.num_classes == 5
    assert len(meta.sample_ids) == 5

    samples = service.get_all_samples()
    assert len(samples) == 5

    sample = service.get_sample("sample_001_airplane")
    assert sample is not None
    assert sample.class_name == "airplane"
