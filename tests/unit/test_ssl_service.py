"""Unit tests for SelfSupervisedService and demo benchmark data."""

from prism.api.ssl_service import SelfSupervisedService, generate_ssl_demo_data
from prism.ssl.reports import SelfSupervisedLearningReport


def test_ssl_demo_payload_generation() -> None:
    """Test full SSL demo dataset generation and JSON serialization."""
    payload = generate_ssl_demo_data()

    assert payload.metadata.experiment_id == "exp_phase18_ssl_suite"
    assert "cnn" in payload.metadata.architectures
    assert "resnet" in payload.metadata.architectures
    assert "vit" in payload.metadata.architectures

    # Check reports exist for all 3 architectures
    assert "cnn" in payload.reports
    assert "resnet" in payload.reports
    assert "vit" in payload.reports

    # Check comparisons
    assert "cnn" in payload.comparisons
    assert "resnet" in payload.comparisons
    assert "vit" in payload.comparisons

    # Check label efficiency
    assert len(payload.label_efficiency["cnn"]) >= 4

    # Check layer probes
    assert len(payload.layer_probes["resnet"]) >= 3

    # Verify JSON serialization round-trip
    json_str = payload.to_json()
    assert len(json_str) > 1000
    roundtrip = payload.__class__.from_json(json_str)
    assert roundtrip.metadata.experiment_id == payload.metadata.experiment_id


def test_ssl_service_query_methods() -> None:
    """Test query methods on SelfSupervisedService instance."""
    payload = generate_ssl_demo_data()
    service = SelfSupervisedService(payload=payload)

    meta = service.get_metadata()
    assert meta is not None
    assert meta.experiment_id == "exp_phase18_ssl_suite"

    cnn_report = service.get_report("cnn")
    assert cnn_report is not None
    assert isinstance(cnn_report, SelfSupervisedLearningReport)

    comp = service.get_comparison("resnet")
    assert comp is not None

    probes = service.get_layer_probes("vit")
    assert len(probes) > 0

    eff = service.get_label_efficiency("cnn")
    assert len(eff) > 0

    geom = service.get_geometry_points("resnet")
    assert len(geom) > 0
