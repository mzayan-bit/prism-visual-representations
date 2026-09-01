"""Unit tests for TransferService queries and JSON schema integrity."""

from prism.api.transfer_service import TransferService, generate_transfer_demo_data
from prism.transfer.reports import TransferLearningReport


def test_transfer_demo_payload_generation() -> None:
    """Test full transfer demo dataset generation and structure."""
    payload = generate_transfer_demo_data()

    assert payload.metadata.experiment_id == "exp_phase17_transfer_suite"
    assert "cnn" in payload.metadata.architectures
    assert "resnet" in payload.metadata.architectures
    assert "vit" in payload.metadata.architectures

    # Check reports exist for all 3 architectures
    cnn_reports = [
        r for r in payload.reports.values() if r.architecture.lower() == "cnn"
    ]
    assert len(cnn_reports) >= 4  # 4 strategies

    # Check shared PCA drifts
    assert "cnn" in payload.shared_pca_drifts
    assert "resnet" in payload.shared_pca_drifts
    assert "vit" in payload.shared_pca_drifts

    # Check data efficiency summaries
    assert "cnn" in payload.data_efficiency
    assert "resnet" in payload.data_efficiency
    assert "vit" in payload.data_efficiency

    # Verify JSON serialization round-trip
    json_str = payload.to_json()
    assert len(json_str) > 1000
    roundtrip = payload.__class__.from_json(json_str)
    assert roundtrip.metadata.experiment_id == payload.metadata.experiment_id


def test_transfer_service_query_methods() -> None:
    """Test query methods on TransferService instance."""
    payload = generate_transfer_demo_data()
    service = TransferService(payload=payload)

    meta = service.get_metadata()
    assert meta is not None
    assert meta.experiment_id == "exp_phase17_transfer_suite"

    cnn_report = service.get_report("cnn", "linear_probe", budget=1.0)
    assert cnn_report is not None
    assert isinstance(cnn_report, TransferLearningReport)

    probes = service.get_layer_probes("cnn")
    assert len(probes) > 0

    eff = service.get_data_efficiency("cnn")
    assert eff is not None

    pca = service.get_shared_pca("cnn")
    assert pca is not None
