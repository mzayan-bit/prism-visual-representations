"""Unit tests for SpatialTransferService, queries, comparisons, and export payloads."""

from prism.api.spatial_service import (
    SpatialTransferService,
    get_default_model_spec,
)
from prism.spatial.enums import (
    PretrainingObjective,
    SpatialTaskType,
    SpatialTransferStrategy,
)


def test_spatial_service_default_specs():
    """Test retrieving default model specifications for spatial transfer."""
    cnn_spec = get_default_model_spec("cnn")
    assert cnn_spec.model_id == "spec_cnn_spatial"
    assert cnn_spec.input_shape == (3, 16, 16)

    resnet_spec = get_default_model_spec("resnet")
    assert resnet_spec.model_id == "spec_resnet_spatial"

    vit_spec = get_default_model_spec("vit")
    assert vit_spec.model_id == "spec_vit_spatial"


def test_spatial_service_run_transfer_study():
    """Test running a transfer study via service."""
    service = SpatialTransferService(seed=42)

    rep = service.run_transfer_study(
        architecture="cnn",
        source_objective=PretrainingObjective.SUPERVISED,
        task_type=SpatialTaskType.OBJECT_DETECTION,
        transfer_strategy=SpatialTransferStrategy.FROZEN_SPATIAL_PROBE,
        epochs=1,
    )
    assert rep.specification.source_objective == PretrainingObjective.SUPERVISED
    assert rep.detection_metrics is not None
    assert rep.detection_metrics.mean_iou is not None


def test_spatial_service_objective_comparison():
    """Test generating cross-objective comparison summary."""
    service = SpatialTransferService(seed=42)

    summary = service.generate_objective_comparison(
        architecture="cnn",
        task_type=SpatialTaskType.OBJECT_DETECTION,
    )
    assert summary.architecture == "cnn"
    assert summary.task_type == SpatialTaskType.OBJECT_DETECTION
    assert "supervised" in summary.reports_by_objective
    assert "simclr" in summary.reports_by_objective
    assert "reconstruction" in summary.reports_by_objective
    assert "scratch" in summary.reports_by_objective
