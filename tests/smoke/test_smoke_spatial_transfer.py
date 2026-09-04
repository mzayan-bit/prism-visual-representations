"""End-to-end smoke test validating Phase 20 Spatial Transfer pipeline."""

from prism.api.spatial_service import get_default_model_spec
from prism.spatial.enums import (
    PretrainingObjective,
    SpatialTaskType,
    SpatialTransferStrategy,
)
from prism.spatial.runner import SpatialTransferRunner
from prism.spatial.specification import SpatialTransferSpecification
from prism.spatial.synthetic import generate_synthetic_spatial_dataset


def test_smoke_spatial_transfer_pipeline():
    """Execute fast detection and segmentation probes on CNN, ResNet, ViT."""
    # 1. Generate deterministic synthetic data
    det_train, seg_train = generate_synthetic_spatial_dataset(
        num_samples=4, image_shape=(3, 16, 16), num_classes=3, seed=42
    )
    det_val, seg_val = generate_synthetic_spatial_dataset(
        num_samples=2, image_shape=(3, 16, 16), num_classes=3, seed=84
    )

    architectures = ["cnn", "resnet", "vit"]

    for arch in architectures:
        model_spec = get_default_model_spec(arch)

        # ----------------------------------------------------
        # Detection Path: Frozen Spatial Probe
        # ----------------------------------------------------
        det_spec = SpatialTransferSpecification.create(
            source_objective=PretrainingObjective.SUPERVISED,
            source_experiment_id=f"smoke_{arch}_det",
            model_spec=model_spec,
            task_type=SpatialTaskType.OBJECT_DETECTION,
            spatial_layer="final_spatial",
            transfer_strategy=SpatialTransferStrategy.FROZEN_SPATIAL_PROBE,
            num_classes=3,
            epochs=1,
            learning_rate=0.01,
            batch_size=2,
            seed=100,
        )
        det_runner = SpatialTransferRunner(det_spec)
        det_report = det_runner.train_and_evaluate(
            train_samples=det_train, eval_samples=det_val
        )

        # Validate detection report & metrics
        assert det_report.detection_metrics is not None
        assert det_report.detection_metrics.mean_iou is not None
        assert det_report.detection_metrics.precision is not None
        assert det_report.detection_metrics.recall is not None
        assert det_report.total_parameters > 0
        assert det_report.trainable_parameters > 0

        # Validate report serialization
        det_json = det_report.model_dump_json()
        assert len(det_json) > 50

        # ----------------------------------------------------
        # Segmentation Path: Partial Fine-Tune
        # ----------------------------------------------------
        seg_spec = SpatialTransferSpecification.create(
            source_objective=PretrainingObjective.RECONSTRUCTION,
            source_experiment_id=f"smoke_{arch}_seg",
            model_spec=model_spec,
            task_type=SpatialTaskType.SEMANTIC_SEGMENTATION,
            spatial_layer="final_spatial",
            transfer_strategy=SpatialTransferStrategy.PARTIAL_FINE_TUNE,
            num_classes=3,
            epochs=1,
            learning_rate=0.01,
            batch_size=2,
            seed=200,
        )
        seg_runner = SpatialTransferRunner(seg_spec)
        seg_report = seg_runner.train_and_evaluate(
            train_samples=seg_train, eval_samples=seg_val
        )

        # Validate segmentation report & metrics
        assert seg_report.segmentation_metrics is not None
        assert seg_report.segmentation_metrics.mean_iou is not None
        assert seg_report.segmentation_metrics.pixel_accuracy is not None
        assert seg_report.segmentation_metrics.per_class_iou is not None
        assert len(seg_report.segmentation_metrics.per_class_iou) == 3

        # Validate report serialization
        seg_json = seg_report.model_dump_json()
        assert len(seg_json) > 50
