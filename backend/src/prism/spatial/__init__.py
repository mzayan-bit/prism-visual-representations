"""PRISM Spatial Representation Transfer Subsystem."""

from prism.spatial.adapter import (
    SpatialRepresentationAdapter,
    get_available_spatial_layers,
)
from prism.spatial.annotations import (
    BoundingBox,
    DetectionAnnotation,
    DetectionPrediction,
    DetectionSample,
    SegmentationSample,
)
from prism.spatial.enums import (
    CoordinateFormat,
    PretrainingObjective,
    SegmentationResizePolicy,
    SpatialTaskType,
    SpatialTransferStrategy,
)
from prism.spatial.heads import (
    GridDetectionHead,
    SegmentationHead,
)
from prism.spatial.losses import (
    GridDetectionLoss,
    PixelCrossEntropyLoss,
)
from prism.spatial.metrics import (
    DetectionEvaluationResult,
    SegmentationConfusionMatrix,
    SegmentationMetricsResult,
    compute_iou_xyxy,
    evaluate_detection_predictions,
)
from prism.spatial.synthetic import generate_synthetic_spatial_dataset

__all__ = [
    "BoundingBox",
    "CoordinateFormat",
    "DetectionAnnotation",
    "DetectionEvaluationResult",
    "DetectionPrediction",
    "DetectionSample",
    "GridDetectionHead",
    "GridDetectionLoss",
    "PixelCrossEntropyLoss",
    "PretrainingObjective",
    "SegmentationConfusionMatrix",
    "SegmentationHead",
    "SegmentationMetricsResult",
    "SegmentationResizePolicy",
    "SegmentationSample",
    "SpatialRepresentationAdapter",
    "SpatialTaskType",
    "SpatialTransferStrategy",
    "compute_iou_xyxy",
    "evaluate_detection_predictions",
    "generate_synthetic_spatial_dataset",
    "get_available_spatial_layers",
]
