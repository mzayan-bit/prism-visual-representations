"""Spatial task types, transfer strategies, and domain enums for PRISM."""

from enum import Enum


class SpatialTaskType(str, Enum):
    """Supported spatial downstream task types."""

    OBJECT_DETECTION = "object_detection"
    SEMANTIC_SEGMENTATION = "semantic_segmentation"


class CoordinateFormat(str, Enum):
    """Bounding box coordinate representation formats."""

    NORMALIZED_XYXY = "normalized_xyxy"
    PIXEL_XYXY = "pixel_xyxy"


class SpatialTransferStrategy(str, Enum):
    """Parameter update policies for spatial representation transfer."""

    FROZEN_SPATIAL_PROBE = "frozen_spatial_probe"
    PARTIAL_FINE_TUNE = "partial_fine_tune"
    FULL_FINE_TUNE = "full_fine_tune"


class PretrainingObjective(str, Enum):
    """Pretraining objectives defining source visual representation characteristics."""

    SUPERVISED = "supervised"
    SIMCLR = "simclr"
    RECONSTRUCTION = "reconstruction"
    SCRATCH = "scratch"


class SegmentationResizePolicy(str, Enum):
    """Spatial interpolation policy for upsampling segmentation logits."""

    NEAREST = "nearest"
    BILINEAR = "bilinear"
