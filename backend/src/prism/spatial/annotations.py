"""Typed annotations, bounding boxes, samples, and predictions for spatial tasks."""

from __future__ import annotations

import math
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from prism.core.errors import ValidationError
from prism.spatial.enums import CoordinateFormat


class BoundingBox(BaseModel):
    """Immutable normalized axis-aligned bounding box [x_min, y_min, x_max, y_max].

    Coordinates are normalized to [0.0, 1.0].
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    x_min: float = Field(..., description="Normalized left coordinate in [0.0, 1.0]")
    y_min: float = Field(..., description="Normalized top coordinate in [0.0, 1.0]")
    x_max: float = Field(..., description="Normalized right coordinate in (0.0, 1.0]")
    y_max: float = Field(..., description="Normalized bottom coordinate in (0.0, 1.0]")
    format: CoordinateFormat = Field(
        default=CoordinateFormat.NORMALIZED_XYXY,
        description="Coordinate representation convention",
    )

    @model_validator(mode="after")
    def validate_box_bounds(self) -> BoundingBox:
        """Validate coordinates are finite, in bounds, and non-inverted."""
        for name, val in [
            ("x_min", self.x_min),
            ("y_min", self.y_min),
            ("x_max", self.x_max),
            ("y_max", self.y_max),
        ]:
            if not isinstance(val, (int, float)) or math.isnan(val) or math.isinf(val):
                raise ValidationError(
                    f"Bounding box {name} must be a finite number, got {val}."
                )

        if not (0.0 <= self.x_min <= 1.0):
            raise ValidationError(f"x_min must be in [0.0, 1.0], got {self.x_min}.")
        if not (0.0 <= self.y_min <= 1.0):
            raise ValidationError(f"y_min must be in [0.0, 1.0], got {self.y_min}.")
        if not (0.0 <= self.x_max <= 1.0):
            raise ValidationError(f"x_max must be in [0.0, 1.0], got {self.x_max}.")
        if not (0.0 <= self.y_max <= 1.0):
            raise ValidationError(f"y_max must be in [0.0, 1.0], got {self.y_max}.")

        if self.x_min >= self.x_max:
            raise ValidationError(
                f"Invalid bounding box: x_min ({self.x_min}) must be strictly less "
                f"than x_max ({self.x_max})."
            )
        if self.y_min >= self.y_max:
            raise ValidationError(
                f"Invalid bounding box: y_min ({self.y_min}) must be strictly less "
                f"than y_max ({self.y_max})."
            )

        return self

    @property
    def width(self) -> float:
        """Normalized bounding box width."""
        return self.x_max - self.x_min

    @property
    def height(self) -> float:
        """Normalized bounding box height."""
        return self.y_max - self.y_min

    @property
    def area(self) -> float:
        """Normalized bounding box area."""
        return self.width * self.height

    @property
    def center(self) -> tuple[float, float]:
        """Normalized center coordinates (center_x, center_y)."""
        return ((self.x_min + self.x_max) / 2.0, (self.y_min + self.y_max) / 2.0)

    def to_tuple(self) -> tuple[float, float, float, float]:
        """Return (x_min, y_min, x_max, y_max)."""
        return (self.x_min, self.y_min, self.x_max, self.y_max)

    def to_pixels(self, img_w: int, img_h: int) -> tuple[float, float, float, float]:
        """Convert normalized box to pixel coordinates."""
        if img_w <= 0 or img_h <= 0:
            raise ValidationError(
                f"Image dimensions must be positive, got ({img_w}, {img_h})."
            )
        return (
            self.x_min * float(img_w),
            self.y_min * float(img_h),
            self.x_max * float(img_w),
            self.y_max * float(img_h),
        )

    @classmethod
    def from_pixels(
        cls,
        x_min_px: float,
        y_min_px: float,
        x_max_px: float,
        y_max_px: float,
        img_w: int,
        img_h: int,
    ) -> BoundingBox:
        """Construct normalized BoundingBox from pixel coordinates."""
        if img_w <= 0 or img_h <= 0:
            raise ValidationError(
                f"Image dimensions must be positive, got ({img_w}, {img_h})."
            )
        return cls(
            x_min=float(x_min_px) / float(img_w),
            y_min=float(y_min_px) / float(img_h),
            x_max=float(x_max_px) / float(img_w),
            y_max=float(y_max_px) / float(img_h),
            format=CoordinateFormat.NORMALIZED_XYXY,
        )


class DetectionAnnotation(BaseModel):
    """Single object annotation with bounding box, class identifier, and frame."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    class_id: int = Field(..., ge=0, description="Target integer category label")
    box: BoundingBox = Field(..., description="Normalized axis-aligned bounding box")
    class_name: str | None = Field(
        default=None, description="Human-readable category name"
    )
    image_width: int = Field(default=32, gt=0, description="Source image pixel width")
    image_height: int = Field(default=32, gt=0, description="Source image pixel height")


class DetectionSample(BaseModel):
    """Structured detection sample containing image tensor and annotations."""

    model_config = ConfigDict(extra="forbid")

    sample_id: str = Field(
        ..., min_length=1, description="Unique identifier for the sample"
    )
    image: list[list[list[float]]] = Field(
        ..., description="3D numerical image tensor of shape [C, H, W]"
    )
    annotations: list[DetectionAnnotation] = Field(
        default_factory=list,
        description="List of target annotations (empty indicates background)",
    )
    dataset_fingerprint: str = Field(
        default="synthetic_fp", description="Deterministic dataset digest"
    )
    split: str = Field(default="train", description="Dataset split (train, val, test)")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Arbitrary sample metadata"
    )

    @field_validator("image")
    @classmethod
    def validate_image_tensor(cls, val: Any) -> list[list[list[float]]]:
        """Validate image is a non-empty 3D tensor [C, H, W]."""
        if not isinstance(val, (list, tuple)) or not val:
            raise ValidationError("Image tensor cannot be empty.")
        if not isinstance(val[0], (list, tuple)) or not val[0]:
            raise ValidationError("Image channel tensor cannot be empty.")
        if not isinstance(val[0][0], (list, tuple)) or not val[0][0]:
            raise ValidationError("Image row tensor cannot be empty.")
        h = len(val[0])
        w = len(val[0][0])
        for ch_idx, ch in enumerate(val):
            if len(ch) != h:
                raise ValidationError(
                    f"Inconsistent height in channel {ch_idx}: {len(ch)} vs {h}."
                )
            for r_idx, row in enumerate(ch):
                if len(row) != w:
                    raise ValidationError(
                        f"Inconsistent width in row {r_idx}: {len(row)} vs {w}."
                    )
        return [[[float(v) for v in row] for row in ch] for ch in val]

    @property
    def image_shape(self) -> tuple[int, int, int]:
        """Return (channels, height, width)."""
        return (len(self.image), len(self.image[0]), len(self.image[0][0]))

    @property
    def num_objects(self) -> int:
        """Return number of annotated objects."""
        return len(self.annotations)


class SegmentationSample(BaseModel):
    """Structured semantic segmentation sample with image and 2D integer class mask."""

    model_config = ConfigDict(extra="forbid")

    sample_id: str = Field(
        ..., min_length=1, description="Unique identifier for the sample"
    )
    image: list[list[list[float]]] = Field(
        ..., description="3D numerical image tensor of shape [C, H, W]"
    )
    mask: list[list[int]] = Field(
        ..., description="2D pixel mask of shape [H, W] containing integer class IDs"
    )
    image_shape: tuple[int, int, int] | None = Field(
        default=None, description="Canonical (channels, height, width)"
    )
    num_classes: int = Field(
        ..., gt=0, description="Total number of valid semantic classes"
    )
    ignore_index: int | None = Field(
        default=None, description="Optional class ID to ignore in loss/metrics"
    )
    dataset_fingerprint: str = Field(
        default="synthetic_fp", description="Deterministic dataset digest"
    )
    split: str = Field(default="train", description="Dataset split (train, val, test)")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Arbitrary sample metadata"
    )

    @model_validator(mode="after")
    def validate_mask_and_image(self) -> SegmentationSample:
        """Validate mask shape matches image shape and mask values are valid."""
        if not self.image or not self.image[0] or not self.image[0][0]:
            raise ValidationError("Image tensor cannot be empty.")
        c = len(self.image)
        h = len(self.image[0])
        w = len(self.image[0][0])
        if self.image_shape is not None and self.image_shape != (c, h, w):
            raise ValidationError(
                f"Image shape {c}x{h}x{w} does not match declared {self.image_shape}."
            )
        if len(self.mask) != h:
            raise ValidationError(
                f"Mask height ({len(self.mask)}) must match image height ({h})."
            )
        for r_idx, row in enumerate(self.mask):
            if len(row) != w:
                raise ValidationError(
                    f"Mask row {r_idx} width ({len(row)}) must match image width ({w})."
                )
            for c_idx, val in enumerate(row):
                if not isinstance(val, int):
                    raise ValidationError(
                        f"Mask value at ({r_idx}, {c_idx}) must be int, "
                        f"got {type(val)}."
                    )
                if self.ignore_index is not None and val == self.ignore_index:
                    continue
                if not (0 <= val < self.num_classes):
                    raise ValidationError(
                        f"Mask val {val} at ({r_idx}, {c_idx}) outside range "
                        f"[0, {self.num_classes - 1}]."
                    )
        return self


class DetectionPrediction(BaseModel):
    """Evaluated object detection prediction output for a sample."""

    model_config = ConfigDict(extra="forbid")

    sample_id: str = Field(..., description="Sample identifier")
    boxes: list[BoundingBox] = Field(
        default_factory=list, description="Predicted bounding boxes"
    )
    objectness_scores: list[float] = Field(
        default_factory=list, description="Objectness confidence scores in [0.0, 1.0]"
    )
    class_probabilities: list[list[float]] = Field(
        default_factory=list, description="Class probability distributions"
    )
    class_ids: list[int] = Field(
        default_factory=list, description="Predicted class category IDs"
    )
    confidences: list[float] = Field(
        default_factory=list, description="Overall confidence (objectness * class_prob)"
    )
    grid_coords: list[tuple[int, int]] = Field(
        default_factory=list, description="Feature grid (row, col) coordinates"
    )
    matched_target_idx: int | None = Field(
        default=None, description="Index of matched ground truth object if matched"
    )
    iou_with_target: float | None = Field(
        default=None, description="IoU with matched target annotation"
    )

    @model_validator(mode="after")
    def validate_prediction_lengths(self) -> DetectionPrediction:
        """Validate list lengths are consistent."""
        n = len(self.boxes)
        if len(self.class_ids) != n:
            raise ValidationError(
                f"Length of class_ids ({len(self.class_ids)}) must match "
                f"length of boxes ({n})."
            )
        if len(self.confidences) != n:
            raise ValidationError(
                f"Length of confidences ({len(self.confidences)}) must match "
                f"length of boxes ({n})."
            )
        if self.objectness_scores and len(self.objectness_scores) != n:
            raise ValidationError(
                f"Length of objectness_scores ({len(self.objectness_scores)}) "
                f"must match length of boxes ({n})."
            )
        return self
