"""Unit tests for spatial annotation contracts."""

import pytest
from pydantic import ValidationError as PydanticValidationError

from prism.core.errors import ValidationError
from prism.spatial.annotations import (
    BoundingBox,
    DetectionAnnotation,
    DetectionPrediction,
    DetectionSample,
    SegmentationSample,
)


def test_bounding_box_valid_cases():
    """Test valid BoundingBox instantiations and geometric properties."""
    box = BoundingBox(x_min=0.1, y_min=0.2, x_max=0.5, y_max=0.8)
    assert box.x_min == pytest.approx(0.1)
    assert box.y_min == pytest.approx(0.2)
    assert box.x_max == pytest.approx(0.5)
    assert box.y_max == pytest.approx(0.8)
    assert box.width == pytest.approx(0.4)
    assert box.height == pytest.approx(0.6)
    assert box.area == pytest.approx(0.24)
    assert box.center[0] == pytest.approx(0.3)
    assert box.center[1] == pytest.approx(0.5)
    assert box.to_tuple() == (0.1, 0.2, 0.5, 0.8)
    assert list(box.to_tuple()) == [0.1, 0.2, 0.5, 0.8]


def test_bounding_box_inverted_and_out_of_bounds_rejection():
    """Test that inverted coordinates raise ValidationError."""
    # Inverted x
    with pytest.raises(ValidationError, match="Invalid bounding box"):
        BoundingBox(x_min=0.6, y_min=0.2, x_max=0.5, y_max=0.8)

    # Equal x (zero area)
    with pytest.raises(ValidationError, match="Invalid bounding box"):
        BoundingBox(x_min=0.5, y_min=0.2, x_max=0.5, y_max=0.8)

    # Inverted y
    with pytest.raises(ValidationError, match="Invalid bounding box"):
        BoundingBox(x_min=0.1, y_min=0.8, x_max=0.5, y_max=0.2)

    # Equal y (zero area)
    with pytest.raises(ValidationError, match="Invalid bounding box"):
        BoundingBox(x_min=0.1, y_min=0.5, x_max=0.5, y_max=0.5)

    # Negative x_min
    with pytest.raises(ValidationError, match="x_min must be in"):
        BoundingBox(x_min=-0.01, y_min=0.2, x_max=0.5, y_max=0.8)

    # x_max > 1.0
    with pytest.raises(ValidationError, match="x_max must be in"):
        BoundingBox(x_min=0.1, y_min=0.2, x_max=1.05, y_max=0.8)

    # Non-finite coordinates (NaN / Inf)
    with pytest.raises(ValidationError, match="must be a finite number"):
        BoundingBox(x_min=float("nan"), y_min=0.2, x_max=0.5, y_max=0.8)

    with pytest.raises(ValidationError, match="must be a finite number"):
        BoundingBox(x_min=0.1, y_min=float("inf"), x_max=0.5, y_max=0.8)


def test_detection_annotation():
    """Test DetectionAnnotation creation and validation."""
    box = BoundingBox(x_min=0.0, y_min=0.0, x_max=1.0, y_max=1.0)
    ann = DetectionAnnotation(
        class_id=2,
        class_name="circle",
        box=box,
        image_width=32,
        image_height=32,
    )
    assert ann.class_id == 2
    assert ann.class_name == "circle"
    assert ann.image_width == 32
    assert ann.image_height == 32

    # Negative class ID rejected
    with pytest.raises((ValidationError, PydanticValidationError), match="class_id"):
        DetectionAnnotation(
            class_id=-1,
            class_name="invalid",
            box=box,
        )


def test_detection_sample_empty_and_multiple():
    """Test DetectionSample with zero, one, and multiple annotations."""
    # Empty annotations (clean empty image support)
    empty_sample = DetectionSample(
        sample_id="empty_01",
        image=[[[0.0 for _ in range(8)] for _ in range(8)]],
        annotations=[],
        split="train",
    )
    assert len(empty_sample.annotations) == 0
    assert empty_sample.sample_id == "empty_01"

    # Multiple annotations
    box1 = BoundingBox(x_min=0.1, y_min=0.1, x_max=0.4, y_max=0.4)
    box2 = BoundingBox(x_min=0.6, y_min=0.6, x_max=0.9, y_max=0.9)
    multi_sample = DetectionSample(
        sample_id="multi_01",
        image=[[[0.5 for _ in range(8)] for _ in range(8)]],
        annotations=[
            DetectionAnnotation(class_id=0, box=box1),
            DetectionAnnotation(class_id=1, box=box2),
        ],
        split="val",
    )
    assert len(multi_sample.annotations) == 2


def test_detection_prediction():
    """Test DetectionPrediction container for inference outputs."""
    box = BoundingBox(x_min=0.2, y_min=0.2, x_max=0.6, y_max=0.6)
    pred = DetectionPrediction(
        sample_id="pred_01",
        boxes=[box],
        class_ids=[1],
        confidences=[0.95],
        objectness_scores=[0.98],
    )
    assert pred.sample_id == "pred_01"
    assert len(pred.boxes) == 1
    assert pred.confidences[0] == pytest.approx(0.95)

    # Length mismatch rejection
    with pytest.raises(ValidationError, match=r"must match length of boxes"):
        DetectionPrediction(
            sample_id="mismatch_01",
            boxes=[box],
            class_ids=[1, 2],
            confidences=[0.9],
            objectness_scores=[0.9],
        )


def test_segmentation_sample_validation():
    """Test SegmentationSample creation, shape validation, and class ID bounds."""
    img = [[[0.1 for _ in range(8)] for _ in range(8)] for _ in range(3)]
    mask = [[0 for _ in range(8)] for _ in range(8)]
    mask[2][2] = 1
    mask[3][3] = 2

    seg = SegmentationSample(
        sample_id="seg_01",
        image=img,
        mask=mask,
        num_classes=3,
        split="train",
    )
    assert seg.sample_id == "seg_01"
    assert seg.num_classes == 3

    # Shape mismatch between image and mask
    invalid_mask_h = [[0 for _ in range(8)] for _ in range(7)]
    with pytest.raises(ValidationError, match=r"Mask height .* image height"):
        SegmentationSample(
            sample_id="bad_shape_01",
            image=img,
            mask=invalid_mask_h,
            num_classes=3,
        )

    # Mask value outside expected class range
    invalid_mask_val = [[0 for _ in range(8)] for _ in range(8)]
    invalid_mask_val[0][0] = 5  # >= num_classes (3)
    with pytest.raises(ValidationError, match="outside range"):
        SegmentationSample(
            sample_id="bad_val_01",
            image=img,
            mask=invalid_mask_val,
            num_classes=3,
        )

    # Negative mask value rejected
    invalid_mask_neg = [[0 for _ in range(8)] for _ in range(8)]
    invalid_mask_neg[0][0] = -1
    with pytest.raises(ValidationError, match="outside range"):
        SegmentationSample(
            sample_id="bad_neg_01",
            image=img,
            mask=invalid_mask_neg,
            num_classes=3,
        )
