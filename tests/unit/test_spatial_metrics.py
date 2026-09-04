"""Unit tests for spatial evaluation metrics."""

import pytest

from prism.spatial.annotations import (
    BoundingBox,
    DetectionAnnotation,
    DetectionPrediction,
    DetectionSample,
)
from prism.spatial.metrics import (
    SegmentationConfusionMatrix,
    compute_iou_xyxy,
    evaluate_detection_predictions,
)


def test_iou_exact_values():
    """Test bounding box IoU calculation across various geometric configurations."""
    # Identical boxes -> 1.0
    b1 = BoundingBox(x_min=0.2, y_min=0.2, x_max=0.8, y_max=0.8)
    b2 = BoundingBox(x_min=0.2, y_min=0.2, x_max=0.8, y_max=0.8)
    assert compute_iou_xyxy(b1, b2) == pytest.approx(1.0)

    # Disjoint boxes -> 0.0
    b3 = BoundingBox(x_min=0.0, y_min=0.0, x_max=0.1, y_max=0.1)
    b4 = BoundingBox(x_min=0.5, y_min=0.5, x_max=0.9, y_max=0.9)
    assert compute_iou_xyxy(b3, b4) == pytest.approx(0.0)

    # Touching edges -> 0.0
    b5 = BoundingBox(x_min=0.0, y_min=0.0, x_max=0.5, y_max=0.5)
    b6 = BoundingBox(x_min=0.5, y_min=0.0, x_max=1.0, y_max=0.5)
    assert compute_iou_xyxy(b5, b6) == pytest.approx(0.0)

    # Partial overlap
    # Box A: [0, 0, 0.5, 0.5] Area = 0.25
    # Box B: [0.25, 0, 0.75, 0.5] Area = 0.25
    # Intersection: [0.25, 0, 0.5, 0.5] Area = 0.125
    # Union: 0.25 + 0.25 - 0.125 = 0.375
    # IoU: 0.125 / 0.375 = 1/3
    b7 = BoundingBox(x_min=0.0, y_min=0.0, x_max=0.5, y_max=0.5)
    b8 = BoundingBox(x_min=0.25, y_min=0.0, x_max=0.75, y_max=0.5)
    assert compute_iou_xyxy(b7, b8) == pytest.approx(1.0 / 3.0)


def test_greedy_detection_matching_and_metrics():
    """Test deterministic greedy 1-to-1 matching and detection precision/recall/IoU."""
    gt_box1 = BoundingBox(x_min=0.1, y_min=0.1, x_max=0.5, y_max=0.5)
    gt_box2 = BoundingBox(x_min=0.6, y_min=0.6, x_max=0.9, y_max=0.9)

    targets = [
        DetectionSample(
            sample_id="eval_01",
            image=[[[0.0 for _ in range(8)] for _ in range(8)]],
            annotations=[
                DetectionAnnotation(class_id=0, box=gt_box1),
                DetectionAnnotation(class_id=1, box=gt_box2),
            ],
        )
    ]

    # Perfect predictions
    pred_box1 = BoundingBox(x_min=0.1, y_min=0.1, x_max=0.5, y_max=0.5)
    pred_box2 = BoundingBox(x_min=0.6, y_min=0.6, x_max=0.9, y_max=0.9)

    preds_perfect = [
        DetectionPrediction(
            sample_id="eval_01",
            boxes=[pred_box1, pred_box2],
            class_ids=[0, 1],
            confidences=[0.9, 0.8],
            objectness_scores=[0.9, 0.8],
        )
    ]

    res_perfect = evaluate_detection_predictions(
        preds_perfect, targets, iou_threshold=0.5
    )
    assert res_perfect.matched_objects == 2
    assert res_perfect.precision == pytest.approx(1.0)
    assert res_perfect.recall == pytest.approx(1.0)
    assert res_perfect.mean_iou == pytest.approx(1.0)
    assert res_perfect.class_accuracy == pytest.approx(1.0)

    # Predictions with 1 false positive and 1 low IoU below threshold
    pred_box_poor = BoundingBox(x_min=0.4, y_min=0.4, x_max=0.6, y_max=0.6)  # low IoU
    pred_box_extra = BoundingBox(x_min=0.0, y_min=0.8, x_max=0.2, y_max=0.9)  # FP
    preds_imperfect = [
        DetectionPrediction(
            sample_id="eval_01",
            boxes=[pred_box1, pred_box_poor, pred_box_extra],
            class_ids=[0, 1, 0],
            confidences=[0.9, 0.5, 0.7],
            objectness_scores=[0.9, 0.5, 0.7],
        )
    ]

    res_imperfect = evaluate_detection_predictions(
        preds_imperfect, targets, iou_threshold=0.5
    )
    assert res_imperfect.matched_objects == 1  # Only pred_box1 matched
    assert res_imperfect.precision == pytest.approx(1.0 / 3.0)
    assert res_imperfect.recall == pytest.approx(1.0 / 2.0)


def test_segmentation_confusion_matrix_and_metrics():
    """Test SegmentationConfusionMatrix, pixel accuracy, class IoU, and mean IoU."""
    cm = SegmentationConfusionMatrix(num_classes=3)

    # True mask:
    # 0 0 1
    # 1 2 2
    # Pred mask:
    # 0 0 1
    # 2 2 2
    # (1 error at (1, 0) where true=1, pred=2)
    pred_mask = [[0, 0, 1], [2, 2, 2]]
    true_mask = [[0, 0, 1], [1, 2, 2]]

    cm.update([pred_mask], [true_mask])

    assert cm.matrix[0][0] == 2  # true 0, pred 0
    assert cm.matrix[1][1] == 1  # true 1, pred 1
    assert cm.matrix[1][2] == 1  # true 1, pred 2
    assert cm.matrix[2][2] == 2  # true 2, pred 2

    res = cm.compute_metrics()
    assert res.total_pixels == 6
    assert res.pixel_accuracy == pytest.approx(5.0 / 6.0)

    # Class 0: TP=2, FP=0, FN=0 -> IoU = 2/2 = 1.0
    assert res.per_class_iou[0] == pytest.approx(1.0)
    # Class 1: TP=1, FP=0, FN=1 -> IoU = 1/2 = 0.5
    assert res.per_class_iou[1] == pytest.approx(0.5)
    # Class 2: TP=2, FP=1, FN=0 -> IoU = 2/3 = 0.6667
    assert res.per_class_iou[2] == pytest.approx(2.0 / 3.0)

    assert res.mean_iou == pytest.approx((1.0 + 0.5 + 2.0 / 3.0) / 3.0)


def test_segmentation_metrics_absent_class_handling():
    """Test that absent classes in ground truth are handled cleanly."""
    cm = SegmentationConfusionMatrix(num_classes=4)
    # Only class 0 and 1 present; class 2 and 3 absent
    pred_mask = [[0, 1], [0, 1]]
    true_mask = [[0, 1], [0, 1]]

    cm.update([pred_mask], [true_mask])
    res = cm.compute_metrics()

    assert res.pixel_accuracy == pytest.approx(1.0)
    assert res.per_class_iou[0] == pytest.approx(1.0)
    assert res.per_class_iou[1] == pytest.approx(1.0)
    # Only active classes (0 and 1) contribute to mean_iou
    assert res.mean_iou == pytest.approx(1.0)
