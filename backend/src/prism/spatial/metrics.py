"""Spatial task evaluation metrics: bounding box IoU, matching, confusion matrix."""

from __future__ import annotations

import math

from pydantic import BaseModel, ConfigDict, Field

from prism.core.errors import ValidationError
from prism.spatial.annotations import BoundingBox, DetectionPrediction, DetectionSample


def compute_iou_xyxy(
    box1: BoundingBox | tuple[float, float, float, float],
    box2: BoundingBox | tuple[float, float, float, float],
) -> float:
    """Compute exact axis-aligned Intersection over Union (IoU)."""
    if isinstance(box1, BoundingBox):
        b1_xmin, b1_ymin, b1_xmax, b1_ymax = box1.to_tuple()
    else:
        b1_xmin, b1_ymin, b1_xmax, b1_ymax = box1

    if isinstance(box2, BoundingBox):
        b2_xmin, b2_ymin, b2_xmax, b2_ymax = box2.to_tuple()
    else:
        b2_xmin, b2_ymin, b2_xmax, b2_ymax = box2

    inter_xmin = max(b1_xmin, b2_xmin)
    inter_ymin = max(b1_ymin, b2_ymin)
    inter_xmax = min(b1_xmax, b2_xmax)
    inter_ymax = min(b1_ymax, b2_ymax)

    inter_w = max(0.0, inter_xmax - inter_xmin)
    inter_h = max(0.0, inter_ymax - inter_ymin)
    inter_area = inter_w * inter_h

    if inter_area <= 0.0:
        return 0.0

    b1_area = max(0.0, (b1_xmax - b1_xmin) * (b1_ymax - b1_ymin))
    b2_area = max(0.0, (b2_xmax - b2_xmin) * (b2_ymax - b2_ymin))
    union_area = b1_area + b2_area - inter_area

    if union_area <= 0.0:
        return 0.0

    return min(1.0, max(0.0, inter_area / union_area))


class DetectionEvaluationResult(BaseModel):
    """Evaluation summary metrics for object detection."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    total_samples: int = Field(..., description="Number of evaluated samples")
    total_targets: int = Field(..., description="Total ground truth objects")
    total_predictions: int = Field(..., description="Total predicted detections")
    matched_objects: int = Field(
        ..., description="Number of matched prediction-target pairs"
    )
    mean_iou: float = Field(
        ..., ge=0.0, le=1.0, description="Mean IoU over matched detections"
    )
    precision: float = Field(
        ..., ge=0.0, le=1.0, description="Precision at IoU threshold"
    )
    recall: float = Field(..., ge=0.0, le=1.0, description="Recall at IoU threshold")
    class_accuracy: float = Field(
        ..., ge=0.0, le=1.0, description="Classification accuracy on matched objects"
    )
    mean_localization_error: float = Field(
        ..., ge=0.0, description="Mean Euclidean distance between matched box centers"
    )
    iou_threshold: float = Field(..., ge=0.0, le=1.0, description="Matching threshold")


def evaluate_detection_predictions(
    predictions: list[DetectionPrediction],
    targets: list[DetectionSample],
    iou_threshold: float = 0.5,
    require_class_match: bool = False,
) -> DetectionEvaluationResult:
    """Evaluate object detections using deterministic greedy 1-to-1 matching."""
    if len(predictions) != len(targets):
        raise ValidationError(
            f"Sample count mismatch: {len(predictions)} preds "
            f"vs {len(targets)} targets."
        )

    total_targets = 0
    total_preds = 0
    matched_count = 0
    matched_ious: list[float] = []
    matched_correct_class = 0
    loc_errors: list[float] = []

    for pred_sample, target_sample in zip(predictions, targets, strict=True):
        gt_annotations = target_sample.annotations
        total_targets += len(gt_annotations)
        total_preds += len(pred_sample.boxes)

        matched_gt_indices: set[int] = set()

        indexed_preds = list(
            zip(
                pred_sample.boxes,
                pred_sample.class_ids,
                pred_sample.confidences,
                strict=True,
            )
        )
        indexed_preds.sort(key=lambda x: x[2], reverse=True)

        for p_box, p_cls, _ in indexed_preds:
            best_iou = -1.0
            best_gt_idx = -1

            for gt_idx, gt_ann in enumerate(gt_annotations):
                if gt_idx in matched_gt_indices:
                    continue
                if require_class_match and p_cls != gt_ann.class_id:
                    continue

                iou = compute_iou_xyxy(p_box, gt_ann.box)
                if iou >= iou_threshold and iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx

            if best_gt_idx >= 0:
                matched_gt_indices.add(best_gt_idx)
                matched_count += 1
                matched_ious.append(best_iou)

                gt_ann = gt_annotations[best_gt_idx]
                if p_cls == gt_ann.class_id:
                    matched_correct_class += 1

                p_cx, p_cy = p_box.center
                gt_cx, gt_cy = gt_ann.box.center
                dist = math.sqrt((p_cx - gt_cx) ** 2 + (p_cy - gt_cy) ** 2)
                loc_errors.append(dist)

    mean_iou = sum(matched_ious) / len(matched_ious) if matched_ious else 0.0
    precision = (
        matched_count / total_preds
        if total_preds > 0
        else (1.0 if total_targets == 0 else 0.0)
    )
    recall = matched_count / total_targets if total_targets > 0 else 1.0
    cls_acc = matched_correct_class / matched_count if matched_count > 0 else 1.0
    mean_loc_err = sum(loc_errors) / len(loc_errors) if loc_errors else 0.0

    return DetectionEvaluationResult(
        total_samples=len(targets),
        total_targets=total_targets,
        total_predictions=total_preds,
        matched_objects=matched_count,
        mean_iou=float(mean_iou),
        precision=float(precision),
        recall=float(recall),
        class_accuracy=float(cls_acc),
        mean_localization_error=float(mean_loc_err),
        iou_threshold=float(iou_threshold),
    )


class SegmentationMetricsResult(BaseModel):
    """Semantic segmentation evaluation metrics and confusion matrix."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    num_classes: int = Field(..., gt=0, description="Total number of evaluated classes")
    total_pixels: int = Field(..., description="Total evaluated valid pixels")
    pixel_accuracy: float = Field(
        ..., ge=0.0, le=1.0, description="Overall pixel accuracy (correct / total)"
    )
    mean_iou: float = Field(
        ..., ge=0.0, le=1.0, description="Mean IoU across evaluated classes"
    )
    per_class_iou: dict[int, float] = Field(..., description="IoU per class ID")
    per_class_dice: dict[int, float] = Field(
        ..., description="Dice coefficient per class ID"
    )
    confusion_matrix: list[list[int]] = Field(
        ...,
        description="K x K confusion matrix (rows = true class, cols = predicted)",
    )


class SegmentationConfusionMatrix:
    """K x K Confusion Matrix accumulator for semantic segmentation."""

    def __init__(self, num_classes: int, ignore_index: int | None = None) -> None:
        if num_classes <= 0:
            raise ValidationError(f"num_classes must be positive, got {num_classes}.")
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.matrix: list[list[int]] = [
            [0 for _ in range(num_classes)] for _ in range(num_classes)
        ]

    def update(
        self,
        predicted_masks: list[list[list[int]]],
        ground_truth_masks: list[list[list[int]]],
    ) -> None:
        """Accumulate counts from batch of predicted and ground truth masks."""
        if len(predicted_masks) != len(ground_truth_masks):
            raise ValidationError(
                f"Batch size mismatch: {len(predicted_masks)} predicted vs "
                f"{len(ground_truth_masks)} ground truth masks."
            )

        for p_mask, gt_mask in zip(predicted_masks, ground_truth_masks, strict=True):
            h = len(gt_mask)
            w = len(gt_mask[0])
            for y in range(h):
                for x in range(w):
                    t_val = gt_mask[y][x]
                    if self.ignore_index is not None and t_val == self.ignore_index:
                        continue
                    p_val = p_mask[y][x]
                    if 0 <= t_val < self.num_classes and 0 <= p_val < self.num_classes:
                        self.matrix[t_val][p_val] += 1

    def compute_metrics(self) -> SegmentationMetricsResult:
        """Compute pixel accuracy, per-class IoU, and mean IoU."""
        total_pixels = sum(sum(row) for row in self.matrix)
        correct_pixels = sum(self.matrix[k][k] for k in range(self.num_classes))

        pixel_acc = correct_pixels / total_pixels if total_pixels > 0 else 0.0

        per_class_iou: dict[int, float] = {}
        per_class_dice: dict[int, float] = {}
        valid_class_ious: list[float] = []

        for k in range(self.num_classes):
            tp = self.matrix[k][k]
            fn = sum(self.matrix[k][c] for c in range(self.num_classes)) - tp
            fp = sum(self.matrix[r][k] for r in range(self.num_classes)) - tp

            denom = tp + fp + fn
            if denom > 0:
                iou = tp / denom
                dice = (
                    (2.0 * tp) / (2.0 * tp + fp + fn)
                    if (2.0 * tp + fp + fn) > 0
                    else 0.0
                )
                per_class_iou[k] = float(iou)
                per_class_dice[k] = float(dice)
                valid_class_ious.append(iou)
            else:
                per_class_iou[k] = 1.0
                per_class_dice[k] = 1.0

        mean_iou = (
            sum(valid_class_ious) / len(valid_class_ious) if valid_class_ious else 1.0
        )

        return SegmentationMetricsResult(
            num_classes=self.num_classes,
            total_pixels=total_pixels,
            pixel_accuracy=float(pixel_acc),
            mean_iou=float(mean_iou),
            per_class_iou=per_class_iou,
            per_class_dice=per_class_dice,
            confusion_matrix=self.matrix,
        )
