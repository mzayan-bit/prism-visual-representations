"""Spatial task loss functions with analytical gradient computation."""

from __future__ import annotations

import math

from prism.core.errors import ValidationError
from prism.spatial.annotations import DetectionSample, SegmentationSample
from prism.spatial.heads import _sigmoid, _softmax


class GridDetectionLoss:
    """Multi-task loss for grid-based object detection."""

    def __init__(
        self,
        lambda_obj: float = 1.0,
        lambda_cls: float = 1.0,
        lambda_box: float = 2.0,
        eps: float = 1e-7,
    ) -> None:
        if lambda_obj < 0.0 or lambda_cls < 0.0 or lambda_box < 0.0:
            raise ValidationError("Loss weight multipliers must be non-negative.")
        self.lambda_obj = lambda_obj
        self.lambda_cls = lambda_cls
        self.lambda_box = lambda_box
        self.eps = eps

    def compute_loss_and_gradients(
        self,
        predictions: list[list[list[list[float]]]],
        targets: list[DetectionSample],
        num_classes: int,
    ) -> tuple[dict[str, float], list[list[list[list[float]]]]]:
        """Compute loss components and analytical gradients."""
        n_samples = len(predictions)
        if n_samples != len(targets):
            raise ValidationError(
                f"Batch size mismatch: {n_samples} preds vs {len(targets)} targets."
            )

        out_channels = len(predictions[0])
        expected_channels = 1 + num_classes + 4
        if out_channels != expected_channels:
            raise ValidationError(
                f"Expected {expected_channels} channels, got {out_channels}."
            )

        h_f = len(predictions[0][0])
        w_f = len(predictions[0][0][0])
        total_cells = n_samples * h_f * w_f

        d_preds: list[list[list[list[float]]]] = [
            [
                [[0.0 for _ in range(w_f)] for _ in range(h_f)]
                for _ in range(out_channels)
            ]
            for _ in range(n_samples)
        ]

        total_obj_loss = 0.0
        total_cls_loss = 0.0
        total_box_loss = 0.0
        num_positives = 0

        for n in range(n_samples):
            sample = targets[n]
            assigned_cells: dict[
                tuple[int, int], tuple[int, float, float, float, float]
            ] = {}

            for ann in sample.annotations:
                c_x, c_y = ann.box.center
                r = max(0, min(h_f - 1, math.floor(c_y * float(h_f))))
                c = max(0, min(w_f - 1, math.floor(c_x * float(w_f))))

                u_x = c_x * float(w_f) - float(c)
                u_y = c_y * float(h_f) - float(r)
                u_x = max(1e-4, min(1.0 - 1e-4, u_x))
                u_y = max(1e-4, min(1.0 - 1e-4, u_y))

                w_tgt = max(1e-4, min(1.0, ann.box.width))
                h_tgt = max(1e-4, min(1.0, ann.box.height))

                if (r, c) not in assigned_cells:
                    assigned_cells[(r, c)] = (ann.class_id, u_x, u_y, w_tgt, h_tgt)

            for r in range(h_f):
                for c in range(w_f):
                    raw_obj = predictions[n][0][r][c]
                    p_obj = _sigmoid(raw_obj)

                    is_pos = (r, c) in assigned_cells
                    y_obj = 1.0 if is_pos else 0.0

                    cell_obj_loss = -(
                        y_obj * math.log(max(self.eps, p_obj))
                        + (1.0 - y_obj) * math.log(max(self.eps, 1.0 - p_obj))
                    )
                    total_obj_loss += cell_obj_loss

                    d_preds[n][0][r][c] = (
                        self.lambda_obj * (p_obj - y_obj) / float(total_cells)
                    )

            for (r, c), (cls_tgt, u_x, u_y, w_tgt, h_tgt) in assigned_cells.items():
                num_positives += 1

                cls_logits = [predictions[n][1 + k][r][c] for k in range(num_classes)]
                cls_probs = _softmax(cls_logits)

                safe_prob = max(self.eps, cls_probs[cls_tgt])
                total_cls_loss += -math.log(safe_prob)

                for k in range(num_classes):
                    y_k = 1.0 if k == cls_tgt else 0.0
                    d_preds[n][1 + k][r][c] = cls_probs[k] - y_k

                tx = predictions[n][1 + num_classes][r][c]
                ty = predictions[n][1 + num_classes + 1][r][c]
                tw = predictions[n][1 + num_classes + 2][r][c]
                th = predictions[n][1 + num_classes + 3][r][c]

                p_x = _sigmoid(tx)
                p_y = _sigmoid(ty)
                p_w = _sigmoid(tw)
                p_h = _sigmoid(th)

                err_x = p_x - u_x
                err_y = p_y - u_y
                err_w = p_w - w_tgt
                err_h = p_h - h_tgt

                cell_box_loss = err_x**2 + err_y**2 + err_w**2 + err_h**2
                total_box_loss += cell_box_loss

                d_preds[n][1 + num_classes][r][c] = 2.0 * err_x * p_x * (1.0 - p_x)
                d_preds[n][1 + num_classes + 1][r][c] = 2.0 * err_y * p_y * (1.0 - p_y)
                d_preds[n][1 + num_classes + 2][r][c] = 2.0 * err_w * p_w * (1.0 - p_w)
                d_preds[n][1 + num_classes + 3][r][c] = 2.0 * err_h * p_h * (1.0 - p_h)

        pos_norm = float(max(1, num_positives))

        mean_obj_loss = total_obj_loss / float(total_cells)
        mean_cls_loss = total_cls_loss / pos_norm
        mean_box_loss = total_box_loss / pos_norm

        for n in range(n_samples):
            for r in range(h_f):
                for c in range(w_f):
                    for k in range(num_classes):
                        d_preds[n][1 + k][r][c] = (
                            self.lambda_cls * d_preds[n][1 + k][r][c] / pos_norm
                        )
                    for b_idx in range(4):
                        ch = 1 + num_classes + b_idx
                        d_preds[n][ch][r][c] = (
                            self.lambda_box * d_preds[n][ch][r][c] / pos_norm
                        )

        total_loss = (
            self.lambda_obj * mean_obj_loss
            + self.lambda_cls * mean_cls_loss
            + self.lambda_box * mean_box_loss
        )

        loss_metrics = {
            "loss": float(total_loss),
            "obj_loss": float(mean_obj_loss),
            "cls_loss": float(mean_cls_loss),
            "box_loss": float(mean_box_loss),
            "num_positive_cells": num_positives,
        }

        return loss_metrics, d_preds


class PixelCrossEntropyLoss:
    """Pixel-wise Softmax Cross-Entropy loss for semantic segmentation."""

    def __init__(self, ignore_index: int | None = None, eps: float = 1e-12) -> None:
        self.ignore_index = ignore_index
        self.eps = eps

    def compute_loss_and_gradients(
        self,
        logits: list[list[list[list[float]]]],
        targets: list[SegmentationSample] | list[list[list[int]]],
    ) -> tuple[dict[str, float], list[list[list[list[float]]]]]:
        """Compute pixel cross-entropy and analytical gradients w.r.t. pixel logits."""
        n_samples = len(logits)
        num_classes = len(logits[0])
        h = len(logits[0][0])
        w = len(logits[0][0][0])

        masks: list[list[list[int]]] = []
        for item in targets:
            if isinstance(item, SegmentationSample):
                masks.append(item.mask)
            else:
                masks.append(item)

        if len(masks) != n_samples:
            raise ValidationError(
                f"Batch size mismatch: {n_samples} logits vs {len(masks)} masks."
            )

        d_logits: list[list[list[list[float]]]] = [
            [[[0.0 for _ in range(w)] for _ in range(h)] for _ in range(num_classes)]
            for _ in range(n_samples)
        ]

        total_loss = 0.0
        valid_pixels = 0

        for n in range(n_samples):
            for y in range(h):
                for x in range(w):
                    target_cls = masks[n][y][x]
                    if (
                        self.ignore_index is not None
                        and target_cls == self.ignore_index
                    ):
                        continue

                    if not (0 <= target_cls < num_classes):
                        raise ValidationError(
                            f"Target class {target_cls} at ({n}, {y}, {x}) "
                            f"out of bounds [0, {num_classes - 1}]."
                        )

                    valid_pixels += 1
                    pixel_logits = [logits[n][k][y][x] for k in range(num_classes)]
                    probs = _softmax(pixel_logits)

                    p_target = max(self.eps, probs[target_cls])
                    total_loss += -math.log(p_target)

                    for k in range(num_classes):
                        y_k = 1.0 if k == target_cls else 0.0
                        d_logits[n][k][y][x] = probs[k] - y_k

        if valid_pixels == 0:
            return {"loss": 0.0, "valid_pixels": 0}, d_logits

        mean_loss = total_loss / float(valid_pixels)

        for n in range(n_samples):
            for k in range(num_classes):
                for y in range(h):
                    for x in range(w):
                        d_logits[n][k][y][x] /= float(valid_pixels)

        return {"loss": float(mean_loss), "valid_pixels": valid_pixels}, d_logits
