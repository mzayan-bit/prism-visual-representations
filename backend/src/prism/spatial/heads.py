"""Lightweight detection and segmentation task heads with analytical backpropagation."""

from __future__ import annotations

import copy
import math
from typing import Any

from prism.core.errors import ValidationError
from prism.models.convolution import Conv2D
from prism.spatial.annotations import BoundingBox, DetectionPrediction
from prism.spatial.enums import SegmentationResizePolicy


def _sigmoid(x: float) -> float:
    """Numerically stable sigmoid function."""
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    else:
        z = math.exp(x)
        return z / (1.0 + z)


def _softmax(logits: list[float]) -> list[float]:
    """Numerically stable softmax for 1D float list."""
    if not logits:
        return []
    max_l = max(logits)
    exp_vals = [math.exp(v - max_l) for v in logits]
    sum_exp = sum(exp_vals)
    if sum_exp == 0.0:
        return [1.0 / len(logits) for _ in logits]
    return [v / sum_exp for v in exp_vals]


class GridDetectionHead:
    """Lightweight grid-based object detection head."""

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        seed: int = 42,
    ) -> None:
        if in_channels <= 0:
            raise ValidationError(f"in_channels must be positive, got {in_channels}.")
        if num_classes <= 0:
            raise ValidationError(f"num_classes must be positive, got {num_classes}.")

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.out_channels = 1 + num_classes + 4  # obj (1) + cls (K) + box (4)

        self.conv = Conv2D(
            in_channels=in_channels,
            out_channels=self.out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
            seed=seed,
            activation="none",
        )
        self._cached_output: list[list[list[list[float]]]] | None = None

    def forward(
        self, features: list[list[list[list[float]]]]
    ) -> list[list[list[list[float]]]]:
        """Compute grid detection output logits [N, 1 + K + 4, H_f, W_f]."""
        output = self.conv.forward(features)
        self._cached_output = output
        return output

    def backward(
        self, d_out: list[list[list[list[float]]]]
    ) -> list[list[list[list[float]]]]:
        """Propagate gradients through 1x1 conv layer back to input features."""
        return self.conv.backward(d_out)

    def zero_grad(self) -> None:
        """Reset parameter gradients to zero."""
        self.conv.zero_grad()

    def get_parameters(self) -> dict[str, Any]:
        """Retrieve trainable parameter dict."""
        return {
            "conv_weights": copy.deepcopy(self.conv.weights),
            "conv_bias": copy.deepcopy(self.conv.bias_weights),
        }

    def set_parameters(self, params: dict[str, Any]) -> None:
        """Load parameter weights into head."""
        if "conv_weights" in params:
            self.conv.weights = copy.deepcopy(params["conv_weights"])
        if "conv_bias" in params:
            self.conv.bias_weights = copy.deepcopy(params["conv_bias"])

    def get_gradients(self) -> dict[str, Any]:
        """Retrieve computed parameter gradients."""
        return {
            "conv_weights": copy.deepcopy(self.conv.grad_weights),
            "conv_bias": copy.deepcopy(self.conv.grad_bias_weights),
        }

    def decode_predictions(
        self,
        outputs: list[list[list[list[float]]]],
        objectness_threshold: float = 0.3,
        sample_ids: list[str] | None = None,
    ) -> list[DetectionPrediction]:
        """Decode raw output tensor [N, O, H_f, W_f] into DetectionPredictions."""
        n_samples = len(outputs)
        h_f = len(outputs[0][0])
        w_f = len(outputs[0][0][0])

        predictions: list[DetectionPrediction] = []

        for n in range(n_samples):
            s_id = (
                sample_ids[n]
                if sample_ids is not None and n < len(sample_ids)
                else f"sample_{n}"
            )
            boxes: list[BoundingBox] = []
            obj_scores: list[float] = []
            cls_probs_list: list[list[float]] = []
            cls_ids: list[int] = []
            confidences: list[float] = []
            grid_coords: list[tuple[int, int]] = []

            for r in range(h_f):
                for c in range(w_f):
                    raw_obj = outputs[n][0][r][c]
                    obj_score = _sigmoid(raw_obj)

                    if obj_score < objectness_threshold:
                        continue

                    raw_cls = [outputs[n][1 + k][r][c] for k in range(self.num_classes)]
                    cls_probs = _softmax(raw_cls)
                    best_cls = max(range(self.num_classes), key=lambda k: cls_probs[k])
                    best_cls_prob = cls_probs[best_cls]

                    tx = outputs[n][1 + self.num_classes][r][c]
                    ty = outputs[n][1 + self.num_classes + 1][r][c]
                    tw = outputs[n][1 + self.num_classes + 2][r][c]
                    th = outputs[n][1 + self.num_classes + 3][r][c]

                    center_x = (float(c) + _sigmoid(tx)) / float(w_f)
                    center_y = (float(r) + _sigmoid(ty)) / float(h_f)
                    box_w = max(0.01, min(1.0, _sigmoid(tw)))
                    box_h = max(0.01, min(1.0, _sigmoid(th)))

                    x_min = max(0.0, min(0.99, center_x - box_w / 2.0))
                    y_min = max(0.0, min(0.99, center_y - box_h / 2.0))
                    x_max = max(x_min + 0.01, min(1.0, center_x + box_w / 2.0))
                    y_max = max(y_min + 0.01, min(1.0, center_y + box_h / 2.0))

                    box = BoundingBox(
                        x_min=x_min,
                        y_min=y_min,
                        x_max=x_max,
                        y_max=y_max,
                    )

                    confidence = obj_score * best_cls_prob

                    boxes.append(box)
                    obj_scores.append(obj_score)
                    cls_probs_list.append(cls_probs)
                    cls_ids.append(best_cls)
                    confidences.append(confidence)
                    grid_coords.append((r, c))

            prediction = DetectionPrediction(
                sample_id=s_id,
                boxes=boxes,
                objectness_scores=obj_scores,
                class_probabilities=cls_probs_list,
                class_ids=cls_ids,
                confidences=confidences,
                grid_coords=grid_coords,
            )
            predictions.append(prediction)

        return predictions


class SegmentationHead:
    """Lightweight semantic segmentation head."""

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        target_spatial_shape: tuple[int, int] = (32, 32),
        resize_policy: SegmentationResizePolicy = SegmentationResizePolicy.NEAREST,
        seed: int = 42,
    ) -> None:
        if in_channels <= 0:
            raise ValidationError(f"in_channels must be positive, got {in_channels}.")
        if num_classes <= 0:
            raise ValidationError(f"num_classes must be positive, got {num_classes}.")

        h_tgt, w_tgt = target_spatial_shape
        if h_tgt <= 0 or w_tgt <= 0:
            raise ValidationError(
                f"target_spatial_shape must be positive, got {target_spatial_shape}."
            )

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.target_height = h_tgt
        self.target_width = w_tgt
        self.resize_policy = resize_policy

        self.conv = Conv2D(
            in_channels=in_channels,
            out_channels=num_classes,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
            seed=seed,
            activation="none",
        )
        self._cached_features_shape: tuple[int, int, int, int] | None = None
        self._cached_conv_out: list[list[list[list[float]]]] | None = None

    def forward(
        self, features: list[list[list[list[float]]]]
    ) -> list[list[list[list[float]]]]:
        """Project features and upsample to target resolution [N, K, H_tgt, W_tgt]."""
        n_samples = len(features)
        c_feat = len(features[0])
        h_f = len(features[0][0])
        w_f = len(features[0][0][0])
        self._cached_features_shape = (n_samples, c_feat, h_f, w_f)

        conv_out = self.conv.forward(features)
        self._cached_conv_out = conv_out

        h_tgt = self.target_height
        w_tgt = self.target_width

        if h_f == h_tgt and w_f == w_tgt:
            return conv_out

        upsampled: list[list[list[list[float]]]] = []

        if self.resize_policy == SegmentationResizePolicy.NEAREST:
            for n in range(n_samples):
                sample_out: list[list[list[float]]] = []
                for k in range(self.num_classes):
                    ch_out: list[list[float]] = []
                    for y in range(h_tgt):
                        r_in = min(h_f - 1, int(y * h_f / h_tgt))
                        row = [
                            conv_out[n][k][r_in][min(w_f - 1, int(x * w_f / w_tgt))]
                            for x in range(w_tgt)
                        ]
                        ch_out.append(row)
                    sample_out.append(ch_out)
                upsampled.append(sample_out)
            return upsampled

        elif self.resize_policy == SegmentationResizePolicy.BILINEAR:
            for n in range(n_samples):
                sample_out = []
                for k in range(self.num_classes):
                    ch_out = []
                    for y in range(h_tgt):
                        v_pos = (float(y) + 0.5) * float(h_f) / float(h_tgt) - 0.5
                        r0 = max(0, min(h_f - 1, math.floor(v_pos)))
                        r1 = max(0, min(h_f - 1, r0 + 1))
                        wr1 = v_pos - float(r0)
                        wr0 = 1.0 - wr1

                        row = []
                        for x in range(w_tgt):
                            u_pos = (float(x) + 0.5) * float(w_f) / float(w_tgt) - 0.5
                            c0 = max(0, min(w_f - 1, math.floor(u_pos)))
                            c1 = max(0, min(w_f - 1, c0 + 1))
                            wc1 = u_pos - float(c0)
                            wc0 = 1.0 - wc1

                            v00 = conv_out[n][k][r0][c0]
                            v01 = conv_out[n][k][r0][c1]
                            v10 = conv_out[n][k][r1][c0]
                            v11 = conv_out[n][k][r1][c1]

                            interpolated = wr0 * (wc0 * v00 + wc1 * v01) + wr1 * (
                                wc0 * v10 + wc1 * v11
                            )
                            row.append(interpolated)
                        ch_out.append(row)
                    sample_out.append(ch_out)
                upsampled.append(sample_out)
            return upsampled

        raise ValidationError(f"Unsupported resize policy: {self.resize_policy}.")

    def backward(
        self, d_logits: list[list[list[list[float]]]]
    ) -> list[list[list[list[float]]]]:
        """Propagate gradients from pixel logits [N, K, H, W] back to features."""
        if self._cached_conv_out is None or self._cached_features_shape is None:
            raise ValidationError("Cannot perform backward before forward pass.")

        n_samples, _, h_f, w_f = self._cached_features_shape
        h_tgt = self.target_height
        w_tgt = self.target_width

        d_conv: list[list[list[list[float]]]] = [
            [
                [[0.0 for _ in range(w_f)] for _ in range(h_f)]
                for _ in range(self.num_classes)
            ]
            for _ in range(n_samples)
        ]

        if h_f == h_tgt and w_f == w_tgt:
            d_conv = d_logits
        elif self.resize_policy == SegmentationResizePolicy.NEAREST:
            for n in range(n_samples):
                for k in range(self.num_classes):
                    for y in range(h_tgt):
                        r_in = min(h_f - 1, int(y * h_f / h_tgt))
                        for x in range(w_tgt):
                            c_in = min(w_f - 1, int(x * w_f / w_tgt))
                            d_conv[n][k][r_in][c_in] += d_logits[n][k][y][x]
        elif self.resize_policy == SegmentationResizePolicy.BILINEAR:
            for n in range(n_samples):
                for k in range(self.num_classes):
                    for y in range(h_tgt):
                        v_pos = (float(y) + 0.5) * float(h_f) / float(h_tgt) - 0.5
                        r0 = max(0, min(h_f - 1, math.floor(v_pos)))
                        r1 = max(0, min(h_f - 1, r0 + 1))
                        wr1 = v_pos - float(r0)
                        wr0 = 1.0 - wr1

                        for x in range(w_tgt):
                            u_pos = (float(x) + 0.5) * float(w_f) / float(w_tgt) - 0.5
                            c0 = max(0, min(w_f - 1, math.floor(u_pos)))
                            c1 = max(0, min(w_f - 1, c0 + 1))
                            wc1 = u_pos - float(c0)
                            wc0 = 1.0 - wc1

                            grad = d_logits[n][k][y][x]
                            d_conv[n][k][r0][c0] += grad * wr0 * wc0
                            d_conv[n][k][r0][c1] += grad * wr0 * wc1
                            d_conv[n][k][r1][c0] += grad * wr1 * wc0
                            d_conv[n][k][r1][c1] += grad * wr1 * wc1

        return self.conv.backward(d_conv)

    def predict_masks(
        self, logits: list[list[list[list[float]]]]
    ) -> list[list[list[int]]]:
        """Convert pixel logits [N, K, H, W] to integer class predictions [N, H, W]."""
        n_samples = len(logits)
        h = len(logits[0][0])
        w = len(logits[0][0][0])

        predictions: list[list[list[int]]] = []
        for n in range(n_samples):
            sample_mask: list[list[int]] = []
            for y in range(h):
                row: list[int] = []
                for x in range(w):
                    best_k = 0
                    best_val = logits[n][0][y][x]
                    for k in range(1, self.num_classes):
                        val = logits[n][k][y][x]
                        if val > best_val:
                            best_val = val
                            best_k = k
                    row.append(best_k)
                sample_mask.append(row)
            predictions.append(sample_mask)
        return predictions

    def zero_grad(self) -> None:
        """Reset parameter gradients."""
        self.conv.zero_grad()

    def get_parameters(self) -> dict[str, Any]:
        """Retrieve trainable parameter dict."""
        return {
            "conv_weights": copy.deepcopy(self.conv.weights),
            "conv_bias": copy.deepcopy(self.conv.bias_weights),
        }

    def set_parameters(self, params: dict[str, Any]) -> None:
        """Load parameter weights into head."""
        if "conv_weights" in params:
            self.conv.weights = copy.deepcopy(params["conv_weights"])
        if "conv_bias" in params:
            self.conv.bias_weights = copy.deepcopy(params["conv_bias"])

    def get_gradients(self) -> dict[str, Any]:
        """Retrieve computed parameter gradients."""
        return {
            "conv_weights": copy.deepcopy(self.conv.grad_weights),
            "conv_bias": copy.deepcopy(self.conv.grad_bias_weights),
        }
