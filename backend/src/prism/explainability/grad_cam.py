"""Grad-CAM (Gradient-weighted Class Activation Mapping) implementation."""

from __future__ import annotations

import math
from typing import Any

from prism.core.errors import ValidationError
from prism.explainability.attribution import (
    AttributionMethod,
    AttributionNormalizationPolicy,
    AttributionResult,
    AttributionSpecification,
    TargetClassMode,
    compute_attribution_statistics,
    normalize_attribution_map,
)
from prism.explainability.gradients import _resolve_target_class
from prism.models.base import BaseVisionModel
from prism.models.transformer import VisionTransformer


def upsample_bilinear_2d(
    matrix_2d: list[list[float]], target_h: int, target_w: int
) -> list[list[float]]:
    """Deterministically upsample 2D matrix [H_in, W_in] to [target_h, target_w].

    Args:
        matrix_2d: Input 2D float grid [H_in, W_in].
        target_h: Destination height.
        target_w: Destination width.

    Returns:
        Upsampled 2D float grid [target_h, target_w].
    """
    if not matrix_2d or not matrix_2d[0]:
        raise ValidationError("matrix_2d must be non-empty.")

    src_h = len(matrix_2d)
    src_w = len(matrix_2d[0])

    if src_h == target_h and src_w == target_w:
        return [[matrix_2d[r][c] for c in range(src_w)] for r in range(src_h)]

    if src_h == 1 and src_w == 1:
        val = matrix_2d[0][0]
        return [[val for _ in range(target_w)] for _ in range(target_h)]

    out: list[list[float]] = []
    scale_r = float(src_h - 1) / float(max(1, target_h - 1))
    scale_c = float(src_w - 1) / float(max(1, target_w - 1))

    for r in range(target_h):
        r_src = r * scale_r
        r0 = math.floor(r_src)
        r1 = min(src_h - 1, r0 + 1)
        dr = r_src - float(r0)

        row: list[float] = []
        for c in range(target_w):
            c_src = c * scale_c
            c0 = math.floor(c_src)
            c1 = min(src_w - 1, c0 + 1)
            dc = c_src - float(c0)

            v00 = matrix_2d[r0][c0]
            v01 = matrix_2d[r0][c1]
            v10 = matrix_2d[r1][c0]
            v11 = matrix_2d[r1][c1]

            val = (
                (1.0 - dr) * (1.0 - dc) * v00
                + (1.0 - dr) * dc * v01
                + dr * (1.0 - dc) * v10
                + dr * dc * v11
            )
            row.append(val)
        out.append(row)

    return out


def compute_grad_cam(
    model: BaseVisionModel,
    image: list[list[list[float]]],
    layer_name: str | None = None,
    specification: AttributionSpecification | None = None,
    target_mode: TargetClassMode = TargetClassMode.PREDICTED_CLASS,
    explicit_target_class: int | None = None,
    true_class: int | None = None,
    normalization: AttributionNormalizationPolicy = (
        AttributionNormalizationPolicy.MIN_MAX_ABSOLUTE
    ),
    sample_id: str = "sample",
) -> AttributionResult:
    """Compute Grad-CAM for convolutional and residual neural networks.

    Args:
        model: Vision model under evaluation (CNN or ResNet).
        image: 3D input image tensor [C, H, W].
        layer_name: Spatial convolutional layer name (e.g. 'final_conv', 'final_stage').
        specification: Optional explicit AttributionSpecification.
        target_mode: Class selection mode if specification not provided.
        explicit_target_class: Target class index when mode is EXPLICIT_CLASS.
        true_class: Ground-truth target class if known.
        normalization: Heatmap normalization policy.
        sample_id: Unique identifier of evaluated sample.

    Returns:
        Standardized AttributionResult envelope.
    """
    if isinstance(model, VisionTransformer):
        raise ValidationError(
            "Grad-CAM is not applicable to VisionTransformer. Use "
            "AttributionMethod.VIT_ATTENTION instead."
        )

    if not image or not image[0] or not image[0][0]:
        raise ValidationError("Input image must be a non-empty 3D tensor [C, H, W].")

    c = len(image)
    h = len(image[0])
    w = len(image[0][0])

    resolved_layer = layer_name or "final_conv"
    if specification is not None and specification.layer_name:
        resolved_layer = specification.layer_name

    if specification is None:
        specification = AttributionSpecification(
            method=AttributionMethod.GRAD_CAM,
            target_mode=target_mode,
            explicit_target_class=explicit_target_class,
            layer_name=resolved_layer,
            normalization=normalization,
        )

    was_training = model.is_training
    model.eval()

    try:
        # 1. Forward pass
        batch_input = [image]
        logits_batch = model.forward(batch_input)
        logits = logits_batch[0]
        num_classes = len(logits)

        # 2. Target class resolution
        target_class = _resolve_target_class(
            logits=logits,
            target_mode=specification.target_mode,
            explicit_target_class=specification.explicit_target_class,
            true_class=true_class,
        )
        predicted_class = max(range(num_classes), key=lambda i: logits[i])

        # 3. Backward pass from one-hot target logit derivative
        d_logits = [[1.0 if idx == target_class else 0.0 for idx in range(num_classes)]]
        model.zero_grad()
        model.backward(d_logits)

        # 4. Extract spatial activations A and gradients dS_c / dA
        if not hasattr(model, "extract_spatial_activation_and_gradient"):
            raise ValidationError(
                f"Model '{model.model_id}' does not support spatial "
                f"gradient extraction."
            )

        act_4d, grad_4d = model.extract_spatial_activation_and_gradient(resolved_layer)
        # Squeeze batch dimension: [1, K, H_feat, W_feat] -> [K, H_feat, W_feat]
        act_3d = act_4d[0]
        grad_3d = grad_4d[0]

        num_channels = len(act_3d)
        feat_h = len(act_3d[0])
        feat_w = len(act_3d[0][0])
        spatial_positions = float(feat_h * feat_w)

        # 5. Channel weights: alpha_k^c = (1 / Z) \sum_{i,j} dS_c / dA_{k,i,j}
        alpha_weights: list[float] = []
        for k in range(num_channels):
            spatial_grad_sum = sum(
                grad_3d[k][r][col] for r in range(feat_h) for col in range(feat_w)
            )
            alpha_k = spatial_grad_sum / spatial_positions
            alpha_weights.append(alpha_k)

        # 6. Weighted combination: L^c = ReLU(\sum_k alpha_k^c A^k)
        cam_low_res: list[list[float]] = []
        for r in range(feat_h):
            cam_row: list[float] = []
            for col in range(feat_w):
                linear_comb = sum(
                    alpha_weights[k] * act_3d[k][r][col] for k in range(num_channels)
                )
                # ReLU activation: keep only positive evidence
                cam_val = max(0.0, linear_comb)
                cam_row.append(cam_val)
            cam_low_res.append(cam_row)

        # 7. Upsample low-resolution CAM to original image dimensions [H, W]
        raw_heatmap = upsample_bilinear_2d(cam_low_res, target_h=h, target_w=w)

        # 8. Normalization
        norm_heatmap = normalize_attribution_map(
            raw_heatmap, policy=specification.normalization
        )

        flat_raw = [raw_heatmap[r][c] for r in range(h) for c in range(w)]
        pos_mass = float(sum(v for v in flat_raw if v > 0.0))
        neg_mass = 0.0  # Grad-CAM uses ReLU, negative mass is strictly 0.0
        abs_mass = pos_mass

        stats = compute_attribution_statistics(norm_heatmap)

        warnings: list[str] = []
        if abs_mass < 1e-9:
            warnings.append("low_attribution_signal")
        if not stats.is_finite:
            warnings.append("non_finite_attribution_values")

        arch_name = (
            model.spec.family.value
            if hasattr(model.spec, "family")
            else "convolutional"
        )

        method_metadata: dict[str, Any] = {
            "layer_name": resolved_layer,
            "feature_map_shape": [num_channels, feat_h, feat_w],
            "channel_count": num_channels,
            "alpha_weights_min": min(alpha_weights) if alpha_weights else 0.0,
            "alpha_weights_max": max(alpha_weights) if alpha_weights else 0.0,
        }

        return AttributionResult(
            sample_id=sample_id,
            model_id=model.model_id,
            architecture=arch_name,
            method=AttributionMethod.GRAD_CAM,
            specification=specification,
            target_class=target_class,
            predicted_class=predicted_class,
            true_class=true_class,
            target_score=float(logits[target_class]),
            predicted_score=float(logits[predicted_class]),
            source_image_shape=[c, h, w],
            attribution_shape=[h, w],
            raw_attribution_map=raw_heatmap,
            normalized_attribution_map=norm_heatmap,
            statistics=stats,
            positive_mass=pos_mass,
            negative_mass=neg_mass,
            absolute_mass=abs_mass,
            method_metadata=method_metadata,
            warnings=warnings,
        )

    finally:
        model.zero_grad()
        if was_training:
            model.train()
