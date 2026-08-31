"""Occlusion sensitivity attribution with sliding window heatmap aggregation."""

from __future__ import annotations

import copy

from prism.core.errors import ValidationError
from prism.explainability.attribution import (
    AttributionMethod,
    AttributionNormalizationPolicy,
    AttributionResult,
    AttributionSpecification,
    OcclusionFillPolicy,
    TargetClassMode,
    compute_attribution_statistics,
    normalize_attribution_map,
)
from prism.explainability.gradients import _resolve_target_class
from prism.models.base import BaseVisionModel


def compute_occlusion_sensitivity(
    model: BaseVisionModel,
    image: list[list[list[float]]],
    specification: AttributionSpecification | None = None,
    target_mode: TargetClassMode = TargetClassMode.PREDICTED_CLASS,
    explicit_target_class: int | None = None,
    true_class: int | None = None,
    window_size: tuple[int, int] = (2, 2),
    stride: tuple[int, int] = (1, 1),
    fill_policy: OcclusionFillPolicy = OcclusionFillPolicy.ZERO,
    max_windows: int = 512,
    normalization: AttributionNormalizationPolicy = (
        AttributionNormalizationPolicy.MIN_MAX_ABSOLUTE
    ),
    sample_id: str = "sample",
) -> AttributionResult:
    """Compute model-agnostic perturbation occlusion sensitivity.

    Args:
        model: Frozen vision model under evaluation.
        image: 3D input image tensor [C, H, W].
        specification: Optional explicit AttributionSpecification.
        target_mode: Class selection mode if specification not provided.
        explicit_target_class: Target class index when mode is EXPLICIT_CLASS.
        true_class: Ground-truth target class if known.
        window_size: (win_h, win_w) spatial window dimensions.
        stride: (stride_h, stride_w) vertical and horizontal strides.
        fill_policy: ZERO (0.0) or IMAGE_MEAN.
        max_windows: Maximum permitted forward passes safeguard.
        normalization: Normalization policy for 2D heatmap.
        sample_id: Unique identifier of evaluated sample.

    Returns:
        Standardized AttributionResult envelope.
    """
    if not image or not image[0] or not image[0][0]:
        raise ValidationError("Input image must be a non-empty 3D tensor [C, H, W].")

    c = len(image)
    h = len(image[0])
    w = len(image[0][0])

    if specification is None:
        specification = AttributionSpecification(
            method=AttributionMethod.OCCLUSION_SENSITIVITY,
            target_mode=target_mode,
            explicit_target_class=explicit_target_class,
            occlusion_window_size=window_size,
            occlusion_stride=stride,
            occlusion_fill=fill_policy,
            occlusion_max_windows=max_windows,
            normalization=normalization,
        )

    win_h, win_w = specification.occlusion_window_size
    str_h, str_w = specification.occlusion_stride

    if win_h <= 0 or win_w <= 0 or str_h <= 0 or str_w <= 0:
        raise ValidationError(
            f"Window dimensions and strides must be positive, "
            f"got window={specification.occlusion_window_size}, "
            f"stride={specification.occlusion_stride}."
        )

    # 1. Compute clean baseline score in evaluation mode
    was_training = model.is_training
    model.eval()

    try:
        clean_logits = model.forward([image])[0]
        num_classes = len(clean_logits)

        target_class = _resolve_target_class(
            logits=clean_logits,
            target_mode=specification.target_mode,
            explicit_target_class=specification.explicit_target_class,
            true_class=true_class,
        )
        predicted_class = max(range(num_classes), key=lambda i: clean_logits[i])
        clean_target_score = clean_logits[target_class]

        # Compute fill values per channel
        if specification.occlusion_fill == OcclusionFillPolicy.IMAGE_MEAN:
            fill_vals = [
                sum(image[ch][r][col] for r in range(h) for col in range(w))
                / float(h * w)
                for ch in range(c)
            ]
        else:
            fill_vals = [0.0] * c

        # 2. Plan sliding window positions
        window_coords: list[tuple[int, int, int, int]] = []
        r_start = 0
        while r_start < h:
            r_end = min(h, r_start + win_h)
            c_start = 0
            while c_start < w:
                c_end = min(w, c_start + win_w)
                window_coords.append((r_start, r_end, c_start, c_end))
                c_start += str_w
            r_start += str_h

        total_windows = len(window_coords)
        if total_windows > specification.occlusion_max_windows:
            raise ValidationError(
                f"Occlusion window count ({total_windows}) exceeds safeguard "
                f"({specification.occlusion_max_windows})."
            )

        # 3. Slide occlusion window and measure target score degradation
        heatmap_accum = [[0.0 for _ in range(w)] for _ in range(h)]
        count_accum = [[0 for _ in range(w)] for _ in range(h)]

        for r_s, r_e, c_s, c_e in window_coords:
            # Create occluded copy
            occ_image = copy.deepcopy(image)
            for ch in range(c):
                f_val = fill_vals[ch]
                for r in range(r_s, r_e):
                    for col in range(c_s, c_e):
                        occ_image[ch][r][col] = f_val

            occ_logits = model.forward([occ_image])[0]
            occ_target_score = occ_logits[target_class]

            # Importance = drop in target logit: S_clean - S_occ
            score_drop = clean_target_score - occ_target_score

            for r in range(r_s, r_e):
                for col in range(c_s, c_e):
                    heatmap_accum[r][col] += score_drop
                    count_accum[r][col] += 1

        # 4. Average overlapping window contributions
        raw_heatmap: list[list[float]] = []
        for r in range(h):
            row: list[float] = []
            for col in range(w):
                cnt = count_accum[r][col]
                avg_val = heatmap_accum[r][col] / float(cnt) if cnt > 0 else 0.0
                row.append(avg_val)
            raw_heatmap.append(row)

        # 5. Normalization
        norm_heatmap = normalize_attribution_map(
            raw_heatmap, policy=specification.normalization
        )

        flat_raw = [raw_heatmap[r][c] for r in range(h) for c in range(w)]
        pos_mass = float(sum(v for v in flat_raw if v > 0.0))
        neg_mass = float(sum(abs(v) for v in flat_raw if v < 0.0))
        abs_mass = float(sum(abs(v) for v in flat_raw))

        stats = compute_attribution_statistics(norm_heatmap)

        warnings: list[str] = []
        if abs_mass < 1e-9:
            warnings.append("low_attribution_signal")
        if not stats.is_finite:
            warnings.append("non_finite_attribution_values")

        arch_name = (
            model.spec.family.value if hasattr(model.spec, "family") else "vision_model"
        )

        return AttributionResult(
            sample_id=sample_id,
            model_id=model.model_id,
            architecture=arch_name,
            method=AttributionMethod.OCCLUSION_SENSITIVITY,
            specification=specification,
            target_class=target_class,
            predicted_class=predicted_class,
            true_class=true_class,
            target_score=float(clean_target_score),
            predicted_score=float(clean_logits[predicted_class]),
            source_image_shape=[c, h, w],
            attribution_shape=[h, w],
            raw_attribution_map=raw_heatmap,
            normalized_attribution_map=norm_heatmap,
            statistics=stats,
            positive_mass=pos_mass,
            negative_mass=neg_mass,
            absolute_mass=abs_mass,
            method_metadata={
                "window_size": list(specification.occlusion_window_size),
                "stride": list(specification.occlusion_stride),
                "fill_policy": specification.occlusion_fill.value,
                "total_windows": total_windows,
            },
            warnings=warnings,
        )

    finally:
        model.zero_grad()
        if was_training:
            model.train()
