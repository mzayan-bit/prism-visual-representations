"""Input gradient saliency and Gradient x Input attribution implementations."""

from __future__ import annotations

from prism.core.errors import ValidationError
from prism.explainability.attribution import (
    AttributionMethod,
    AttributionNormalizationPolicy,
    AttributionResult,
    AttributionSpecification,
    ChannelReductionPolicy,
    TargetClassMode,
    compute_attribution_statistics,
    normalize_attribution_map,
    reduce_channels,
)
from prism.models.base import BaseVisionModel


def _resolve_target_class(
    logits: list[float],
    target_mode: TargetClassMode,
    explicit_target_class: int | None = None,
    true_class: int | None = None,
) -> int:
    """Resolve target class index based on declared mode and available metadata."""
    num_classes = len(logits)
    if target_mode == TargetClassMode.PREDICTED_CLASS:
        # Argmax of logits
        max_val = logits[0]
        max_idx = 0
        for idx in range(1, num_classes):
            if logits[idx] > max_val:
                max_val = logits[idx]
                max_idx = idx
        return max_idx

    elif target_mode == TargetClassMode.TRUE_CLASS:
        if true_class is None:
            raise ValidationError(
                "Cannot compute attribution with TargetClassMode.TRUE_CLASS "
                "when true_class is None."
            )
        if true_class < 0 or true_class >= num_classes:
            raise ValidationError(
                f"true_class index {true_class} out of bounds [0, {num_classes - 1}]."
            )
        return true_class

    elif target_mode == TargetClassMode.EXPLICIT_CLASS:
        if explicit_target_class is None:
            raise ValidationError(
                "Cannot compute attribution with TargetClassMode.EXPLICIT_CLASS "
                "when explicit_target_class is None."
            )
        if explicit_target_class < 0 or explicit_target_class >= num_classes:
            raise ValidationError(
                f"explicit_target_class index {explicit_target_class} out of bounds "
                f"[0, {num_classes - 1}]."
            )
        return explicit_target_class

    else:
        raise ValidationError(f"Unsupported TargetClassMode: {target_mode}")


def compute_input_gradient_saliency(
    model: BaseVisionModel,
    image: list[list[list[float]]],
    specification: AttributionSpecification | None = None,
    target_mode: TargetClassMode = TargetClassMode.PREDICTED_CLASS,
    explicit_target_class: int | None = None,
    true_class: int | None = None,
    channel_reduction: ChannelReductionPolicy = ChannelReductionPolicy.ABS_MAX,
    normalization: AttributionNormalizationPolicy = (
        AttributionNormalizationPolicy.MIN_MAX_ABSOLUTE
    ),
    sample_id: str = "sample",
) -> AttributionResult:
    """Compute vanilla input gradient saliency dS_c / dx via analytical backprop.

    Args:
        model: Frozen vision model under evaluation.
        image: 3D input image tensor [C, H, W].
        specification: Optional explicit AttributionSpecification.
        target_mode: Class selection mode if specification not provided.
        explicit_target_class: Target class index when mode is EXPLICIT_CLASS.
        true_class: Ground-truth target class if known.
        channel_reduction: Strategy to reduce channels to 2D heatmap.
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
            method=AttributionMethod.INPUT_GRADIENT,
            target_mode=target_mode,
            explicit_target_class=explicit_target_class,
            channel_reduction=channel_reduction,
            normalization=normalization,
        )

    # 1. Forward pass in evaluation mode (frozen parameters and running stats)
    was_training = model.is_training
    model.eval()

    try:
        # Wrap single sample into batch of size 1: [1, C, H, W]
        batch_input = [image]
        logits_batch = model.forward(batch_input)
        logits = logits_batch[0]
        num_classes = len(logits)

        # 2. Resolve target class and predicted class
        target_class = _resolve_target_class(
            logits=logits,
            target_mode=specification.target_mode,
            explicit_target_class=specification.explicit_target_class,
            true_class=true_class,
        )
        predicted_class = max(range(num_classes), key=lambda i: logits[i])

        # 3. Create one-hot derivative vector: d(S_target) / d(logits) = 1.0 at target
        d_logits = [[1.0 if idx == target_class else 0.0 for idx in range(num_classes)]]

        # 4. Backward pass to obtain input gradient
        model.zero_grad()
        model.backward(d_logits)

        cached_grad = model.get_last_input_gradient()
        if cached_grad is None:
            raise ValidationError(
                f"Model '{model.model_id}' did not cache input gradients."
            )
        grad_3d = cached_grad[0]  # [C, H, W]

        # 5. Channel reduction: [C, H, W] -> [H, W]
        raw_heatmap = reduce_channels(grad_3d, policy=specification.channel_reduction)

        # 6. Normalization
        norm_heatmap = normalize_attribution_map(
            raw_heatmap, policy=specification.normalization
        )

        # 7. Mass and statistics
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
            method=AttributionMethod.INPUT_GRADIENT,
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
            method_metadata={"reduction": specification.channel_reduction.value},
            warnings=warnings,
        )

    finally:
        model.zero_grad()
        if was_training:
            model.train()


def compute_gradient_x_input(
    model: BaseVisionModel,
    image: list[list[list[float]]],
    specification: AttributionSpecification | None = None,
    target_mode: TargetClassMode = TargetClassMode.PREDICTED_CLASS,
    explicit_target_class: int | None = None,
    true_class: int | None = None,
    channel_reduction: ChannelReductionPolicy = ChannelReductionPolicy.ABS_MAX,
    normalization: AttributionNormalizationPolicy = (
        AttributionNormalizationPolicy.MIN_MAX_ABSOLUTE
    ),
    sample_id: str = "sample",
) -> AttributionResult:
    """Compute Gradient x Input attribution (dS_c / dx) * x elementwise.

    Args:
        model: Frozen vision model under evaluation.
        image: 3D input image tensor [C, H, W].
        specification: Optional explicit AttributionSpecification.
        target_mode: Class selection mode if specification not provided.
        explicit_target_class: Target class index when mode is EXPLICIT_CLASS.
        true_class: Ground-truth target class if known.
        channel_reduction: Strategy to reduce channels to 2D heatmap.
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
            method=AttributionMethod.GRADIENT_X_INPUT,
            target_mode=target_mode,
            explicit_target_class=explicit_target_class,
            channel_reduction=channel_reduction,
            normalization=normalization,
        )

    was_training = model.is_training
    model.eval()

    try:
        batch_input = [image]
        logits_batch = model.forward(batch_input)
        logits = logits_batch[0]
        num_classes = len(logits)

        target_class = _resolve_target_class(
            logits=logits,
            target_mode=specification.target_mode,
            explicit_target_class=specification.explicit_target_class,
            true_class=true_class,
        )
        predicted_class = max(range(num_classes), key=lambda i: logits[i])

        d_logits = [[1.0 if idx == target_class else 0.0 for idx in range(num_classes)]]

        model.zero_grad()
        model.backward(d_logits)

        cached_grad = model.get_last_input_gradient()
        if cached_grad is None:
            raise ValidationError(
                f"Model '{model.model_id}' did not cache input gradients."
            )
        grad_3d = cached_grad[0]  # [C, H, W]

        # Elementwise product: (dS_c / dx) * x
        gxi_3d: list[list[list[float]]] = []
        for ch_idx in range(c):
            ch_plane: list[list[float]] = []
            for r in range(h):
                row = [
                    grad_3d[ch_idx][r][col] * image[ch_idx][r][col] for col in range(w)
                ]
                ch_plane.append(row)
            gxi_3d.append(ch_plane)

        # Channel reduction
        raw_heatmap = reduce_channels(gxi_3d, policy=specification.channel_reduction)

        # Normalization
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
            method=AttributionMethod.GRADIENT_X_INPUT,
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
            method_metadata={"reduction": specification.channel_reduction.value},
            warnings=warnings,
        )

    finally:
        model.zero_grad()
        if was_training:
            model.train()
