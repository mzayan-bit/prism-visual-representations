"""Vision Transformer CLS-to-patch attention attribution implementation."""

from __future__ import annotations

from typing import Any

from prism.core.errors import ValidationError
from prism.explainability.attribution import (
    AttributionMethod,
    AttributionNormalizationPolicy,
    AttributionResult,
    AttributionSpecification,
    TargetClassMode,
    ViTAttentionHeadPolicy,
    compute_attribution_statistics,
    normalize_attribution_map,
)
from prism.explainability.grad_cam import upsample_bilinear_2d
from prism.explainability.gradients import _resolve_target_class
from prism.models.base import BaseVisionModel
from prism.models.transformer import VisionTransformer


def compute_vit_attention_attribution(
    model: BaseVisionModel,
    image: list[list[list[float]]],
    specification: AttributionSpecification | None = None,
    target_mode: TargetClassMode = TargetClassMode.PREDICTED_CLASS,
    explicit_target_class: int | None = None,
    true_class: int | None = None,
    head_policy: ViTAttentionHeadPolicy = ViTAttentionHeadPolicy.MEAN_HEADS,
    head_index: int | None = None,
    layer_index: int = -1,
    normalization: AttributionNormalizationPolicy = (
        AttributionNormalizationPolicy.MIN_MAX_ABSOLUTE
    ),
    sample_id: str = "sample",
) -> AttributionResult:
    """Compute CLS-to-patch spatial attention attribution for Vision Transformers.

    Args:
        model: VisionTransformer model under evaluation.
        image: 3D input image tensor [C, H, W].
        specification: Optional explicit AttributionSpecification.
        target_mode: Class selection mode if specification not provided.
        explicit_target_class: Target class index when mode is EXPLICIT_CLASS.
        true_class: Ground-truth target class if known.
        head_policy: Multi-head aggregation policy (MEAN_HEADS or SPECIFIC_HEAD).
        head_index: Head index when policy is SPECIFIC_HEAD.
        layer_index: Transformer encoder layer index (-1 for final layer).
        normalization: Heatmap normalization policy.
        sample_id: Unique identifier of evaluated sample.

    Returns:
        Standardized AttributionResult envelope.
    """
    if not isinstance(model, VisionTransformer):
        raise ValidationError(
            f"ViT Attention Attribution is only supported for VisionTransformer, "
            f"got {type(model).__name__}."
        )

    if not image or not image[0] or not image[0][0]:
        raise ValidationError("Input image must be a non-empty 3D tensor [C, H, W].")

    c = len(image)
    h = len(image[0])
    w = len(image[0][0])

    if specification is None:
        specification = AttributionSpecification(
            method=AttributionMethod.VIT_ATTENTION,
            target_mode=target_mode,
            explicit_target_class=explicit_target_class,
            vit_head_policy=head_policy,
            vit_head_index=head_index,
            vit_layer_index=layer_index,
            normalization=normalization,
        )

    was_training = model.is_training
    model.eval()

    try:
        # 1. Forward pass
        logits_batch = model.forward([image])
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

        # 3. Resolve encoder layer
        num_encoder_layers = len(model.encoder.blocks)
        resolved_layer_idx = specification.vit_layer_index
        if resolved_layer_idx < 0:
            resolved_layer_idx = num_encoder_layers + resolved_layer_idx

        if resolved_layer_idx < 0 or resolved_layer_idx >= num_encoder_layers:
            raise ValidationError(
                f"Invalid vit_layer_index {specification.vit_layer_index}. "
                f"Model has {num_encoder_layers} encoder layers."
            )

        target_block = model.encoder.blocks[resolved_layer_idx]
        attn_weights = target_block.last_attention_weights
        if attn_weights is None:
            raise ValidationError(
                f"No attention weights recorded in encoder block {resolved_layer_idx}."
            )

        # Shape: [1, num_heads, seq_len, seq_len]
        sample_attn = attn_weights[0]  # [num_heads, seq_len, seq_len]
        num_heads = len(sample_attn)
        seq_len = len(sample_attn[0])
        num_patches = model.geometry.total_patches
        grid_rows = model.geometry.patches_per_column
        grid_cols = model.geometry.patches_per_row

        if seq_len != num_patches + 1:
            raise ValidationError(
                f"Sequence length ({seq_len}) does not match "
                f"patches ({num_patches}) + 1."
            )

        # 4. Extract CLS query row (index 0) to patch keys (indices 1 .. seq_len-1)
        if specification.vit_head_policy == ViTAttentionHeadPolicy.SPECIFIC_HEAD:
            head_idx = specification.vit_head_index
            if head_idx is None or head_idx < 0 or head_idx >= num_heads:
                raise ValidationError(
                    f"vit_head_index {head_idx} is out of bounds [0, {num_heads - 1}]."
                )
            patch_attentions = [
                sample_attn[head_idx][0][1 + p] for p in range(num_patches)
            ]
        else:
            # MEAN_HEADS: average over all heads
            patch_attentions = []
            for p in range(num_patches):
                avg_p = sum(
                    sample_attn[head][0][1 + p] for head in range(num_heads)
                ) / float(num_heads)
                patch_attentions.append(avg_p)

        # 5. Reshape 1D patch vector to 2D patch grid [grid_rows, grid_cols]
        patch_grid_2d: list[list[float]] = []
        for r in range(grid_rows):
            row = [patch_attentions[r * grid_cols + col] for col in range(grid_cols)]
            patch_grid_2d.append(row)

        # 6. Upsample patch grid to input image dimensions [H, W]
        raw_heatmap = upsample_bilinear_2d(patch_grid_2d, target_h=h, target_w=w)

        # 7. Normalization
        norm_heatmap = normalize_attribution_map(
            raw_heatmap, policy=specification.normalization
        )

        flat_raw = [raw_heatmap[r][c] for r in range(h) for c in range(w)]
        pos_mass = float(sum(v for v in flat_raw if v > 0.0))
        neg_mass = 0.0
        abs_mass = pos_mass

        stats = compute_attribution_statistics(norm_heatmap)

        warnings: list[str] = []
        if abs_mass < 1e-9:
            warnings.append("low_attribution_signal")
        if not stats.is_finite:
            warnings.append("non_finite_attribution_values")

        method_metadata: dict[str, Any] = {
            "encoder_layer_index": resolved_layer_idx,
            "head_policy": specification.vit_head_policy.value,
            "head_index": specification.vit_head_index,
            "total_heads": num_heads,
            "patch_grid_shape": [grid_rows, grid_cols],
            "total_patches": num_patches,
            "cls_to_cls_excluded": True,
        }

        return AttributionResult(
            sample_id=sample_id,
            model_id=model.model_id,
            architecture="transformer",
            method=AttributionMethod.VIT_ATTENTION,
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
