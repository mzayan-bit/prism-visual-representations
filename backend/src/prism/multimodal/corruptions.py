"""Robustness and Representation Drift of Multimodal Vision-Language Alignment."""

from __future__ import annotations

import math
from typing import Any

from prism.core.errors import ValidationError
from prism.multimodal.contracts import ClassPrompt, VisionLanguageSample
from prism.multimodal.embeddings import TextEncoder, VisualProjectionHead
from prism.multimodal.evaluation import (
    evaluate_cross_modal_retrieval,
    evaluate_zero_shot_classification,
)
from prism.multimodal.tokenizer import SimpleTokenizer
from prism.robustness.corruptions import (
    apply_brightness_shift,
    apply_gaussian_noise,
    apply_rectangular_occlusion,
    apply_spatial_blur,
)
from prism.ssl.adapter import RepresentationEncoder
from prism.ssl.projection import normalize_embeddings


def apply_multimodal_image_corruption(
    image: list[list[list[float]]],
    corruption_type: str,
    severity: int = 2,
    seed: int = 42,
) -> list[list[list[float]]]:
    """Apply visual corruption to image while keeping paired text fixed."""
    if corruption_type == "gaussian_noise":
        sigmas = {1: 0.05, 2: 0.12, 3: 0.22, 4: 0.35, 5: 0.50}
        sigma = sigmas.get(severity, 0.12)
        return apply_gaussian_noise(image, sigma=sigma, seed=seed)
    elif corruption_type == "blur":
        params = {1: (3, 0.8), 2: (3, 1.2), 3: (5, 1.8), 4: (5, 2.5), 5: (7, 3.2)}
        ksize, sigma = params.get(severity, (3, 1.2))
        return apply_spatial_blur(image, kernel_size=ksize, sigma=sigma)
    elif corruption_type == "brightness":
        deltas = {1: 0.15, 2: 0.25, 3: 0.40, 4: 0.55, 5: 0.70}
        delta = deltas.get(severity, 0.25)
        return apply_brightness_shift(image, delta=delta)
    elif corruption_type == "occlusion":
        ratios = {1: 0.08, 2: 0.15, 3: 0.25, 4: 0.38, 5: 0.50}
        ratio = ratios.get(severity, 0.15)
        return apply_rectangular_occlusion(image, area_ratio=ratio, seed=seed)
    elif corruption_type == "clean":
        return [list(r) for r in image]
    else:
        raise ValidationError(f"Unsupported corruption type: {corruption_type}")


def evaluate_multimodal_alignment_robustness(
    samples: list[VisionLanguageSample],
    visual_encoder: RepresentationEncoder,
    visual_projection: VisualProjectionHead,
    text_encoder: TextEncoder,
    tokenizer: SimpleTokenizer,
    class_prompts: list[ClassPrompt] | None = None,
    corruptions: list[str] | None = None,
    severity: int = 2,
    seed: int = 42,
) -> dict[str, Any]:
    """Quantify degradation of alignment, retrieval, and zero-shot under corruption."""
    if not samples:
        raise ValidationError("Cannot evaluate robustness on empty sample list.")

    corruptions_to_eval = corruptions or [
        "clean",
        "gaussian_noise",
        "blur",
        "brightness",
        "occlusion",
    ]
    n = len(samples)
    sample_ids = [s.sample_id for s in samples]
    true_labels = [s.class_label if s.class_label is not None else 0 for s in samples]
    class_names = sorted({s.class_name or f"class_{s.class_label}" for s in samples})

    # 1. Encode Fixed Text Modality
    tokenized = [tokenizer.encode(s.text) for s in samples]
    token_ids = [t.token_ids for t in tokenized]
    masks = [t.attention_mask for t in tokenized]
    raw_text, _ = text_encoder.forward(token_ids, masks)
    fixed_text_embeds, _ = normalize_embeddings(raw_text)

    # 2. Encode Clean Image Modality
    clean_images = [s.image for s in samples]
    clean_h = visual_encoder.forward(clean_images)
    clean_z = visual_projection.forward(clean_h)
    clean_image_embeds, _ = normalize_embeddings(clean_z)

    results_by_corruption: dict[str, Any] = {}

    for c_name in corruptions_to_eval:
        if c_name == "clean":
            corr_images = clean_images
        else:
            corr_images = [
                apply_multimodal_image_corruption(
                    s.image, c_name, severity=severity, seed=seed + idx
                )
                for idx, s in enumerate(samples)
            ]

        corr_h = visual_encoder.forward(corr_images)
        corr_z = visual_projection.forward(corr_h)
        corr_image_embeds, _ = normalize_embeddings(corr_z)

        # 3. Compute Metrics
        paired_cosines: list[float] = []
        visual_drifts: list[float] = []
        alignment_drifts: list[float] = []

        for i in range(n):
            v_clean = clean_image_embeds[i]
            v_corr = corr_image_embeds[i]
            t_fixed = fixed_text_embeds[i]

            cos_paired = sum(a * b for a, b in zip(v_corr, t_fixed, strict=True))
            paired_cosines.append(cos_paired)

            v_drift = math.sqrt(
                sum((a - b) ** 2 for a, b in zip(v_clean, v_corr, strict=True))
            )
            visual_drifts.append(v_drift)

            align_drift = math.sqrt(
                sum((a - b) ** 2 for a, b in zip(v_corr, t_fixed, strict=True))
            )
            alignment_drifts.append(align_drift)

        corr_retrieval, _, _ = evaluate_cross_modal_retrieval(
            image_embeddings=corr_image_embeds,
            text_embeddings=fixed_text_embeds,
            sample_ids=sample_ids,
        )

        corr_zero_shot = None
        if class_prompts:
            corr_zero_shot = evaluate_zero_shot_classification(
                image_embeddings=corr_image_embeds,
                class_prompts=class_prompts,
                text_encoder=text_encoder,
                tokenizer=tokenizer,
                true_labels=true_labels,
                class_names=class_names,
            )

        mean_paired_cos = sum(paired_cosines) / float(n)
        cosine_drop = sum(
            (
                sum(
                    a * b
                    for a, b in zip(
                        clean_image_embeds[i], fixed_text_embeds[i], strict=True
                    )
                )
                - paired_cosines[i]
            )
            for i in range(n)
        ) / float(n)

        results_by_corruption[c_name] = {
            "corruption": c_name,
            "severity": severity if c_name != "clean" else 0,
            "mean_paired_cosine": mean_paired_cos,
            "cosine_drop": max(0.0, cosine_drop),
            "mean_visual_drift": sum(visual_drifts) / float(n),
            "mean_alignment_drift": sum(alignment_drifts) / float(n),
            "image_to_text_r1": corr_retrieval.image_to_text_r1,
            "image_to_text_r3": corr_retrieval.image_to_text_r3,
            "image_to_text_mrr": corr_retrieval.image_to_text_mrr,
            "zero_shot_accuracy": (corr_zero_shot.accuracy if corr_zero_shot else None),
        }

    return {
        "corruptions": corruptions_to_eval,
        "results": results_by_corruption,
    }
