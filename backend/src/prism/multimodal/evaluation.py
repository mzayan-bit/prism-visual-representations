"""Evaluation, Retrieval, Zero-Shot Matching, and Shared Geometry."""

from __future__ import annotations

import math
from typing import Any

from prism.core.errors import ValidationError
from prism.multimodal.contracts import (
    ClassPrompt,
    CrossModalCentroidAlignment,
    CrossModalRetrievalSummary,
    MultimodalCollapseSummary,
    RetrievalResult,
    ZeroShotClassificationSummary,
)
from prism.multimodal.embeddings import TextEncoder
from prism.multimodal.enums import RetrievalDirection
from prism.multimodal.tokenizer import SimpleTokenizer
from prism.representations.pca import PrincipalComponentAnalysis
from prism.ssl.projection import normalize_embeddings


def evaluate_cross_modal_retrieval(
    image_embeddings: list[list[float]],
    text_embeddings: list[list[float]],
    sample_ids: list[str],
) -> tuple[
    CrossModalRetrievalSummary,
    list[RetrievalResult],
    list[RetrievalResult],
]:
    """Evaluate Image-to-Text and Text-to-Image retrieval.

    Parameters
    ----------
    image_embeddings : list[list[float]]
        Unit-norm image embeddings [N x D].
    text_embeddings : list[list[float]]
        Unit-norm text embeddings [N x D].
    sample_ids : list[str]
        Unique sample identifiers of length N.

    Returns
    -------
    tuple[CrossModalRetrievalSummary, list[RetrievalResult], list[RetrievalResult]]
        Aggregate summary, i2t individual results, and t2i individual results.
    """
    n = len(image_embeddings)
    if n == 0:
        raise ValidationError("Cannot evaluate retrieval on empty embeddings list.")
    if len(text_embeddings) != n or len(sample_ids) != n:
        msg = (
            f"Mismatched input lengths: images={n}, "
            f"texts={len(text_embeddings)}, ids={len(sample_ids)}"
        )
        raise ValidationError(msg)

    # Compute full N x N similarity matrix S[i][j] = image_i . text_j
    sim_matrix: list[list[float]] = [[0.0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        vi = image_embeddings[i]
        for j in range(n):
            tj = text_embeddings[j]
            sim_matrix[i][j] = sum(a * b for a, b in zip(vi, tj, strict=True))

    # 1. Image-to-Text Retrieval
    i2t_results: list[RetrievalResult] = []
    i2t_ranks: list[int] = []

    for i in range(n):
        scores = [(sim_matrix[i][j], sample_ids[j], j) for j in range(n)]
        scores.sort(key=lambda x: (x[0], x[1]), reverse=True)

        ranked_ids = [s[1] for s in scores]
        ranked_sims = [s[0] for s in scores]

        matched_idx = next(idx for idx, s in enumerate(scores) if s[2] == i)
        matched_rank = matched_idx + 1
        i2t_ranks.append(matched_rank)

        top_k = {
            1: matched_rank <= 1,
            3: matched_rank <= min(3, n),
            5: matched_rank <= min(5, n),
        }

        i2t_results.append(
            RetrievalResult(
                query_modality=RetrievalDirection.IMAGE_TO_TEXT,
                query_sample_id=sample_ids[i],
                ranked_candidate_sample_ids=ranked_ids,
                similarities=ranked_sims,
                matched_pair_rank=matched_rank,
                top_k_success=top_k,
                candidate_count=n,
            )
        )

    # 2. Text-to-Image Retrieval
    t2i_results: list[RetrievalResult] = []
    t2i_ranks: list[int] = []

    for j in range(n):
        scores = [(sim_matrix[i][j], sample_ids[i], i) for i in range(n)]
        scores.sort(key=lambda x: (x[0], x[1]), reverse=True)

        ranked_ids = [s[1] for s in scores]
        ranked_sims = [s[0] for s in scores]

        matched_idx = next(idx for idx, s in enumerate(scores) if s[2] == j)
        matched_rank = matched_idx + 1
        t2i_ranks.append(matched_rank)

        top_k = {
            1: matched_rank <= 1,
            3: matched_rank <= min(3, n),
            5: matched_rank <= min(5, n),
        }

        t2i_results.append(
            RetrievalResult(
                query_modality=RetrievalDirection.TEXT_TO_IMAGE,
                query_sample_id=sample_ids[j],
                ranked_candidate_sample_ids=ranked_ids,
                similarities=ranked_sims,
                matched_pair_rank=matched_rank,
                top_k_success=top_k,
                candidate_count=n,
            )
        )

    # Aggregate metrics
    summary = CrossModalRetrievalSummary(
        image_to_text_r1=sum(1.0 for r in i2t_ranks if r <= 1) / float(n),
        image_to_text_r3=sum(1.0 for r in i2t_ranks if r <= 3) / float(n),
        image_to_text_r5=sum(1.0 for r in i2t_ranks if r <= 5) / float(n),
        image_to_text_mrr=sum(1.0 / float(r) for r in i2t_ranks) / float(n),
        text_to_image_r1=sum(1.0 for r in t2i_ranks if r <= 1) / float(n),
        text_to_image_r3=sum(1.0 for r in t2i_ranks if r <= 3) / float(n),
        text_to_image_r5=sum(1.0 for r in t2i_ranks if r <= 5) / float(n),
        text_to_image_mrr=sum(1.0 / float(r) for r in t2i_ranks) / float(n),
        sample_count=n,
        candidate_count=n,
    )

    return summary, i2t_results, t2i_results


def evaluate_zero_shot_classification(
    image_embeddings: list[list[float]],
    class_prompts: list[ClassPrompt],
    text_encoder: TextEncoder,
    tokenizer: SimpleTokenizer,
    true_labels: list[int],
    class_names: list[str],
    prompt_template: str = "a photo of a {class_name}",
) -> ZeroShotClassificationSummary:
    """Perform zero-shot classification via class prompt matching."""
    n = len(image_embeddings)
    k = len(class_prompts)
    if n == 0 or k == 0:
        raise ValidationError("Cannot evaluate zero-shot on empty image or class list.")

    # 1. Encode Class Prompts
    tokenized_prompts = [tokenizer.encode(cp.rendered_text) for cp in class_prompts]
    token_ids = [tp.token_ids for tp in tokenized_prompts]
    attention_masks = [tp.attention_mask for tp in tokenized_prompts]

    raw_text_embeds, _ = text_encoder.forward(token_ids, attention_masks)
    norm_class_embeds, _ = normalize_embeddings(raw_text_embeds)

    # 2. Score Images against Class Prompts
    predictions: list[int] = []
    correct_count = 0
    per_class_correct: dict[str, int] = dict.fromkeys(class_names, 0)
    per_class_total: dict[str, int] = dict.fromkeys(class_names, 0)

    confusion_matrix = [[0 for _ in range(k)] for _ in range(k)]

    for i in range(n):
        vi = image_embeddings[i]
        sims = [
            sum(a * b for a, b in zip(vi, norm_class_embeds[c_idx], strict=True))
            for c_idx in range(k)
        ]
        pred_class = max(range(k), key=lambda c: sims[c])
        predictions.append(pred_class)

        target = true_labels[i]
        if 0 <= target < k:
            confusion_matrix[target][pred_class] += 1
            target_name = class_names[target]
            per_class_total[target_name] += 1
            if pred_class == target:
                correct_count += 1
                per_class_correct[target_name] += 1

    accuracy = correct_count / float(n) if n > 0 else 0.0
    per_class_acc = {
        c: (per_class_correct[c] / float(per_class_total[c]))
        if per_class_total[c] > 0
        else 0.0
        for c in class_names
    }

    return ZeroShotClassificationSummary(
        prompt_template=prompt_template,
        class_count=k,
        accuracy=accuracy,
        per_class_accuracy=per_class_acc,
        confusion_matrix=confusion_matrix,
        class_names=class_names,
        top_3_accuracy=None,
    )


def evaluate_prompt_sensitivity(
    image_embeddings: list[list[float]],
    prompt_templates: list[str],
    class_names: list[str],
    text_encoder: TextEncoder,
    tokenizer: SimpleTokenizer,
    true_labels: list[int],
) -> dict[str, Any]:
    """Compare multiple prompt templates to quantify zero-shot prediction stability."""
    results_by_template: dict[str, ZeroShotClassificationSummary] = {}
    preds_by_template: dict[str, list[int]] = {}
    class_embeds_by_template: dict[str, list[list[float]]] = {}

    for template in prompt_templates:
        prompts = [
            ClassPrompt(
                class_id=idx,
                class_name=cname,
                prompt_template=template,
                rendered_text=template.format(
                    class_name=cname,
                    color=cname.split("_")[0] if "_" in cname else cname,
                    shape=cname.split("_")[1] if "_" in cname else "object",
                    position="center",
                    size="medium",
                ),
                prompt_id=f"p_{idx}_{hash(template)}",
            )
            for idx, cname in enumerate(class_names)
        ]

        toks = [tokenizer.encode(cp.rendered_text) for cp in prompts]
        raw_embeds, _ = text_encoder.forward(
            [t.token_ids for t in toks], [t.attention_mask for t in toks]
        )
        norm_embeds, _ = normalize_embeddings(raw_embeds)
        class_embeds_by_template[template] = norm_embeds

        summary = evaluate_zero_shot_classification(
            image_embeddings=image_embeddings,
            class_prompts=prompts,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            true_labels=true_labels,
            class_names=class_names,
            prompt_template=template,
        )
        results_by_template[template] = summary

        preds: list[int] = []
        for vi in image_embeddings:
            sims = [
                sum(a * b for a, b in zip(vi, norm_embeds[c], strict=True))
                for c in range(len(class_names))
            ]
            preds.append(max(range(len(class_names)), key=lambda c: sims[c]))
        preds_by_template[template] = preds

    pairwise_agreements: dict[str, float] = {}
    templates_list = list(prompt_templates)
    for i in range(len(templates_list)):
        t1 = templates_list[i]
        for j in range(i + 1, len(templates_list)):
            t2 = templates_list[j]
            p1 = preds_by_template[t1]
            p2 = preds_by_template[t2]
            agree = sum(1.0 for a, b in zip(p1, p2, strict=True) if a == b) / float(
                len(p1)
            )
            pairwise_agreements[f"{t1} <-> {t2}"] = agree

    return {
        "templates": templates_list,
        "results": {t: res.to_dict() for t, res in results_by_template.items()},
        "pairwise_agreements": pairwise_agreements,
    }


def compute_shared_multimodal_geometry(
    image_embeddings: list[list[float]],
    text_embeddings: list[list[float]],
    sample_ids: list[str],
    class_labels: list[int] | None = None,
    class_names: list[str] | None = None,
) -> dict[str, Any]:
    """Fit a single joint PCA projection on combined image + text embeddings."""
    n = len(image_embeddings)
    if n == 0 or len(text_embeddings) != n:
        raise ValidationError(
            "Image and text embeddings must be non-empty and equal length."
        )

    dim = len(image_embeddings[0])

    # 1. Combined Dataset for Shared PCA: [Image_1..N, Text_1..N]
    combined_vectors = image_embeddings + text_embeddings
    pca = PrincipalComponentAnalysis(n_components=2)
    pca_coords = pca.fit_transform(combined_vectors)

    image_coords = pca_coords[:n]
    text_coords = pca_coords[n:]

    # 2. Paired Distances
    paired_distances: list[float] = []
    paired_cosines: list[float] = []
    for i in range(n):
        vi = image_embeddings[i]
        ti = text_embeddings[i]
        euc_dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(vi, ti, strict=True)))
        cos_sim = sum(a * b for a, b in zip(vi, ti, strict=True))
        paired_distances.append(euc_dist)
        paired_cosines.append(cos_sim)

    # 3. Centroid Alignment per Semantic Class
    centroid_alignments: list[CrossModalCentroidAlignment] = []
    if class_labels is not None and class_names is not None:
        unique_classes = sorted(set(class_labels))
        for c_idx in unique_classes:
            c_name = (
                class_names[c_idx] if c_idx < len(class_names) else f"class_{c_idx}"
            )
            member_indices = [
                idx for idx, lab in enumerate(class_labels) if lab == c_idx
            ]
            if not member_indices:
                continue

            # Image centroid
            img_centroid = [0.0 for _ in range(dim)]
            for m in member_indices:
                for d in range(dim):
                    img_centroid[d] += image_embeddings[m][d]
            img_centroid = [val / float(len(member_indices)) for val in img_centroid]

            # Text centroid
            txt_centroid = [0.0 for _ in range(dim)]
            for m in member_indices:
                for d in range(dim):
                    txt_centroid[d] += text_embeddings[m][d]
            txt_centroid = [val / float(len(member_indices)) for val in txt_centroid]

            # Centroid metrics
            c_dist = math.sqrt(
                sum(
                    (a - b) ** 2
                    for a, b in zip(img_centroid, txt_centroid, strict=True)
                )
            )
            norm_i = math.sqrt(sum(a * a for a in img_centroid)) + 1e-12
            norm_t = math.sqrt(sum(a * a for a in txt_centroid)) + 1e-12
            c_cos = sum(
                a * b for a, b in zip(img_centroid, txt_centroid, strict=True)
            ) / (norm_i * norm_t)

            centroid_alignments.append(
                CrossModalCentroidAlignment(
                    class_name=c_name,
                    image_centroid=img_centroid,
                    text_centroid=txt_centroid,
                    euclidean_distance=c_dist,
                    cosine_similarity=c_cos,
                )
            )

    # 4. Cross-Modal Nearest Neighbors
    image_to_text_nn: list[dict[str, Any]] = []
    for i in range(min(n, 20)):
        vi = image_embeddings[i]
        sims = [
            (
                sum(a * b for a, b in zip(vi, text_embeddings[j], strict=True)),
                sample_ids[j],
                j == i,
            )
            for j in range(n)
        ]
        sims.sort(key=lambda x: x[0], reverse=True)
        image_to_text_nn.append(
            {
                "query_sample_id": sample_ids[i],
                "nearest_neighbors": [
                    {"sample_id": s[1], "similarity": s[0], "is_paired": s[2]}
                    for s in sims[:5]
                ],
            }
        )

    return {
        "explained_variance_ratio": pca.explained_variance_ratio or [0.0, 0.0],
        "image_pca_coordinates": image_coords,
        "text_pca_coordinates": text_coords,
        "paired_distances": paired_distances,
        "paired_cosines": paired_cosines,
        "mean_paired_distance": sum(paired_distances) / float(n),
        "mean_paired_cosine": sum(paired_cosines) / float(n),
        "centroid_alignments": [ca.to_dict() for ca in centroid_alignments],
        "image_to_text_nn": image_to_text_nn,
    }


def compute_multimodal_collapse_diagnostics(
    image_embeddings: list[list[float]],
    text_embeddings: list[list[float]],
) -> MultimodalCollapseSummary:
    """Compute dimensional collapse and representation diversity diagnostics."""
    n = len(image_embeddings)
    if n == 0:
        raise ValidationError(
            "Cannot compute collapse diagnostics on empty embeddings."
        )
    dim = len(image_embeddings[0])

    vis_dim_vars: list[float] = []
    for d in range(dim):
        vals = [image_embeddings[i][d] for i in range(n)]
        mean_v = sum(vals) / float(n)
        var_v = sum((x - mean_v) ** 2 for x in vals) / float(n)
        vis_dim_vars.append(var_v)
    mean_vis_var = sum(vis_dim_vars) / float(dim)
    vis_std = math.sqrt(mean_vis_var)

    txt_dim_vars: list[float] = []
    for d in range(dim):
        vals = [text_embeddings[i][d] for i in range(n)]
        mean_t = sum(vals) / float(n)
        var_t = sum((x - mean_t) ** 2 for x in vals) / float(n)
        txt_dim_vars.append(var_t)
    mean_txt_var = sum(txt_dim_vars) / float(dim)
    txt_std = math.sqrt(mean_txt_var)

    pos_sims: list[float] = []
    neg_sims: list[float] = []
    for i in range(n):
        for j in range(n):
            sim = sum(
                a * b
                for a, b in zip(image_embeddings[i], text_embeddings[j], strict=True)
            )
            if i == j:
                pos_sims.append(sim)
            else:
                neg_sims.append(sim)

    matched_sim = sum(pos_sims) / float(len(pos_sims)) if pos_sims else 1.0
    unmatched_sim = sum(neg_sims) / float(len(neg_sims)) if neg_sims else 0.0
    gap = matched_sim - unmatched_sim

    vis_pairwise: list[float] = []
    txt_pairwise: list[float] = []
    for i in range(n):
        for j in range(i + 1, n):
            v_sim = sum(
                a * b
                for a, b in zip(image_embeddings[i], image_embeddings[j], strict=True)
            )
            t_sim = sum(
                a * b
                for a, b in zip(text_embeddings[i], text_embeddings[j], strict=True)
            )
            vis_pairwise.append(v_sim)
            txt_pairwise.append(t_sim)

    mean_vis_pair = (
        sum(vis_pairwise) / float(len(vis_pairwise)) if vis_pairwise else 0.0
    )
    mean_txt_pair = (
        sum(txt_pairwise) / float(len(txt_pairwise)) if txt_pairwise else 0.0
    )

    is_collapsed = vis_std < 0.01 or txt_std < 0.01 or gap < 0.05

    return MultimodalCollapseSummary(
        visual_dim_variance=mean_vis_var,
        visual_feature_std=vis_std,
        visual_pairwise_similarity=mean_vis_pair,
        text_dim_variance=mean_txt_var,
        text_feature_std=txt_std,
        text_pairwise_similarity=mean_txt_pair,
        matched_similarity=matched_sim,
        unmatched_similarity=unmatched_sim,
        similarity_gap=gap,
        is_collapsed=is_collapsed,
    )
