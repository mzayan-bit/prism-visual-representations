"""Exact AUROC, AUPR, and threshold selection for binary OOD evaluation."""

from __future__ import annotations

import math
from collections.abc import Sequence

from prism.core.errors import ValidationError
from prism.uncertainty.contracts import OODBinaryEvaluationSummary
from prism.uncertainty.enums import OODScoreMethod, ThresholdPolicy


def compute_auroc(id_scores: Sequence[float], ood_scores: Sequence[float]) -> float:
    """Compute exact Area Under the ROC Curve (AUROC) via rank-sum integration.

    Conventions:
    - Positive class: Out-of-Distribution (OOD)
    - Negative class: In-Distribution (ID)
    - Assumption: Higher score = more OOD-like (standard polarity).
    - Deterministic tie handling with exact fractional average ranks.

    Parameters
    ----------
    id_scores : Sequence[float]
        OOD novelty scores for in-distribution samples (negatives).
    ood_scores : Sequence[float]
        OOD novelty scores for out-of-distribution samples (positives).

    Returns
    -------
    float
        Exact AUROC score in [0.0, 1.0].
    """
    n_neg = len(id_scores)
    n_pos = len(ood_scores)

    if n_neg == 0 or n_pos == 0:
        raise ValidationError(
            f"ID (N={n_neg}) and OOD (N={n_pos}) scores must be non-empty for AUROC."
        )

    # 1. Combine all samples: (score, is_positive)
    combined: list[tuple[float, int]] = []
    for s in id_scores:
        if math.isnan(s) or math.isinf(s):
            raise ValidationError(f"Non-finite score in ID scores: {s}")
        combined.append((s, 0))
    for s in ood_scores:
        if math.isnan(s) or math.isinf(s):
            raise ValidationError(f"Non-finite score in OOD scores: {s}")
        combined.append((s, 1))

    # 2. Sort ascending by score
    combined.sort(key=lambda x: x[0])
    total_n = len(combined)

    # 3. Assign 1-indexed ranks with average rank for ties
    ranks: list[float] = [0.0] * total_n
    i = 0
    while i < total_n:
        j = i
        while j < total_n and combined[j][0] == combined[i][0]:
            j += 1
        # Indices i .. j-1 are tied
        # 1-indexed ranks range from (i+1) to j
        avg_rank = (float(i + 1) + float(j)) / 2.0
        for k in range(i, j):
            ranks[k] = avg_rank
        i = j

    # 4. Sum ranks of positive (OOD) samples
    pos_rank_sum = sum(ranks[idx] for idx in range(total_n) if combined[idx][1] == 1)

    # 5. Mann-Whitney U formula for AUROC
    u_stat = pos_rank_sum - (float(n_pos * (n_pos + 1)) / 2.0)
    auroc = u_stat / float(n_pos * n_neg)

    return max(0.0, min(1.0, auroc))


def compute_aupr(id_scores: Sequence[float], ood_scores: Sequence[float]) -> float:
    """Compute Area Under the Precision-Recall Curve (AUPR) with OOD as positive class.

    Parameters
    ----------
    id_scores : Sequence[float]
        OOD novelty scores for ID samples.
    ood_scores : Sequence[float]
        OOD novelty scores for OOD samples.

    Returns
    -------
    float
        AUPR score in [0.0, 1.0].
    """
    n_neg = len(id_scores)
    n_pos = len(ood_scores)

    if n_neg == 0 or n_pos == 0:
        return 0.0

    # Combine and sort descending by score
    combined = [(s, 0) for s in id_scores] + [(s, 1) for s in ood_scores]
    combined.sort(key=lambda x: x[0], reverse=True)

    tp = 0
    fp = 0
    precisions = [1.0]
    recalls = [0.0]

    for _, is_pos in combined:
        if is_pos == 1:
            tp += 1
        else:
            fp += 1
        precision = float(tp) / float(tp + fp)
        recall = float(tp) / float(n_pos)
        precisions.append(precision)
        recalls.append(recall)

    # Trapezoidal integration under PR curve
    aupr = 0.0
    for i in range(1, len(recalls)):
        dr = recalls[i] - recalls[i - 1]
        avg_p = (precisions[i] + precisions[i - 1]) / 2.0
        aupr += avg_p * dr

    return max(0.0, min(1.0, aupr))


def select_ood_threshold(
    id_scores: Sequence[float],
    policy: ThresholdPolicy = ThresholdPolicy.TARGET_ID_TPR,
    target_id_tpr: float = 0.95,
    fixed_threshold: float = 0.5,
) -> float:
    """Select decision threshold theta on in-distribution reference/validation scores.

    Parameters
    ----------
    id_scores : Sequence[float]
        In-distribution score distribution.
    policy : ThresholdPolicy
        Policy for threshold derivation.
    target_id_tpr : float
        Target ID acceptance rate (e.g. 0.95 for 95% ID TPR).
    fixed_threshold : float
        Explicit constant threshold.

    Returns
    -------
    float
        Selected scalar decision threshold.
    """
    if policy == ThresholdPolicy.FIXED:
        return fixed_threshold

    if not id_scores:
        return fixed_threshold

    sorted_id = sorted(id_scores)
    n = len(sorted_id)

    if policy in (ThresholdPolicy.TARGET_ID_TPR, ThresholdPolicy.VALIDATION_QUANTILE):
        # Samples with score <= theta are classified as ID
        # To accept target_id_tpr fraction of ID samples:
        idx = math.floor(target_id_tpr * float(n - 1))
        idx = max(0, min(n - 1, idx))
        return sorted_id[idx]

    return fixed_threshold


def evaluate_ood_binary_classification(
    id_scores: Sequence[float],
    ood_scores: Sequence[float],
    score_method: OODScoreMethod,
    threshold_policy: ThresholdPolicy = ThresholdPolicy.TARGET_ID_TPR,
    target_id_tpr: float = 0.95,
    fixed_threshold: float = 0.5,
) -> OODBinaryEvaluationSummary:
    """Perform complete binary evaluation (AUROC, AUPR, operating threshold metrics)."""
    if not id_scores or not ood_scores:
        raise ValidationError(
            "Cannot evaluate OOD binary classification with empty scores."
        )

    auroc = compute_auroc(id_scores, ood_scores)
    aupr = compute_aupr(id_scores, ood_scores)

    threshold = select_ood_threshold(
        id_scores=id_scores,
        policy=threshold_policy,
        target_id_tpr=target_id_tpr,
        fixed_threshold=fixed_threshold,
    )

    n_id = len(id_scores)
    n_ood = len(ood_scores)

    # Sample predicted as OOD if score > threshold
    true_positives = sum(1 for s in ood_scores if s > threshold)
    false_positives = sum(1 for s in id_scores if s > threshold)
    true_negatives = n_id - false_positives

    tpr = float(true_positives) / float(n_ood)
    fpr = float(false_positives) / float(n_id)
    det_acc = float(true_positives + true_negatives) / float(n_id + n_ood)

    mean_id = sum(id_scores) / float(n_id)
    mean_ood = sum(ood_scores) / float(n_ood)
    gap = mean_ood - mean_id

    return OODBinaryEvaluationSummary(
        score_method=score_method,
        auroc=auroc,
        aupr=aupr,
        threshold=threshold,
        threshold_policy=threshold_policy,
        tpr_at_threshold=tpr,
        fpr_at_threshold=fpr,
        detection_accuracy_at_threshold=det_acc,
        id_sample_count=n_id,
        ood_sample_count=n_ood,
        mean_id_score=mean_id,
        mean_ood_score=mean_ood,
        score_separation_gap=gap,
    )
