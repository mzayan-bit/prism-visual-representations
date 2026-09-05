"""Unit tests for AUROC, AUPR, and binary decision threshold metrics."""

from __future__ import annotations

import pytest

from prism.uncertainty.contracts import OODBinaryEvaluationSummary
from prism.uncertainty.enums import OODScoreMethod, ThresholdPolicy
from prism.uncertainty.metrics import (
    compute_aupr,
    compute_auroc,
    evaluate_ood_binary_classification,
    select_ood_threshold,
)


def test_auroc_perfect_separation() -> None:
    """Verify AUROC is 1.0 when OOD scores are strictly higher than all ID scores."""
    id_scores = [0.1, 0.2, 0.3, 0.4]
    ood_scores = [0.5, 0.6, 0.7, 0.8]

    auroc = compute_auroc(id_scores, ood_scores)
    assert auroc == pytest.approx(1.0, abs=1e-5)


def test_auroc_inverted_separation() -> None:
    """Verify AUROC is 0.0 when OOD scores are strictly lower than ID scores."""
    id_scores = [0.8, 0.9, 1.0]
    ood_scores = [0.1, 0.2, 0.3]

    auroc = compute_auroc(id_scores, ood_scores)
    assert auroc == pytest.approx(0.0, abs=1e-5)


def test_auroc_random_separation() -> None:
    """Verify AUROC is ~0.5 on symmetric overlapping scores."""
    id_scores = [0.1, 0.3, 0.5, 0.7]
    ood_scores = [0.1, 0.3, 0.5, 0.7]

    auroc = compute_auroc(id_scores, ood_scores)
    assert auroc == pytest.approx(0.5, abs=1e-4)


def test_aupr_computation() -> None:
    """Verify Area Under Precision-Recall curve."""
    id_scores = [0.1, 0.2, 0.3]
    ood_scores = [0.7, 0.8, 0.9]

    aupr = compute_aupr(id_scores, ood_scores)
    assert aupr is not None
    assert aupr == pytest.approx(1.0, rel=1e-3)


def test_select_ood_threshold_policies() -> None:
    """Verify threshold selection policies."""
    id_scores = [0.1, 0.2, 0.3, 0.4, 0.5]

    # Target ID TPR (e.g. 80% of ID accepted -> threshold accepts 4/5)
    t_target = select_ood_threshold(
        id_scores,
        policy=ThresholdPolicy.TARGET_ID_TPR,
        target_id_tpr=0.80,
    )
    assert 0.3 <= t_target <= 0.5

    # Validation quantile
    t_quant = select_ood_threshold(
        id_scores,
        policy=ThresholdPolicy.VALIDATION_QUANTILE,
        target_id_tpr=0.90,
    )
    assert t_quant >= 0.3

    # Fixed threshold
    t_fixed = select_ood_threshold(
        id_scores, policy=ThresholdPolicy.FIXED, fixed_threshold=0.55
    )
    assert t_fixed == 0.55


def test_evaluate_ood_binary_classification() -> None:
    """Verify full OODBinaryEvaluationSummary computation."""
    id_scores = [0.1, 0.2, 0.3, 0.4]
    ood_scores = [0.6, 0.7, 0.8, 0.9]

    summary = evaluate_ood_binary_classification(
        id_scores=id_scores,
        ood_scores=ood_scores,
        score_method=OODScoreMethod.MAX_SOFTMAX_PROBABILITY,
        threshold_policy=ThresholdPolicy.TARGET_ID_TPR,
        target_id_tpr=0.95,
    )

    assert isinstance(summary, OODBinaryEvaluationSummary)
    assert summary.auroc == pytest.approx(1.0, abs=1e-5)
    assert summary.id_sample_count == 4
    assert summary.ood_sample_count == 4
    assert summary.score_separation_gap > 0.0
    assert summary.tpr_at_threshold >= 0.9
    assert summary.fpr_at_threshold <= 0.30
