"""Unit tests and numerical gradient checks for contrastive loss."""

from __future__ import annotations

import copy
import math

import pytest

from prism.multimodal.loss import SymmetricContrastiveLoss


def test_symmetric_contrastive_loss_finite_and_symmetric() -> None:
    """Verify loss computation, symmetry, and telemetry metrics."""
    loss_fn = SymmetricContrastiveLoss(temperature=0.2)

    # 3 samples, dimension 4 (already unit normalized)
    v = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
    ]
    t = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
    ]

    loss, _d_v, _d_t, metrics = loss_fn(v, t)

    assert loss > 0.0
    assert math.isfinite(loss)
    # Perfect diagonal alignment -> matched similarity should be 1.0, unmatched 0.0
    assert pytest.approx(metrics["matched_similarity"], abs=1e-6) == 1.0
    assert pytest.approx(metrics["unmatched_similarity"], abs=1e-6) == 0.0
    assert pytest.approx(metrics["similarity_gap"], abs=1e-6) == 1.0
    assert (
        pytest.approx(metrics["image_to_text_loss"], abs=1e-6)
        == metrics["text_to_image_loss"]
    )


def test_symmetric_loss_numerical_gradients() -> None:
    """Validate analytical gradients of loss w.r.t v and t using finite diffs."""
    loss_fn = SymmetricContrastiveLoss(temperature=0.5)

    # 3 samples, 2D unit vectors
    raw_v = [[0.6, 0.8], [-0.8, 0.6], [0.0, 1.0]]
    raw_t = [[0.8, 0.6], [-0.6, 0.8], [1.0, 0.0]]

    # L2 normalize
    v = [[x / math.sqrt(sum(a * a for a in row)) for x in row] for row in raw_v]
    t = [[x / math.sqrt(sum(a * a for a in row)) for x in row] for row in raw_t]

    _loss, d_v, d_t, _ = loss_fn(v, t)

    eps = 1e-6

    # 1. Finite-difference gradient for v[1][0]
    target_i = 1
    target_d = 0
    orig_v_val = v[target_i][target_d]

    v_pos = copy.deepcopy(v)
    v_pos[target_i][target_d] = orig_v_val + eps
    loss_pos, _, _, _ = loss_fn(v_pos, t)

    v_neg = copy.deepcopy(v)
    v_neg[target_i][target_d] = orig_v_val - eps
    loss_neg, _, _, _ = loss_fn(v_neg, t)

    num_grad_v = (loss_pos - loss_neg) / (2.0 * eps)
    assert pytest.approx(d_v[target_i][target_d], rel=1e-4, abs=1e-4) == num_grad_v

    # 2. Finite-difference gradient for t[0][1]
    target_j = 0
    target_d = 1
    orig_t_val = t[target_j][target_d]

    t_pos = copy.deepcopy(t)
    t_pos[target_j][target_d] = orig_t_val + eps
    loss_pos_t, _, _, _ = loss_fn(v, t_pos)

    t_neg = copy.deepcopy(t)
    t_neg[target_j][target_d] = orig_t_val - eps
    loss_neg_t, _, _, _ = loss_fn(v, t_neg)

    num_grad_t = (loss_pos_t - loss_neg_t) / (2.0 * eps)
    assert pytest.approx(d_t[target_j][target_d], rel=1e-4, abs=1e-4) == num_grad_t


def test_label_independence_invariant() -> None:
    """Verify class labels never affect contrastive loss computation."""
    loss_fn = SymmetricContrastiveLoss(temperature=0.1)

    v = [[0.707, 0.707], [-0.707, 0.707]]
    t = [[0.707, 0.707], [-0.707, 0.707]]

    # Loss computation takes only image and text normalized embeddings
    loss1, d_v1, d_t1, _ = loss_fn(v, t)
    loss2, d_v2, d_t2, _ = loss_fn(v, t)

    assert loss1 == loss2
    assert d_v1 == d_v2
    assert d_t1 == d_t2
