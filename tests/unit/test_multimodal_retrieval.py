"""Unit tests for Image-to-Text and Text-to-Image cross-modal retrieval."""

from __future__ import annotations

import pytest

from prism.core.errors import ValidationError
from prism.multimodal.enums import RetrievalDirection
from prism.multimodal.evaluation import evaluate_cross_modal_retrieval


def test_perfect_diagonal_retrieval() -> None:
    """Verify perfect retrieval when embeddings match exactly on the diagonal."""
    n = 4
    # Orthogonal embeddings
    v = [[1.0 if i == d else 0.0 for d in range(n)] for i in range(n)]
    t = [[1.0 if i == d else 0.0 for d in range(n)] for i in range(n)]
    sample_ids = [f"sample_{i}" for i in range(n)]

    summary, i2t_res, t2i_res = evaluate_cross_modal_retrieval(v, t, sample_ids)

    # Both directions must achieve R@1 = 1.0 and MRR = 1.0
    assert summary.image_to_text_r1 == 1.0
    assert summary.image_to_text_mrr == 1.0
    assert summary.text_to_image_r1 == 1.0
    assert summary.text_to_image_mrr == 1.0

    for res in i2t_res:
        assert res.matched_pair_rank == 1
        assert res.top_k_success[1] is True
        assert res.query_modality == RetrievalDirection.IMAGE_TO_TEXT

    for res in t2i_res:
        assert res.matched_pair_rank == 1
        assert res.top_k_success[1] is True
        assert res.query_modality == RetrievalDirection.TEXT_TO_IMAGE


def test_imperfect_retrieval_ranking() -> None:
    """Verify rank calculation when matched candidate is ranked lower."""
    sample_ids = ["s0", "s1", "s2"]
    # Image 0 is closer to Text 1 than Text 0
    v = [
        [0.0, 1.0, 0.0],  # Closer to t[1]
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]
    t = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]

    summary, i2t_res, _ = evaluate_cross_modal_retrieval(v, t, sample_ids)

    # For query image s0: sim with t0 is 0.0, sim with t1 is 1.0
    # True text is s0 (t0), so rank is > 1
    assert i2t_res[0].matched_pair_rank > 1
    assert summary.image_to_text_r1 < 1.0


def test_empty_embeddings_validation() -> None:
    """Verify ValidationError on empty input lists."""
    with pytest.raises(ValidationError):
        evaluate_cross_modal_retrieval([], [], [])
