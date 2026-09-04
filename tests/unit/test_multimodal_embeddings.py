"""Unit tests and numerical gradient checks for multimodal embeddings."""

from __future__ import annotations

import copy

import pytest

from prism.multimodal.embeddings import (
    MaskedMeanPooling,
    MultimodalProjectionHead,
    TextEncoder,
    TokenEmbeddingTable,
)


def test_token_embedding_forward_and_backward() -> None:
    """Verify TokenEmbeddingTable forward lookup and backward accumulation."""
    vocab_size = 10
    emb_dim = 4
    table = TokenEmbeddingTable(vocab_size=vocab_size, embedding_dim=emb_dim, seed=42)

    # Batch of 2 samples, sequence length 3
    token_ids = [[2, 4, 0], [4, 5, 0]]
    out = table.forward(token_ids)

    assert len(out) == 2
    assert len(out[0]) == 3
    assert len(out[0][0]) == emb_dim

    # Token 4 appears at (0, 1) and (1, 0)
    assert out[0][1] == table.weights[4]
    assert out[1][0] == table.weights[4]

    # Backward pass with dummy gradient d_out
    table.zero_grad()
    d_out = [
        [[1.0] * emb_dim, [2.0] * emb_dim, [0.0] * emb_dim],
        [[3.0] * emb_dim, [0.5] * emb_dim, [0.0] * emb_dim],
    ]
    table.backward(d_out)

    grads = table.get_gradients()["embedding_weights"]
    # Token 4 grad should be 2.0 + 3.0 = 5.0
    for d in range(emb_dim):
        assert pytest.approx(grads[4][d], abs=1e-6) == 5.0
    # Token 2 grad should be 1.0
    for d in range(emb_dim):
        assert pytest.approx(grads[2][d], abs=1e-6) == 1.0
    # Token 5 grad should be 0.5
    for d in range(emb_dim):
        assert pytest.approx(grads[5][d], abs=1e-6) == 0.5


def test_token_embedding_numerical_gradient() -> None:
    """Validate analytical gradients of TokenEmbeddingTable using finite diffs."""
    vocab_size = 6
    emb_dim = 4
    table = TokenEmbeddingTable(vocab_size=vocab_size, embedding_dim=emb_dim, seed=42)

    token_ids = [[1, 3, 2], [3, 4, 1]]
    table.forward(token_ids)

    # Synthetic scalar loss: sum(H^2)
    # dL / dH = 2 * H
    embeds = table.forward(token_ids)
    d_out = [
        [[2.0 * embeds[i][seq_idx][d] for d in range(emb_dim)] for seq_idx in range(3)]
        for i in range(2)
    ]
    table.zero_grad()
    table.backward(d_out)
    analytical_grads = table.get_gradients()["embedding_weights"]

    # Finite difference check on table.weights[3][1]
    eps = 1e-6
    target_v = 3
    target_d = 1

    orig_val = table.weights[target_v][target_d]

    table.weights[target_v][target_d] = orig_val + eps
    out_pos = table.forward(token_ids)
    loss_pos = sum(x * x for seq in out_pos for vec in seq for x in vec)

    table.weights[target_v][target_d] = orig_val - eps
    out_neg = table.forward(token_ids)
    loss_neg = sum(x * x for seq in out_neg for vec in seq for x in vec)

    table.weights[target_v][target_d] = orig_val  # Reset

    numerical_grad = (loss_pos - loss_neg) / (2.0 * eps)
    assert (
        pytest.approx(analytical_grads[target_v][target_d], rel=1e-4, abs=1e-4)
        == numerical_grad
    )


def test_masked_mean_pooling_ignores_padding() -> None:
    """Verify MaskedMeanPooling computes mean strictly over valid tokens."""
    pooling = MaskedMeanPooling()

    # 1 sample, seq_len 3, dim 2
    # embeddings: [[1.0, 2.0], [3.0, 4.0], [100.0, 200.0]] (last token is PAD)
    embeddings = [[[1.0, 2.0], [3.0, 4.0], [100.0, 200.0]]]
    masks = [[1, 1, 0]]

    pooled = pooling.forward(embeddings, masks)
    # Expected mean over first two tokens: (1+3)/2 = 2.0, (2+4)/2 = 3.0
    assert pytest.approx(pooled[0][0], abs=1e-6) == 2.0
    assert pytest.approx(pooled[0][1], abs=1e-6) == 3.0

    # Backward gradient check
    d_pooled = [[1.0, 1.0]]
    d_emb = pooling.backward(d_pooled)
    # Valid tokens should receive gradient (1.0 / 2.0) = 0.5
    assert pytest.approx(d_emb[0][0][0], abs=1e-6) == 0.5
    assert pytest.approx(d_emb[0][1][0], abs=1e-6) == 0.5
    # PAD token should receive 0 gradient
    assert d_emb[0][2][0] == 0.0
    assert d_emb[0][2][1] == 0.0


def test_projection_head_numerical_gradient() -> None:
    """Verify analytical gradients of MultimodalProjectionHead via finite diffs."""
    in_dim = 4
    out_dim = 3
    proj = MultimodalProjectionHead(
        in_dim=in_dim, out_dim=out_dim, use_mlp=True, seed=42
    )

    x = [[0.5, -0.2, 0.8, 0.1], [-0.4, 0.3, 0.0, 0.7]]
    z = proj.forward(x)

    # Loss = sum(z^2) -> dL / dz = 2 * z
    d_z = [[2.0 * val for val in row] for row in z]
    proj.zero_grad()
    d_x = proj.backward(d_z)

    # Finite difference on input x[0][2]
    eps = 1e-6
    x_pos = copy.deepcopy(x)
    x_pos[0][2] += eps
    z_pos = proj.forward(x_pos)
    loss_pos = sum(v * v for row in z_pos for v in row)

    x_neg = copy.deepcopy(x)
    x_neg[0][2] -= eps
    z_neg = proj.forward(x_neg)
    loss_neg = sum(v * v for row in z_neg for v in row)

    num_grad_x = (loss_pos - loss_neg) / (2.0 * eps)
    assert pytest.approx(d_x[0][2], rel=1e-4, abs=1e-4) == num_grad_x


def test_text_encoder_composite_forward_backward() -> None:
    """Verify TextEncoder forward and backward execution."""
    encoder = TextEncoder(vocab_size=8, text_dim=6, shared_dim=4, seed=42)

    token_ids = [[2, 4, 3, 0], [2, 5, 3, 0]]
    masks = [[1, 1, 1, 0], [1, 1, 1, 0]]

    shared, pooled = encoder.forward(token_ids, masks)
    assert len(shared) == 2
    assert len(shared[0]) == 4
    assert len(pooled[0]) == 6

    encoder.zero_grad()
    d_shared = [[1.0] * 4, [0.5] * 4]
    encoder.backward(d_shared)

    grads = encoder.get_gradients()
    assert "embedding_weights" in grads
    assert "proj_w1" in grads
