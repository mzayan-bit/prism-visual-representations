"""Smoke test for Vision Transformer patch, embedding, and attention foundations."""

import math

import pytest

from prism.models.attention import MultiHeadSelfAttention
from prism.models.patches import (
    ClassToken,
    ImagePatchExtractor,
    LearnablePositionalEmbedding,
    PatchEmbedding,
    PatchGeometry,
    patches_to_image,
)
from prism.representations.attention import (
    AttentionTensorSummary,
    compute_attention_entropy,
    compute_diagonal_attention_mass,
    summarize_attention_weights,
)


@pytest.mark.smoke
def test_smoke_attention_foundations_pipeline() -> None:
    """Validate vision transformer pipeline from pixels to contextual tokens."""
    # 1. Deterministic synthetic image batch: N=2, C=3, H=8, W=8
    x_img = [
        [
            [
                [0.1 * ((n + 1) * (c + 1) + h * 8 + w) for w in range(8)]
                for h in range(8)
            ]
            for c in range(3)
        ]
        for n in range(2)
    ]

    # 2. Patch Geometry Definition
    geom = PatchGeometry.create(image_size=(8, 8), patch_size=(4, 4), channels=3)
    assert geom.total_patches == 4
    assert geom.flattened_patch_dimension == 48

    # 3. Patch Extraction
    patch_ext = ImagePatchExtractor(geometry=geom)
    patches = patch_ext.extract_patches(x_img)
    assert len(patches) == 2 and len(patches[0]) == 4 and len(patches[0][0]) == 48

    # 4. Patch Reconstruction Identity Check
    reconstructed = patches_to_image(patches, geometry=geom)
    assert reconstructed == x_img

    # 5. Patch Embedding (48 -> embed_dim 16)
    patch_emb = PatchEmbedding(in_features=48, embed_dim=16, bias=True, seed=42)
    e_tokens = patch_emb.forward(patches)
    assert len(e_tokens) == 2 and len(e_tokens[0]) == 4 and len(e_tokens[0][0]) == 16

    # 6. Class Token (Prepend [1, 1, 16] CLS -> 5 tokens)
    cls_layer = ClassToken(embed_dim=16, seed=42, init_std=0.02)
    z_tokens = cls_layer.forward(e_tokens)
    assert len(z_tokens) == 2 and len(z_tokens[0]) == 5 and len(z_tokens[0][0]) == 16

    # 7. Positional Embedding (Add 1D learned position embeddings)
    pos_layer = LearnablePositionalEmbedding(
        num_positions=5, embed_dim=16, seed=42, init_std=0.02
    )
    y_tokens = pos_layer.forward(z_tokens)
    assert len(y_tokens) == 2 and len(y_tokens[0]) == 5 and len(y_tokens[0][0]) == 16

    # 8. Multi-Head Self-Attention (16 dim, 2 heads -> D_head = 8)
    mhsa = MultiHeadSelfAttention(embed_dim=16, num_heads=2, bias=True, seed=42)
    context_tokens = mhsa.forward(y_tokens)
    assert (
        len(context_tokens) == 2
        and len(context_tokens[0]) == 5
        and len(context_tokens[0][0]) == 16
    )

    # 9. Intermediate Representation Extraction
    assert mhsa.last_q is not None
    assert mhsa.last_k is not None
    assert mhsa.last_v is not None
    assert mhsa.last_head_outputs is not None
    assert mhsa.last_concat is not None
    assert mhsa.last_attention_weights is not None

    # 10. Attention Representation Summary, Entropy, and Diagonal Mass
    summary = summarize_attention_weights(mhsa.last_attention_weights)
    assert isinstance(summary, AttentionTensorSummary)
    assert summary.tensor_shape == (2, 2, 5, 5)
    assert summary.batch_size == 2
    assert summary.num_heads == 2
    assert summary.seq_len == 5
    assert summary.is_row_normalized is True
    assert summary.is_finite is True
    assert 0.0 <= summary.min_value <= summary.max_value <= 1.0

    entropies = compute_attention_entropy(mhsa.last_attention_weights)
    assert len(entropies) == 2 and len(entropies[0]) == 2 and len(entropies[0][0]) == 5

    diag_mass = compute_diagonal_attention_mass(mhsa.last_attention_weights)
    assert 0.0 <= diag_mass <= 1.0

    # 11. Synthetic Loss & Analytical Backpropagation
    total_elements = 2 * 5 * 16
    d_out = [
        [
            [(context_tokens[n][s][d] - 0.5) / float(total_elements) for d in range(16)]
            for s in range(5)
        ]
        for n in range(2)
    ]

    dy = mhsa.backward(d_out)
    assert len(dy) == 2 and len(dy[0]) == 5 and len(dy[0][0]) == 16

    dz = pos_layer.backward(dy)
    assert len(dz) == 2 and len(dz[0]) == 5 and len(dz[0][0]) == 16

    de = cls_layer.backward(dz)
    assert len(de) == 2 and len(de[0]) == 4 and len(de[0][0]) == 16

    dp = patch_emb.backward(de)
    assert len(dp) == 2 and len(dp[0]) == 4 and len(dp[0][0]) == 48

    dx = patch_ext.backward(dp)
    assert (
        len(dx) == 2
        and len(dx[0]) == 3
        and len(dx[0][0]) == 8
        and len(dx[0][0][0]) == 8
    )

    # 12. Verify Gradients and Parameters
    mhsa_grads = mhsa.get_gradients()
    assert sum(sum(abs(v) for v in row) for row in mhsa_grads["w_q"]) > 0.0
    assert sum(sum(abs(v) for v in row) for row in mhsa_grads["w_k"]) > 0.0
    assert sum(sum(abs(v) for v in row) for row in mhsa_grads["w_v"]) > 0.0
    assert sum(sum(abs(v) for v in row) for row in mhsa_grads["w_o"]) > 0.0

    pos_grads = pos_layer.get_gradients()
    assert sum(abs(v) for row in pos_grads["embeddings"][0] for v in row) > 0.0

    cls_grads = cls_layer.get_gradients()
    assert sum(abs(v) for v in cls_grads["token"][0][0]) > 0.0

    pe_grads = patch_emb.get_gradients()
    assert sum(sum(abs(v) for v in row) for row in pe_grads["weights"]) > 0.0

    # Image gradients are finite
    for n in range(2):
        for c in range(3):
            for h in range(8):
                for w in range(8):
                    val = dx[n][c][h][w]
                    assert not math.isnan(val) and not math.isinf(val)
