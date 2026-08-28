"""Unit tests for patch extraction, embedding, class token, and pos embeddings."""

import copy

import pytest

from prism.core.errors import ValidationError
from prism.models.patches import (
    ClassToken,
    PatchEmbedding,
    PatchExtractor,
    PositionalEmbedding,
    ensure_3d_tensor,
)


@pytest.mark.unit
def test_ensure_3d_tensor_validation() -> None:
    """Verify ensure_3d_tensor accepts valid 3D/2D data and rejects invalid shapes."""
    # 3D tensor
    valid_3d = [[[1.0, 2.0], [3.0, 4.0]]]
    res_3d = ensure_3d_tensor(valid_3d)
    assert len(res_3d) == 1 and len(res_3d[0]) == 2 and len(res_3d[0][0]) == 2

    # 2D tensor wrapped to 3D
    valid_2d = [[1.0, 2.0], [3.0, 4.0]]
    res_2d = ensure_3d_tensor(valid_2d)
    assert len(res_2d) == 1 and len(res_2d[0]) == 2

    # Rejection of None, empty, non-finite
    with pytest.raises(ValidationError, match="Input tensor cannot be None"):
        ensure_3d_tensor(None)

    with pytest.raises(ValidationError, match="Tensor batch cannot be empty"):
        ensure_3d_tensor([])

    with pytest.raises(ValidationError, match="Non-finite or non-numeric"):
        ensure_3d_tensor([[[float("nan"), 1.0]]])


@pytest.mark.unit
def test_patch_extractor_shapes_and_row_major_ordering() -> None:
    """Verify PatchExtractor divides images into row-major flattened patches."""
    # 1 sample, 1 channel, 4x4 image with pixel values 0..15
    # Patch size: 2x2 -> 4 patches, each patch dim = 1 * 2 * 2 = 4
    img = [
        [
            [
                [0.0, 1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0, 7.0],
                [8.0, 9.0, 10.0, 11.0],
                [12.0, 13.0, 14.0, 15.0],
            ]
        ]
    ]
    extractor = PatchExtractor(patch_size=2)
    patches = extractor.forward(img)

    assert len(patches) == 1
    assert len(patches[0]) == 4  # 4 patches
    assert len(patches[0][0]) == 4  # 4 values per patch

    # Patch 0 (top-left): [0, 1, 4, 5]
    assert patches[0][0] == [0.0, 1.0, 4.0, 5.0]
    # Patch 1 (top-right): [2, 3, 6, 7]
    assert patches[0][1] == [2.0, 3.0, 6.0, 7.0]
    # Patch 2 (bottom-left): [8, 9, 12, 13]
    assert patches[0][2] == [8.0, 9.0, 12.0, 13.0]
    # Patch 3 (bottom-right): [10, 11, 14, 15]
    assert patches[0][3] == [10.0, 11.0, 14.0, 15.0]


@pytest.mark.unit
def test_patch_extractor_non_square_and_multichannel() -> None:
    """Verify PatchExtractor handles rectangular images and multi-channel inputs."""
    # Batch=2, C=3, H=6, W=8, Patch=(3, 4) -> Grid: 2x2 = 4 patches, D_patch = 36
    batch_img = [
        [[[float(n + c + h + w) for w in range(8)] for h in range(6)] for c in range(3)]
        for n in range(2)
    ]
    extractor = PatchExtractor(patch_size=(3, 4))
    patches = extractor.forward(batch_img)

    assert len(patches) == 2
    assert len(patches[0]) == 4
    assert len(patches[0][0]) == 36


@pytest.mark.unit
def test_patch_extractor_rejections() -> None:
    """Verify PatchExtractor validates patch divisibility and positive dimensions."""
    with pytest.raises(ValidationError, match="Patch dimensions must be positive"):
        PatchExtractor(patch_size=0)

    extractor = PatchExtractor(patch_size=4)
    # Image H=7 (not divisible by 4)
    img_bad = [[[[1.0] * 8 for _ in range(7)]]]
    with pytest.raises(ValidationError, match="not divisible by patch height"):
        extractor.forward(img_bad)

    # Non-finite pixel
    img_nan = [[[[float("nan")] * 8 for _ in range(8)]]]
    with pytest.raises(ValidationError, match="Non-finite value"):
        extractor.forward(img_nan)


@pytest.mark.unit
def test_patch_extractor_backward_reconstruction() -> None:
    """Verify PatchExtractor backward reconstructs exact spatial 4D gradients."""
    img = [
        [
            [
                [1.0, 2.0, 3.0, 4.0],
                [5.0, 6.0, 7.0, 8.0],
                [9.0, 10.0, 11.0, 12.0],
                [13.0, 14.0, 15.0, 16.0],
            ]
        ]
    ]
    extractor = PatchExtractor(patch_size=2)
    _ = extractor.forward(img)

    # Synthetic upstream gradient dPatches
    d_patches = [
        [
            [1.0, 1.0, 1.0, 1.0],
            [2.0, 2.0, 2.0, 2.0],
            [3.0, 3.0, 3.0, 3.0],
            [4.0, 4.0, 4.0, 4.0],
        ]
    ]
    dx = extractor.backward(d_patches)

    assert len(dx) == 1 and len(dx[0]) == 1 and len(dx[0][0]) == 4
    # Check spatial placement
    assert dx[0][0][0][0] == 1.0
    assert dx[0][0][0][2] == 2.0
    assert dx[0][0][2][0] == 3.0
    assert dx[0][0][2][2] == 4.0


@pytest.mark.unit
def test_patch_embedding_forward_and_backward() -> None:
    """Verify PatchEmbedding linear projection and analytical parameter gradients."""
    # N=2, L=3, D_patch=4, D_embed=6
    pe = PatchEmbedding(in_features=4, embed_dim=6, bias=True, seed=42)
    p_in = [
        [[0.1, 0.2, -0.3, 0.4], [0.5, -0.1, 0.2, 0.3], [-0.2, 0.4, 0.1, -0.5]],
        [[0.2, -0.3, 0.1, 0.5], [-0.4, 0.2, -0.1, 0.3], [0.3, 0.1, -0.2, 0.4]],
    ]
    out = pe.forward(p_in)
    assert len(out) == 2 and len(out[0]) == 3 and len(out[0][0]) == 6

    # Upstream gradient dOut
    d_out = [
        [[1.0] * 6, [0.5] * 6, [-0.5] * 6],
        [[0.2] * 6, [-0.2] * 6, [0.8] * 6],
    ]
    dp = pe.backward(d_out)

    assert len(dp) == 2 and len(dp[0]) == 3 and len(dp[0][0]) == 4
    grads = pe.get_gradients()
    assert "weights" in grads and "bias" in grads
    assert len(grads["weights"]) == 4 and len(grads["weights"][0]) == 6
    assert len(grads["bias"]) == 6


@pytest.mark.unit
def test_patch_embedding_numerical_gradients() -> None:
    """Verify PatchEmbedding analytical gradients match finite-difference gradients."""
    pe = PatchEmbedding(in_features=3, embed_dim=4, bias=True, seed=42)
    p_in = [[[0.5, -0.2, 0.8], [0.1, 0.4, -0.6]]]

    # Scalar objective: L = sum(out)
    _ = pe.forward(p_in)
    d_out = [[[1.0] * 4, [1.0] * 4]]
    dp_analytical = pe.backward(d_out)
    grads_analytical = pe.get_gradients()

    eps = 1e-6
    # 1. Check dP
    for seq_idx in range(2):
        for k in range(3):
            p_plus = copy.deepcopy(p_in)
            p_minus = copy.deepcopy(p_in)
            p_plus[0][seq_idx][k] += eps
            p_minus[0][seq_idx][k] -= eps

            out_plus = pe.forward(p_plus)
            out_minus = pe.forward(p_minus)
            loss_plus = sum(sum(row) for sample in out_plus for row in sample)
            loss_minus = sum(sum(row) for sample in out_minus for row in sample)
            grad_num = (loss_plus - loss_minus) / (2.0 * eps)
            assert dp_analytical[0][seq_idx][k] == pytest.approx(
                grad_num, rel=1e-3, abs=1e-4
            )

    # 2. Check dW_E
    for k in range(3):
        for d in range(4):
            pe.weights[k][d] += eps
            out_plus = pe.forward(p_in)
            loss_plus = sum(sum(row) for sample in out_plus for row in sample)
            pe.weights[k][d] -= 2.0 * eps
            out_minus = pe.forward(p_in)
            loss_minus = sum(sum(row) for sample in out_minus for row in sample)
            pe.weights[k][d] += eps  # restore

            grad_num = (loss_plus - loss_minus) / (2.0 * eps)
            assert grads_analytical["weights"][k][d] == pytest.approx(
                grad_num, rel=1e-3, abs=1e-4
            )


@pytest.mark.unit
def test_class_token_batch_accumulation_and_gradients() -> None:
    """Verify ClassToken prepends token to sequences and accumulates batch gradients."""
    cls_layer = ClassToken(embed_dim=4, seed=42, init_std=0.02)
    # Batch=3, L=2, D=4
    e_in = [
        [[1.0, 1.0, 1.0, 1.0], [2.0, 2.0, 2.0, 2.0]],
        [[3.0, 3.0, 3.0, 3.0], [4.0, 4.0, 4.0, 4.0]],
        [[5.0, 5.0, 5.0, 5.0], [6.0, 6.0, 6.0, 6.0]],
    ]
    z_out = cls_layer.forward(e_in)

    assert len(z_out) == 3
    assert len(z_out[0]) == 3  # L+1 = 3
    # Check CLS token placed at index 0 across batch
    for n in range(3):
        assert z_out[n][0] == cls_layer.token[0][0]
        assert z_out[n][1] == e_in[n][0]
        assert z_out[n][2] == e_in[n][1]

    # Backward pass with synthetic gradient
    dz = [
        [[1.0, 2.0, 3.0, 4.0], [0.5, 0.5, 0.5, 0.5], [0.1, 0.1, 0.1, 0.1]],
        [[2.0, 3.0, 4.0, 5.0], [0.5, 0.5, 0.5, 0.5], [0.1, 0.1, 0.1, 0.1]],
        [[3.0, 4.0, 5.0, 6.0], [0.5, 0.5, 0.5, 0.5], [0.1, 0.1, 0.1, 0.1]],
    ]
    de = cls_layer.backward(dz)

    # Check patch gradient extracted correctly
    assert len(de) == 3 and len(de[0]) == 2
    assert de[0][0] == [0.5, 0.5, 0.5, 0.5]
    assert de[0][1] == [0.1, 0.1, 0.1, 0.1]

    # Check CLS parameter gradient accumulated across all 3 batch elements
    grads = cls_layer.get_gradients()
    expected_cls_grad = [
        1.0 + 2.0 + 3.0,  # 6.0
        2.0 + 3.0 + 4.0,  # 9.0
        3.0 + 4.0 + 5.0,  # 12.0
        4.0 + 5.0 + 6.0,  # 15.0
    ]
    assert grads["token"][0][0] == expected_cls_grad


@pytest.mark.unit
def test_positional_embedding_forward_and_backward() -> None:
    """Verify PositionalEmbedding adds position vectors and accumulates gradients."""
    pos_layer = PositionalEmbedding(num_positions=4, embed_dim=3, seed=42)
    # Batch=2, S=3, D=3
    z_in = [
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
        [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]],
    ]
    y_out = pos_layer.forward(z_in)

    assert len(y_out) == 2 and len(y_out[0]) == 3 and len(y_out[0][0]) == 3
    for s in range(3):
        assert y_out[0][s][0] == pytest.approx(
            z_in[0][s][0] + pos_layer.embeddings[0][s][0]
        )

    # Sequence length exceeding num_positions
    with pytest.raises(ValidationError, match=r"Sequence length .* exceeds"):
        pos_layer.forward([[[1.0, 2.0, 3.0]] * 5])

    # Backward pass
    dy = [
        [[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0]],
        [[4.0, 4.0, 4.0], [5.0, 5.0, 5.0], [6.0, 6.0, 6.0]],
    ]
    dz = pos_layer.backward(dy)

    # dZ = dY
    assert dz == dy

    # dP_pos accumulated across batch
    grads = pos_layer.get_gradients()
    assert grads["embeddings"][0][0] == [5.0, 5.0, 5.0]
    assert grads["embeddings"][0][1] == [7.0, 7.0, 7.0]
    assert grads["embeddings"][0][2] == [9.0, 9.0, 9.0]
