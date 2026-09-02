"""Unit tests for deterministic masking context, patch mask partitioning."""

import pytest
from pydantic import ValidationError as PydanticValidationError

from prism.reconstruction.context import MaskingContext
from prism.reconstruction.mask import PatchMask, generate_patch_mask
from prism.reconstruction.tokens import LearnableMaskToken


def test_masking_context_validation() -> None:
    """Validate mask ratio bounds."""
    with pytest.raises(PydanticValidationError):
        MaskingContext(
            global_seed=42,
            sample_id="s1",
            mask_ratio=0.0,
        )

    with pytest.raises(PydanticValidationError):
        MaskingContext(
            global_seed=42,
            sample_id="s1",
            mask_ratio=1.0,
        )

    ctx = MaskingContext(
        global_seed=42,
        sample_id="s1",
        mask_ratio=0.5,
    )
    assert ctx.mask_ratio == 0.5
    s1 = ctx.derive_seed_int("stream_a")
    s2 = ctx.derive_seed_int("stream_a")
    s3 = ctx.derive_seed_int("stream_b")
    assert s1 == s2
    assert s1 != s3


def test_generate_patch_mask_deterministic_partition() -> None:
    """Verify exact partition count, disjoint sets, and repeatability."""
    t = 16
    ctx1 = MaskingContext(global_seed=100, sample_id="cifar_1", mask_ratio=0.25)
    mask1 = generate_patch_mask(ctx1, total_patches=t)

    # 16 * 0.25 = 4 masked
    assert mask1.total_patches == 16
    assert mask1.num_masked == 4
    assert mask1.num_visible == 12
    assert len(set(mask1.masked_indices)) == 4
    assert len(set(mask1.visible_indices)) == 12

    # Disjoint union
    assert set(mask1.masked_indices) & set(mask1.visible_indices) == set()
    assert set(mask1.masked_indices) | set(mask1.visible_indices) == set(range(16))

    # Determinism
    mask1_repeat = generate_patch_mask(ctx1, total_patches=t)
    assert mask1.masked_indices == mask1_repeat.masked_indices
    assert mask1.visible_indices == mask1_repeat.visible_indices

    # Different sample produces different mask
    ctx2 = MaskingContext(global_seed=100, sample_id="cifar_2", mask_ratio=0.25)
    mask2 = generate_patch_mask(ctx2, total_patches=t)
    assert mask1.masked_indices != mask2.masked_indices


def test_patch_mask_serialization() -> None:
    """Verify JSON roundtrip for PatchMask."""
    ctx = MaskingContext(global_seed=42, sample_id="test_sample", mask_ratio=0.5)
    mask = generate_patch_mask(ctx, total_patches=8)
    json_str = mask.to_json()

    restored = PatchMask.from_json(json_str)
    assert restored.total_patches == mask.total_patches
    assert restored.masked_indices == mask.masked_indices
    assert restored.visible_indices == mask.visible_indices
    assert restored.sample_id == mask.sample_id


def test_learnable_mask_token_lifecycle() -> None:
    """Test mask token initialization, token substitution, and gradient routing."""
    embed_dim = 8
    token_module = LearnableMaskToken(embed_dim=embed_dim, std=0.02, seed=42)

    params = token_module.get_parameters()
    assert "mask_token" in params
    assert len(params["mask_token"]) == embed_dim

    # Patch tokens: 4 tokens of dimension 8
    patch_tokens = [[float(t * 10 + d) for d in range(embed_dim)] for t in range(4)]
    masked_indices = [1, 3]

    replaced = token_module.replace_masked_patches(patch_tokens, masked_indices)
    assert replaced[0] == patch_tokens[0]
    assert replaced[1] == params["mask_token"]
    assert replaced[2] == patch_tokens[2]
    assert replaced[3] == params["mask_token"]

    # Backward gradient accumulation
    d_tokens = [[1.0 for _ in range(embed_dim)] for _ in range(4)]
    d_inputs = token_module.backward_masked_tokens(d_tokens, masked_indices)

    # Gradients on unmasked positions flow to input
    assert d_inputs[0] == [1.0] * embed_dim
    assert d_inputs[2] == [1.0] * embed_dim
    # Gradients on masked positions do not flow to inputs
    assert d_inputs[1] == [0.0] * embed_dim
    assert d_inputs[3] == [0.0] * embed_dim

    # Gradients on masked positions accumulated into mask token:
    # 2 masked patches * 1.0 = 2.0
    grads = token_module.get_gradients()
    for g in grads["mask_token"]:
        assert pytest.approx(g, abs=1e-6) == 2.0
