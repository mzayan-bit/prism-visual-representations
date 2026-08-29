"""Unit tests for VisionTransformer representation extraction and attention profiles."""

import pytest

from prism.core.enums import ModelFamily, TaskType
from prism.core.errors import ValidationError
from prism.models.specifications import ModelSpecification
from prism.models.transformer import VisionTransformer
from prism.representations.attention import (
    compare_transformer_attention_profiles,
    compute_transformer_attention_profile,
)


def _create_vit() -> VisionTransformer:
    spec = ModelSpecification(
        model_id="repr-vit",
        name="Representation Test ViT",
        family=ModelFamily.VISION_TRANSFORMER,
        architecture="vit_tiny",
        compatible_tasks=[TaskType.CLASSIFICATION],
        input_shape=(1, 8, 8),
        num_classes=2,
        hyperparameters={
            "patch_size": 4,
            "embed_dim": 8,
            "num_heads": 2,
            "depth": 2,
            "mlp_ratio": 2.0,
            "norm_eps": 1e-5,
            "activation": "gelu",
        },
    )
    return VisionTransformer(spec=spec, seed=42)


def test_vit_intermediate_representation_extraction() -> None:
    """Test extracting representations at each distinct architectural point."""
    model = _create_vit()
    x = [[[[float(r + c) for c in range(8)] for r in range(8)]] for _ in range(2)]

    # 1. Input Spatial [N=2, C=1, H=8, W=8]
    r_inp = model.extract_representations(x, layer="input_spatial")
    assert len(r_inp) == 2
    assert len(r_inp[0]) == 1
    assert len(r_inp[0][0]) == 8

    # 2. Extracted Patches [N=2, T=4, D_patch=16]
    r_patches = model.extract_representations(x, layer="patches")
    assert len(r_patches) == 2
    assert len(r_patches[0]) == 4
    assert len(r_patches[0][0]) == 16

    # 3. Patch Embeddings [N=2, T=4, D_model=8]
    r_emb = model.extract_representations(x, layer="patch_embeddings")
    assert len(r_emb) == 2
    assert len(r_emb[0]) == 4
    assert len(r_emb[0][0]) == 8

    # 4. Tokens Pre Position (with CLS) [N=2, T+1=5, D_model=8]
    r_pre_pos = model.extract_representations(x, layer="tokens_pre_position")
    assert len(r_pre_pos) == 2
    assert len(r_pre_pos[0]) == 5
    assert len(r_pre_pos[0][0]) == 8

    # 5. Tokens Post Position [N=2, T+1=5, D_model=8]
    r_post_pos = model.extract_representations(x, layer="tokens_post_position")
    assert len(r_post_pos) == 2
    assert len(r_post_pos[0]) == 5

    # 6. Encoder Block 0 Output [N=2, T+1=5, D_model=8]
    r_enc0 = model.extract_representations(x, layer="encoder_0_output")
    assert len(r_enc0) == 2
    assert len(r_enc0[0]) == 5

    # 7. Final Tokens [N=2, T+1=5, D_model=8]
    r_final = model.extract_representations(x, layer="final_tokens")
    assert len(r_final) == 2
    assert len(r_final[0]) == 5

    # 8. CLS Representation [N=2, D_model=8]
    r_cls = model.extract_representations(x, layer="cls_representation")
    assert len(r_cls) == 2
    assert len(r_cls[0]) == 8

    # 9. Patch Tokens [N=2, T=4, D_model=8]
    r_patch_toks = model.extract_representations(x, layer="patch_tokens")
    assert len(r_patch_toks) == 2
    assert len(r_patch_toks[0]) == 4
    assert len(r_patch_toks[0][0]) == 8

    # 10. Logits [N=2, num_classes=2]
    r_logits = model.extract_representations(x, layer="logits")
    assert len(r_logits) == 2
    assert len(r_logits[0]) == 2


def test_vit_attention_profile_computation() -> None:
    """Test generating TransformerAttentionProfile across encoder depth."""
    model = _create_vit()
    x = [[[[1.0 for _ in range(8)] for _ in range(8)]] for _ in range(2)]
    _ = model.forward(x)

    attn_weights = model.get_attention_weights()
    profile = compute_transformer_attention_profile(
        attn_weights, model_id=model.model_id
    )

    assert profile.depth == 2
    assert profile.num_heads == 2
    assert len(profile.layer_summaries) == 2
    assert len(profile.layer_mean_entropies) == 2
    assert len(profile.layer_diagonal_masses) == 2
    assert all(ent >= 0.0 for ent in profile.layer_mean_entropies)

    # Serialization test
    d = profile.to_dict()
    restored = profile.from_dict(d)
    assert restored.depth == profile.depth


def test_vit_attention_profile_comparison() -> None:
    """Test comparing two attention profiles across layers."""
    model_a = _create_vit()
    model_b = _create_vit()

    x = [[[[1.0 for _ in range(8)] for _ in range(8)]] for _ in range(2)]
    _ = model_a.forward(x)
    _ = model_b.forward(x)

    p_a = compute_transformer_attention_profile(
        model_a.get_attention_weights(), model_id="vit-a"
    )
    p_b = compute_transformer_attention_profile(
        model_b.get_attention_weights(), model_id="vit-b"
    )

    comparison = compare_transformer_attention_profiles(p_a, p_b)
    assert comparison["depth"] == 2
    assert len(comparison["layer_comparisons"]) == 2
    assert len(comparison["layer_entropy_deltas"]) == 2


def test_vit_invalid_representation_layer() -> None:
    """Test requesting unsupported layer name raises ValidationError."""
    model = _create_vit()
    x = [[[[1.0 for _ in range(8)] for _ in range(8)]] for _ in range(2)]

    with pytest.raises(ValidationError):
        model.extract_representations(x, layer="nonexistent_layer_123")
