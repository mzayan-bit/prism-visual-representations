"""Unit tests for multimodal data contracts and serialization."""

from __future__ import annotations

import pytest

from prism.multimodal.contracts import (
    RetrievalResult,
    TokenizedText,
    VisionLanguageBatch,
    VisionLanguageSample,
    ZeroShotClassificationSummary,
)
from prism.multimodal.enums import (
    MultimodalFailureType,
    MultimodalTaskType,
    PretrainingObjective,
    RetrievalDirection,
    SpecialToken,
)


def test_vision_language_sample_contract() -> None:
    """Verify VisionLanguageSample instantiation, serialization, and roundtrip."""
    sample = VisionLanguageSample(
        sample_id="vl_sample_001",
        image=[[[0.5, 0.2], [0.1, 0.8]]],
        text="a red square on the left",
        captions=["a red square on the left", "red square"],
        class_label=2,
        class_name="red_square",
        dataset_fingerprint="fp12345",
        split="train",
        pair_identity="pair_001",
        metadata={"shape": "square", "color": "red"},
    )

    data = sample.to_dict()
    assert data["sample_id"] == "vl_sample_001"
    assert data["text"] == "a red square on the left"
    assert data["class_label"] == 2
    assert data["class_name"] == "red_square"

    # Roundtrip from dict (providing original image tensor)
    restored = VisionLanguageSample.from_dict(data, image=sample.image)
    assert restored.sample_id == sample.sample_id
    assert restored.text == sample.text
    assert restored.class_label == sample.class_label
    assert restored.pair_identity == sample.pair_identity
    assert restored.image == sample.image


def test_tokenized_text_contract() -> None:
    """Verify TokenizedText serialization and roundtrip."""
    tok_text = TokenizedText(
        original_text="a blue circle",
        token_strings=["<BOS>", "a", "blue", "circle", "<EOS>", "<PAD>"],
        token_ids=[2, 4, 5, 6, 3, 0],
        sequence_length=5,
        attention_mask=[1, 1, 1, 1, 1, 0],
    )

    data = tok_text.to_dict()
    assert data["sequence_length"] == 5
    assert len(data["token_ids"]) == 6

    restored = TokenizedText.from_dict(data)
    assert restored.original_text == tok_text.original_text
    assert restored.token_ids == tok_text.token_ids
    assert restored.attention_mask == tok_text.attention_mask


def test_vision_language_batch_validation() -> None:
    """Verify batch size assertions on mismatched dimensions."""
    tok = TokenizedText(
        original_text="test",
        token_strings=["<BOS>", "test", "<EOS>"],
        token_ids=[2, 4, 3],
        sequence_length=3,
        attention_mask=[1, 1, 1],
    )

    # Mismatched image count vs batch_size
    with pytest.raises(ValueError, match="Batch image count"):
        VisionLanguageBatch(
            sample_ids=["s1"],
            images=[],
            texts=["test"],
            tokenized_texts=[tok],
            token_ids=[[2, 4, 3]],
            attention_masks=[[1, 1, 1]],
            pair_mapping=[0],
            batch_size=1,
        )


def test_retrieval_and_zeroshot_contracts() -> None:
    """Verify retrieval and zero-shot dataclasses roundtripping."""
    ret_res = RetrievalResult(
        query_modality=RetrievalDirection.IMAGE_TO_TEXT,
        query_sample_id="q01",
        ranked_candidate_sample_ids=["c01", "c02"],
        similarities=[0.92, 0.45],
        matched_pair_rank=1,
        top_k_success={1: True, 3: True, 5: True},
        candidate_count=2,
    )
    r_dict = ret_res.to_dict()
    r_restored = RetrievalResult.from_dict(r_dict)
    assert r_restored.matched_pair_rank == 1
    assert r_restored.query_modality == RetrievalDirection.IMAGE_TO_TEXT

    zs_sum = ZeroShotClassificationSummary(
        prompt_template="a photo of a {class_name}",
        class_count=2,
        accuracy=1.0,
        per_class_accuracy={"red_square": 1.0, "blue_circle": 1.0},
        confusion_matrix=[[5, 0], [0, 5]],
        class_names=["red_square", "blue_circle"],
    )
    zs_dict = zs_sum.to_dict()
    zs_restored = ZeroShotClassificationSummary.from_dict(zs_dict)
    assert zs_restored.accuracy == 1.0
    assert zs_restored.confusion_matrix == [[5, 0], [0, 5]]


def test_enums_and_taxonomy() -> None:
    """Verify all required enum constants exist and are well-formed."""
    assert SpecialToken.PAD.value == "<PAD>"
    assert SpecialToken.BOS.value == "<BOS>"
    assert SpecialToken.EOS.value == "<EOS>"
    assert SpecialToken.UNK.value == "<UNK>"

    assert MultimodalFailureType.PAIR_MISALIGNMENT.value == "pair_misalignment"
    assert MultimodalFailureType.MODALITY_COLLAPSE.value == "modality_collapse"
    assert (
        MultimodalTaskType.VISION_LANGUAGE_ALIGNMENT.value
        == "vision_language_alignment"
    )
    assert PretrainingObjective.VISION_LANGUAGE.value == "vision_language"
