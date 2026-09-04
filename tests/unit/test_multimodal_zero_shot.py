"""Unit tests for zero-shot classification and prompt sensitivity."""

from __future__ import annotations

from prism.multimodal.contracts import ClassPrompt
from prism.multimodal.embeddings import TextEncoder
from prism.multimodal.evaluation import (
    evaluate_prompt_sensitivity,
    evaluate_zero_shot_classification,
)
from prism.multimodal.tokenizer import SimpleTokenizer, Vocabulary


def test_zero_shot_classification_flow() -> None:
    """Verify zero-shot class matching without training a classifier."""
    vocab = Vocabulary(["a", "photo", "of", "red", "square", "blue", "circle"])
    tokenizer = SimpleTokenizer(vocab, max_length=8)
    text_encoder = TextEncoder(vocab_size=vocab.size, text_dim=8, shared_dim=4, seed=42)

    class_names = ["red_square", "blue_circle"]
    prompts = [
        ClassPrompt(
            class_id=0,
            class_name="red_square",
            prompt_template="a photo of {class_name}",
            rendered_text="a photo of red square",
            prompt_id="p0",
        ),
        ClassPrompt(
            class_id=1,
            class_name="blue_circle",
            prompt_template="a photo of {class_name}",
            rendered_text="a photo of blue circle",
            prompt_id="p1",
        ),
    ]

    # Dummy image embeddings (2 samples, 4D)
    image_embeddings = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
    ]
    true_labels = [0, 1]

    summary = evaluate_zero_shot_classification(
        image_embeddings=image_embeddings,
        class_prompts=prompts,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        true_labels=true_labels,
        class_names=class_names,
    )

    assert summary.class_count == 2
    assert 0.0 <= summary.accuracy <= 1.0
    assert len(summary.confusion_matrix) == 2
    assert len(summary.confusion_matrix[0]) == 2


def test_prompt_sensitivity_analysis() -> None:
    """Verify prompt sensitivity comparison across multiple templates."""
    vocab = Vocabulary(
        [
            "a",
            "photo",
            "of",
            "image",
            "an",
            "red",
            "square",
            "blue",
            "circle",
            "on",
            "the",
            "center",
        ]
    )
    tokenizer = SimpleTokenizer(vocab, max_length=10)
    text_encoder = TextEncoder(vocab_size=vocab.size, text_dim=8, shared_dim=4, seed=42)

    templates = [
        "a photo of a {class_name}",
        "an image of a {class_name}",
    ]
    class_names = ["red_square", "blue_circle"]
    image_embeddings = [[0.5, 0.5, 0.5, 0.5], [-0.5, -0.5, 0.5, 0.5]]
    true_labels = [0, 1]

    res = evaluate_prompt_sensitivity(
        image_embeddings=image_embeddings,
        prompt_templates=templates,
        class_names=class_names,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        true_labels=true_labels,
    )

    assert len(res["templates"]) == 2
    assert len(res["results"]) == 2
    assert len(res["pairwise_agreements"]) == 1
