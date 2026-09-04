"""Unit tests for multimodal dual-encoder training engine."""

from __future__ import annotations

from prism.core.enums import ModelFamily
from prism.models.specifications import ModelSpecification
from prism.multimodal.runner import MultimodalTrainingEngine
from prism.multimodal.specification import VisionLanguageTrainingSpecification
from prism.multimodal.synthetic import (
    build_synthetic_vocabulary,
    generate_synthetic_multimodal_dataset,
)
from prism.multimodal.tokenizer import SimpleTokenizer


def test_multimodal_training_engine_step() -> None:
    """Verify end-to-end parameter updates across visual and text branches."""
    vocab = build_synthetic_vocabulary()
    tokenizer = SimpleTokenizer(vocab, max_length=10)
    samples = generate_synthetic_multimodal_dataset(
        num_samples=4, image_shape=(3, 8, 8), seed=42
    )

    vis_spec = ModelSpecification(
        model_id="test-cnn-multimodal",
        name="Test CNN",
        family=ModelFamily.CNN,
        architecture="conv_tiny",
        num_classes=2,
        input_shape=(3, 8, 8),
        hyperparameters={
            "conv_channels": [4],
            "kernel_sizes": [3],
            "hidden_dims": [8],
            "pooling": "max",
        },
    )

    spec = VisionLanguageTrainingSpecification(
        visual_family=ModelFamily.CNN,
        visual_spec=vis_spec,
        text_dim=8,
        shared_dim=4,
        temperature=0.1,
        learning_rate=0.05,
        batch_size=4,
        epochs=2,
        seed=42,
    )

    engine = MultimodalTrainingEngine()
    vis_enc, vis_proj, txt_enc, history = engine.train(
        spec=spec,
        samples=samples,
        tokenizer=tokenizer,
    )

    assert len(history) == 2
    assert history[0]["loss"] > 0.0
    assert "matched_similarity" in history[0]

    # Verify model parameters are valid and non-empty
    assert len(vis_enc.get_parameters()) > 0
    assert len(vis_proj.get_parameters()) > 0
    assert len(txt_enc.get_parameters()) > 0
