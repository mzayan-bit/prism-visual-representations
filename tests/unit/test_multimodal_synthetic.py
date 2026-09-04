"""Unit tests for synthetic vision-language dataset generator."""

from __future__ import annotations

import pytest

from prism.core.errors import ValidationError
from prism.multimodal.synthetic import (
    COLOR_MAP,
    COLORS,
    SHAPES,
    build_synthetic_vocabulary,
    generate_synthetic_multimodal_dataset,
    render_synthetic_image,
)


def test_render_synthetic_image_shapes() -> None:
    """Verify deterministic image rendering for different shapes."""
    for shape in SHAPES:
        for color in COLORS:
            img = render_synthetic_image(
                shape=shape,
                color=color,
                position="center",
                size="large",
                image_shape=(3, 16, 16),
            )
            assert len(img) == 3
            assert len(img[0]) == 16
            assert len(img[0][0]) == 16

            # Check that foreground pixel color matches color map
            expected_rgb = COLOR_MAP[color]
            # Center pixel (8, 8) must have the shape color
            assert pytest.approx(img[0][8][8], abs=1e-3) == expected_rgb[0]
            assert pytest.approx(img[1][8][8], abs=1e-3) == expected_rgb[1]
            assert pytest.approx(img[2][8][8], abs=1e-3) == expected_rgb[2]


def test_generate_synthetic_multimodal_dataset() -> None:
    """Verify generated samples have matching text descriptions and metadata."""
    samples = generate_synthetic_multimodal_dataset(num_samples=12, seed=42)
    assert len(samples) == 12

    for s in samples:
        # Text must be non-empty
        assert len(s.text) > 0
        # Color and shape in metadata must appear in canonical text
        color = s.metadata["color"]
        shape = s.metadata["shape"]
        assert color in s.text
        assert shape in s.text
        assert s.class_label is not None
        assert s.class_name == f"{color}_{shape}"
        assert len(s.captions) >= 3


def test_build_synthetic_vocabulary() -> None:
    """Verify all synthetic attributes and template words are in vocabulary."""
    vocab = build_synthetic_vocabulary()
    for color in COLORS:
        assert color in vocab.tokens
    for shape in SHAPES:
        assert shape in vocab.tokens
    assert "photo" in vocab.tokens
    assert "image" in vocab.tokens


def test_invalid_sample_count() -> None:
    """Verify error on invalid num_samples."""
    with pytest.raises(ValidationError):
        generate_synthetic_multimodal_dataset(num_samples=0)
