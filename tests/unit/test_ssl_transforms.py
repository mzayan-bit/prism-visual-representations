"""Unit tests for deterministic self-supervised visual augmentations."""

from prism.ssl.context import AugmentationContext
from prism.ssl.transforms import (
    AugmentationPolicy,
    ColorJitter,
    Grayscale,
    RandomCropWithPadding,
    RandomHorizontalFlip,
)


def _make_dummy_image(c: int = 3, h: int = 8, w: int = 8) -> list[list[list[float]]]:
    return [
        [[float(row * w + col) / float(h * w) for col in range(w)] for row in range(h)]
        for _ in range(c)
    ]


def test_augmentation_context_determinism() -> None:
    """Test that context derives deterministic seeds across identical contexts."""
    ctx1 = AugmentationContext(
        global_seed=42, sample_id="s1", epoch=0, view_index=0, transform_index=0
    )
    ctx2 = AugmentationContext(
        global_seed=42, sample_id="s1", epoch=0, view_index=0, transform_index=0
    )
    ctx3 = AugmentationContext(
        global_seed=42, sample_id="s1", epoch=0, view_index=1, transform_index=0
    )

    assert ctx1.derive_seed("flip") == ctx2.derive_seed("flip")
    assert ctx1.derive_seed("flip") != ctx3.derive_seed("flip")


def test_random_horizontal_flip_operation() -> None:
    """Test horizontal flip flips columns correctly."""
    img = _make_dummy_image(c=1, h=4, w=4)
    # Set known pattern: left column 1.0, others 0.0
    img[0][0] = [1.0, 0.0, 0.0, 0.0]

    flip = RandomHorizontalFlip(p=1.0)
    ctx = AugmentationContext(
        global_seed=123, sample_id="s_flip", epoch=0, view_index=0, transform_index=0
    )

    out, trace = flip(img, ctx)
    assert trace.applied is True
    assert out[0][0] == [0.0, 0.0, 0.0, 1.0]
    # Verify original unchanged
    assert img[0][0] == [1.0, 0.0, 0.0, 0.0]


def test_random_crop_with_padding() -> None:
    """Test random crop preserves spatial dimensions."""
    c, h, w = 3, 8, 8
    img = _make_dummy_image(c, h, w)
    crop = RandomCropWithPadding(padding=2, pad_mode="reflect")
    ctx = AugmentationContext(
        global_seed=456, sample_id="s_crop", epoch=0, view_index=0, transform_index=0
    )

    out, trace = crop(img, ctx)
    assert trace.applied is True
    assert len(out) == c
    assert len(out[0]) == h
    assert len(out[0][0]) == w


def test_color_jitter_range() -> None:
    """Test color jitter values stay in [0.0, 1.0]."""
    img = _make_dummy_image(3, 6, 6)
    jitter = ColorJitter(brightness=0.3, contrast=0.3)
    ctx = AugmentationContext(
        global_seed=789, sample_id="s_jitter", epoch=0, view_index=0, transform_index=0
    )

    out, trace = jitter(img, ctx)
    assert trace.applied is True
    for ch in range(3):
        for r in range(6):
            for col in range(6):
                assert 0.0 <= out[ch][r][col] <= 1.0


def test_grayscale_conversion() -> None:
    """Test grayscale converts 3 channels to identical luminance planes."""
    img = _make_dummy_image(3, 4, 4)
    gray = Grayscale(p=1.0)
    ctx = AugmentationContext(
        global_seed=999, sample_id="s_gray", epoch=0, view_index=0, transform_index=0
    )

    out, trace = gray(img, ctx)
    assert trace.applied is True
    assert len(out) == 3
    # Channels should be identical
    assert out[0] == out[1] == out[2]


def test_augmentation_policy_traces() -> None:
    """Test sequential policy application and trace recording."""
    img = _make_dummy_image(3, 8, 8)
    policy = AugmentationPolicy(
        [
            RandomHorizontalFlip(p=1.0),
            RandomCropWithPadding(padding=1),
            ColorJitter(brightness=0.1, contrast=0.1),
        ]
    )
    ctx = AugmentationContext(
        global_seed=42, sample_id="s_policy", epoch=0, view_index=0, transform_index=0
    )

    _out, traces = policy.apply(img, ctx)
    assert len(traces) == 3
    assert traces[0].transform_name == "RandomHorizontalFlip"
    assert traces[1].transform_name == "RandomCropWithPadding"
    assert traces[2].transform_name == "ColorJitter"
