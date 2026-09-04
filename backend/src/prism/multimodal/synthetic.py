"""Synthetic Vision-Language Dataset Generator for PRISM Multimodal Research."""

from __future__ import annotations

import hashlib

from prism.core.errors import ValidationError
from prism.multimodal.contracts import VisionLanguageSample
from prism.multimodal.tokenizer import Vocabulary

SHAPES = ["square", "circle", "triangle"]
COLORS = ["red", "green", "blue", "yellow"]
POSITIONS = ["left", "center", "right"]
SIZES = ["small", "large"]

COLOR_MAP: dict[str, tuple[float, float, float]] = {
    "red": (0.9, 0.1, 0.1),
    "green": (0.1, 0.8, 0.1),
    "blue": (0.1, 0.2, 0.9),
    "yellow": (0.9, 0.8, 0.1),
}

PROMPT_TEMPLATES = [
    "a {color} {shape} on the {position}",
    "an image of a {color} {shape} on the {position}",
    "a photo of a {color} {shape}",
    "{color} {shape}",
    "a {size} {color} {shape}",
]


def render_synthetic_image(
    shape: str,
    color: str,
    position: str,
    size: str,
    image_shape: tuple[int, int, int] = (3, 32, 32),
) -> list[list[list[float]]]:
    """Render a deterministic 3D image tensor (C, H, W) for geometric properties."""
    c, h, w = image_shape
    rgb = COLOR_MAP.get(color, (0.5, 0.5, 0.5))

    # Determine center coordinates
    cy = h // 2
    if position == "left":
        cx = w // 4
    elif position == "right":
        cx = (3 * w) // 4
    else:
        cx = w // 2

    # Determine radius
    radius = 5 if size == "small" else 9

    # Create background canvas (dark neutral gray)
    canvas = [[[0.05 for _ in range(w)] for _ in range(h)] for _ in range(c)]

    for y in range(h):
        for x in range(w):
            dx = x - cx
            dy = y - cy
            is_inside = False

            if shape == "square":
                is_inside = abs(dx) <= radius and abs(dy) <= radius
            elif shape == "circle":
                is_inside = (dx * dx + dy * dy) <= (radius * radius)
            elif shape == "triangle":
                # Upright triangle
                if -radius <= dy <= radius:
                    max_w = (radius - dy) * 0.75
                    is_inside = abs(dx) <= max_w
            else:
                is_inside = abs(dx) <= radius and abs(dy) <= radius

            if is_inside:
                for ch in range(c):
                    canvas[ch][y][x] = rgb[ch]

    return canvas


def generate_synthetic_multimodal_dataset(
    num_samples: int = 60,
    image_shape: tuple[int, int, int] = (3, 32, 32),
    seed: int = 42,
    split: str = "train",
) -> list[VisionLanguageSample]:
    """Generate deterministic paired synthetic image-text samples.

    Parameters
    ----------
    num_samples : int
        Number of paired samples to generate.
    image_shape : tuple[int, int, int]
        (C, H, W) of synthesized images.
    seed : int
        Deterministic random seed.
    split : str
        Dataset split name.

    Returns
    -------
    list[VisionLanguageSample]
        Paired multimodal samples.
    """
    if num_samples <= 0:
        raise ValidationError(f"num_samples must be positive, got {num_samples}")

    samples: list[VisionLanguageSample] = []

    # All unique class definitions: color x shape
    classes = [f"{c}_{s}" for c in sorted(COLORS) for s in sorted(SHAPES)]
    class_to_id = {cls_name: idx for idx, cls_name in enumerate(classes)}

    dataset_fingerprint = hashlib.sha256(
        f"synth_vl_{num_samples}_{image_shape}_{seed}_{split}".encode()
    ).hexdigest()[:16]

    for i in range(num_samples):
        # Select attributes deterministically
        color = COLORS[i % len(COLORS)]
        shape = SHAPES[(i // len(COLORS)) % len(SHAPES)]
        position = POSITIONS[(i // (len(COLORS) * len(SHAPES))) % len(POSITIONS)]
        size = SIZES[(i // (len(COLORS) * len(SHAPES) * len(POSITIONS))) % len(SIZES)]

        # Generate image
        image = render_synthetic_image(shape, color, position, size, image_shape)

        # Generate compositional captions
        canonical_caption = f"a {color} {shape} on the {position}"
        all_captions = [
            f"a {color} {shape} on the {position}",
            f"an image of a {color} {shape} on the {position}",
            f"a photo of a {color} {shape}",
            f"{color} {shape}",
            f"a {size} {color} {shape}",
        ]

        class_name = f"{color}_{shape}"
        class_label = class_to_id[class_name]
        sample_id = f"vl_{split}_{i:04d}"
        pair_identity = f"pair_{class_name}_{position}_{size}_{i}"

        sample = VisionLanguageSample(
            sample_id=sample_id,
            image=image,
            text=canonical_caption,
            captions=all_captions,
            class_label=class_label,
            class_name=class_name,
            dataset_fingerprint=dataset_fingerprint,
            split=split,
            pair_identity=pair_identity,
            metadata={
                "color": color,
                "shape": shape,
                "position": position,
                "size": size,
                "class_name": class_name,
            },
        )
        samples.append(sample)

    return samples


def build_synthetic_vocabulary() -> Vocabulary:
    """Construct a deterministic vocabulary for synthetic shapes and templates."""
    words: set[str] = set()
    for color in COLORS:
        words.add(color)
    for shape in SHAPES:
        words.add(shape)
    for pos in POSITIONS:
        words.add(pos)
    for size in SIZES:
        words.add(size)

    # Add words appearing in prompt templates and standard prompts
    extra_words = [
        "a",
        "an",
        "the",
        "on",
        "in",
        "image",
        "photo",
        "of",
        "rendered",
        "picture",
        "visual",
        "representation",
        "view",
        "scene",
        "showing",
    ]
    for w in extra_words:
        words.add(w)

    return Vocabulary(sorted(words))
