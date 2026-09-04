"""Deterministic synthetic spatial dataset generator for detection and segmentation."""

from __future__ import annotations

import hashlib
import random

from prism.core.errors import ValidationError
from prism.spatial.annotations import (
    BoundingBox,
    DetectionAnnotation,
    DetectionSample,
    SegmentationSample,
)


def generate_synthetic_spatial_dataset(
    num_samples: int = 20,
    image_shape: tuple[int, int, int] = (3, 32, 32),
    num_classes: int = 3,
    max_objects_per_image: int = 2,
    seed: int = 42,
    split: str = "train",
) -> tuple[list[DetectionSample], list[SegmentationSample]]:
    """Generate deterministic synthetic spatial samples.

    Parameters
    ----------
    num_samples : int
        Number of spatial samples to synthesize.
    image_shape : tuple[int, int, int]
        (channels, height, width) of synthesized images.
    num_classes : int
        Total categories (0 is background, 1..K-1 are objects).
    max_objects_per_image : int
        Maximum number of geometric objects to insert per sample.
    seed : int
        Deterministic random seed.
    split : str
        Dataset partition identifier ('train', 'val', 'test').

    Returns
    -------
    tuple[list[DetectionSample], list[SegmentationSample]]
        Aligned detection and segmentation sample collections.
    """
    if num_samples <= 0:
        raise ValidationError(f"num_samples must be positive, got {num_samples}.")
    if len(image_shape) != 3:
        raise ValidationError(
            f"image_shape must be 3-tuple (C, H, W), got {image_shape}."
        )
    c_img, h_img, w_img = image_shape
    if c_img <= 0 or h_img <= 0 or w_img <= 0:
        raise ValidationError(
            f"All dimensions in image_shape must be positive, got {image_shape}."
        )
    if num_classes < 2:
        raise ValidationError(
            f"num_classes must be >= 2 (background + object), got {num_classes}."
        )

    rng = random.Random(seed)

    class_colors: list[list[float]] = []
    for cls_idx in range(num_classes):
        if cls_idx == 0:
            class_colors.append([0.05 for _ in range(c_img)])
        else:
            color = [
                0.2 + 0.6 * ((cls_idx * 37 + ch * 59) % 100) / 100.0
                for ch in range(c_img)
            ]
            class_colors.append(color)

    fp_hasher = hashlib.sha256()
    fp_hasher.update(
        f"synthetic_spatial_{seed}_{num_samples}_{image_shape}_{num_classes}_{split}".encode()
    )
    dataset_fingerprint = fp_hasher.hexdigest()[:16]

    detection_samples: list[DetectionSample] = []
    segmentation_samples: list[SegmentationSample] = []

    for s_idx in range(num_samples):
        sample_id = f"syn_spatial_{split}_{s_idx:04d}"

        image: list[list[list[float]]] = [
            [
                [
                    class_colors[0][ch] + 0.02 * (rng.random() - 0.5)
                    for _ in range(w_img)
                ]
                for _ in range(h_img)
            ]
            for ch in range(c_img)
        ]

        mask: list[list[int]] = [[0 for _ in range(w_img)] for _ in range(h_img)]

        if num_samples > 5 and s_idx % 10 == 0:
            n_objs = 0
        else:
            n_objs = rng.randint(1, max_objects_per_image)

        annotations: list[DetectionAnnotation] = []
        occupied_regions: list[tuple[int, int, int, int]] = []

        for _obj_idx in range(n_objs):
            target_class = rng.randint(1, num_classes - 1)
            class_name = f"object_type_{target_class}"

            min_dim = max(3, min(h_img, w_img) // 8)
            max_dim = max(min_dim + 2, min(h_img, w_img) // 2)

            box_h = rng.randint(min_dim, max_dim)
            box_w = rng.randint(min_dim, max_dim)

            placed = False
            for _ in range(20):
                top = rng.randint(1, h_img - box_h - 1)
                left = rng.randint(1, w_img - box_w - 1)
                bottom = top + box_h
                right = left + box_w

                overlap = False
                for o_top, o_left, o_bottom, o_right in occupied_regions:
                    if not (
                        right <= o_left
                        or left >= o_right
                        or bottom <= o_top
                        or top >= o_bottom
                    ):
                        overlap = True
                        break

                if not overlap:
                    occupied_regions.append((top, left, bottom, right))
                    placed = True
                    break

            if not placed:
                continue

            obj_color = class_colors[target_class]
            for r in range(top, bottom):
                for c in range(left, right):
                    mask[r][c] = target_class
                    for ch in range(c_img):
                        image[ch][r][c] = obj_color[ch] + 0.05 * (rng.random() - 0.5)

            box = BoundingBox.from_pixels(
                x_min_px=float(left),
                y_min_px=float(top),
                x_max_px=float(right),
                y_max_px=float(bottom),
                img_w=w_img,
                img_h=h_img,
            )

            annotation = DetectionAnnotation(
                class_id=target_class,
                box=box,
                class_name=class_name,
                image_width=w_img,
                image_height=h_img,
            )
            annotations.append(annotation)

        det_sample = DetectionSample(
            sample_id=sample_id,
            image=image,
            annotations=annotations,
            dataset_fingerprint=dataset_fingerprint,
            split=split,
            metadata={"num_objects": len(annotations), "seed": seed},
        )
        detection_samples.append(det_sample)

        seg_sample = SegmentationSample(
            sample_id=sample_id,
            image=image,
            mask=mask,
            image_shape=(c_img, h_img, w_img),
            num_classes=num_classes,
            dataset_fingerprint=dataset_fingerprint,
            split=split,
            metadata={"num_objects": len(annotations), "seed": seed},
        )
        segmentation_samples.append(seg_sample)

    return detection_samples, segmentation_samples
