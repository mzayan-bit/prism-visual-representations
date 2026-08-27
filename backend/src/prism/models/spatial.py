"""Spatial dimension utilities, shape calculations, and receptive field tracking."""

import math
from typing import Any

from prism.core.errors import ValidationError


def normalize_spatial_pair(
    param: int | tuple[int, int], name: str = "parameter"
) -> tuple[int, int]:
    """Normalize integer or 2-tuple parameter to (height, width) with validation."""
    if isinstance(param, int):
        if param < 0:
            raise ValidationError(f"{name} must be non-negative, got {param}.")
        return (param, param)
    if isinstance(param, (tuple, list)) and len(param) == 2:
        h, w = int(param[0]), int(param[1])
        if h < 0 or w < 0:
            raise ValidationError(
                f"{name} dimensions must be non-negative, got {param}."
            )
        return (h, w)
    raise ValidationError(
        f"{name} must be an integer or a 2-tuple of integers, got {type(param)}."
    )


def compute_conv2d_output_shape(
    input_height: int,
    input_width: int,
    kernel_size: int | tuple[int, int],
    stride: int | tuple[int, int] = 1,
    padding: int | tuple[int, int] = 0,
) -> tuple[int, int]:
    """Compute 2D spatial output dimensions (H_out, W_out) for convolution.

    Formula:
        H_out = floor((H_in + 2*pad_h - kernel_h) / stride_h) + 1
        W_out = floor((W_in + 2*pad_w - kernel_w) / stride_w) + 1
    """
    if input_height <= 0 or input_width <= 0:
        raise ValidationError(
            f"Input spatial dimensions must be positive, "
            f"got ({input_height}, {input_width})."
        )

    k_h, k_w = normalize_spatial_pair(kernel_size, "kernel_size")
    s_h, s_w = normalize_spatial_pair(stride, "stride")
    p_h, p_w = normalize_spatial_pair(padding, "padding")

    if k_h <= 0 or k_w <= 0:
        raise ValidationError(
            f"kernel_size must be strictly positive, got ({k_h}, {k_w})."
        )
    if s_h <= 0 or s_w <= 0:
        raise ValidationError(
            f"stride must be strictly positive, got ({s_h}, {s_w})."
        )

    eff_h = input_height + 2 * p_h
    eff_w = input_width + 2 * p_w

    if k_h > eff_h:
        raise ValidationError(
            f"Kernel height ({k_h}) exceeds padded input height ({eff_h})."
        )
    if k_w > eff_w:
        raise ValidationError(
            f"Kernel width ({k_w}) exceeds padded input width ({eff_w})."
        )

    out_h = math.floor((eff_h - k_h) / s_h) + 1
    out_w = math.floor((eff_w - k_w) / s_w) + 1

    if out_h <= 0 or out_w <= 0:
        raise ValidationError(
            f"Calculated non-positive output shape: ({out_h}, {out_w})."
        )

    return (out_h, out_w)


def compute_pool2d_output_shape(
    input_height: int,
    input_width: int,
    kernel_size: int | tuple[int, int] = 2,
    stride: int | tuple[int, int] = 2,
    padding: int | tuple[int, int] = 0,
) -> tuple[int, int]:
    """Compute 2D spatial output dimensions for spatial pooling."""
    return compute_conv2d_output_shape(
        input_height=input_height,
        input_width=input_width,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
    )


def compute_receptive_field(
    stages: list[tuple[int, int]],
) -> tuple[int, int]:
    """Compute effective receptive field size and jump across a sequence of stages.

    Parameters
    ----------
    stages : list[tuple[int, int]]
        List of (kernel_size, stride) for each sequential stage.

    Returns
    -------
    tuple[int, int]
        (receptive_field_size, effective_jump)
    """
    rf = 1
    jump = 1

    for idx, (k, s) in enumerate(stages):
        if k <= 0 or s <= 0:
            raise ValidationError(
                f"Stage {idx} parameters must be positive, "
                f"got kernel={k}, stride={s}."
            )
        rf += (k - 1) * jump
        jump *= s

    return (rf, jump)


def ensure_4d_tensor(data: Any) -> list[list[list[list[float]]]]:
    """Validate and normalize nested data into 4D tensor [N, C, H, W]."""
    if data is None:
        raise ValidationError("Input tensor cannot be None.")

    # Single 3D image [C, H, W] -> Wrap into [1, C, H, W]
    if isinstance(data, (list, tuple)):
        if not data:
            raise ValidationError("Tensor batch cannot be empty.")

        first_elem = data[0]
        if isinstance(first_elem, (list, tuple)) and first_elem:
            second_elem = first_elem[0]
            if isinstance(second_elem, (list, tuple)) and second_elem:
                third_elem = second_elem[0]
                if isinstance(third_elem, (list, tuple)):
                    # Already 4D: [N, C, H, W]
                    return [
                        [
                            [
                                [float(val) for val in row]
                                for row in ch
                            ]
                            for ch in sample
                        ]
                        for sample in data
                    ]
                else:
                    # 3D: [C, H, W] -> wrap to [1, C, H, W]
                    single_sample = [
                        [[float(val) for val in row] for row in ch]
                        for ch in data
                    ]
                    return [single_sample]

    raise ValidationError(
        "Expected 4D batch [N, C, H, W] or 3D sample [C, H, W] nested list structure."
    )
