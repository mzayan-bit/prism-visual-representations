"""Explicit 2D Convolution layer with analytical backpropagation."""

import copy
from typing import Any

from prism.core.errors import ValidationError
from prism.models.initialization import initialize_conv2d_parameters
from prism.models.spatial import (
    compute_conv2d_output_shape,
    ensure_4d_tensor,
    normalize_spatial_pair,
)


def _pad_4d_tensor(
    x: list[list[list[list[float]]]], pad_h: int, pad_w: int
) -> list[list[list[list[float]]]]:
    """Zero-pad a 4D tensor [N, C, H, W] by pad_h and pad_w on spatial boundaries."""
    if pad_h == 0 and pad_w == 0:
        return x

    n_samples = len(x)
    c_channels = len(x[0])
    h_in = len(x[0][0])
    w_in = len(x[0][0][0])

    padded: list[list[list[list[float]]]] = []
    for n in range(n_samples):
        sample_padded: list[list[list[float]]] = []
        for c in range(c_channels):
            channel_padded: list[list[float]] = []
            # Top padding rows
            for _ in range(pad_h):
                channel_padded.append([0.0 for _ in range(w_in + 2 * pad_w)])
            # Middle rows with left/right padding
            for h in range(h_in):
                row = (
                    [0.0 for _ in range(pad_w)]
                    + list(x[n][c][h])
                    + [0.0 for _ in range(pad_w)]
                )
                channel_padded.append(row)
            # Bottom padding rows
            for _ in range(pad_h):
                channel_padded.append([0.0 for _ in range(w_in + 2 * pad_w)])
            sample_padded.append(channel_padded)
        padded.append(sample_padded)
    return padded


class Conv2D:
    """Explicit 2D multi-channel convolutional layer."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        stride: int | tuple[int, int] = 1,
        padding: int | tuple[int, int] = 0,
        bias: bool = True,
        seed: int = 42,
        activation: str = "relu",
    ) -> None:
        if in_channels <= 0:
            raise ValidationError(f"in_channels must be positive, got {in_channels}.")
        if out_channels <= 0:
            raise ValidationError(f"out_channels must be positive, got {out_channels}.")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.k_h, self.k_w = normalize_spatial_pair(kernel_size, "kernel_size")
        self.s_h, self.s_w = normalize_spatial_pair(stride, "stride")
        self.p_h, self.p_w = normalize_spatial_pair(padding, "padding")
        self.use_bias = bias

        self.weights, self.bias_weights = initialize_conv2d_parameters(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(self.k_h, self.k_w),
            seed=seed,
            bias=bias,
            activation=activation,
        )

        self.zero_grad()
        self._cached_x: list[list[list[list[float]]]] | None = None
        self._cached_x_pad: list[list[list[list[float]]]] | None = None

    def forward(self, inputs: Any) -> list[list[list[list[float]]]]:
        """Compute 2D convolution forward pass producing [N, C_out, H_out, W_out]."""
        x_4d = ensure_4d_tensor(inputs)
        n_samples = len(x_4d)
        c_in = len(x_4d[0])

        if c_in != self.in_channels:
            raise ValidationError(
                f"Input has {c_in} channels, expected in_channels={self.in_channels}."
            )

        h_in = len(x_4d[0][0])
        w_in = len(x_4d[0][0][0])

        h_out, w_out = compute_conv2d_output_shape(
            input_height=h_in,
            input_width=w_in,
            kernel_size=(self.k_h, self.k_w),
            stride=(self.s_h, self.s_w),
            padding=(self.p_h, self.p_w),
        )

        x_pad = _pad_4d_tensor(x_4d, self.p_h, self.p_w)
        self._cached_x = x_4d
        self._cached_x_pad = x_pad

        # Output shape: [N, C_out, H_out, W_out]
        output: list[list[list[list[float]]]] = []

        for n in range(n_samples):
            sample_out: list[list[list[float]]] = []
            for f in range(self.out_channels):
                filter_weights = self.weights[f]
                bias_val = self.bias_weights[f] if self.use_bias else 0.0

                channel_out: list[list[float]] = []
                for i in range(h_out):
                    row_out: list[float] = []
                    h_start = i * self.s_h
                    for j in range(w_out):
                        w_start = j * self.s_w

                        # Sum over C_in and spatial receptive field K_h x K_w
                        accum = bias_val
                        for c in range(self.in_channels):
                            for kh in range(self.k_h):
                                for kw in range(self.k_w):
                                    x_val = x_pad[n][c][h_start + kh][w_start + kw]
                                    w_val = filter_weights[c][kh][kw]
                                    accum += x_val * w_val
                        row_out.append(accum)
                    channel_out.append(row_out)
                sample_out.append(channel_out)
            output.append(sample_out)

        return output

    def backward(
        self, d_out: list[list[list[list[float]]]]
    ) -> list[list[list[list[float]]]]:
        """Compute parameter gradients (weights, bias) and propagate dX backwards."""
        if self._cached_x is None or self._cached_x_pad is None:
            raise ValidationError("Cannot execute backward pass before forward pass.")

        n_samples = len(d_out)
        c_out = len(d_out[0])
        h_out = len(d_out[0][0])
        w_out = len(d_out[0][0][0])

        if c_out != self.out_channels:
            raise ValidationError(
                f"d_out has {c_out} channels, expected {self.out_channels}."
            )

        h_pad = len(self._cached_x_pad[0][0])
        w_pad = len(self._cached_x_pad[0][0][0])

        # Initialize padded input gradient accumulator [N, C_in, H_pad, W_pad]
        dx_pad: list[list[list[list[float]]]] = [
            [
                [[0.0 for _ in range(w_pad)] for _ in range(h_pad)]
                for _ in range(self.in_channels)
            ]
            for _ in range(n_samples)
        ]

        # 1. Gradients w.r.t Bias and Weights & Accumulate dX_pad
        for n in range(n_samples):
            for f in range(self.out_channels):
                filter_weights = self.weights[f]

                for i in range(h_out):
                    h_start = i * self.s_h
                    for j in range(w_out):
                        w_start = j * self.s_w
                        dout_val = d_out[n][f][i][j]

                        if self.use_bias:
                            self.grad_bias_weights[f] += dout_val

                        for c in range(self.in_channels):
                            for kh in range(self.k_h):
                                for kw in range(self.k_w):
                                    x_val = self._cached_x_pad[n][c][h_start + kh][
                                        w_start + kw
                                    ]
                                    self.grad_weights[f][c][kh][kw] += (
                                        dout_val * x_val
                                    )
                                    dx_pad[n][c][h_start + kh][w_start + kw] += (
                                        dout_val * filter_weights[c][kh][kw]
                                    )

        # 2. Crop/Unpad dX_pad to match original input shape [N, C_in, H_in, W_in]
        h_in = len(self._cached_x[0][0])
        w_in = len(self._cached_x[0][0][0])

        dx: list[list[list[list[float]]]] = []
        for n in range(n_samples):
            sample_dx: list[list[list[float]]] = []
            for c in range(self.in_channels):
                ch_dx: list[list[float]] = []
                for h in range(h_in):
                    row_dx = dx_pad[n][c][self.p_h + h][self.p_w : self.p_w + w_in]
                    ch_dx.append(row_dx)
                sample_dx.append(ch_dx)
            dx.append(sample_dx)

        return dx

    def zero_grad(self) -> None:
        """Clear computed gradients for weights and bias."""
        self.grad_weights: list[list[list[list[float]]]] = [
            [
                [[0.0 for _ in range(self.k_w)] for _ in range(self.k_h)]
                for _ in range(self.in_channels)
            ]
            for _ in range(self.out_channels)
        ]
        self.grad_bias_weights: list[float] = (
            [0.0 for _ in range(self.out_channels)] if self.use_bias else []
        )

    def get_parameters(self) -> dict[str, Any]:
        """Return parameters dictionary."""
        params: dict[str, Any] = {"weights": copy.deepcopy(self.weights)}
        if self.use_bias:
            params["bias"] = list(self.bias_weights)
        return params

    def set_parameters(self, params: dict[str, Any]) -> None:
        """Set parameters from dictionary."""
        if "weights" in params:
            self.weights = copy.deepcopy(params["weights"])
        if "bias" in params and self.use_bias:
            self.bias_weights = list(params["bias"])

    def get_gradients(self) -> dict[str, Any]:
        """Return gradients dictionary."""
        grads: dict[str, Any] = {"grad_weights": copy.deepcopy(self.grad_weights)}
        if self.use_bias:
            grads["grad_bias"] = list(self.grad_bias_weights)
        return grads
