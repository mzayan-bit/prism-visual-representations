"""Spatial pooling layers (MaxPool2D and AvgPool2D) with analytical backpropagation."""

from typing import Any

from prism.core.errors import ValidationError
from prism.models.convolution import _pad_4d_tensor
from prism.models.spatial import (
    compute_pool2d_output_shape,
    ensure_4d_tensor,
    normalize_spatial_pair,
)


class MaxPool2D:
    """2D Spatial Max Pooling layer with exact argmax gradient routing."""

    def __init__(
        self,
        kernel_size: int | tuple[int, int] = 2,
        stride: int | tuple[int, int] = 2,
        padding: int | tuple[int, int] = 0,
    ) -> None:
        self.k_h, self.k_w = normalize_spatial_pair(kernel_size, "kernel_size")
        self.s_h, self.s_w = normalize_spatial_pair(stride, "stride")
        self.p_h, self.p_w = normalize_spatial_pair(padding, "padding")

        if self.k_h <= 0 or self.k_w <= 0:
            raise ValidationError(
                f"Pooling kernel_size must be positive, got ({self.k_h}, {self.k_w})."
            )
        if self.s_h <= 0 or self.s_w <= 0:
            raise ValidationError(
                f"Pooling stride must be positive, got ({self.s_h}, {self.s_w})."
            )

        self._cached_x: list[list[list[list[float]]]] | None = None
        self._cached_argmax: list[list[list[list[tuple[int, int]]]]] | None = None

    def forward(self, inputs: Any) -> list[list[list[list[float]]]]:
        """Compute MaxPool2D forward pass producing [N, C, H_out, W_out]."""
        x_4d = ensure_4d_tensor(inputs)
        n_samples = len(x_4d)
        c_channels = len(x_4d[0])
        h_in = len(x_4d[0][0])
        w_in = len(x_4d[0][0][0])

        h_out, w_out = compute_pool2d_output_shape(
            input_height=h_in,
            input_width=w_in,
            kernel_size=(self.k_h, self.k_w),
            stride=(self.s_h, self.s_w),
            padding=(self.p_h, self.p_w),
        )

        x_pad = _pad_4d_tensor(x_4d, self.p_h, self.p_w)
        self._cached_x = x_4d

        output: list[list[list[list[float]]]] = []
        argmax_map: list[list[list[list[tuple[int, int]]]]] = []

        for n in range(n_samples):
            sample_out: list[list[list[float]]] = []
            sample_argmax: list[list[list[tuple[int, int]]]] = []

            for c in range(c_channels):
                ch_out: list[list[float]] = []
                ch_argmax: list[list[tuple[int, int]]] = []

                for i in range(h_out):
                    row_out: list[float] = []
                    row_argmax: list[tuple[int, int]] = []
                    h_start = i * self.s_h

                    for j in range(w_out):
                        w_start = j * self.s_w

                        # Find maximum in spatial window
                        max_val = float("-inf")
                        max_coords = (h_start, w_start)

                        for kh in range(self.k_h):
                            for kw in range(self.k_w):
                                cur_h = h_start + kh
                                cur_w = w_start + kw
                                val = x_pad[n][c][cur_h][cur_w]
                                if val > max_val:
                                    max_val = val
                                    max_coords = (cur_h, cur_w)

                        row_out.append(max_val)
                        row_argmax.append(max_coords)

                    ch_out.append(row_out)
                    ch_argmax.append(row_argmax)

                sample_out.append(ch_out)
                sample_argmax.append(ch_argmax)

            output.append(sample_out)
            argmax_map.append(sample_argmax)

        self._cached_argmax = argmax_map
        return output

    def backward(
        self, d_out: list[list[list[list[float]]]]
    ) -> list[list[list[list[float]]]]:
        """Route upstream gradients d_out to argmax locations in input tensor."""
        if self._cached_x is None or self._cached_argmax is None:
            raise ValidationError(
                "Cannot perform MaxPool2D backward pass before forward pass."
            )

        n_samples = len(d_out)
        c_channels = len(d_out[0])
        h_out = len(d_out[0][0])
        w_out = len(d_out[0][0][0])

        h_in = len(self._cached_x[0][0])
        w_in = len(self._cached_x[0][0][0])
        h_pad = h_in + 2 * self.p_h
        w_pad = w_in + 2 * self.p_w

        # Initialize padded input gradient accumulator [N, C, H_pad, W_pad]
        dx_pad: list[list[list[list[float]]]] = [
            [
                [[0.0 for _ in range(w_pad)] for _ in range(h_pad)]
                for _ in range(c_channels)
            ]
            for _ in range(n_samples)
        ]

        for n in range(n_samples):
            for c in range(c_channels):
                for i in range(h_out):
                    for j in range(w_out):
                        dout_val = d_out[n][c][i][j]
                        max_h, max_w = self._cached_argmax[n][c][i][j]
                        dx_pad[n][c][max_h][max_w] += dout_val

        # Unpad dx_pad to match input shape [N, C, H_in, W_in]
        dx: list[list[list[list[float]]]] = []
        for n in range(n_samples):
            sample_dx: list[list[list[float]]] = []
            for c in range(c_channels):
                ch_dx: list[list[float]] = []
                for h in range(h_in):
                    row = dx_pad[n][c][self.p_h + h][self.p_w : self.p_w + w_in]
                    ch_dx.append(row)
                sample_dx.append(ch_dx)
            dx.append(sample_dx)

        return dx


class AvgPool2D:
    """2D Spatial Average Pooling layer."""

    def __init__(
        self,
        kernel_size: int | tuple[int, int] = 2,
        stride: int | tuple[int, int] = 2,
        padding: int | tuple[int, int] = 0,
    ) -> None:
        self.k_h, self.k_w = normalize_spatial_pair(kernel_size, "kernel_size")
        self.s_h, self.s_w = normalize_spatial_pair(stride, "stride")
        self.p_h, self.p_w = normalize_spatial_pair(padding, "padding")

        if self.k_h <= 0 or self.k_w <= 0:
            raise ValidationError(
                f"AvgPool kernel_size must be positive, got ({self.k_h}, {self.k_w})."
            )
        if self.s_h <= 0 or self.s_w <= 0:
            raise ValidationError(
                f"AvgPool stride must be positive, got ({self.s_h}, {self.s_w})."
            )

        self._cached_x: list[list[list[list[float]]]] | None = None

    def forward(self, inputs: Any) -> list[list[list[list[float]]]]:
        """Compute AvgPool2D forward pass producing [N, C, H_out, W_out]."""
        x_4d = ensure_4d_tensor(inputs)
        n_samples = len(x_4d)
        c_channels = len(x_4d[0])
        h_in = len(x_4d[0][0])
        w_in = len(x_4d[0][0][0])

        h_out, w_out = compute_pool2d_output_shape(
            input_height=h_in,
            input_width=w_in,
            kernel_size=(self.k_h, self.k_w),
            stride=(self.s_h, self.s_w),
            padding=(self.p_h, self.p_w),
        )

        x_pad = _pad_4d_tensor(x_4d, self.p_h, self.p_w)
        self._cached_x = x_4d
        window_area = float(self.k_h * self.k_w)

        output: list[list[list[list[float]]]] = []
        for n in range(n_samples):
            sample_out: list[list[list[float]]] = []
            for c in range(c_channels):
                ch_out: list[list[float]] = []
                for i in range(h_out):
                    row_out: list[float] = []
                    h_start = i * self.s_h
                    for j in range(w_out):
                        w_start = j * self.s_w
                        accum = 0.0
                        for kh in range(self.k_h):
                            for kw in range(self.k_w):
                                accum += x_pad[n][c][h_start + kh][w_start + kw]
                        row_out.append(accum / window_area)
                    ch_out.append(row_out)
                sample_out.append(ch_out)
            output.append(sample_out)

        return output

    def backward(
        self, d_out: list[list[list[list[float]]]]
    ) -> list[list[list[list[float]]]]:
        """Distribute upstream gradients d_out uniformly across pooling windows."""
        if self._cached_x is None:
            raise ValidationError(
                "Cannot perform AvgPool2D backward pass before forward pass."
            )

        n_samples = len(d_out)
        c_channels = len(d_out[0])
        h_out = len(d_out[0][0])
        w_out = len(d_out[0][0][0])

        h_in = len(self._cached_x[0][0])
        w_in = len(self._cached_x[0][0][0])
        h_pad = h_in + 2 * self.p_h
        w_pad = w_in + 2 * self.p_w

        window_area = float(self.k_h * self.k_w)
        dx_pad: list[list[list[list[float]]]] = [
            [
                [[0.0 for _ in range(w_pad)] for _ in range(h_pad)]
                for _ in range(c_channels)
            ]
            for _ in range(n_samples)
        ]

        for n in range(n_samples):
            for c in range(c_channels):
                for i in range(h_out):
                    h_start = i * self.s_h
                    for j in range(w_out):
                        w_start = j * self.s_w
                        grad_val = d_out[n][c][i][j] / window_area
                        for kh in range(self.k_h):
                            for kw in range(self.k_w):
                                dx_pad[n][c][h_start + kh][w_start + kw] += grad_val

        # Unpad dx_pad to [N, C, H_in, W_in]
        dx: list[list[list[list[float]]]] = []
        for n in range(n_samples):
            sample_dx: list[list[list[float]]] = []
            for c in range(c_channels):
                ch_dx: list[list[float]] = []
                for h in range(h_in):
                    row = dx_pad[n][c][self.p_h + h][self.p_w : self.p_w + w_in]
                    ch_dx.append(row)
                sample_dx.append(ch_dx)
            dx.append(sample_dx)

        return dx
