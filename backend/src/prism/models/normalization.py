"""Batch normalization layers for vector and convolutional visual representations."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Any

from prism.core.errors import ConfigurationError, ValidationError


class BaseNormalization(ABC):
    """Abstract base contract for normalization layers in PRISM."""

    def __init__(self) -> None:
        self._is_training: bool = True

    @property
    def is_training(self) -> bool:
        """Return True if normalization is in training mode."""
        return self._is_training

    def train(self, mode: bool = True) -> BaseNormalization:
        """Set layer to training mode (updates running statistics)."""
        self._is_training = mode
        return self

    def eval(self) -> BaseNormalization:
        """Set layer to evaluation mode (uses running statistics)."""
        self._is_training = False
        return self

    @abstractmethod
    def forward(self, x: Any) -> Any:
        """Compute normalization forward pass."""
        ...

    @abstractmethod
    def backward(self, d_out: Any) -> Any:
        """Compute analytic gradient w.r.t input and accumulate parameter gradients."""
        ...

    @abstractmethod
    def zero_grad(self) -> None:
        """Reset parameter gradients to zero."""
        ...

    @abstractmethod
    def get_parameters(self) -> dict[str, Any]:
        """Return trainable affine parameters (gamma, beta)."""
        ...

    @abstractmethod
    def set_parameters(self, params: dict[str, Any]) -> None:
        """Load trainable affine parameters."""
        ...

    @abstractmethod
    def get_gradients(self) -> dict[str, Any]:
        """Return parameter gradients."""
        ...

    @abstractmethod
    def get_state(self) -> dict[str, Any]:
        """Return non-trainable state (running statistics, batch count)."""
        ...

    @abstractmethod
    def set_state(self, state: dict[str, Any]) -> None:
        """Load non-trainable state."""
        ...


class BatchNorm1D(BaseNormalization):
    """Batch Normalization over 2D vector features [N, D].

    Normalizes each feature dimension across the batch:
        x_hat = (x - mean) / sqrt(var + eps)
        y = gamma * x_hat + beta
    """

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
    ) -> None:
        super().__init__()
        if num_features <= 0:
            raise ValidationError(
                f"num_features must be positive, got {num_features}."
            )
        if eps <= 0.0:
            raise ValidationError(f"eps must be positive, got {eps}.")
        if momentum < 0.0 or momentum > 1.0:
            raise ValidationError(
                f"momentum must be in [0.0, 1.0], got {momentum}."
            )

        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine

        # Trainable Affine Parameters
        if self.affine:
            self.gamma: list[float] = [1.0] * num_features
            self.beta: list[float] = [0.0] * num_features
            self.grad_gamma: list[float] = [0.0] * num_features
            self.grad_beta: list[float] = [0.0] * num_features
        else:
            self.gamma = [1.0] * num_features
            self.beta = [0.0] * num_features
            self.grad_gamma = [0.0] * num_features
            self.grad_beta = [0.0] * num_features

        # Non-trainable Running Statistics
        self.running_mean: list[float] = [0.0] * num_features
        self.running_var: list[float] = [1.0] * num_features
        self.num_batches_tracked: int = 0

        # Forward Pass Caching for Backward
        self._cached_x: list[list[float]] | None = None
        self._cached_x_hat: list[list[float]] | None = None
        self._cached_mean: list[float] | None = None
        self._cached_var: list[float] | None = None
        self._cached_inv_std: list[float] | None = None

    def forward(self, x: list[list[float]]) -> list[list[float]]:
        """Normalize batch of vector features [N, D]."""
        if not isinstance(x, list) or not x:
            raise ValidationError("Input batch to BatchNorm1D cannot be empty.")

        n_samples = len(x)
        if len(x[0]) != self.num_features:
            raise ValidationError(
                f"Expected feature dimension {self.num_features}, got {len(x[0])}."
            )

        out: list[list[float]] = []

        if self.is_training:
            # Training Mode: compute batch statistics
            if n_samples == 1:
                # Small-batch behavior: for N=1 in training, variance is zero
                mean = list(x[0])
                var = [0.0] * self.num_features
            else:
                mean = [0.0] * self.num_features
                for n in range(n_samples):
                    for d in range(self.num_features):
                        mean[d] += x[n][d]
                mean = [m / n_samples for m in mean]

                var = [0.0] * self.num_features
                for n in range(n_samples):
                    for d in range(self.num_features):
                        diff = x[n][d] - mean[d]
                        var[d] += diff * diff
                var = [v / n_samples for v in var]

            # Update running statistics using exponential moving average
            for d in range(self.num_features):
                self.running_mean[d] = (
                    1.0 - self.momentum
                ) * self.running_mean[d] + self.momentum * mean[d]
                self.running_var[d] = (
                    1.0 - self.momentum
                ) * self.running_var[d] + self.momentum * var[d]

            self.num_batches_tracked += 1

            # Compute normalized x_hat and output y
            inv_std = [1.0 / math.sqrt(v + self.eps) for v in var]
            x_hat_batch: list[list[float]] = []

            for n in range(n_samples):
                x_hat_row: list[float] = []
                out_row: list[float] = []
                for d in range(self.num_features):
                    x_hat_val = (x[n][d] - mean[d]) * inv_std[d]
                    x_hat_row.append(x_hat_val)
                    if self.affine:
                        out_val = self.gamma[d] * x_hat_val + self.beta[d]
                    else:
                        out_val = x_hat_val
                    out_row.append(out_val)
                x_hat_batch.append(x_hat_row)
                out.append(out_row)

            self._cached_x = x
            self._cached_x_hat = x_hat_batch
            self._cached_mean = mean
            self._cached_var = var
            self._cached_inv_std = inv_std
        else:
            # Evaluation Mode: use running statistics (do not update running stats)
            inv_std = [1.0 / math.sqrt(v + self.eps) for v in self.running_var]
            for n in range(n_samples):
                out_row = []
                for d in range(self.num_features):
                    x_hat_val = (
                        x[n][d] - self.running_mean[d]
                    ) * inv_std[d]
                    if self.affine:
                        out_val = self.gamma[d] * x_hat_val + self.beta[d]
                    else:
                        out_val = x_hat_val
                    out_row.append(out_val)
                out.append(out_row)

        return out

    def backward(self, d_out: list[list[float]]) -> list[list[float]]:
        """Compute analytic gradient w.r.t input and accumulate parameter gradients."""
        if (
            self._cached_x is None
            or self._cached_x_hat is None
            or self._cached_inv_std is None
        ):
            raise ValidationError(
                "Cannot perform backward pass before forward pass in training mode."
            )

        n_samples = len(d_out)
        d = self.num_features

        # 1. Parameter Gradients: d_gamma and d_beta
        if self.affine:
            for n in range(n_samples):
                for j in range(d):
                    dout_val = d_out[n][j]
                    self.grad_beta[j] += dout_val
                    self.grad_gamma[j] += dout_val * self._cached_x_hat[n][j]

        # 2. Input Gradient: dX
        # Simplified batchnorm gradient formula:
        # dX = (gamma * inv_std / N) * [ N*d_out - sum(d_out) - x_hat*sum(d_out*x_hat) ]
        if n_samples == 1:
            return [[0.0 for _ in range(d)] for _ in range(n_samples)]

        # Intermediate sums per feature
        sum_dout = [0.0] * d
        sum_dout_xhat = [0.0] * d
        for n in range(n_samples):
            for j in range(d):
                dout_val = d_out[n][j]
                sum_dout[j] += dout_val
                sum_dout_xhat[j] += dout_val * self._cached_x_hat[n][j]

        dx: list[list[float]] = []
        for n in range(n_samples):
            dx_row: list[float] = []
            for j in range(d):
                gamma_val = self.gamma[j] if self.affine else 1.0
                scale = (gamma_val * self._cached_inv_std[j]) / float(n_samples)
                val = scale * (
                    float(n_samples) * d_out[n][j]
                    - sum_dout[j]
                    - self._cached_x_hat[n][j] * sum_dout_xhat[j]
                )
                dx_row.append(val)
            dx.append(dx_row)

        return dx

    def zero_grad(self) -> None:
        """Reset parameter gradients."""
        self.grad_gamma = [0.0] * self.num_features
        self.grad_beta = [0.0] * self.num_features

    def get_parameters(self) -> dict[str, Any]:
        """Return trainable parameters."""
        if not self.affine:
            return {}
        return {
            "gamma": list(self.gamma),
            "beta": list(self.beta),
        }

    def set_parameters(self, params: dict[str, Any]) -> None:
        """Load trainable parameters."""
        if self.affine:
            if "gamma" in params:
                self.gamma = list(params["gamma"])
            if "beta" in params:
                self.beta = list(params["beta"])

    def get_gradients(self) -> dict[str, Any]:
        """Return parameter gradients."""
        if not self.affine:
            return {}
        return {
            "grad_gamma": list(self.grad_gamma),
            "grad_beta": list(self.grad_beta),
        }

    def get_state(self) -> dict[str, Any]:
        """Return non-trainable running state."""
        return {
            "running_mean": list(self.running_mean),
            "running_var": list(self.running_var),
            "num_batches_tracked": self.num_batches_tracked,
        }

    def set_state(self, state: dict[str, Any]) -> None:
        """Load non-trainable running state."""
        if "running_mean" in state:
            self.running_mean = list(state["running_mean"])
        if "running_var" in state:
            self.running_var = list(state["running_var"])
        if "num_batches_tracked" in state:
            self.num_batches_tracked = int(state["num_batches_tracked"])


class BatchNorm2D(BaseNormalization):
    """Batch Normalization over 4D spatial convolutional feature maps [N, C, H, W].

    Computes normalization statistics channel-wise across batch and spatial coordinates:
        M = N * H * W
        mean_c = (1/M) * sum_{n, h, w} X_{n, c, h, w}
        var_c = (1/M) * sum_{n, h, w} (X_{n, c, h, w} - mean_c)^2
        x_hat = (X - mean_c) / sqrt(var_c + eps)
        Y = gamma_c * x_hat + beta_c
    """

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
    ) -> None:
        super().__init__()
        if num_features <= 0:
            raise ValidationError(
                f"num_features (channels) must be positive, got {num_features}."
            )
        if eps <= 0.0:
            raise ValidationError(f"eps must be positive, got {eps}.")
        if momentum < 0.0 or momentum > 1.0:
            raise ValidationError(
                f"momentum must be in [0.0, 1.0], got {momentum}."
            )

        self.num_features = num_features  # Number of channels C
        self.eps = eps
        self.momentum = momentum
        self.affine = affine

        # Trainable Affine Parameters (1D per channel)
        self.gamma: list[float] = [1.0] * num_features
        self.beta: list[float] = [0.0] * num_features
        self.grad_gamma: list[float] = [0.0] * num_features
        self.grad_beta: list[float] = [0.0] * num_features

        # Non-trainable Running Statistics (1D per channel)
        self.running_mean: list[float] = [0.0] * num_features
        self.running_var: list[float] = [1.0] * num_features
        self.num_batches_tracked: int = 0

        # Forward Pass Caching for Backward
        self._cached_x: list[list[list[list[float]]]] | None = None
        self._cached_x_hat: list[list[list[list[float]]]] | None = None
        self._cached_mean: list[float] | None = None
        self._cached_var: list[float] | None = None
        self._cached_inv_std: list[float] | None = None
        self._cached_m: int = 0

    def forward(
        self, x: list[list[list[list[float]]]]
    ) -> list[list[list[list[float]]]]:
        """Normalize batch of 4D spatial feature tensors [N, C, H, W]."""
        if not isinstance(x, list) or not x:
            raise ValidationError("Input tensor to BatchNorm2D cannot be empty.")

        n_samples = len(x)
        c_channels = len(x[0])
        h_len = len(x[0][0])
        w_len = len(x[0][0][0])

        if c_channels != self.num_features:
            raise ValidationError(
                f"Expected {self.num_features} channels, got {c_channels}."
            )

        m = n_samples * h_len * w_len  # Total elements per channel
        out_4d: list[list[list[list[float]]]] = []

        if self.is_training:
            # 1. Compute Channel-Wise Mean across N * H * W
            mean = [0.0] * self.num_features
            for n in range(n_samples):
                for c in range(self.num_features):
                    for h in range(h_len):
                        for w in range(w_len):
                            mean[c] += x[n][c][h][w]
            mean = [m_val / float(m) for m_val in mean]

            # 2. Compute Channel-Wise Variance across N * H * W
            var = [0.0] * self.num_features
            for n in range(n_samples):
                for c in range(self.num_features):
                    for h in range(h_len):
                        for w in range(w_len):
                            diff = x[n][c][h][w] - mean[c]
                            var[c] += diff * diff
            var = [v_val / float(m) for v_val in var]

            # 3. Update Running Statistics
            for c in range(self.num_features):
                self.running_mean[c] = (
                    1.0 - self.momentum
                ) * self.running_mean[c] + self.momentum * mean[c]
                self.running_var[c] = (
                    1.0 - self.momentum
                ) * self.running_var[c] + self.momentum * var[c]

            self.num_batches_tracked += 1

            # 4. Normalize & Apply Affine
            inv_std = [1.0 / math.sqrt(v + self.eps) for v in var]
            x_hat_4d: list[list[list[list[float]]]] = []

            for n in range(n_samples):
                sample_hat: list[list[list[float]]] = []
                sample_out: list[list[list[float]]] = []

                for c in range(self.num_features):
                    c_hat: list[list[float]] = []
                    c_out: list[list[float]] = []
                    gamma_c = self.gamma[c] if self.affine else 1.0
                    beta_c = self.beta[c] if self.affine else 0.0
                    inv_s = inv_std[c]
                    mu = mean[c]

                    for h in range(h_len):
                        row_hat: list[float] = []
                        row_out: list[float] = []
                        for w in range(w_len):
                            hat_val = (x[n][c][h][w] - mu) * inv_s
                            out_val = gamma_c * hat_val + beta_c
                            row_hat.append(hat_val)
                            row_out.append(out_val)
                        c_hat.append(row_hat)
                        c_out.append(row_out)
                    sample_hat.append(c_hat)
                    sample_out.append(c_out)
                x_hat_4d.append(sample_hat)
                out_4d.append(sample_out)

            self._cached_x = x
            self._cached_x_hat = x_hat_4d
            self._cached_mean = mean
            self._cached_var = var
            self._cached_inv_std = inv_std
            self._cached_m = m
        else:
            # Evaluation Mode: use running statistics
            inv_std = [1.0 / math.sqrt(v + self.eps) for v in self.running_var]
            for n in range(n_samples):
                sample_out = []
                for c in range(self.num_features):
                    c_out = []
                    gamma_c = self.gamma[c] if self.affine else 1.0
                    beta_c = self.beta[c] if self.affine else 0.0
                    inv_s = inv_std[c]
                    mu = self.running_mean[c]

                    for h in range(h_len):
                        row_out = []
                        for w in range(w_len):
                            hat_val = (x[n][c][h][w] - mu) * inv_s
                            out_val = gamma_c * hat_val + beta_c
                            row_out.append(out_val)
                        c_out.append(row_out)
                    sample_out.append(c_out)
                out_4d.append(sample_out)

        return out_4d

    def backward(
        self, d_out: list[list[list[list[float]]]]
    ) -> list[list[list[list[float]]]]:
        """Compute channel-wise backward gradients w.r.t gamma, beta, and input dX."""
        if (
            self._cached_x is None
            or self._cached_x_hat is None
            or self._cached_inv_std is None
        ):
            raise ValidationError(
                "Cannot perform backward pass before forward pass in training mode."
            )

        n_samples = len(d_out)
        c_channels = len(d_out[0])
        h_len = len(d_out[0][0])
        w_len = len(d_out[0][0][0])
        m = self._cached_m

        # 1. Parameter Gradients: d_gamma and d_beta per channel
        if self.affine:
            for n in range(n_samples):
                for c in range(self.num_features):
                    for h in range(h_len):
                        for w in range(w_len):
                            dout_val = d_out[n][c][h][w]
                            self.grad_beta[c] += dout_val
                            self.grad_gamma[c] += (
                                dout_val * self._cached_x_hat[n][c][h][w]
                            )

        # 2. Input Gradient: dX
        if m == 1:
            return [
                [
                    [[0.0 for _ in range(w_len)] for _ in range(h_len)]
                    for _ in range(c_channels)
                ]
                for _ in range(n_samples)
            ]

        # Channel-wise reduction sums
        sum_dout = [0.0] * self.num_features
        sum_dout_xhat = [0.0] * self.num_features

        for n in range(n_samples):
            for c in range(self.num_features):
                for h in range(h_len):
                    for w in range(w_len):
                        dout_val = d_out[n][c][h][w]
                        sum_dout[c] += dout_val
                        sum_dout_xhat[c] += (
                            dout_val * self._cached_x_hat[n][c][h][w]
                        )

        dx: list[list[list[list[float]]]] = []
        for n in range(n_samples):
            sample_dx: list[list[list[float]]] = []
            for c in range(self.num_features):
                c_dx: list[list[float]] = []
                gamma_c = self.gamma[c] if self.affine else 1.0
                scale = (gamma_c * self._cached_inv_std[c]) / float(m)

                for h in range(h_len):
                    row_dx: list[float] = []
                    for w in range(w_len):
                        dout_val = d_out[n][c][h][w]
                        xhat_val = self._cached_x_hat[n][c][h][w]
                        val = scale * (
                            float(m) * dout_val
                            - sum_dout[c]
                            - xhat_val * sum_dout_xhat[c]
                        )
                        row_dx.append(val)
                    c_dx.append(row_dx)
                sample_dx.append(c_dx)
            dx.append(sample_dx)

        return dx

    def zero_grad(self) -> None:
        """Reset parameter gradients."""
        self.grad_gamma = [0.0] * self.num_features
        self.grad_beta = [0.0] * self.num_features

    def get_parameters(self) -> dict[str, Any]:
        """Return trainable parameters."""
        if not self.affine:
            return {}
        return {
            "gamma": list(self.gamma),
            "beta": list(self.beta),
        }

    def set_parameters(self, params: dict[str, Any]) -> None:
        """Load trainable parameters."""
        if self.affine:
            if "gamma" in params:
                self.gamma = list(params["gamma"])
            if "beta" in params:
                self.beta = list(params["beta"])

    def get_gradients(self) -> dict[str, Any]:
        """Return parameter gradients."""
        if not self.affine:
            return {}
        return {
            "grad_gamma": list(self.grad_gamma),
            "grad_beta": list(self.grad_beta),
        }

    def get_state(self) -> dict[str, Any]:
        """Return non-trainable running state."""
        return {
            "running_mean": list(self.running_mean),
            "running_var": list(self.running_var),
            "num_batches_tracked": self.num_batches_tracked,
        }

    def set_state(self, state: dict[str, Any]) -> None:
        """Load non-trainable running state."""
        if "running_mean" in state:
            self.running_mean = list(state["running_mean"])
        if "running_var" in state:
            self.running_var = list(state["running_var"])
        if "num_batches_tracked" in state:
            self.num_batches_tracked = int(state["num_batches_tracked"])


def get_normalization(
    norm_type: str | None,
    num_features: int,
    is_spatial: bool = False,
    eps: float = 1e-5,
    momentum: float = 0.1,
    affine: bool = True,
) -> BaseNormalization | None:
    """Factory function creating a normalization layer if configured."""
    if not norm_type or norm_type.strip().lower() in ("none", "null", ""):
        return None

    norm = norm_type.strip().lower()
    if norm in ("batch_norm", "batchnorm", "bn"):
        if is_spatial:
            return BatchNorm2D(
                num_features=num_features,
                eps=eps,
                momentum=momentum,
                affine=affine,
            )
        else:
            return BatchNorm1D(
                num_features=num_features,
                eps=eps,
                momentum=momentum,
                affine=affine,
            )

    raise ConfigurationError(
        f"Unsupported normalization '{norm_type}'. Supported: 'none', 'batch_norm'."
    )
