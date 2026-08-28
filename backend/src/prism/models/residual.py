"""Residual connections, explicit skip shortcuts, and residual blocks."""

from __future__ import annotations

import copy
import math
from typing import Any

from prism.core.errors import ValidationError
from prism.models.activations import BaseActivation, get_activation
from prism.models.convolution import Conv2D
from prism.models.normalization import BaseNormalization, get_normalization
from prism.models.spatial import ensure_4d_tensor


class ResidualAdd:
    """Explicit elementwise residual addition operator Z = A + B."""

    def __init__(self) -> None:
        self._cached_shape: tuple[int, int, int, int] | None = None

    def forward(
        self,
        a: list[list[list[list[float]]]],
        b: list[list[list[list[float]]]],
    ) -> list[list[list[list[float]]]]:
        """Compute elementwise addition Z = A + B with strict shape validation."""
        a_4d = ensure_4d_tensor(a)
        b_4d = ensure_4d_tensor(b)

        n_a, c_a, h_a, w_a = (
            len(a_4d),
            len(a_4d[0]),
            len(a_4d[0][0]),
            len(a_4d[0][0][0]),
        )
        n_b, c_b, h_b, w_b = (
            len(b_4d),
            len(b_4d[0]),
            len(b_4d[0][0]),
            len(b_4d[0][0][0]),
        )

        if (n_a, c_a, h_a, w_a) != (n_b, c_b, h_b, w_b):
            raise ValidationError(
                f"Incompatible shapes for residual addition: "
                f"main path ({n_a}, {c_a}, {h_a}, {w_a}) vs "
                f"shortcut path ({n_b}, {c_b}, {h_b}, {w_b})."
            )

        self._cached_shape = (n_a, c_a, h_a, w_a)
        out_4d: list[list[list[list[float]]]] = []

        for n in range(n_a):
            sample_out: list[list[list[float]]] = []
            for c in range(c_a):
                channel_out: list[list[float]] = []
                for h in range(h_a):
                    row_out: list[float] = []
                    for w in range(w_a):
                        val_a = a_4d[n][c][h][w]
                        val_b = b_4d[n][c][h][w]
                        if (
                            math.isnan(val_a)
                            or math.isinf(val_a)
                            or math.isnan(val_b)
                            or math.isinf(val_b)
                        ):
                            raise ValidationError(
                                f"Non-finite value in residual addition at "
                                f"({n}, {c}, {h}, {w}): a={val_a}, b={val_b}."
                            )
                        row_out.append(val_a + val_b)
                    channel_out.append(row_out)
                sample_out.append(channel_out)
            out_4d.append(sample_out)

        return out_4d

    def backward(
        self, d_out: list[list[list[list[float]]]]
    ) -> tuple[list[list[list[list[float]]]], list[list[list[list[float]]]]]:
        """Route upstream gradient dZ to both paths: dA = dZ, dB = dZ."""
        d_4d = ensure_4d_tensor(d_out)
        if self._cached_shape is not None:
            n_d, c_d, h_d, w_d = (
                len(d_4d),
                len(d_4d[0]),
                len(d_4d[0][0]),
                len(d_4d[0][0][0]),
            )
            if (n_d, c_d, h_d, w_d) != self._cached_shape:
                raise ValidationError(
                    f"Gradient shape mismatch in ResidualAdd backward: "
                    f"expected {self._cached_shape}, got ({n_d}, {c_d}, {h_d}, {w_d})."
                )

        # Deep copy to ensure neither branch can mutate the other's gradient
        d_a = copy.deepcopy(d_4d)
        d_b = copy.deepcopy(d_4d)
        return d_a, d_b


class IdentityShortcut:
    """Parameter-free identity skip connection S(x) = x."""

    def __init__(self) -> None:
        self._is_training: bool = True

    @property
    def is_training(self) -> bool:
        return self._is_training

    def train(self, mode: bool = True) -> IdentityShortcut:
        self._is_training = mode
        return self

    def eval(self) -> IdentityShortcut:
        self._is_training = False
        return self

    def forward(
        self, x: list[list[list[list[float]]]]
    ) -> list[list[list[list[float]]]]:
        """Pass input through unchanged."""
        return ensure_4d_tensor(x)

    def backward(
        self, d_out: list[list[list[list[float]]]]
    ) -> list[list[list[list[float]]]]:
        """Propagate upstream gradient through identity shortcut unchanged."""
        return ensure_4d_tensor(d_out)

    def zero_grad(self) -> None:
        pass

    def get_parameters(self) -> dict[str, Any]:
        return {}

    def set_parameters(self, params: dict[str, Any]) -> None:
        _ = params

    def get_gradients(self) -> dict[str, Any]:
        return {}

    def get_state(self) -> dict[str, Any]:
        return {}

    def set_state(self, state: dict[str, Any]) -> None:
        _ = state


class ProjectionShortcut:
    """Explicit projection skip connection S(x) = Norm(Conv2D(x, 1x1, stride=s))."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        normalization: str = "batch_norm",
        norm_eps: float = 1e-5,
        norm_momentum: float = 0.1,
        norm_affine: bool = True,
        seed: int = 42,
    ) -> None:
        if in_channels <= 0:
            raise ValidationError(f"in_channels must be positive, got {in_channels}.")
        if out_channels <= 0:
            raise ValidationError(f"out_channels must be positive, got {out_channels}.")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride

        # 1x1 Convolution for linear channel/spatial projection
        self.conv = Conv2D(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=stride,
            padding=0,
            bias=True,
            seed=seed,
            activation="linear",
        )

        # Optional Normalization on Shortcut
        self.norm: BaseNormalization | None = get_normalization(
            norm_type=normalization,
            num_features=out_channels,
            is_spatial=True,
            eps=norm_eps,
            momentum=norm_momentum,
            affine=norm_affine,
        )

        self._cached_conv_out: list[list[list[list[float]]]] | None = None
        self._is_training: bool = True

    @property
    def is_training(self) -> bool:
        return self._is_training

    def train(self, mode: bool = True) -> ProjectionShortcut:
        self._is_training = mode
        if self.norm is not None:
            self.norm.train(mode)
        return self

    def eval(self) -> ProjectionShortcut:
        self._is_training = False
        if self.norm is not None:
            self.norm.eval()
        return self

    def forward(
        self, x: list[list[list[list[float]]]]
    ) -> list[list[list[list[float]]]]:
        """Compute projection forward pass."""
        conv_out = self.conv.forward(x)
        self._cached_conv_out = conv_out
        if self.norm is not None:
            norm_res = self.norm.forward(conv_out)
            return ensure_4d_tensor(norm_res)
        return conv_out

    def backward(
        self, d_out: list[list[list[list[float]]]]
    ) -> list[list[list[list[float]]]]:
        """Compute backward gradients through shortcut norm and 1x1 convolution."""
        d_conv_out = self.norm.backward(d_out) if self.norm is not None else d_out
        return self.conv.backward(d_conv_out)

    def zero_grad(self) -> None:
        self.conv.zero_grad()
        if self.norm is not None:
            self.norm.zero_grad()

    def get_parameters(self) -> dict[str, Any]:
        params: dict[str, Any] = {
            "proj_conv_weights": copy.deepcopy(self.conv.weights),
        }
        if self.conv.use_bias:
            params["proj_conv_bias"] = list(self.conv.bias_weights)
        if self.norm is not None:
            for k, v in self.norm.get_parameters().items():
                params[f"proj_norm_{k}"] = copy.deepcopy(v)
        return params

    def set_parameters(self, params: dict[str, Any]) -> None:
        if "proj_conv_weights" in params:
            self.conv.weights = copy.deepcopy(params["proj_conv_weights"])
        if "proj_conv_bias" in params and self.conv.use_bias:
            self.conv.bias_weights = list(params["proj_conv_bias"])
        if self.norm is not None:
            norm_p = {}
            if "proj_norm_gamma" in params:
                norm_p["gamma"] = copy.deepcopy(params["proj_norm_gamma"])
            if "proj_norm_beta" in params:
                norm_p["beta"] = copy.deepcopy(params["proj_norm_beta"])
            self.norm.set_parameters(norm_p)

    def get_gradients(self) -> dict[str, Any]:
        grads: dict[str, Any] = {
            "grad_proj_conv_weights": copy.deepcopy(self.conv.grad_weights),
        }
        if self.conv.use_bias:
            grads["grad_proj_conv_bias"] = list(self.conv.grad_bias_weights)
        if self.norm is not None:
            for k, v in self.norm.get_gradients().items():
                grads[f"grad_proj_norm_{k.replace('grad_', '')}"] = copy.deepcopy(v)
        return grads

    def get_state(self) -> dict[str, Any]:
        state: dict[str, Any] = {}
        if self.norm is not None:
            for k, v in self.norm.get_state().items():
                state[f"proj_norm_{k}"] = copy.deepcopy(v)
        return state

    def set_state(self, state: dict[str, Any]) -> None:
        if self.norm is not None:
            norm_s = {}
            if "proj_norm_running_mean" in state:
                norm_s["running_mean"] = copy.deepcopy(state["proj_norm_running_mean"])
            if "proj_norm_running_var" in state:
                norm_s["running_var"] = copy.deepcopy(state["proj_norm_running_var"])
            if "proj_norm_num_batches_tracked" in state:
                norm_s["num_batches_tracked"] = state["proj_norm_num_batches_tracked"]
            self.norm.set_state(norm_s)


class ResidualBlock:
    """Basic 2-Convolution Residual Block: y = activation(F(x) + S(x))."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        normalization: str = "batch_norm",
        norm_eps: float = 1e-5,
        norm_momentum: float = 0.1,
        norm_affine: bool = True,
        activation: str = "relu",
        seed: int = 42,
    ) -> None:
        if in_channels <= 0:
            raise ValidationError(f"in_channels must be positive, got {in_channels}.")
        if out_channels <= 0:
            raise ValidationError(f"out_channels must be positive, got {out_channels}.")
        if stride <= 0:
            raise ValidationError(f"stride must be positive, got {stride}.")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.activation_name = activation
        self.normalization_name = normalization

        # Main Path: Conv1 (3x3, stride=s) -> Norm1 -> Act1 -> Conv2 (3x3) -> Norm2
        seed_c1 = (seed * 10007 + 11) & 0x7FFFFFFF
        seed_c2 = (seed * 10007 + 23) & 0x7FFFFFFF
        seed_proj = (seed * 10007 + 37) & 0x7FFFFFFF

        self.conv1 = Conv2D(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=True,
            seed=seed_c1,
            activation=activation,
        )
        self.norm1: BaseNormalization | None = get_normalization(
            norm_type=normalization,
            num_features=out_channels,
            is_spatial=True,
            eps=norm_eps,
            momentum=norm_momentum,
            affine=norm_affine,
        )
        self.act1: BaseActivation = get_activation(activation)

        self.conv2 = Conv2D(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
            seed=seed_c2,
            activation=activation,
        )
        self.norm2: BaseNormalization | None = get_normalization(
            norm_type=normalization,
            num_features=out_channels,
            is_spatial=True,
            eps=norm_eps,
            momentum=norm_momentum,
            affine=norm_affine,
        )

        # Shortcut Path: Identity if shape preserves, Projection if dims change
        self.has_projection = (in_channels != out_channels) or (stride != 1)
        self.shortcut: IdentityShortcut | ProjectionShortcut
        if self.has_projection:
            self.shortcut = ProjectionShortcut(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                normalization=normalization,
                norm_eps=norm_eps,
                norm_momentum=norm_momentum,
                norm_affine=norm_affine,
                seed=seed_proj,
            )
        else:
            self.shortcut = IdentityShortcut()

        # Residual Addition & Final Activation
        self.residual_add = ResidualAdd()
        self.act2: BaseActivation = get_activation(activation)

        self._is_training: bool = True

        # Forward caches for backpropagation and representations
        self._cached_x: list[list[list[list[float]]]] | None = None
        self._cached_conv1_out: list[list[list[list[float]]]] | None = None
        self._cached_norm1_out: list[list[list[list[float]]]] | None = None
        self._cached_act1_out: list[list[list[list[float]]]] | None = None
        self._cached_conv2_out: list[list[list[list[float]]]] | None = None
        self._cached_norm2_out: list[list[list[list[float]]]] | None = None
        self._cached_shortcut_out: list[list[list[list[float]]]] | None = None
        self._cached_add_out: list[list[list[list[float]]]] | None = None
        self._cached_final_out: list[list[list[list[float]]]] | None = None

    @property
    def is_training(self) -> bool:
        return self._is_training

    def train(self, mode: bool = True) -> ResidualBlock:
        self._is_training = mode
        if self.norm1 is not None:
            self.norm1.train(mode)
        if self.norm2 is not None:
            self.norm2.train(mode)
        self.shortcut.train(mode)
        return self

    def eval(self) -> ResidualBlock:
        self._is_training = False
        if self.norm1 is not None:
            self.norm1.eval()
        if self.norm2 is not None:
            self.norm2.eval()
        self.shortcut.eval()
        return self

    def forward(
        self, x: list[list[list[list[float]]]]
    ) -> list[list[list[list[float]]]]:
        """Compute residual block forward pass: y = act(F(x) + S(x))."""
        x_4d = ensure_4d_tensor(x)
        self._cached_x = x_4d

        # 1. Main Path: Conv1 -> Norm1 -> Act1 -> Conv2 -> Norm2
        conv1_out = self.conv1.forward(x_4d)
        norm1_out = (
            self.norm1.forward(conv1_out) if self.norm1 is not None else conv1_out
        )
        act1_out = self.act1.forward(norm1_out)

        conv2_out = self.conv2.forward(act1_out)
        norm2_out = (
            self.norm2.forward(conv2_out) if self.norm2 is not None else conv2_out
        )

        # 2. Shortcut Path
        shortcut_out = self.shortcut.forward(x_4d)

        # 3. Residual Addition: Z = F(x) + S(x)
        add_out = self.residual_add.forward(norm2_out, shortcut_out)

        # 4. Final Activation: Y = act(Z)
        final_out_res = self.act2.forward(add_out)
        final_out = ensure_4d_tensor(final_out_res)

        self._cached_conv1_out = conv1_out
        self._cached_norm1_out = norm1_out
        self._cached_act1_out = act1_out
        self._cached_conv2_out = conv2_out
        self._cached_norm2_out = norm2_out
        self._cached_shortcut_out = shortcut_out
        self._cached_add_out = add_out
        self._cached_final_out = final_out

        return final_out

    def backward(
        self, d_out: list[list[list[list[float]]]]
    ) -> list[list[list[list[float]]]]:
        """Propagate gradients backward through post-act, addition, and paths."""
        if (
            self._cached_x is None
            or self._cached_norm1_out is None
            or self._cached_add_out is None
        ):
            raise ValidationError("Cannot perform backward pass before forward pass.")

        # 1. Final Activation Backward: d_add = dL/dZ
        d_add_res = self.act2.backward(self._cached_add_out, d_out)
        d_add = ensure_4d_tensor(d_add_res)

        # 2. Residual Addition Backward: routes d_add to both paths
        d_main, d_shortcut = self.residual_add.backward(d_add)

        # 3. Main Path Backward: Norm2 -> Conv2 -> Act1 -> Norm1 -> Conv1
        d_conv2_out = self.norm2.backward(d_main) if self.norm2 is not None else d_main
        d_act1_out = self.conv2.backward(d_conv2_out)
        d_norm1_out_res = self.act1.backward(self._cached_norm1_out, d_act1_out)
        d_norm1_out = ensure_4d_tensor(d_norm1_out_res)
        d_conv1_out = (
            self.norm1.backward(d_norm1_out) if self.norm1 is not None else d_norm1_out
        )
        d_x_main = self.conv1.backward(d_conv1_out)

        # 4. Shortcut Path Backward
        d_x_shortcut = self.shortcut.backward(d_shortcut)

        # 5. Sum Input Gradients: dX = dX_main + dX_shortcut
        n_samples = len(d_x_main)
        c_in = len(d_x_main[0])
        h_in = len(d_x_main[0][0])
        w_in = len(d_x_main[0][0][0])

        dx: list[list[list[list[float]]]] = []
        for n in range(n_samples):
            sample_dx: list[list[list[float]]] = []
            for c in range(c_in):
                channel_dx: list[list[float]] = []
                for h in range(h_in):
                    row_dx = [
                        d_x_main[n][c][h][w] + d_x_shortcut[n][c][h][w]
                        for w in range(w_in)
                    ]
                    channel_dx.append(row_dx)
                sample_dx.append(channel_dx)
            dx.append(sample_dx)

        return dx

    def zero_grad(self) -> None:
        self.conv1.zero_grad()
        if self.norm1 is not None:
            self.norm1.zero_grad()
        self.conv2.zero_grad()
        if self.norm2 is not None:
            self.norm2.zero_grad()
        self.shortcut.zero_grad()

    def get_parameters(self, prefix: str = "") -> dict[str, Any]:
        params: dict[str, Any] = {}
        p_str = f"{prefix}_" if prefix else ""

        # Conv1
        params[f"{p_str}conv1_weights"] = copy.deepcopy(self.conv1.weights)
        if self.conv1.use_bias:
            params[f"{p_str}conv1_bias"] = list(self.conv1.bias_weights)

        # Norm1
        if self.norm1 is not None:
            for k, v in self.norm1.get_parameters().items():
                params[f"{p_str}norm1_{k}"] = copy.deepcopy(v)

        # Conv2
        params[f"{p_str}conv2_weights"] = copy.deepcopy(self.conv2.weights)
        if self.conv2.use_bias:
            params[f"{p_str}conv2_bias"] = list(self.conv2.bias_weights)

        # Norm2
        if self.norm2 is not None:
            for k, v in self.norm2.get_parameters().items():
                params[f"{p_str}norm2_{k}"] = copy.deepcopy(v)

        # Shortcut
        for k, v in self.shortcut.get_parameters().items():
            params[f"{p_str}{k}"] = copy.deepcopy(v)

        return params

    def set_parameters(self, params: dict[str, Any], prefix: str = "") -> None:
        p_str = f"{prefix}_" if prefix else ""

        # Conv1
        if f"{p_str}conv1_weights" in params:
            self.conv1.weights = copy.deepcopy(params[f"{p_str}conv1_weights"])
        if f"{p_str}conv1_bias" in params and self.conv1.use_bias:
            self.conv1.bias_weights = list(params[f"{p_str}conv1_bias"])

        # Norm1
        if self.norm1 is not None:
            n1_p = {}
            if f"{p_str}norm1_gamma" in params:
                n1_p["gamma"] = copy.deepcopy(params[f"{p_str}norm1_gamma"])
            if f"{p_str}norm1_beta" in params:
                n1_p["beta"] = copy.deepcopy(params[f"{p_str}norm1_beta"])
            self.norm1.set_parameters(n1_p)

        # Conv2
        if f"{p_str}conv2_weights" in params:
            self.conv2.weights = copy.deepcopy(params[f"{p_str}conv2_weights"])
        if f"{p_str}conv2_bias" in params and self.conv2.use_bias:
            self.conv2.bias_weights = list(params[f"{p_str}conv2_bias"])

        # Norm2
        if self.norm2 is not None:
            n2_p = {}
            if f"{p_str}norm2_gamma" in params:
                n2_p["gamma"] = copy.deepcopy(params[f"{p_str}norm2_gamma"])
            if f"{p_str}norm2_beta" in params:
                n2_p["beta"] = copy.deepcopy(params[f"{p_str}norm2_beta"])
            self.norm2.set_parameters(n2_p)

        # Shortcut
        sc_p = {}
        for k, v in params.items():
            if k.startswith(f"{p_str}proj_"):
                sc_key = k[len(p_str) :]
                sc_p[sc_key] = copy.deepcopy(v)
        if sc_p:
            self.shortcut.set_parameters(sc_p)

    def get_gradients(self, prefix: str = "") -> dict[str, Any]:
        grads: dict[str, Any] = {}
        p_str = f"{prefix}_" if prefix else ""

        grads[f"grad_{p_str}conv1_weights"] = copy.deepcopy(self.conv1.grad_weights)
        if self.conv1.use_bias:
            grads[f"grad_{p_str}conv1_bias"] = list(self.conv1.grad_bias_weights)

        if self.norm1 is not None:
            for k, v in self.norm1.get_gradients().items():
                grads[f"grad_{p_str}norm1_{k.replace('grad_', '')}"] = copy.deepcopy(v)

        grads[f"grad_{p_str}conv2_weights"] = copy.deepcopy(self.conv2.grad_weights)
        if self.conv2.use_bias:
            grads[f"grad_{p_str}conv2_bias"] = list(self.conv2.grad_bias_weights)

        if self.norm2 is not None:
            for k, v in self.norm2.get_gradients().items():
                grads[f"grad_{p_str}norm2_{k.replace('grad_', '')}"] = copy.deepcopy(v)

        for k, v in self.shortcut.get_gradients().items():
            grads[f"grad_{p_str}{k.replace('grad_', '')}"] = copy.deepcopy(v)

        return grads

    def get_state(self, prefix: str = "") -> dict[str, Any]:
        state: dict[str, Any] = {}
        p_str = f"{prefix}_" if prefix else ""

        if self.norm1 is not None:
            for k, v in self.norm1.get_state().items():
                state[f"{p_str}norm1_{k}"] = copy.deepcopy(v)

        if self.norm2 is not None:
            for k, v in self.norm2.get_state().items():
                state[f"{p_str}norm2_{k}"] = copy.deepcopy(v)

        for k, v in self.shortcut.get_state().items():
            state[f"{p_str}{k}"] = copy.deepcopy(v)

        return state

    def set_state(self, state: dict[str, Any], prefix: str = "") -> None:
        p_str = f"{prefix}_" if prefix else ""

        if self.norm1 is not None:
            n1_s = {}
            if f"{p_str}norm1_running_mean" in state:
                n1_s["running_mean"] = copy.deepcopy(
                    state[f"{p_str}norm1_running_mean"]
                )
            if f"{p_str}norm1_running_var" in state:
                n1_s["running_var"] = copy.deepcopy(state[f"{p_str}norm1_running_var"])
            if f"{p_str}norm1_num_batches_tracked" in state:
                n1_s["num_batches_tracked"] = state[f"{p_str}norm1_num_batches_tracked"]
            self.norm1.set_state(n1_s)

        if self.norm2 is not None:
            n2_s = {}
            if f"{p_str}norm2_running_mean" in state:
                n2_s["running_mean"] = copy.deepcopy(
                    state[f"{p_str}norm2_running_mean"]
                )
            if f"{p_str}norm2_running_var" in state:
                n2_s["running_var"] = copy.deepcopy(state[f"{p_str}norm2_running_var"])
            if f"{p_str}norm2_num_batches_tracked" in state:
                n2_s["num_batches_tracked"] = state[f"{p_str}norm2_num_batches_tracked"]
            self.norm2.set_state(n2_s)

        sc_s = {}
        for k, v in state.items():
            if k.startswith(f"{p_str}proj_"):
                sc_key = k[len(p_str) :]
                sc_s[sc_key] = copy.deepcopy(v)
        if sc_s:
            self.shortcut.set_state(sc_s)
