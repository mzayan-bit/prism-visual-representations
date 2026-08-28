"""Multi-Layer Perceptron (MLP) vision baseline model with optional normalization."""

import copy
import random
from collections.abc import Sequence
from typing import Any

from prism.core.errors import ValidationError
from prism.models.activations import BaseActivation, get_activation
from prism.models.base import BaseVisionModel
from prism.models.initialization import initialize_mlp_parameters
from prism.models.linear import _flatten_single_input
from prism.models.normalization import BaseNormalization, get_normalization
from prism.models.specifications import ModelSpecification


class MultiLayerPerceptron(BaseVisionModel):
    """Deep MLP with activations, dropout, and optional normalization."""

    def __init__(
        self,
        spec: ModelSpecification,
        seed: int = 42,
    ) -> None:
        super().__init__(spec)
        if spec.num_classes is None or spec.num_classes <= 0:
            raise ValidationError(
                f"num_classes must be positive, got {spec.num_classes}."
            )

        # 1. Dimensionality Calculation
        if "in_features" in spec.hyperparameters:
            self.in_features: int = int(spec.hyperparameters["in_features"])
        else:
            self.in_features = 1
            for dim in spec.input_shape:
                self.in_features *= dim

        raw_hidden = spec.hyperparameters.get("hidden_dims", [128])
        if not raw_hidden:
            raise ValidationError("hidden_dims cannot be empty for an MLP.")
        self.hidden_dims: list[int] = [int(h) for h in raw_hidden]
        for idx, h in enumerate(self.hidden_dims):
            if h <= 0:
                raise ValidationError(f"hidden_dims[{idx}] must be positive, got {h}.")

        self.num_classes_val: int = spec.num_classes
        act_name = str(spec.hyperparameters.get("activation", "relu"))
        self.activation: BaseActivation = get_activation(act_name)

        dropout_p = float(spec.hyperparameters.get("dropout", 0.0))
        if dropout_p < 0.0 or dropout_p >= 1.0:
            raise ValidationError(
                f"Dropout probability must be in [0.0, 1.0), got {dropout_p}."
            )
        self.dropout: float = dropout_p
        self.seed: int = seed

        # Normalization Hyperparameters
        self.normalization_name: str = str(
            spec.hyperparameters.get("normalization", "none")
        ).lower()
        self.norm_eps: float = float(spec.hyperparameters.get("norm_eps", 1e-5))
        self.norm_momentum: float = float(
            spec.hyperparameters.get("norm_momentum", 0.1)
        )
        self.norm_affine: bool = bool(spec.hyperparameters.get("norm_affine", True))

        # 2. Parameter Initialization
        self.layer_weights, self.layer_biases = initialize_mlp_parameters(
            in_features=self.in_features,
            hidden_dims=self.hidden_dims,
            num_classes=self.num_classes_val,
            seed=seed,
            activation=act_name,
        )

        self.num_layers = len(self.layer_weights)  # hidden layers + 1 output layer

        # 3. Normalization Layers for Hidden Stages
        self.norm_layers: list[BaseNormalization | None] = []
        for h_dim in self.hidden_dims:
            norm = get_normalization(
                self.normalization_name,
                num_features=h_dim,
                is_spatial=False,
                eps=self.norm_eps,
                momentum=self.norm_momentum,
                affine=self.norm_affine,
            )
            self.norm_layers.append(norm)

        self.zero_grad()

        # Cached tensors for backpropagation
        self._cached_h: list[list[list[float]]] = []
        self._cached_z: list[list[list[float]]] = []
        self._cached_norm_out: list[list[list[float]] | None] = []
        self._cached_masks: list[list[list[float]] | None] = []

        # Internal step counter for reproducible dropout seeding
        self._step_counter: int = 0

    def train(self, mode: bool = True) -> "MultiLayerPerceptron":
        """Set training mode across MLP and normalization layers."""
        super().train(mode)
        for norm in self.norm_layers:
            if norm is not None:
                norm.train(mode)
        return self

    def eval(self) -> "MultiLayerPerceptron":
        """Set evaluation mode across MLP and normalization layers."""
        super().eval()
        for norm in self.norm_layers:
            if norm is not None:
                norm.eval()
        return self

    def _prepare_inputs(self, inputs: Any) -> list[list[float]]:
        """Flatten and validate batch of inputs into shape [B, in_features]."""
        if inputs is None:
            raise ValidationError("Model input cannot be None.")

        if isinstance(inputs, Sequence) and not isinstance(inputs, (str, bytes)):
            raw_batch = list(inputs)
        else:
            raw_batch = [inputs]

        if not raw_batch:
            raise ValidationError("Batch cannot be empty.")

        x_flat: list[list[float]] = []
        for idx, item in enumerate(raw_batch):
            flat = _flatten_single_input(item)
            if len(flat) != self.in_features:
                raise ValidationError(
                    f"Sample at batch index {idx} has feature dimension {len(flat)}, "
                    f"expected in_features={self.in_features}."
                )
            x_flat.append(flat)
        return x_flat

    def _matmul_add_bias(
        self, x: list[list[float]], weights: list[list[float]], bias: list[float]
    ) -> list[list[float]]:
        """Compute matrix-vector product Z = XW + b."""
        batch_size = len(x)
        fan_in = len(weights)
        fan_out = len(bias)

        out: list[list[float]] = []
        for i in range(batch_size):
            row_out: list[float] = []
            x_row = x[i]
            for c in range(fan_out):
                val = bias[c]
                for d in range(fan_in):
                    val += x_row[d] * weights[d][c]
                row_out.append(val)
            out.append(row_out)
        return out

    def forward(self, inputs: Any) -> list[list[float]]:
        """Execute forward pass and produce raw output logits [B, num_classes]."""
        h_current = self._prepare_inputs(inputs)
        batch_size = len(h_current)

        if self.is_training:
            self._step_counter += 1

        self._cached_h = [h_current]
        self._cached_z = []
        self._cached_norm_out = []
        self._cached_masks = []

        # Forward through hidden layers: Linear -> (Norm) -> Act -> (Dropout)
        num_hidden = self.num_layers - 1
        for l_idx in range(num_hidden):
            w = self.layer_weights[l_idx]
            b = self.layer_biases[l_idx]
            norm = self.norm_layers[l_idx]

            z = self._matmul_add_bias(h_current, w, b)
            self._cached_z.append(z)

            # Optional Normalization
            if norm is not None:
                norm_out = norm.forward(z)
                act_in = norm_out
            else:
                norm_out = None
                act_in = z

            self._cached_norm_out.append(norm_out)

            # Activation
            a = self.activation.forward(act_in)

            # Apply dropout in training mode
            if self.is_training and self.dropout > 0.0:
                p_keep = 1.0 - self.dropout
                inv_p_keep = 1.0 / p_keep
                fan_out = len(b)

                layer_seed = (
                    (self.seed * 1000003) ^ (l_idx * 10007) ^ (self._step_counter * 31)
                ) & 0x7FFFFFFF
                rng = random.Random(layer_seed)

                mask: list[list[float]] = []
                h_next: list[list[float]] = []
                for i in range(batch_size):
                    m_row = [
                        inv_p_keep if rng.random() < p_keep else 0.0
                        for _ in range(fan_out)
                    ]
                    mask.append(m_row)
                    h_next.append([a[i][j] * m_row[j] for j in range(fan_out)])

                self._cached_masks.append(mask)
                h_current = h_next
            else:
                self._cached_masks.append(None)
                h_current = a

            self._cached_h.append(h_current)

        # Output Layer: raw unnormalized logits
        w_out = self.layer_weights[-1]
        b_out = self.layer_biases[-1]
        logits = self._matmul_add_bias(h_current, w_out, b_out)
        self._cached_z.append(logits)

        return logits

    def extract_representations(self, inputs: Any, layer: str = "final_hidden") -> Any:
        """Extract intermediate activations without modifying model state."""
        norm_layer = layer.lower().strip()
        was_training = self.is_training
        self.eval()

        try:
            h_current = self._prepare_inputs(inputs)

            if norm_layer in ("input_flat", "input"):
                return h_current

            num_hidden = self.num_layers - 1
            for l_idx in range(num_hidden):
                w = self.layer_weights[l_idx]
                b = self.layer_biases[l_idx]
                norm = self.norm_layers[l_idx]

                z = self._matmul_add_bias(h_current, w, b)
                if norm_layer == f"hidden_{l_idx}_pre_norm":
                    return z

                if norm is not None:
                    norm_out = norm.forward(z)
                    act_in = norm_out
                else:
                    norm_out = None
                    act_in = z

                if norm_layer == f"hidden_{l_idx}_post_norm":
                    return norm_out if norm_out is not None else z

                a = self.activation.forward(act_in)
                h_current = a

                if norm_layer == f"hidden_{l_idx}":
                    return h_current

            if norm_layer in ("final_hidden", f"hidden_{num_hidden - 1}"):
                return h_current

            if norm_layer == "logits":
                w_out = self.layer_weights[-1]
                b_out = self.layer_biases[-1]
                return self._matmul_add_bias(h_current, w_out, b_out)

            valid = (
                ["input_flat", "final_hidden", "logits"]
                + [f"hidden_{i}" for i in range(num_hidden)]
                + [f"hidden_{i}_pre_norm" for i in range(num_hidden)]
                + [f"hidden_{i}_post_norm" for i in range(num_hidden)]
            )
            raise ValidationError(
                f"Unknown layer '{layer}' for MultiLayerPerceptron. Supported: {valid}"
            )
        finally:
            if was_training:
                self.train()

    def backward(self, d_logits: list[list[float]]) -> None:
        """Propagate gradients backward through output layer and all hidden layers."""
        if not self._cached_h or not self._cached_z:
            raise ValidationError("Cannot perform backward pass before forward pass.")

        batch_size = len(d_logits)
        cached_bs = len(self._cached_h[0])
        if cached_bs != batch_size:
            raise ValidationError(
                f"d_logits size ({batch_size}) != cached input size ({cached_bs})."
            )

        # 1. Output layer gradient
        w_out = self.layer_weights[-1]
        h_final = self._cached_h[-1]
        fan_in_out = len(w_out)
        fan_out = len(self.layer_biases[-1])

        for d in range(fan_in_out):
            for c in range(fan_out):
                val = 0.0
                for i in range(batch_size):
                    val += h_final[i][d] * d_logits[i][c]
                self.grad_weights[-1][d][c] += val

        for c in range(fan_out):
            val_b = 0.0
            for i in range(batch_size):
                val_b += d_logits[i][c]
            self.grad_biases[-1][c] += val_b

        # dH_final = d_logits @ W_out.T
        d_h_current: list[list[float]] = []
        for i in range(batch_size):
            row_dh: list[float] = []
            for d in range(fan_in_out):
                sum_val = 0.0
                for c in range(fan_out):
                    sum_val += d_logits[i][c] * w_out[d][c]
                row_dh.append(sum_val)
            d_h_current.append(row_dh)

        # 2. Hidden layers backward (reversed)
        num_hidden = self.num_layers - 1
        for l_idx in reversed(range(num_hidden)):
            norm = self.norm_layers[l_idx]
            z = self._cached_z[l_idx]
            norm_out = self._cached_norm_out[l_idx]
            mask = self._cached_masks[l_idx]
            h_in = self._cached_h[l_idx]
            w = self.layer_weights[l_idx]

            fan_in = len(w)
            fan_out_h = len(self.layer_biases[l_idx])

            # Backward through dropout mask
            if mask is not None:
                d_a = [
                    [d_h_current[i][j] * mask[i][j] for j in range(fan_out_h)]
                    for i in range(batch_size)
                ]
            else:
                d_a = d_h_current

            # Backward through activation
            act_in = norm_out if norm is not None else z
            d_act_in = self.activation.backward(act_in, d_a)

            # Backward through normalization
            d_z = norm.backward(d_act_in) if norm is not None else d_act_in

            # Accumulate weight and bias gradients
            for d in range(fan_in):
                for c in range(fan_out_h):
                    val_w = 0.0
                    for i in range(batch_size):
                        val_w += h_in[i][d] * d_z[i][c]
                    self.grad_weights[l_idx][d][c] += val_w

            for c in range(fan_out_h):
                val_bh = 0.0
                for i in range(batch_size):
                    val_bh += d_z[i][c]
                self.grad_biases[l_idx][c] += val_bh

            # dH_prev = dZ @ W.T
            d_h_prev: list[list[float]] = []
            for i in range(batch_size):
                row_prev: list[float] = []
                for d in range(fan_in):
                    sum_h = 0.0
                    for c in range(fan_out_h):
                        sum_h += d_z[i][c] * w[d][c]
                    row_prev.append(sum_h)
                d_h_prev.append(row_prev)

            d_h_current = d_h_prev

    def zero_grad(self) -> None:
        """Reset parameter gradients."""
        self.grad_weights: list[list[list[float]]] = [
            [[0.0 for _ in range(len(w[0]))] for _ in range(len(w))]
            for w in self.layer_weights
        ]
        self.grad_biases: list[list[float]] = [
            [0.0 for _ in range(len(b))] for b in self.layer_biases
        ]
        for norm in self.norm_layers:
            if norm is not None:
                norm.zero_grad()

    def get_parameters(self) -> dict[str, Any]:
        """Return trainable parameters mapping."""
        params: dict[str, Any] = {}
        for idx, (w, b) in enumerate(
            zip(self.layer_weights, self.layer_biases, strict=True)
        ):
            is_out = idx == (self.num_layers - 1)
            suffix = "out" if is_out else str(idx)
            params[f"weights_{suffix}"] = copy.deepcopy(w)
            params[f"bias_{suffix}"] = list(b)

        for idx, norm in enumerate(self.norm_layers):
            if norm is not None:
                for k, v in norm.get_parameters().items():
                    params[f"norm_{idx}_{k}"] = copy.deepcopy(v)
        return params

    def set_parameters(self, params: dict[str, Any]) -> None:
        """Load trainable parameters."""
        for idx in range(self.num_layers):
            is_out = idx == (self.num_layers - 1)
            suffix = "out" if is_out else str(idx)
            w_key = f"weights_{suffix}"
            b_key = f"bias_{suffix}"
            if w_key in params:
                self.layer_weights[idx] = copy.deepcopy(params[w_key])
            if b_key in params:
                self.layer_biases[idx] = list(params[b_key])

        for idx, norm in enumerate(self.norm_layers):
            if norm is not None:
                norm_dict = {}
                gamma_key = f"norm_{idx}_gamma"
                beta_key = f"norm_{idx}_beta"
                if gamma_key in params:
                    norm_dict["gamma"] = copy.deepcopy(params[gamma_key])
                if beta_key in params:
                    norm_dict["beta"] = copy.deepcopy(params[beta_key])
                norm.set_parameters(norm_dict)

    def get_gradients(self) -> dict[str, Any]:
        """Return computed gradients mapping."""
        grads: dict[str, Any] = {}
        for idx, (w, b) in enumerate(
            zip(self.grad_weights, self.grad_biases, strict=True)
        ):
            is_out = idx == (self.num_layers - 1)
            suffix = "out" if is_out else str(idx)
            grads[f"grad_weights_{suffix}"] = copy.deepcopy(w)
            grads[f"grad_bias_{suffix}"] = list(b)

        for idx, norm in enumerate(self.norm_layers):
            if norm is not None:
                for k, v in norm.get_gradients().items():
                    grads[f"grad_norm_{idx}_{k.replace('grad_', '')}"] = copy.deepcopy(
                        v
                    )
        return grads

    def get_state(self) -> dict[str, Any]:
        """Return non-trainable running state."""
        state: dict[str, Any] = {}
        for idx, norm in enumerate(self.norm_layers):
            if norm is not None:
                for k, v in norm.get_state().items():
                    state[f"norm_{idx}_{k}"] = copy.deepcopy(v)
        return state

    def set_state(self, state: dict[str, Any]) -> None:
        """Load non-trainable running state."""
        for idx, norm in enumerate(self.norm_layers):
            if norm is not None:
                n_state = {}
                mean_key = f"norm_{idx}_running_mean"
                var_key = f"norm_{idx}_running_var"
                batch_key = f"norm_{idx}_num_batches_tracked"
                if mean_key in state:
                    n_state["running_mean"] = copy.deepcopy(state[mean_key])
                if var_key in state:
                    n_state["running_var"] = copy.deepcopy(state[var_key])
                if batch_key in state:
                    n_state["num_batches_tracked"] = state[batch_key]
                norm.set_state(n_state)
