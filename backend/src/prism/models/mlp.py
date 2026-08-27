"""Multi-Layer Perceptron (MLP) vision baseline model."""

import random
from collections.abc import Sequence
from typing import Any

from prism.core.errors import ValidationError
from prism.models.activations import BaseActivation, get_activation
from prism.models.base import BaseVisionModel
from prism.models.initialization import initialize_mlp_parameters
from prism.models.linear import _flatten_single_input
from prism.models.specifications import ModelSpecification


class MultiLayerPerceptron(BaseVisionModel):
    """Deep Multi-Layer Perceptron with non-linear activations and optional dropout."""

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

        # 2. Parameter Initialization
        self.layer_weights, self.layer_biases = initialize_mlp_parameters(
            in_features=self.in_features,
            hidden_dims=self.hidden_dims,
            num_classes=self.num_classes_val,
            seed=seed,
            activation=act_name,
        )

        self.num_layers = len(self.layer_weights)  # hidden layers + 1 output layer
        self.zero_grad()

        # Cached tensors for backpropagation
        self._cached_h: list[list[list[float]]] = []
        self._cached_z: list[list[list[float]]] = []
        self._cached_masks: list[list[list[float]] | None] = []

        # Internal step counter for reproducible dropout seeding
        self._step_counter: int = 0

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

        self._cached_h = [h_current]
        self._cached_z = []
        self._cached_masks = []

        # Forward through hidden layers
        for l_idx in range(self.num_layers - 1):
            w = self.layer_weights[l_idx]
            b = self.layer_biases[l_idx]

            z = self._matmul_add_bias(h_current, w, b)
            self._cached_z.append(z)

            a = self.activation.forward(z)

            # Apply dropout in training mode
            if self.is_training and self.dropout > 0.0:
                p_keep = 1.0 - self.dropout
                inv_p_keep = 1.0 / p_keep
                fan_out = len(b)

                # Deterministic dropout seed derived from model seed, layer, and step
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

        if self.is_training:
            self._step_counter += 1

        # Output layer (raw logits, no activation or dropout)
        w_out = self.layer_weights[-1]
        b_out = self.layer_biases[-1]
        logits = self._matmul_add_bias(h_current, w_out, b_out)
        self._cached_z.append(logits)

        return logits

    def extract_representations(
        self, inputs: Any, layer: str = "final_hidden"
    ) -> Any:
        """Extract intermediate activations without modifying model state."""
        norm_layer = layer.lower().strip()
        h_current = self._prepare_inputs(inputs)

        if norm_layer in ("input_flat", "input"):
            return h_current

        num_hidden = self.num_layers - 1
        for l_idx in range(num_hidden):
            w = self.layer_weights[l_idx]
            b = self.layer_biases[l_idx]
            z = self._matmul_add_bias(h_current, w, b)
            a = self.activation.forward(z)

            # In eval mode (or during representation extraction), dropout is identity
            h_current = a

            if norm_layer == f"hidden_{l_idx}":
                return h_current

        if norm_layer in ("final_hidden", f"hidden_{num_hidden - 1}"):
            return h_current

        if norm_layer == "logits":
            w_out = self.layer_weights[-1]
            b_out = self.layer_biases[-1]
            return self._matmul_add_bias(h_current, w_out, b_out)

        valid = ["input_flat", "final_hidden", "logits"] + [
            f"hidden_{i}" for i in range(num_hidden)
        ]
        raise ValidationError(
            f"Unknown layer '{layer}' for MultiLayerPerceptron. Supported: {valid}"
        )

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

        # grad_weights_out = H_final^T @ d_logits [fan_in, num_classes]
        for d in range(fan_in_out):
            for c in range(fan_out):
                val = 0.0
                for i in range(batch_size):
                    val += h_final[i][d] * d_logits[i][c]
                self.grad_layer_weights[-1][d][c] += val

        for c in range(fan_out):
            val_b = 0.0
            for i in range(batch_size):
                val_b += d_logits[i][c]
            self.grad_layer_biases[-1][c] += val_b

        # Compute dH_final = d_logits @ W_out^T [B, fan_in_out]
        dh_current: list[list[float]] = []
        for i in range(batch_size):
            row_dh: list[float] = []
            for d in range(fan_in_out):
                val_dh = 0.0
                for c in range(fan_out):
                    val_dh += d_logits[i][c] * w_out[d][c]
                row_dh.append(val_dh)
            dh_current.append(row_dh)

        # 2. Propagate backward through hidden layers
        num_hidden = self.num_layers - 1
        for l_idx in range(num_hidden - 1, -1, -1):
            w = self.layer_weights[l_idx]
            z = self._cached_z[l_idx]
            h_prev = self._cached_h[l_idx]
            mask = self._cached_masks[l_idx]

            fan_in = len(w)
            fan_out_l = len(w[0])

            # Apply dropout mask to gradient if active
            if mask is not None:
                da = [
                    [dh_current[i][j] * mask[i][j] for j in range(fan_out_l)]
                    for i in range(batch_size)
                ]
            else:
                da = dh_current

            # Backprop through activation: dz = activation.backward(z, da)
            dz = self.activation.backward(z, da)

            # Layer parameter gradients
            for d in range(fan_in):
                for c in range(fan_out_l):
                    grad_w = 0.0
                    for i in range(batch_size):
                        grad_w += h_prev[i][d] * dz[i][c]
                    self.grad_layer_weights[l_idx][d][c] += grad_w

            for c in range(fan_out_l):
                grad_b = 0.0
                for i in range(batch_size):
                    grad_b += dz[i][c]
                self.grad_layer_biases[l_idx][c] += grad_b

            # Upstream gradient for previous hidden layer: dh_prev = dz @ W^T
            if l_idx > 0:
                dh_prev_mat: list[list[float]] = []
                for i in range(batch_size):
                    row_dh_prev: list[float] = []
                    for d in range(fan_in):
                        val_prev = 0.0
                        for c in range(fan_out_l):
                            val_prev += dz[i][c] * w[d][c]
                        row_dh_prev.append(val_prev)
                    dh_prev_mat.append(row_dh_prev)
                dh_current = dh_prev_mat

    def zero_grad(self) -> None:
        """Clear all stored parameter gradients across all layers."""
        self.grad_layer_weights = [
            [[0.0 for _ in range(len(w))] for w in self.layer_weights[layer_i]]
            for layer_i in range(self.num_layers)
        ]
        self.grad_layer_biases = [
            [0.0 for _ in range(len(b))] for b in self.layer_biases
        ]

    def get_parameters(self) -> dict[str, Any]:
        """Return parameters dictionary mapping layer names to weight and bias lists."""
        params: dict[str, Any] = {}
        for l_idx in range(self.num_layers):
            tag = "out" if l_idx == self.num_layers - 1 else str(l_idx)
            params[f"weights_{tag}"] = [row[:] for row in self.layer_weights[l_idx]]
            params[f"bias_{tag}"] = self.layer_biases[l_idx][:]
        return params

    def set_parameters(self, params: dict[str, Any]) -> None:
        """Load parameters from dictionary mapping."""
        for l_idx in range(self.num_layers):
            tag = "out" if l_idx == self.num_layers - 1 else str(l_idx)
            w_key = f"weights_{tag}"
            b_key = f"bias_{tag}"
            if w_key in params:
                self.layer_weights[l_idx] = [row[:] for row in params[w_key]]
            if b_key in params:
                self.layer_biases[l_idx] = list(params[b_key])

    def get_gradients(self) -> dict[str, Any]:
        """Return computed gradients mapping layer names to gradient lists."""
        grads: dict[str, Any] = {}
        for l_idx in range(self.num_layers):
            tag = "out" if l_idx == self.num_layers - 1 else str(l_idx)
            grads[f"grad_weights_{tag}"] = [
                row[:] for row in self.grad_layer_weights[l_idx]
            ]
            grads[f"grad_bias_{tag}"] = self.grad_layer_biases[l_idx][:]
        return grads
