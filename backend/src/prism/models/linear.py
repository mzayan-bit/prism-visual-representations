"""Multiclass Linear Softmax Classifier baseline implementation."""

from collections.abc import Sequence
from typing import Any

from prism.core.errors import ValidationError
from prism.models.base import BaseVisionModel
from prism.models.initialization import initialize_linear_parameters
from prism.models.specifications import ModelSpecification


def _flatten_single_input(data: Any) -> list[float]:
    """Recursively flatten multidimensional nested structures into a 1D float list."""
    if data is None:
        return []
    if isinstance(data, (int, float)):
        return [float(data)]
    if isinstance(data, (list, tuple)):
        flattened: list[float] = []
        for item in data:
            flattened.extend(_flatten_single_input(item))
        return flattened
    # Fallback for duck-typed objects or iterables
    try:
        flattened = []
        for item in data:
            flattened.extend(_flatten_single_input(item))
        return flattened
    except Exception as exc:
        raise ValidationError(
            f"Unsupported input data type for flattening: {type(data)}"
        ) from exc


class LinearSoftmaxClassifier(BaseVisionModel):
    """Multiclass Linear Softmax Classifier: scores = xW + b."""

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

        # Calculate flattened feature dimensionality D
        if "in_features" in spec.hyperparameters:
            self.in_features: int = int(spec.hyperparameters["in_features"])
        else:
            self.in_features = 1
            for dim in spec.input_shape:
                self.in_features *= dim

        self.num_classes_val: int = spec.num_classes
        scale = spec.hyperparameters.get("init_scale", None)

        self.weights, self.bias = initialize_linear_parameters(
            in_features=self.in_features,
            num_classes=self.num_classes_val,
            seed=seed,
            strategy=spec.initialization,
            scale=scale,
        )

        self.zero_grad()
        self._last_x: list[list[float]] | None = None

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

    def forward(self, inputs: Any) -> list[list[float]]:
        """Compute forward pass producing raw linear logits [B, num_classes]."""
        x_flat = self._prepare_inputs(inputs)
        self._last_x = x_flat
        batch_size = len(x_flat)

        # Matrix multiply: Z = XW + b
        # X: [B, D], W: [D, C], b: [C] -> Z: [B, C]
        logits: list[list[float]] = []
        for i in range(batch_size):
            row_logits: list[float] = []
            x_row = x_flat[i]
            for c in range(self.num_classes_val):
                logit_val = self.bias[c]
                for d in range(self.in_features):
                    logit_val += x_row[d] * self.weights[d][c]
                row_logits.append(logit_val)
            logits.append(row_logits)

        return logits

    def extract_representations(self, inputs: Any, layer: str = "final_hidden") -> Any:
        """Extract representations at specified layer ('input_flat', 'final_hidden')."""
        valid_layers = ("input_flat", "input", "final_hidden", "logits")
        norm_layer = layer.lower().strip()
        if norm_layer not in valid_layers:
            raise ValidationError(
                f"Unknown layer '{layer}' for LinearSoftmaxClassifier. "
                f"Supported layers: {valid_layers}"
            )

        x_flat = self._prepare_inputs(inputs)
        if norm_layer in ("input_flat", "input", "final_hidden"):
            return x_flat

        # 'logits'
        return self.forward(inputs)

    def backward(self, d_logits: list[list[float]]) -> None:
        """Compute parameter gradients given loss derivatives d_logits [B, C]."""
        if self._last_x is None:
            raise ValidationError("Cannot perform backward pass before forward pass.")

        batch_size = len(self._last_x)
        if len(d_logits) != batch_size:
            raise ValidationError(
                f"d_logits size ({len(d_logits)}) != input size ({batch_size})."
            )

        # Gradients:
        # grad_weights = X^T @ d_logits [D, C]
        # grad_bias = sum(d_logits, axis=0) [C]
        for d in range(self.in_features):
            for c in range(self.num_classes_val):
                grad_val = 0.0
                for i in range(batch_size):
                    grad_val += self._last_x[i][d] * d_logits[i][c]
                self.grad_weights[d][c] += grad_val

        for c in range(self.num_classes_val):
            bias_grad = 0.0
            for i in range(batch_size):
                bias_grad += d_logits[i][c]
            self.grad_bias[c] += bias_grad

    def zero_grad(self) -> None:
        """Clear all stored parameter gradients."""
        self.grad_weights = [
            [0.0 for _ in range(self.num_classes_val)] for _ in range(self.in_features)
        ]
        self.grad_bias = [0.0 for _ in range(self.num_classes_val)]

    def get_parameters(self) -> dict[str, Any]:
        """Return model parameters."""
        return {
            "weights": [row[:] for row in self.weights],
            "bias": self.bias[:],
        }

    def set_parameters(self, params: dict[str, Any]) -> None:
        """Set model parameters."""
        if "weights" in params:
            self.weights = [row[:] for row in params["weights"]]
        if "bias" in params:
            self.bias = list(params["bias"])

    def get_gradients(self) -> dict[str, Any]:
        """Return computed parameter gradients."""
        return {
            "grad_weights": [row[:] for row in self.grad_weights],
            "grad_bias": self.grad_bias[:],
        }
