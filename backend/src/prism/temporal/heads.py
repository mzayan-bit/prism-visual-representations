"""Downstream temporal classification heads and unified temporal models."""

from __future__ import annotations

import math
import random
from typing import Any

from prism.temporal.adapter import TemporalFrameEncoder
from prism.temporal.aggregators import BaseTemporalAggregator


class TemporalClassificationHead:
    """Linear classifier projecting sequence representations [N, D] to logits [N, K]."""

    def __init__(self, input_dim: int, num_classes: int, seed: int = 42) -> None:
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.seed = seed

        rng = random.Random(seed)
        bound = 1.0 / math.sqrt(max(1, input_dim))

        self.weights: list[list[float]] = [
            [rng.uniform(-bound, bound) for _ in range(input_dim)]
            for _ in range(num_classes)
        ]
        self.bias: list[float] = [0.0 for _ in range(num_classes)]

        self.grad_weights: list[list[float]] = [
            [0.0 for _ in range(input_dim)] for _ in range(num_classes)
        ]
        self.grad_bias: list[float] = [0.0 for _ in range(num_classes)]

        self._cached_inputs: list[list[float]] = []

    def get_parameters(self) -> dict[str, Any]:
        return {
            "weights": [[float(v) for v in row] for row in self.weights],
            "bias": [float(b) for b in self.bias],
        }

    def set_parameters(self, params: dict[str, Any]) -> None:
        if "weights" in params:
            self.weights = [[float(v) for v in row] for row in params["weights"]]
        if "bias" in params:
            self.bias = [float(b) for b in params["bias"]]

    def get_gradients(self) -> dict[str, Any]:
        return {
            "weights": [[float(v) for v in row] for row in self.grad_weights],
            "bias": [float(b) for b in self.grad_bias],
        }

    def zero_grad(self) -> None:
        self.grad_weights = [
            [0.0 for _ in range(self.input_dim)] for _ in range(self.num_classes)
        ]
        self.grad_bias = [0.0 for _ in range(self.num_classes)]

    def forward(self, inputs: list[list[float]]) -> list[list[float]]:
        """Compute linear projection: logits = inputs @ W^T + bias."""
        self._cached_inputs = inputs
        n_samples = len(inputs)
        logits: list[list[float]] = []

        for i in range(n_samples):
            x = inputs[i]
            row_logits = [
                sum(self.weights[c][d] * x[d] for d in range(self.input_dim))
                + self.bias[c]
                for c in range(self.num_classes)
            ]
            logits.append(row_logits)

        return logits

    def compute_loss_and_grad(
        self,
        logits: list[list[float]],  # N x K
        targets: list[int],  # N
    ) -> tuple[float, list[list[float]]]:
        """Compute numerically stable Cross-Entropy loss and logits gradient."""
        n_samples = len(logits)
        if n_samples == 0:
            return 0.0, []

        total_loss = 0.0
        d_logits: list[list[float]] = []

        for i in range(n_samples):
            row = logits[i]
            target = targets[i]

            max_l = max(row)
            exp_row = [math.exp(val - max_l) for val in row]
            sum_exp = sum(exp_row)
            log_sum_exp = max_l + math.log(sum_exp)

            loss_i = log_sum_exp - row[target]
            total_loss += loss_i

            probs = [e / sum_exp for e in exp_row]
            d_row = [float(p) for p in probs]
            d_row[target] -= 1.0
            d_logits.append([val / n_samples for val in d_row])

        mean_loss = total_loss / n_samples
        return mean_loss, d_logits

    def backward(self, d_logits: list[list[float]]) -> list[list[float]]:
        """Backpropagate upstream gradient through classifier to sequence inputs."""
        n_samples = len(d_logits)
        inputs = self._cached_inputs

        d_inputs: list[list[float]] = []

        for i in range(n_samples):
            x = inputs[i]
            g = d_logits[i]

            for c in range(self.num_classes):
                self.grad_bias[c] += g[c]
                for d in range(self.input_dim):
                    self.grad_weights[c][d] += g[c] * x[d]

            row_dx = [
                sum(g[c] * self.weights[c][d] for c in range(self.num_classes))
                for d in range(self.input_dim)
            ]
            d_inputs.append(row_dx)

        return d_inputs


class TemporalRepresentationModel:
    """Unified temporal model: frame encoder + temporal aggregator + classifier."""

    def __init__(
        self,
        frame_encoder: TemporalFrameEncoder,
        aggregator: BaseTemporalAggregator,
        classifier: TemporalClassificationHead,
        train_encoder: bool = False,
    ) -> None:
        self.frame_encoder = frame_encoder
        self.aggregator = aggregator
        self.classifier = classifier
        self.train_encoder = train_encoder
        self._is_training: bool = True

    @property
    def is_training(self) -> bool:
        return self._is_training

    def train(self, mode: bool = True) -> TemporalRepresentationModel:
        self._is_training = mode
        if self.train_encoder:
            self.frame_encoder.train(mode)
        else:
            self.frame_encoder.eval()
        return self

    def eval(self) -> TemporalRepresentationModel:
        self._is_training = False
        self.frame_encoder.eval()
        return self

    def forward(
        self,
        videos: list[list[list[list[list[float]]]]],  # N x T x C x H x W
        mask: list[list[float]] | None = None,
    ) -> list[list[float]]:
        """Compute end-to-end forward pass: videos -> frames -> sequence -> logits."""
        frame_feats = self.frame_encoder.forward(videos)
        seq_feats = self.aggregator.forward(frame_feats, mask=mask)
        logits = self.classifier.forward(seq_feats)
        return logits

    def extract_frame_representations(
        self,
        videos: list[list[list[list[list[float]]]]],
    ) -> list[list[list[float]]]:
        """Extract multi-frame representation tensor [N, T, D]."""
        return self.frame_encoder.forward(videos)

    def extract_sequence_representations(
        self,
        videos: list[list[list[list[list[float]]]]],
        mask: list[list[float]] | None = None,
    ) -> list[list[float]]:
        """Extract pooled/recurrent sequence representation vectors [N, D]."""
        frame_feats = self.frame_encoder.forward(videos)
        return self.aggregator.forward(frame_feats, mask=mask)

    def backward(self, d_logits: list[list[float]]) -> None:
        """Propagate gradients backward through classifier and aggregator."""
        d_seq = self.classifier.backward(d_logits)
        d_frames = self.aggregator.backward(d_seq)

        if self.train_encoder and hasattr(self.frame_encoder.model, "backward"):
            flat_df: list[list[float]] = []
            for seq in d_frames:
                for frame_grad in seq:
                    flat_df.append(frame_grad)
            self.frame_encoder.model.backward(flat_df)

    def zero_grad(self) -> None:
        """Clear gradients in classifier, temporal aggregator, and image encoder."""
        self.classifier.zero_grad()
        self.aggregator.zero_grad()
        if self.train_encoder and hasattr(self.frame_encoder.model, "zero_grad"):
            self.frame_encoder.model.zero_grad()

    def get_parameters(self) -> dict[str, Any]:
        """Return parameters for temporal components (and encoder if trainable)."""
        params: dict[str, Any] = {
            "aggregator": self.aggregator.get_parameters(),
            "classifier": self.classifier.get_parameters(),
        }
        if self.train_encoder and hasattr(self.frame_encoder.model, "get_parameters"):
            params["encoder"] = self.frame_encoder.model.get_parameters()
        return params

    def set_parameters(self, params: dict[str, Any]) -> None:
        """Set parameters for temporal components and encoder."""
        if "aggregator" in params:
            self.aggregator.set_parameters(params["aggregator"])
        if "classifier" in params:
            self.classifier.set_parameters(params["classifier"])
        if (
            "encoder" in params
            and self.train_encoder
            and hasattr(self.frame_encoder.model, "set_parameters")
        ):
            self.frame_encoder.model.set_parameters(params["encoder"])

    def get_gradients(self) -> dict[str, Any]:
        """Return gradients for temporal components and encoder."""
        grads: dict[str, Any] = {
            "aggregator": self.aggregator.get_gradients(),
            "classifier": self.classifier.get_gradients(),
        }
        if self.train_encoder and hasattr(self.frame_encoder.model, "get_gradients"):
            grads["encoder"] = self.frame_encoder.model.get_gradients()
        return grads
