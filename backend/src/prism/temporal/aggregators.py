"""Temporal aggregation strategies and vanilla SimpleRNN with exact BPTT."""

from __future__ import annotations

import math
import random
from abc import ABC, abstractmethod
from typing import Any

from prism.temporal.contracts import RNNDynamicsSummary, TemporalWeightSummary
from prism.temporal.enums import RNNAggregationMode, TemporalAggregationType


class BaseTemporalAggregator(ABC):
    """Abstract base contract for temporal sequence aggregation."""

    def __init__(self, agg_type: TemporalAggregationType) -> None:
        self.agg_type = agg_type

    @abstractmethod
    def forward(
        self,
        features: list[list[list[float]]],  # N x T x D
        mask: list[list[float]] | None = None,  # N x T
    ) -> list[list[float]]:  # N x D
        """Aggregate frame representation sequences into sequence representations."""
        ...

    @abstractmethod
    def backward(
        self,
        d_out: list[list[float]],  # N x D
    ) -> list[list[list[float]]]:  # N x T x D
        """Backpropagate upstream loss derivatives through aggregator to frames."""
        ...

    def get_parameters(self) -> dict[str, Any]:
        """Return trainable aggregator parameters."""
        return {}

    def set_parameters(self, params: dict[str, Any]) -> None:
        """Set trainable aggregator parameters."""
        _ = params

    def get_gradients(self) -> dict[str, Any]:
        """Return parameter gradients."""
        return {}

    def zero_grad(self) -> None:
        """Clear parameter gradients."""
        return None


class MeanTemporalPooling(BaseTemporalAggregator):
    """Computes feature-wise mean across valid sequence frames."""

    def __init__(self) -> None:
        super().__init__(TemporalAggregationType.MEAN_POOL)
        self._cached_t: int = 0
        self._cached_dim: int = 0
        self._cached_mask: list[list[float]] | None = None

    def forward(
        self,
        features: list[list[list[float]]],  # N x T x D
        mask: list[list[float]] | None = None,
    ) -> list[list[float]]:
        if not features or not features[0]:
            return []

        n_samples = len(features)
        t_steps = len(features[0])
        dim = len(features[0][0])

        self._cached_t = t_steps
        self._cached_dim = dim
        self._cached_mask = mask

        outputs: list[list[float]] = []
        for i in range(n_samples):
            sum_vec = [0.0] * dim
            valid_count = 0.0
            for t in range(t_steps):
                m_val = mask[i][t] if mask is not None else 1.0
                if m_val > 0.0:
                    valid_count += m_val
                    for d in range(dim):
                        sum_vec[d] += features[i][t][d] * m_val

            norm_factor = max(1.0, valid_count)
            outputs.append([v / norm_factor for v in sum_vec])

        return outputs

    def backward(
        self,
        d_out: list[list[float]],  # N x D
    ) -> list[list[list[float]]]:  # N x T x D
        n_samples = len(d_out)
        t_steps = self._cached_t
        dim = self._cached_dim
        mask = self._cached_mask

        d_features: list[list[list[float]]] = []
        for i in range(n_samples):
            valid_count = 0.0
            for t in range(t_steps):
                m_val = mask[i][t] if mask is not None else 1.0
                valid_count += m_val
            norm_factor = max(1.0, valid_count)

            seq_grad: list[list[float]] = []
            for t in range(t_steps):
                m_val = mask[i][t] if mask is not None else 1.0
                frame_grad = [(d_out[i][d] / norm_factor) * m_val for d in range(dim)]
                seq_grad.append(frame_grad)
            d_features.append(seq_grad)

        return d_features


class MaxTemporalPooling(BaseTemporalAggregator):
    """Computes feature-wise maximum across time with deterministic argmax routing."""

    def __init__(self) -> None:
        super().__init__(TemporalAggregationType.MAX_POOL)
        self._cached_argmax: list[list[int]] = []
        self._cached_t: int = 0
        self._cached_dim: int = 0

    def forward(
        self,
        features: list[list[list[float]]],  # N x T x D
        mask: list[list[float]] | None = None,
    ) -> list[list[float]]:
        if not features or not features[0]:
            return []

        n_samples = len(features)
        t_steps = len(features[0])
        dim = len(features[0][0])

        self._cached_t = t_steps
        self._cached_dim = dim
        self._cached_argmax = []

        outputs: list[list[float]] = []
        for i in range(n_samples):
            max_vec = [-float("inf")] * dim
            argmax_vec = [0] * dim
            for d in range(dim):
                for t in range(t_steps):
                    m_val = mask[i][t] if mask is not None else 1.0
                    if m_val > 0.0:
                        val = features[i][t][d]
                        if val > max_vec[d]:
                            max_vec[d] = val
                            argmax_vec[d] = t
                if max_vec[d] == -float("inf"):
                    max_vec[d] = 0.0
                    argmax_vec[d] = 0
            outputs.append(max_vec)
            self._cached_argmax.append(argmax_vec)

        return outputs

    def backward(
        self,
        d_out: list[list[float]],  # N x D
    ) -> list[list[list[float]]]:  # N x T x D
        n_samples = len(d_out)
        t_steps = self._cached_t
        dim = self._cached_dim

        d_features: list[list[list[float]]] = []
        for i in range(n_samples):
            seq_grad = [[0.0 for _ in range(dim)] for _ in range(t_steps)]
            for d in range(dim):
                best_t = self._cached_argmax[i][d]
                seq_grad[best_t][d] += d_out[i][d]
            d_features.append(seq_grad)

        return d_features


class LastFramePooling(BaseTemporalAggregator):
    """Selects the final valid frame representation from each sequence."""

    def __init__(self) -> None:
        super().__init__(TemporalAggregationType.LAST_FRAME)
        self._cached_last_idx: list[int] = []
        self._cached_t: int = 0
        self._cached_dim: int = 0

    def forward(
        self,
        features: list[list[list[float]]],  # N x T x D
        mask: list[list[float]] | None = None,
    ) -> list[list[float]]:
        if not features or not features[0]:
            return []

        n_samples = len(features)
        t_steps = len(features[0])
        dim = len(features[0][0])

        self._cached_t = t_steps
        self._cached_dim = dim
        self._cached_last_idx = []

        outputs: list[list[float]] = []
        for i in range(n_samples):
            last_valid = t_steps - 1
            if mask is not None:
                for t in reversed(range(t_steps)):
                    if mask[i][t] > 0.0:
                        last_valid = t
                        break
            self._cached_last_idx.append(last_valid)
            outputs.append([features[i][last_valid][d] for d in range(dim)])

        return outputs

    def backward(
        self,
        d_out: list[list[float]],  # N x D
    ) -> list[list[list[float]]]:  # N x T x D
        n_samples = len(d_out)
        t_steps = self._cached_t
        dim = self._cached_dim

        d_features: list[list[list[float]]] = []
        for i in range(n_samples):
            last_valid = self._cached_last_idx[i]
            seq_grad = [[0.0 for _ in range(dim)] for _ in range(t_steps)]
            for d in range(dim):
                seq_grad[last_valid][d] = d_out[i][d]
            d_features.append(seq_grad)

        return d_features


class LearnedTemporalPooling(BaseTemporalAggregator):
    """Learned attention temporal pooling with analytical gradients."""

    def __init__(self, input_dim: int, seed: int = 42) -> None:
        super().__init__(TemporalAggregationType.LEARNED_TEMPORAL_POOLING)
        self.input_dim = input_dim
        self.seed = seed

        rng = random.Random(seed)
        bound = 1.0 / math.sqrt(max(1, input_dim))
        self.weights: list[float] = [
            rng.uniform(-bound, bound) for _ in range(input_dim)
        ]
        self.bias: float = 0.0

        self.grad_weights: list[float] = [0.0] * input_dim
        self.grad_bias: float = 0.0

        self._cached_features: list[list[list[float]]] = []
        self._cached_alphas: list[list[float]] = []
        self._cached_scores: list[list[float]] = []
        self._cached_mask: list[list[float]] | None = None

    def get_parameters(self) -> dict[str, Any]:
        return {
            "weights": [float(w) for w in self.weights],
            "bias": float(self.bias),
        }

    def set_parameters(self, params: dict[str, Any]) -> None:
        if "weights" in params:
            self.weights = [float(w) for w in params["weights"]]
        if "bias" in params:
            self.bias = float(params["bias"])

    def get_gradients(self) -> dict[str, Any]:
        return {
            "weights": [float(g) for g in self.grad_weights],
            "bias": float(self.grad_bias),
        }

    def zero_grad(self) -> None:
        self.grad_weights = [0.0] * self.input_dim
        self.grad_bias = 0.0

    def forward(
        self,
        features: list[list[list[float]]],  # N x T x D
        mask: list[list[float]] | None = None,
    ) -> list[list[float]]:
        if not features or not features[0]:
            return []

        n_samples = len(features)
        t_steps = len(features[0])
        dim = len(features[0][0])

        self._cached_features = features
        self._cached_mask = mask
        self._cached_alphas = []
        self._cached_scores = []

        outputs: list[list[float]] = []
        for i in range(n_samples):
            scores: list[float] = []
            for t in range(t_steps):
                score = self.bias
                for d in range(dim):
                    score += self.weights[d] * features[i][t][d]
                if mask is not None and mask[i][t] <= 0.0:
                    score = -1e9
                scores.append(score)
            self._cached_scores.append(scores)

            max_s = max(scores)
            exp_scores = [math.exp(s - max_s) if s > -1e8 else 0.0 for s in scores]
            sum_exp = max(1e-12, sum(exp_scores))
            alphas = [e / sum_exp for e in exp_scores]
            self._cached_alphas.append(alphas)

            agg_vec = [0.0] * dim
            for t in range(t_steps):
                a_t = alphas[t]
                for d in range(dim):
                    agg_vec[d] += a_t * features[i][t][d]
            outputs.append(agg_vec)

        return outputs

    def backward(
        self,
        d_out: list[list[float]],  # N x D
    ) -> list[list[list[float]]]:  # N x T x D
        n_samples = len(d_out)
        t_steps = len(self._cached_features[0])
        dim = len(self._cached_features[0][0])

        d_features: list[list[list[float]]] = []

        for i in range(n_samples):
            alphas = self._cached_alphas[i]
            features_i = self._cached_features[i]
            d_out_i = d_out[i]

            g_alpha: list[float] = [0.0] * t_steps
            for t in range(t_steps):
                for d in range(dim):
                    g_alpha[t] += d_out_i[d] * features_i[t][d]

            sum_alpha_g = sum(alphas[k] * g_alpha[k] for k in range(t_steps))
            g_s: list[float] = [
                alphas[t] * (g_alpha[t] - sum_alpha_g) for t in range(t_steps)
            ]

            for t in range(t_steps):
                self.grad_bias += g_s[t]
                for d in range(dim):
                    self.grad_weights[d] += g_s[t] * features_i[t][d]

            seq_grad: list[list[float]] = []
            for t in range(t_steps):
                frame_grad = [
                    alphas[t] * d_out_i[d] + g_s[t] * self.weights[d]
                    for d in range(dim)
                ]
                seq_grad.append(frame_grad)
            d_features.append(seq_grad)

        return d_features

    def get_weight_summary(self, sample_idx: int = 0) -> TemporalWeightSummary:
        """Compute summary metrics of learned temporal weights for a sample."""
        if not self._cached_alphas or sample_idx >= len(self._cached_alphas):
            return TemporalWeightSummary(
                weights=[],
                entropy=0.0,
                max_weight_timestep=0,
                max_weight=0.0,
            )

        alphas = self._cached_alphas[sample_idx]
        entropy = 0.0
        max_val = -1.0
        max_t = 0
        for t, a in enumerate(alphas):
            if a > max_val:
                max_val = a
                max_t = t
            if a > 1e-12:
                entropy -= a * math.log(a)

        return TemporalWeightSummary(
            weights=[float(a) for a in alphas],
            entropy=float(entropy),
            max_weight_timestep=int(max_t),
            max_weight=float(max_val),
        )


class SimpleRNN(BaseTemporalAggregator):
    """Vanilla RNN (h_t = tanh(W_x x_t + W_h h_{t-1} + b)) with exact BPTT."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        mode: RNNAggregationMode = RNNAggregationMode.LAST_HIDDEN,
        seed: int = 42,
    ) -> None:
        super().__init__(TemporalAggregationType.SIMPLE_RNN)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.mode = mode
        self.seed = seed

        rng = random.Random(seed)
        bound_x = 1.0 / math.sqrt(max(1, input_dim))
        bound_h = 1.0 / math.sqrt(max(1, hidden_dim))

        self.W_x: list[list[float]] = [
            [rng.uniform(-bound_x, bound_x) for _ in range(input_dim)]
            for _ in range(hidden_dim)
        ]
        self.W_h: list[list[float]] = [
            [rng.uniform(-bound_h, bound_h) for _ in range(hidden_dim)]
            for _ in range(hidden_dim)
        ]
        self.bias: list[float] = [0.0 for _ in range(hidden_dim)]

        self.grad_W_x: list[list[float]] = [
            [0.0 for _ in range(input_dim)] for _ in range(hidden_dim)
        ]
        self.grad_W_h: list[list[float]] = [
            [0.0 for _ in range(hidden_dim)] for _ in range(hidden_dim)
        ]
        self.grad_bias: list[float] = [0.0 for _ in range(hidden_dim)]

        self._cached_inputs: list[list[list[float]]] = []
        self._cached_hidden: list[list[list[float]]] = []
        self._cached_mask: list[list[float]] | None = None

    def get_parameters(self) -> dict[str, Any]:
        return {
            "W_x": [[float(v) for v in row] for row in self.W_x],
            "W_h": [[float(v) for v in row] for row in self.W_h],
            "bias": [float(b) for b in self.bias],
        }

    def set_parameters(self, params: dict[str, Any]) -> None:
        if "W_x" in params:
            self.W_x = [[float(v) for v in row] for row in params["W_x"]]
        if "W_h" in params:
            self.W_h = [[float(v) for v in row] for row in params["W_h"]]
        if "bias" in params:
            self.bias = [float(b) for b in params["bias"]]

    def get_gradients(self) -> dict[str, Any]:
        return {
            "W_x": [[float(v) for v in row] for row in self.grad_W_x],
            "W_h": [[float(v) for v in row] for row in self.grad_W_h],
            "bias": [float(b) for b in self.grad_bias],
        }

    def zero_grad(self) -> None:
        self.grad_W_x = [
            [0.0 for _ in range(self.input_dim)] for _ in range(self.hidden_dim)
        ]
        self.grad_W_h = [
            [0.0 for _ in range(self.hidden_dim)] for _ in range(self.hidden_dim)
        ]
        self.grad_bias = [0.0 for _ in range(self.hidden_dim)]

    def forward(
        self,
        features: list[list[list[float]]],  # N x T x D
        mask: list[list[float]] | None = None,
    ) -> list[list[float]]:  # N x hidden_dim
        if not features or not features[0]:
            return []

        n_samples = len(features)
        t_steps = len(features[0])

        self._cached_inputs = features
        self._cached_mask = mask
        self._cached_hidden = []

        outputs: list[list[float]] = []

        for i in range(n_samples):
            h_seq: list[list[float]] = []
            h_prev = [0.0] * self.hidden_dim

            for t in range(t_steps):
                x_t = features[i][t]
                m_val = mask[i][t] if mask is not None else 1.0

                if m_val > 0.0:
                    h_curr: list[float] = [0.0] * self.hidden_dim
                    for j in range(self.hidden_dim):
                        act = self.bias[j]
                        for d in range(self.input_dim):
                            act += self.W_x[j][d] * x_t[d]
                        for k in range(self.hidden_dim):
                            act += self.W_h[j][k] * h_prev[k]
                        h_curr[j] = math.tanh(act)
                    h_seq.append(h_curr)
                    h_prev = h_curr
                else:
                    h_seq.append(h_prev)

            self._cached_hidden.append(h_seq)

            if self.mode == RNNAggregationMode.LAST_HIDDEN:
                last_idx = t_steps - 1
                if mask is not None:
                    for t in reversed(range(t_steps)):
                        if mask[i][t] > 0.0:
                            last_idx = t
                            break
                outputs.append(list(h_seq[last_idx]))
            else:
                sum_h = [0.0] * self.hidden_dim
                valid_count = 0.0
                for t in range(t_steps):
                    m_val = mask[i][t] if mask is not None else 1.0
                    if m_val > 0.0:
                        valid_count += m_val
                        for j in range(self.hidden_dim):
                            sum_h[j] += h_seq[t][j]
                norm_f = max(1.0, valid_count)
                outputs.append([v / norm_f for v in sum_h])

        return outputs

    def backward(
        self,
        d_out: list[list[float]],  # N x hidden_dim
    ) -> list[list[list[float]]]:  # N x T x D
        n_samples = len(d_out)
        t_steps = len(self._cached_inputs[0])
        mask = self._cached_mask

        d_inputs: list[list[list[float]]] = []

        for i in range(n_samples):
            h_seq = self._cached_hidden[i]
            x_seq = self._cached_inputs[i]

            delta_h: list[list[float]] = [
                [0.0] * self.hidden_dim for _ in range(t_steps)
            ]

            if self.mode == RNNAggregationMode.LAST_HIDDEN:
                last_idx = t_steps - 1
                if mask is not None:
                    for t in reversed(range(t_steps)):
                        if mask[i][t] > 0.0:
                            last_idx = t
                            break
                for j in range(self.hidden_dim):
                    delta_h[last_idx][j] += d_out[i][j]
            else:
                valid_count = 0.0
                for t in range(t_steps):
                    m_val = mask[i][t] if mask is not None else 1.0
                    valid_count += m_val
                norm_f = max(1.0, valid_count)
                for t in range(t_steps):
                    m_val = mask[i][t] if mask is not None else 1.0
                    if m_val > 0.0:
                        for j in range(self.hidden_dim):
                            delta_h[t][j] += d_out[i][j] / norm_f

            seq_dx: list[list[float]] = [[0.0] * self.input_dim for _ in range(t_steps)]

            for t in reversed(range(t_steps)):
                m_val = mask[i][t] if mask is not None else 1.0
                if m_val <= 0.0:
                    continue

                h_t = h_seq[t]
                h_prev = h_seq[t - 1] if t > 0 else [0.0] * self.hidden_dim
                x_t = x_seq[t]

                delta_a: list[float] = [0.0] * self.hidden_dim
                for j in range(self.hidden_dim):
                    delta_a[j] = delta_h[t][j] * (1.0 - h_t[j] * h_t[j])

                for j in range(self.hidden_dim):
                    self.grad_bias[j] += delta_a[j]
                    for d in range(self.input_dim):
                        self.grad_W_x[j][d] += delta_a[j] * x_t[d]
                    for k in range(self.hidden_dim):
                        self.grad_W_h[j][k] += delta_a[j] * h_prev[k]

                for d in range(self.input_dim):
                    for j in range(self.hidden_dim):
                        seq_dx[t][d] += self.W_x[j][d] * delta_a[j]

                if t > 0:
                    for k in range(self.hidden_dim):
                        for j in range(self.hidden_dim):
                            delta_h[t - 1][k] += self.W_h[j][k] * delta_a[j]

            d_inputs.append(seq_dx)

        return d_inputs

    def get_dynamics_summary(self, sample_idx: int = 0) -> RNNDynamicsSummary:
        """Compute hidden state norms across time for recurrent dynamics inspection."""
        if not self._cached_hidden or sample_idx >= len(self._cached_hidden):
            return RNNDynamicsSummary(
                hidden_norms=[],
                mean_norm=0.0,
                max_norm=0.0,
                final_norm=0.0,
            )

        h_seq = self._cached_hidden[sample_idx]
        norms = [math.sqrt(sum(val * val for val in h_t)) for h_t in h_seq]
        mean_n = sum(norms) / max(1, len(norms)) if norms else 0.0
        max_n = max(norms) if norms else 0.0
        final_n = norms[-1] if norms else 0.0

        return RNNDynamicsSummary(
            hidden_norms=[float(n) for n in norms],
            mean_norm=float(mean_n),
            max_norm=float(max_n),
            final_norm=float(final_n),
        )
