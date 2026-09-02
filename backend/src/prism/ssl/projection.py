"""SimCLR non-linear projection head and analytical L2 vector normalization."""

from __future__ import annotations

import copy
import math
import random
from typing import Any


def normalize_embeddings(
    embeddings: list[list[float]], eps: float = 1e-8
) -> tuple[list[list[float]], list[float]]:
    """Normalize each vector in batch to unit L2 norm: z_hat = z / (||z||_2 + eps).

    Returns:
        tuple of (normalized_embeddings, l2_norms)
    """
    normalized: list[list[float]] = []
    norms: list[float] = []

    for vec in embeddings:
        sq_sum = sum(x * x for x in vec)
        norm = math.sqrt(sq_sum)
        norms.append(norm)
        denom = norm + eps
        normalized.append([x / denom for x in vec])

    return normalized, norms


def backward_normalize_embeddings(
    d_normalized: list[list[float]],
    unnormalized: list[list[float]],
    norms: list[float],
    eps: float = 1e-8,
) -> list[list[float]]:
    """Analytical gradient of L2 vector normalization.

    d_z = (1 / (norm + eps)) * [ d_z_hat - (d_z_hat . z_hat) * z_hat ]
    """
    d_unnormalized: list[list[float]] = []

    for d_hat, z, norm in zip(d_normalized, unnormalized, norms, strict=True):
        denom = norm + eps
        z_hat = [x / denom for x in z]
        dot_product = sum(dh * zh for dh, zh in zip(d_hat, z_hat, strict=True))

        d_z = [
            (dh - dot_product * zh) / denom for dh, zh in zip(d_hat, z_hat, strict=True)
        ]
        d_unnormalized.append(d_z)

    return d_unnormalized


class SimCLRProjectionHead:
    """Non-linear MLP projection head mapping features to metric space.

    Architecture:
        h (in_dim) -> Linear -> ReLU -> Linear -> z (out_dim)
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        activation: str = "relu",
        seed: int = 42,
    ) -> None:
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.activation = activation.lower()
        self.seed = seed

        rng1 = random.Random(seed)
        std1 = math.sqrt(2.0 / float(in_dim))
        self.w1: list[list[float]] = [
            [rng1.gauss(0.0, std1) for _ in range(in_dim)] for _ in range(hidden_dim)
        ]
        self.b1: list[float] = [0.0 for _ in range(hidden_dim)]

        rng2 = random.Random(seed + 1)
        std2 = math.sqrt(2.0 / float(hidden_dim + out_dim))
        self.w2: list[list[float]] = [
            [rng2.gauss(0.0, std2) for _ in range(hidden_dim)] for _ in range(out_dim)
        ]
        self.b2: list[float] = [0.0 for _ in range(out_dim)]

        # Gradients
        self.grad_w1: list[list[float]] = [
            [0.0 for _ in range(self.in_dim)] for _ in range(self.hidden_dim)
        ]
        self.grad_b1: list[float] = [0.0 for _ in range(self.hidden_dim)]
        self.grad_w2: list[list[float]] = [
            [0.0 for _ in range(self.hidden_dim)] for _ in range(self.out_dim)
        ]
        self.grad_b2: list[float] = [0.0 for _ in range(self.out_dim)]

        # Forward caches
        self._cached_h: list[list[float]] = []
        self._cached_pre_act: list[list[float]] = []
        self._cached_post_act: list[list[float]] = []
        self._cached_z: list[list[float]] = []

    def zero_grad(self) -> None:
        """Reset parameter gradients to zero."""
        self.grad_w1 = [
            [0.0 for _ in range(self.in_dim)] for _ in range(self.hidden_dim)
        ]
        self.grad_b1 = [0.0 for _ in range(self.hidden_dim)]
        self.grad_w2 = [
            [0.0 for _ in range(self.hidden_dim)] for _ in range(self.out_dim)
        ]
        self.grad_b2 = [0.0 for _ in range(self.out_dim)]

    def forward(self, h_batch: list[list[float]]) -> list[list[float]]:
        """Forward pass through projection head."""
        self._cached_h = copy.deepcopy(h_batch)
        self._cached_pre_act = []
        self._cached_post_act = []
        self._cached_z = []

        batch_size = len(h_batch)

        for b_idx in range(batch_size):
            h = h_batch[b_idx]

            # Layer 1: z1 = W1 @ h + b1
            z1: list[float] = []
            for j in range(self.hidden_dim):
                val = self.b1[j]
                for i in range(self.in_dim):
                    val += self.w1[j][i] * h[i]
                z1.append(val)
            self._cached_pre_act.append(z1)

            # Activation: a1 = ReLU(z1) or GELU(z1)
            a1: list[float] = []
            for val in z1:
                if self.activation == "relu":
                    a1.append(val if val > 0.0 else 0.0)
                elif self.activation == "gelu":
                    a1.append(
                        0.5
                        * val
                        * (
                            1.0
                            + math.tanh(
                                math.sqrt(2.0 / math.pi) * (val + 0.044715 * (val**3))
                            )
                        )
                    )
                else:
                    a1.append(val)
            self._cached_post_act.append(a1)

            # Layer 2: z2 = W2 @ a1 + b2
            z2: list[float] = []
            for j in range(self.out_dim):
                val = self.b2[j]
                for i in range(self.hidden_dim):
                    val += self.w2[j][i] * a1[i]
                z2.append(val)
            self._cached_z.append(z2)

        return self._cached_z

    def backward(self, d_z_batch: list[list[float]]) -> list[list[float]]:
        """Analytical backpropagation through projection head.

        Computes gradients for W2, b2, W1, b1 and returns input gradient d_h_batch.
        """
        batch_size = len(d_z_batch)
        d_h_batch: list[list[float]] = []

        for b_idx in range(batch_size):
            h = self._cached_h[b_idx]
            z1 = self._cached_pre_act[b_idx]
            a1 = self._cached_post_act[b_idx]
            dz2 = d_z_batch[b_idx]

            # 1. Gradients for Layer 2 (W2, b2)
            for j in range(self.out_dim):
                self.grad_b2[j] += dz2[j]
                for i in range(self.hidden_dim):
                    self.grad_w2[j][i] += dz2[j] * a1[i]

            # 2. Backprop into a1: da1 = W2.T @ dz2
            da1: list[float] = [0.0 for _ in range(self.hidden_dim)]
            for i in range(self.hidden_dim):
                val = 0.0
                for j in range(self.out_dim):
                    val += self.w2[j][i] * dz2[j]
                da1[i] = val

            # 3. Backprop through activation: dz1 = da1 * d_act(z1)
            dz1: list[float] = []
            for i in range(self.hidden_dim):
                val = z1[i]
                if self.activation == "relu":
                    grad_act = 1.0 if val > 0.0 else 0.0
                elif self.activation == "gelu":
                    # Analytical derivative of GELU approximation
                    u = math.sqrt(2.0 / math.pi) * (val + 0.044715 * (val**3))
                    tanh_u = math.tanh(u)
                    dtanh = 1.0 - tanh_u * tanh_u
                    du = math.sqrt(2.0 / math.pi) * (1.0 + 3.0 * 0.044715 * (val**2))
                    grad_act = 0.5 * (1.0 + tanh_u) + 0.5 * val * dtanh * du
                else:
                    grad_act = 1.0
                dz1.append(da1[i] * grad_act)

            # 4. Gradients for Layer 1 (W1, b1)
            for j in range(self.hidden_dim):
                self.grad_b1[j] += dz1[j]
                for i in range(self.in_dim):
                    self.grad_w1[j][i] += dz1[j] * h[i]

            # 5. Backprop into h: dh = W1.T @ dz1
            dh: list[float] = [0.0 for _ in range(self.in_dim)]
            for i in range(self.in_dim):
                val = 0.0
                for j in range(self.hidden_dim):
                    val += self.w1[j][i] * dz1[j]
                dh[i] = val
            d_h_batch.append(dh)

        return d_h_batch

    def get_parameters(self, prefix: str = "projection") -> dict[str, Any]:
        """Return parameters dictionary."""
        return {
            f"{prefix}_w1": copy.deepcopy(self.w1),
            f"{prefix}_b1": list(self.b1),
            f"{prefix}_w2": copy.deepcopy(self.w2),
            f"{prefix}_b2": list(self.b2),
        }

    def set_parameters(
        self, params: dict[str, Any], prefix: str = "projection"
    ) -> None:
        """Set parameters from dictionary."""
        if f"{prefix}_w1" in params:
            self.w1 = copy.deepcopy(params[f"{prefix}_w1"])
        if f"{prefix}_b1" in params:
            self.b1 = list(params[f"{prefix}_b1"])
        if f"{prefix}_w2" in params:
            self.w2 = copy.deepcopy(params[f"{prefix}_w2"])
        if f"{prefix}_b2" in params:
            self.b2 = list(params[f"{prefix}_b2"])

    def get_gradients(self, prefix: str = "projection") -> dict[str, Any]:
        """Return gradients dictionary."""
        return {
            f"{prefix}_w1": copy.deepcopy(self.grad_w1),
            f"{prefix}_b1": list(self.grad_b1),
            f"{prefix}_w2": copy.deepcopy(self.grad_w2),
            f"{prefix}_b2": list(self.grad_b2),
        }
