"""Token Embedding Layer, Masked Mean Pooling, Projection Heads, and Text Encoder."""

from __future__ import annotations

import copy
import math
import random
from typing import Any

from prism.core.errors import ValidationError


class TokenEmbeddingTable:
    """Learnable token embedding matrix with analytical gradient accumulation."""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        seed: int = 42,
        pad_idx: int = 0,
    ) -> None:
        if vocab_size <= 0:
            raise ValidationError(f"vocab_size must be positive, got {vocab_size}")
        if embedding_dim <= 0:
            raise ValidationError(
                f"embedding_dim must be positive, got {embedding_dim}"
            )

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.pad_idx = pad_idx

        rng = random.Random(seed)
        std = 1.0 / math.sqrt(embedding_dim)

        # Initialize weights
        self.weights: list[list[float]] = [
            [rng.gauss(0.0, std) for _ in range(embedding_dim)]
            for _ in range(vocab_size)
        ]
        # Zero out PAD token embedding initially
        if 0 <= pad_idx < vocab_size:
            self.weights[pad_idx] = [0.0 for _ in range(embedding_dim)]

        self.grad_weights: list[list[float]] = [
            [0.0 for _ in range(embedding_dim)] for _ in range(vocab_size)
        ]
        self._cached_token_ids: list[list[int]] | None = None

    def zero_grad(self) -> None:
        """Reset gradient buffer to zero."""
        for v in range(self.vocab_size):
            for d in range(self.embedding_dim):
                self.grad_weights[v][d] = 0.0

    def forward(self, token_ids: list[list[int]]) -> list[list[list[float]]]:
        """Look up embeddings for a batch of token ID sequences.

        Args:
            token_ids: Shape (N, L)

        Returns:
            Embeddings of shape (N, L, D_text)
        """
        self._cached_token_ids = token_ids
        n_samples = len(token_ids)
        out: list[list[list[float]]] = []

        for i in range(n_samples):
            seq_embeds: list[list[float]] = []
            for tid in token_ids[i]:
                if 0 <= tid < self.vocab_size:
                    seq_embeds.append(list(self.weights[tid]))
                else:
                    # Fallback to zero vector if invalid
                    seq_embeds.append([0.0 for _ in range(self.embedding_dim)])
            out.append(seq_embeds)

        return out

    def backward(self, d_out: list[list[list[float]]]) -> None:
        """Accumulate gradients into token embedding table.

        Args:
            d_out: Upstream gradient of shape (N, L, D_text)
        """
        if self._cached_token_ids is None:
            raise RuntimeError("Cannot call backward before forward.")

        for i, seq_ids in enumerate(self._cached_token_ids):
            for l_idx, tid in enumerate(seq_ids):
                if 0 <= tid < self.vocab_size:
                    for d in range(self.embedding_dim):
                        self.grad_weights[tid][d] += d_out[i][l_idx][d]

    def get_parameters(self) -> dict[str, list[list[float]]]:
        """Return parameters dictionary."""
        return {"embedding_weights": copy.deepcopy(self.weights)}

    def set_parameters(self, params: dict[str, list[list[float]]]) -> None:
        """Update parameters."""
        if "embedding_weights" in params:
            self.weights = copy.deepcopy(params["embedding_weights"])

    def get_gradients(self) -> dict[str, list[list[float]]]:
        """Return computed gradients."""
        return {"embedding_weights": copy.deepcopy(self.grad_weights)}


class MaskedMeanPooling:
    """Computes sequence-level mean representation ignoring padding tokens."""

    def __init__(self) -> None:
        self._cached_masks: list[list[int]] | None = None
        self._cached_counts: list[int] | None = None

    def forward(
        self,
        embeddings: list[list[list[float]]],
        attention_masks: list[list[int]],
    ) -> list[list[float]]:
        """Compute masked mean representation.

        Args:
            embeddings: (N, L, D)
            attention_masks: (N, L) with 1 for active token, 0 for PAD

        Returns:
            Pooled representations of shape (N, D)
        """
        self._cached_masks = attention_masks
        n_samples = len(embeddings)
        dim = len(embeddings[0][0]) if n_samples > 0 and embeddings[0] else 0

        pooled: list[list[float]] = []
        counts: list[int] = []

        for i in range(n_samples):
            mask = attention_masks[i]
            valid_count = sum(mask)
            counts.append(max(1, valid_count))

            denom = float(max(1, valid_count))
            acc = [0.0 for _ in range(dim)]

            for l_idx, m_val in enumerate(mask):
                if m_val > 0 and l_idx < len(embeddings[i]):
                    tok_vec = embeddings[i][l_idx]
                    for d in range(dim):
                        acc[d] += tok_vec[d]

            pooled.append([val / denom for val in acc])

        self._cached_counts = counts
        return pooled

    def backward(self, d_pooled: list[list[float]]) -> list[list[list[float]]]:
        """Backpropagate upstream gradients to token embeddings.

        Args:
            d_pooled: Upstream gradient of shape (N, D)

        Returns:
            Gradient w.r.t token embeddings of shape (N, L, D)
        """
        if self._cached_masks is None or self._cached_counts is None:
            raise RuntimeError("Cannot call backward before forward.")

        n_samples = len(d_pooled)
        dim = len(d_pooled[0]) if n_samples > 0 else 0
        d_embeddings: list[list[list[float]]] = []

        for i in range(n_samples):
            mask = self._cached_masks[i]
            denom = float(self._cached_counts[i])
            seq_grad: list[list[float]] = []

            for m_val in mask:
                if m_val > 0:
                    scale = 1.0 / denom
                    seq_grad.append([d_pooled[i][d] * scale for d in range(dim)])
                else:
                    seq_grad.append([0.0 for _ in range(dim)])

            d_embeddings.append(seq_grad)

        return d_embeddings


class MultimodalProjectionHead:
    """Linear or MLP projection head mapping features to shared metric space."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: int | None = None,
        use_mlp: bool = False,
        seed: int = 42,
    ) -> None:
        if in_dim <= 0 or out_dim <= 0:
            raise ValidationError(
                f"Dimensions must be positive: in={in_dim}, out={out_dim}"
            )

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.use_mlp = use_mlp
        self.hidden_dim = hidden_dim or in_dim

        rng = random.Random(seed)

        if not self.use_mlp:
            # Linear projection: W (out_dim, in_dim), b (out_dim)
            std = 1.0 / math.sqrt(in_dim)
            self.w1 = [
                [rng.gauss(0.0, std) for _ in range(in_dim)] for _ in range(out_dim)
            ]
            self.b1 = [0.0 for _ in range(out_dim)]
            self.grad_w1 = [[0.0 for _ in range(in_dim)] for _ in range(out_dim)]
            self.grad_b1 = [0.0 for _ in range(out_dim)]
        else:
            # 2-layer MLP with ReLU
            std1 = 1.0 / math.sqrt(in_dim)
            self.w1 = [
                [rng.gauss(0.0, std1) for _ in range(in_dim)]
                for _ in range(self.hidden_dim)
            ]
            self.b1 = [0.0 for _ in range(self.hidden_dim)]
            self.grad_w1 = [
                [0.0 for _ in range(in_dim)] for _ in range(self.hidden_dim)
            ]
            self.grad_b1 = [0.0 for _ in range(self.hidden_dim)]

            std2 = 1.0 / math.sqrt(self.hidden_dim)
            self.w2 = [
                [rng.gauss(0.0, std2) for _ in range(self.hidden_dim)]
                for _ in range(out_dim)
            ]
            self.b2 = [0.0 for _ in range(out_dim)]
            self.grad_w2 = [
                [0.0 for _ in range(self.hidden_dim)] for _ in range(out_dim)
            ]
            self.grad_b2 = [0.0 for _ in range(out_dim)]

        self._cached_x: list[list[float]] | None = None
        self._cached_h1: list[list[float]] | None = None
        self._cached_a1: list[list[float]] | None = None

    def zero_grad(self) -> None:
        """Reset parameter gradients."""
        for r in range(len(self.grad_w1)):
            for c in range(len(self.grad_w1[0])):
                self.grad_w1[r][c] = 0.0
        for i in range(len(self.grad_b1)):
            self.grad_b1[i] = 0.0

        if self.use_mlp:
            for r in range(len(self.grad_w2)):
                for c in range(len(self.grad_w2[0])):
                    self.grad_w2[r][c] = 0.0
            for i in range(len(self.grad_b2)):
                self.grad_b2[i] = 0.0

    def forward(self, x: list[list[float]]) -> list[list[float]]:
        """Project batch of vectors (N, in_dim) to (N, out_dim)."""
        self._cached_x = x
        n_samples = len(x)

        if not self.use_mlp:
            out: list[list[float]] = []
            for i in range(n_samples):
                xi = x[i]
                zi = [
                    sum(self.w1[r][c] * xi[c] for c in range(self.in_dim)) + self.b1[r]
                    for r in range(self.out_dim)
                ]
                out.append(zi)
            return out
        else:
            # Layer 1: Linear -> ReLU
            h1: list[list[float]] = []
            a1: list[list[float]] = []
            for i in range(n_samples):
                xi = x[i]
                lin1 = [
                    sum(self.w1[r][c] * xi[c] for c in range(self.in_dim)) + self.b1[r]
                    for r in range(self.hidden_dim)
                ]
                act1 = [max(0.0, val) for val in lin1]
                h1.append(lin1)
                a1.append(act1)
            self._cached_h1 = h1
            self._cached_a1 = a1

            # Layer 2: Linear
            out2: list[list[float]] = []
            for i in range(n_samples):
                ai = a1[i]
                zi = [
                    sum(self.w2[r][c] * ai[c] for c in range(self.hidden_dim))
                    + self.b2[r]
                    for r in range(self.out_dim)
                ]
                out2.append(zi)
            return out2

    def backward(self, d_out: list[list[float]]) -> list[list[float]]:
        """Analytical backward returning d_x and accumulating parameter gradients."""
        if self._cached_x is None:
            raise RuntimeError("Cannot call backward before forward.")

        n_samples = len(d_out)

        if not self.use_mlp:
            d_x: list[list[float]] = [
                [0.0 for _ in range(self.in_dim)] for _ in range(n_samples)
            ]

            for i in range(n_samples):
                xi = self._cached_x[i]
                doi = d_out[i]
                for r in range(self.out_dim):
                    grad_r = doi[r]
                    self.grad_b1[r] += grad_r
                    for c in range(self.in_dim):
                        self.grad_w1[r][c] += grad_r * xi[c]
                        d_x[i][c] += grad_r * self.w1[r][c]

            return d_x
        else:
            assert self._cached_a1 is not None and self._cached_h1 is not None
            d_a1: list[list[float]] = [
                [0.0 for _ in range(self.hidden_dim)] for _ in range(n_samples)
            ]

            # Layer 2 gradients
            for i in range(n_samples):
                ai = self._cached_a1[i]
                doi = d_out[i]
                for r in range(self.out_dim):
                    grad_r = doi[r]
                    self.grad_b2[r] += grad_r
                    for c in range(self.hidden_dim):
                        self.grad_w2[r][c] += grad_r * ai[c]
                        d_a1[i][c] += grad_r * self.w2[r][c]

            # ReLU backward: d_h1 = d_a1 * 1(h1 > 0)
            d_h1: list[list[float]] = []
            for i in range(n_samples):
                dh_i = [
                    d_a1[i][c] if self._cached_h1[i][c] > 0.0 else 0.0
                    for c in range(self.hidden_dim)
                ]
                d_h1.append(dh_i)

            # Layer 1 gradients
            d_x = [[0.0 for _ in range(self.in_dim)] for _ in range(n_samples)]
            for i in range(n_samples):
                xi = self._cached_x[i]
                dhi = d_h1[i]
                for r in range(self.hidden_dim):
                    grad_r = dhi[r]
                    self.grad_b1[r] += grad_r
                    for c in range(self.in_dim):
                        self.grad_w1[r][c] += grad_r * xi[c]
                        d_x[i][c] += grad_r * self.w1[r][c]

            return d_x

    def get_parameters(self) -> dict[str, Any]:
        """Return parameters dictionary."""
        params: dict[str, Any] = {
            "w1": copy.deepcopy(self.w1),
            "b1": copy.deepcopy(self.b1),
        }
        if self.use_mlp:
            params["w2"] = copy.deepcopy(self.w2)
            params["b2"] = copy.deepcopy(self.b2)
        return params

    def set_parameters(self, params: dict[str, Any]) -> None:
        """Update parameters."""
        if "w1" in params:
            self.w1 = copy.deepcopy(params["w1"])
        if "b1" in params:
            self.b1 = copy.deepcopy(params["b1"])
        if self.use_mlp:
            if "w2" in params:
                self.w2 = copy.deepcopy(params["w2"])
            if "b2" in params:
                self.b2 = copy.deepcopy(params["b2"])

    def get_gradients(self) -> dict[str, Any]:
        """Return computed gradients."""
        grads: dict[str, Any] = {
            "w1": copy.deepcopy(self.grad_w1),
            "b1": copy.deepcopy(self.grad_b1),
        }
        if self.use_mlp:
            grads["w2"] = (copy.deepcopy(self.grad_w2),)
            grads["b2"] = (copy.deepcopy(self.grad_b2),)
        return grads


class VisualProjectionHead(MultimodalProjectionHead):
    """Projection head for visual encoder representation to shared multimodal space."""

    pass


class TextProjectionHead(MultimodalProjectionHead):
    """Projection head for text encoder representation to shared multimodal space."""

    pass


class TextEncoder:
    """Bag-of-words text encoder: TokenEmbeddings -> Pooling -> Projection."""

    def __init__(
        self,
        vocab_size: int,
        text_dim: int = 32,
        shared_dim: int = 16,
        use_mlp: bool = False,
        seed: int = 42,
    ) -> None:
        self.vocab_size = vocab_size
        self.text_dim = text_dim
        self.shared_dim = shared_dim

        self.embedding_table = TokenEmbeddingTable(
            vocab_size=vocab_size,
            embedding_dim=text_dim,
            seed=seed,
            pad_idx=0,
        )
        self.pooling = MaskedMeanPooling()
        self.projection = TextProjectionHead(
            in_dim=text_dim,
            out_dim=shared_dim,
            use_mlp=use_mlp,
            seed=seed + 1,
        )

    def zero_grad(self) -> None:
        """Zero all gradients."""
        self.embedding_table.zero_grad()
        self.projection.zero_grad()

    def forward(
        self,
        token_ids: list[list[int]],
        attention_masks: list[list[int]],
    ) -> tuple[list[list[float]], list[list[float]]]:
        """Encode token ID sequences into pooled and projected shared embeddings.

        Returns:
            tuple of (shared_embeddings [N x D_shared], pooled_embeddings [N x D_text])
        """
        # 1. Lookup embeddings: (N, L, D_text)
        embeds = self.embedding_table.forward(token_ids)

        # 2. Masked Mean Pooling: (N, D_text)
        pooled = self.pooling.forward(embeds, attention_masks)

        # 3. Project to shared space: (N, D_shared)
        shared = self.projection.forward(pooled)

        return shared, pooled

    def backward(self, d_shared: list[list[float]]) -> None:
        """Backpropagate gradient through projection, pooling, and token embeddings."""
        # 1. Projection backward -> d_pooled (N, D_text)
        d_pooled = self.projection.backward(d_shared)

        # 2. Pooling backward -> d_embeds (N, L, D_text)
        d_embeds = self.pooling.backward(d_pooled)

        # 3. Embedding table backward (accumulates gradients)
        self.embedding_table.backward(d_embeds)

    def get_parameters(self) -> dict[str, Any]:
        """Return parameters dictionary."""
        params = {}
        params.update(self.embedding_table.get_parameters())
        for k, v in self.projection.get_parameters().items():
            params[f"proj_{k}"] = v
        return params

    def set_parameters(self, params: dict[str, Any]) -> None:
        """Update parameters."""
        emb_params = {k: v for k, v in params.items() if k.startswith("embedding_")}
        self.embedding_table.set_parameters(emb_params)

        proj_params = {
            k.replace("proj_", ""): v
            for k, v in params.items()
            if k.startswith("proj_")
        }
        if proj_params:
            self.projection.set_parameters(proj_params)

    def get_gradients(self) -> dict[str, Any]:
        """Return computed gradients."""
        grads = {}
        grads.update(self.embedding_table.get_gradients())
        for k, v in self.projection.get_gradients().items():
            grads[f"proj_{k}"] = v
        return grads
