"""Learnable mask token representation, parameter discovery, and gradient routing."""

from __future__ import annotations

import copy
import random
from typing import Any

from prism.core.errors import ValidationError


class LearnableMaskToken:
    """Learnable embedding vector replacing masked patch tokens in ViT architectures.

    Shape: 1 x D_model.
    Supports deterministic initialization, gradient accumulation, parameter discovery,
    and state restoration.
    """

    def __init__(
        self,
        embed_dim: int,
        std: float = 0.02,
        seed: int = 42,
    ) -> None:
        if embed_dim <= 0:
            raise ValidationError(f"embed_dim must be positive, got {embed_dim}.")

        self.embed_dim = embed_dim
        self.std = std
        self.seed = seed

        rng = random.Random(seed)
        self.token: list[float] = [rng.gauss(0.0, std) for _ in range(embed_dim)]
        self.grad_token: list[float] = [0.0] * embed_dim

    def zero_grad(self) -> None:
        """Reset parameter gradient accumulator to zeros."""
        self.grad_token = [0.0] * self.embed_dim

    def get_parameters(self) -> dict[str, Any]:
        """Expose trainable parameters for optimizer integration."""
        return {"mask_token": copy.deepcopy(self.token)}

    def set_parameters(self, params: dict[str, Any]) -> None:
        """Update trainable parameters from dictionary."""
        if "mask_token" in params:
            val = params["mask_token"]
            if not isinstance(val, list) or len(val) != self.embed_dim:
                raise ValidationError(
                    f"mask_token shape mismatch: expected ({self.embed_dim},), "
                    f"got {len(val) if isinstance(val, list) else type(val)}."
                )
            self.token = copy.deepcopy(val)

    def get_gradients(self) -> dict[str, Any]:
        """Expose computed parameter gradients."""
        return {"mask_token": copy.deepcopy(self.grad_token)}

    def replace_masked_patches(
        self,
        patch_tokens: list[list[float]],
        masked_indices: list[int],
    ) -> list[list[float]]:
        """Return a new token sequence with masked indices replaced by mask token.

        Parameters
        ----------
        patch_tokens : list[list[float]]
            Sequence of embedded patch tokens [T x D].
        masked_indices : list[int]
            List of patch indices to replace with mask token.

        Returns
        -------
        list[list[float]]
            New sequence with masked patch tokens substituted.
        """
        masked_set = set(masked_indices)
        output_tokens: list[list[float]] = []
        for i, tok in enumerate(patch_tokens):
            if i in masked_set:
                output_tokens.append(list(self.token))
            else:
                output_tokens.append(list(tok))
        return output_tokens

    def backward_masked_tokens(
        self,
        d_tokens: list[list[float]],
        masked_indices: list[int],
    ) -> list[list[float]]:
        """Route token gradients and accumulate gradients for the mask token.

        Gradients on masked positions flow into `grad_token`, while upstream gradients
        for the original patch tokens at masked positions are zeroed out (since they
        were masked out and received no upstream signal).

        Parameters
        ----------
        d_tokens : list[list[float]]
            Upstream gradients w.r.t token sequence [T x D].
        masked_indices : list[int]
            Indices of patches that were masked.

        Returns
        -------
        list[list[float]]
            Gradients w.r.t original unmasked input patch tokens [T x D].
        """
        masked_set = set(masked_indices)
        d_inputs: list[list[float]] = []

        for i, d_tok in enumerate(d_tokens):
            if i in masked_set:
                # Accumulate into mask token parameter gradient
                for d in range(self.embed_dim):
                    self.grad_token[d] += d_tok[d]
                # Original patch received no forward signal
                d_inputs.append([0.0] * self.embed_dim)
            else:
                # Visible patch propagates upstream gradient directly
                d_inputs.append(list(d_tok))

        return d_inputs
