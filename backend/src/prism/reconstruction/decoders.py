"""Reconstruction decoders for patch-based and spatial representations."""

from __future__ import annotations

import copy
import math
import random
from typing import Any

from prism.core.errors import ValidationError


class PatchReconstructionDecoder:
    """Linear projection decoder mapping latent tokens to raw patch pixel space.

    Formula:
        p_hat_i = h_i * W_dec + b_dec
        where:
            h_i in R^(D_model)
            W_dec in R^(D_model x D_patch)
            b_dec in R^(D_patch)
            p_hat_i in R^(D_patch)
    """

    def __init__(
        self,
        in_features: int,
        patch_dim: int,
        bias: bool = True,
        seed: int = 42,
    ) -> None:
        if in_features <= 0:
            raise ValidationError(f"in_features must be positive, got {in_features}.")
        if patch_dim <= 0:
            raise ValidationError(f"patch_dim must be positive, got {patch_dim}.")

        self.in_features = in_features
        self.patch_dim = patch_dim
        self.use_bias = bias
        self.seed = seed

        rng = random.Random(seed)
        std = math.sqrt(2.0 / float(in_features + patch_dim))

        self.weights: list[list[float]] = [
            [rng.gauss(0.0, std) for _ in range(patch_dim)] for _ in range(in_features)
        ]
        self.bias_vec: list[float] = [0.0] * patch_dim if bias else []

        self.zero_grad()
        self._cached_inputs: list[list[list[float]]] | None = None

    def zero_grad(self) -> None:
        """Reset parameter gradient buffers to zero."""
        self.grad_weights: list[list[float]] = [
            [0.0] * self.patch_dim for _ in range(self.in_features)
        ]
        self.grad_bias: list[float] = [0.0] * self.patch_dim if self.use_bias else []

    def get_parameters(self) -> dict[str, Any]:
        """Return copy of trainable parameter matrices."""
        params: dict[str, Any] = {"weights": copy.deepcopy(self.weights)}
        if self.use_bias:
            params["bias"] = copy.deepcopy(self.bias_vec)
        return params

    def set_parameters(self, params: dict[str, Any]) -> None:
        """Set parameter matrices from dictionary."""
        if "weights" in params:
            w = params["weights"]
            if (
                not isinstance(w, list)
                or len(w) != self.in_features
                or (w and len(w[0]) != self.patch_dim)
            ):
                raise ValidationError(
                    f"Decoder weights mismatch: expected ({self.in_features}, "
                    f"{self.patch_dim})."
                )
            self.weights = copy.deepcopy(w)
        if self.use_bias and "bias" in params:
            b = params["bias"]
            if not isinstance(b, list) or len(b) != self.patch_dim:
                raise ValidationError(
                    f"Decoder bias mismatch: expected ({self.patch_dim},)."
                )
            self.bias_vec = copy.deepcopy(b)

    def get_gradients(self) -> dict[str, Any]:
        """Return copy of accumulated parameter gradients."""
        grads: dict[str, Any] = {"weights": copy.deepcopy(self.grad_weights)}
        if self.use_bias:
            grads["bias"] = copy.deepcopy(self.grad_bias)
        return grads

    def forward(
        self, token_sequence: list[list[list[float]]]
    ) -> list[list[list[float]]]:
        """Project tokens [N x T x D_model] to patch space [N x T x D_patch]."""
        self._cached_inputs = token_sequence
        n_samples = len(token_sequence)
        reconstructed: list[list[list[float]]] = []

        for n in range(n_samples):
            sample_tokens = token_sequence[n]
            n_tokens = len(sample_tokens)
            sample_out: list[list[float]] = []

            for t in range(n_tokens):
                tok = sample_tokens[t]
                row: list[float] = [0.0] * self.patch_dim
                for p in range(self.patch_dim):
                    dot = sum(
                        tok[d] * self.weights[d][p] for d in range(self.in_features)
                    )
                    if self.use_bias:
                        dot += self.bias_vec[p]
                    row[p] = dot
                sample_out.append(row)
            reconstructed.append(sample_out)

        return reconstructed

    def backward(
        self, d_reconstructed: list[list[list[float]]]
    ) -> list[list[list[float]]]:
        """Backpropagate upstream patch gradients into tokens and parameters.

        Parameters
        ----------
        d_reconstructed : list[list[list[float]]]
            Gradients w.r.t reconstructed patches [N x T x D_patch].

        Returns
        -------
        list[list[list[float]]]
            Gradients w.r.t input token sequence [N x T x D_model].
        """
        if self._cached_inputs is None:
            raise ValidationError("Cannot run backward before forward pass.")

        n_samples = len(d_reconstructed)
        d_tokens: list[list[list[float]]] = []

        for n in range(n_samples):
            cached_tokens = self._cached_inputs[n]
            d_patches = d_reconstructed[n]
            n_tokens = len(d_patches)
            sample_d_tokens: list[list[float]] = []

            for t in range(n_tokens):
                d_p = d_patches[t]
                tok = cached_tokens[t]

                # 1. Input token gradient: d_tok = d_p * W_dec^T
                d_tok: list[float] = [0.0] * self.in_features
                for d in range(self.in_features):
                    d_tok[d] = sum(
                        d_p[p] * self.weights[d][p] for p in range(self.patch_dim)
                    )
                sample_d_tokens.append(d_tok)

                # 2. Parameter gradients accumulation
                for d in range(self.in_features):
                    tok_d = tok[d]
                    for p in range(self.patch_dim):
                        self.grad_weights[d][p] += tok_d * d_p[p]

                if self.use_bias:
                    for p in range(self.patch_dim):
                        self.grad_bias[p] += d_p[p]

            d_tokens.append(sample_d_tokens)

        return d_tokens


class SpatialReconstructionDecoder:
    """Linear spatial decoder mapping latent vectors to reconstructed image tensors.

    Maps latent representation h in R^D to spatial image tensor x_hat in R^(C x H x W).
    """

    def __init__(
        self,
        in_features: int,
        output_shape: tuple[int, int, int],
        bias: bool = True,
        seed: int = 42,
    ) -> None:
        c, h, w = output_shape
        if in_features <= 0:
            raise ValidationError(f"in_features must be positive, got {in_features}.")
        if c <= 0 or h <= 0 or w <= 0:
            raise ValidationError(
                f"output_shape dimensions must be positive, got {output_shape}."
            )

        self.in_features = in_features
        self.output_shape = (c, h, w)
        self.total_pixels = c * h * w
        self.use_bias = bias

        rng = random.Random(seed)
        std = math.sqrt(2.0 / float(in_features + self.total_pixels))

        self.weights: list[list[float]] = [
            [rng.gauss(0.0, std) for _ in range(self.total_pixels)]
            for _ in range(in_features)
        ]
        self.bias_vec: list[float] = [0.0] * self.total_pixels if bias else []

        self.zero_grad()
        self._cached_latent: list[list[float]] | None = None

    def zero_grad(self) -> None:
        """Reset parameter gradients."""
        self.grad_weights: list[list[float]] = [
            [0.0] * self.total_pixels for _ in range(self.in_features)
        ]
        self.grad_bias: list[float] = [0.0] * self.total_pixels if self.use_bias else []

    def get_parameters(self) -> dict[str, Any]:
        """Return trainable parameters."""
        params: dict[str, Any] = {"weights": copy.deepcopy(self.weights)}
        if self.use_bias:
            params["bias"] = copy.deepcopy(self.bias_vec)
        return params

    def set_parameters(self, params: dict[str, Any]) -> None:
        """Update trainable parameters."""
        if "weights" in params:
            self.weights = copy.deepcopy(params["weights"])
        if self.use_bias and "bias" in params:
            self.bias_vec = copy.deepcopy(params["bias"])

    def get_gradients(self) -> dict[str, Any]:
        """Return computed parameter gradients."""
        grads: dict[str, Any] = {"weights": copy.deepcopy(self.grad_weights)}
        if self.use_bias:
            grads["bias"] = copy.deepcopy(self.grad_bias)
        return grads

    def forward(self, latents: list[list[float]]) -> list[list[list[list[float]]]]:
        """Project latent vectors [N x D] to image tensors [N x C x H x W]."""
        self._cached_latent = latents
        n_samples = len(latents)
        c, h, w = self.output_shape
        outputs: list[list[list[list[float]]]] = []

        for n in range(n_samples):
            vec = latents[n]
            flat_out: list[float] = [0.0] * self.total_pixels
            for p in range(self.total_pixels):
                dot = sum(vec[d] * self.weights[d][p] for d in range(self.in_features))
                if self.use_bias:
                    dot += self.bias_vec[p]
                flat_out[p] = dot

            # Reshape flat_out into [C x H x W]
            img: list[list[list[float]]] = []
            idx = 0
            for _ in range(c):
                ch_mat: list[list[float]] = []
                for _ in range(h):
                    row = flat_out[idx : idx + w]
                    ch_mat.append(row)
                    idx += w
                img.append(ch_mat)
            outputs.append(img)

        return outputs

    def backward(self, d_images: list[list[list[list[float]]]]) -> list[list[float]]:
        """Backpropagate image gradients into latent representations and parameters."""
        if self._cached_latent is None:
            raise ValidationError("Cannot run backward before forward pass.")

        n_samples = len(d_images)
        c, h, _w = self.output_shape
        d_latents: list[list[float]] = []

        for n in range(n_samples):
            img_grad = d_images[n]
            latent = self._cached_latent[n]

            # Flatten image gradient into 1D [total_pixels]
            flat_grad: list[float] = []
            for ch in range(c):
                for r in range(h):
                    flat_grad.extend(img_grad[ch][r])

            # Latent gradient
            d_vec: list[float] = [0.0] * self.in_features
            for d in range(self.in_features):
                d_vec[d] = sum(
                    flat_grad[p] * self.weights[d][p] for p in range(self.total_pixels)
                )
            d_latents.append(d_vec)

            # Accumulate parameter gradients
            for d in range(self.in_features):
                l_d = latent[d]
                for p in range(self.total_pixels):
                    self.grad_weights[d][p] += l_d * flat_grad[p]

            if self.use_bias:
                for p in range(self.total_pixels):
                    self.grad_bias[p] += flat_grad[p]

        return d_latents
