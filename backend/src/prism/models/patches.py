"""Explicit patch representation abstractions for Vision Transformers."""

import copy
import math
import random
from typing import Any

from prism.core.errors import ValidationError
from prism.models.spatial import ensure_4d_tensor, normalize_spatial_pair


def ensure_3d_tensor(data: Any) -> list[list[list[float]]]:
    """Validate and normalize nested list structure into 3D tensor [N, L, D]."""
    if data is None:
        raise ValidationError("Input tensor cannot be None.")

    if not isinstance(data, (list, tuple)):
        raise ValidationError("Expected 3D nested list [N, L, D].")

    if not data:
        raise ValidationError("Tensor batch cannot be empty.")

    first_elem = data[0]
    if not isinstance(first_elem, (list, tuple)):
        raise ValidationError("Expected 3D nested list [N, L, D], got 1D sequence.")

    if not first_elem:
        raise ValidationError("Sequence length L cannot be 0.")

    second_elem = first_elem[0]
    if not isinstance(second_elem, (list, tuple)):
        # 2D tensor [L, D] -> wrap to [1, L, D]
        sample_3d: list[list[float]] = []
        for row in data:
            if not isinstance(row, (list, tuple)):
                raise ValidationError("Inconsistent row structure in 2D tensor.")
            row_floats: list[float] = []
            for val in row:
                if (
                    not isinstance(val, (int, float))
                    or math.isnan(val)
                    or math.isinf(val)
                ):
                    raise ValidationError(
                        f"Non-finite or non-numeric scalar in tensor: {val}"
                    )
                row_floats.append(float(val))
            sample_3d.append(row_floats)
        return [sample_3d]

    # Already 3D: validate every scalar
    tensor_3d: list[list[list[float]]] = []
    expected_l = len(first_elem)
    expected_d = len(second_elem)

    for n_idx, sample in enumerate(data):
        if not isinstance(sample, (list, tuple)) or len(sample) != expected_l:
            actual_len = len(sample) if isinstance(sample, (list, tuple)) else "invalid"
            raise ValidationError(
                f"Sample at batch index {n_idx} has sequence length {actual_len}, "
                f"expected {expected_l}."
            )
        sample_rows: list[list[float]] = []
        for l_idx, row in enumerate(sample):
            if not isinstance(row, (list, tuple)) or len(row) != expected_d:
                actual_d = len(row) if isinstance(row, (list, tuple)) else "invalid"
                raise ValidationError(
                    f"Token at index ({n_idx}, {l_idx}) has dim {actual_d}, "
                    f"expected {expected_d}."
                )
            row_floats = []
            for val in row:
                if (
                    not isinstance(val, (int, float))
                    or math.isnan(val)
                    or math.isinf(val)
                ):
                    raise ValidationError(
                        f"Non-finite or non-numeric scalar in tensor: {val}"
                    )
                row_floats.append(float(val))
            sample_rows.append(row_floats)
        tensor_3d.append(sample_rows)

    return tensor_3d


class PatchExtractor:
    """Extract non-overlapping 2D image patches into flattened sequence tokens.

    Maps image tensor X in R^(N x C x H x W) to patch tokens P in R^(N x L x D_patch)
    using row-major spatial ordering (left-to-right across width, top-to-bottom across
    height).
    """

    def __init__(self, patch_size: int | tuple[int, int]) -> None:
        self.p_h, self.p_w = normalize_spatial_pair(patch_size, "patch_size")
        if self.p_h <= 0 or self.p_w <= 0:
            raise ValidationError(
                f"Patch dimensions must be positive, got ({self.p_h}, {self.p_w})."
            )
        self._cached_input_shape: tuple[int, int, int, int] | None = None

    def forward(self, inputs: Any) -> list[list[list[float]]]:
        """Extract patches from 4D image batch producing [N, L, D_patch]."""
        x_4d = ensure_4d_tensor(inputs)
        n_samples = len(x_4d)
        c_channels = len(x_4d[0])
        h_img = len(x_4d[0][0])
        w_img = len(x_4d[0][0][0])

        if c_channels <= 0:
            raise ValidationError(
                f"Input channel count must be positive, got {c_channels}."
            )
        if h_img <= 0 or w_img <= 0:
            raise ValidationError(
                f"Input spatial dimensions must be positive, got ({h_img}, {w_img})."
            )

        if h_img % self.p_h != 0:
            raise ValidationError(
                f"Image height ({h_img}) is not divisible by patch height ({self.p_h})."
            )
        if w_img % self.p_w != 0:
            raise ValidationError(
                f"Image width ({w_img}) is not divisible by patch width ({self.p_w})."
            )

        grid_h = h_img // self.p_h
        grid_w = w_img // self.p_w

        # Cache exact input shape for analytical backward reconstruction
        self._cached_input_shape = (n_samples, c_channels, h_img, w_img)

        extracted: list[list[list[float]]] = []
        for n in range(n_samples):
            sample_patches: list[list[float]] = []
            for r in range(grid_h):
                h_start = r * self.p_h
                for c in range(grid_w):
                    w_start = c * self.p_w
                    # Flatten patch in channel-major spatial order: C x P_h x P_w
                    patch_vec: list[float] = []
                    for ch in range(c_channels):
                        for ph in range(self.p_h):
                            for pw in range(self.p_w):
                                val = x_4d[n][ch][h_start + ph][w_start + pw]
                                if math.isnan(val) or math.isinf(val):
                                    raise ValidationError(
                                        f"Non-finite value at ({n}, {ch}, "
                                        f"{h_start + ph}, {w_start + pw}): {val}"
                                    )
                                patch_vec.append(val)
                    sample_patches.append(patch_vec)
            extracted.append(sample_patches)

        return extracted

    def backward(self, d_out: Any) -> list[list[list[list[float]]]]:
        """Reconstruct 4D image gradient [N, C, H, W] from patch gradients."""
        if self._cached_input_shape is None:
            raise ValidationError("Cannot run backward before forward pass.")

        d_3d = ensure_3d_tensor(d_out)
        n_samples, c_channels, h_img, w_img = self._cached_input_shape
        grid_h = h_img // self.p_h
        grid_w = w_img // self.p_w
        expected_l = grid_h * grid_w
        expected_d = c_channels * self.p_h * self.p_w

        if len(d_3d) != n_samples:
            raise ValidationError(
                f"Upstream batch size ({len(d_3d)}) does not match forward batch size "
                f"({n_samples})."
            )
        if len(d_3d[0]) != expected_l:
            raise ValidationError(
                f"Upstream patch count ({len(d_3d[0])}) does not match expected "
                f"({expected_l})."
            )
        if len(d_3d[0][0]) != expected_d:
            raise ValidationError(
                f"Upstream patch dimension ({len(d_3d[0][0])}) does not match "
                f"expected ({expected_d})."
            )

        # Initialize zero gradients for reconstructed 4D image
        dx: list[list[list[list[float]]]] = [
            [
                [[0.0 for _ in range(w_img)] for _ in range(h_img)]
                for _ in range(c_channels)
            ]
            for _ in range(n_samples)
        ]

        for n in range(n_samples):
            patch_idx = 0
            for r in range(grid_h):
                h_start = r * self.p_h
                for c in range(grid_w):
                    w_start = c * self.p_w
                    d_patch_vec = d_3d[n][patch_idx]
                    flat_idx = 0
                    for ch in range(c_channels):
                        for ph in range(self.p_h):
                            for pw in range(self.p_w):
                                dx[n][ch][h_start + ph][w_start + pw] = d_patch_vec[
                                    flat_idx
                                ]
                                flat_idx += 1
                    patch_idx += 1

        return dx


class PatchEmbedding:
    """Linear projection layer: E = P W_E + b_E mapping patch vectors to embeddings."""

    def __init__(
        self,
        in_features: int,
        embed_dim: int,
        bias: bool = True,
        seed: int = 42,
    ) -> None:
        if in_features <= 0:
            raise ValidationError(f"in_features must be positive, got {in_features}.")
        if embed_dim <= 0:
            raise ValidationError(f"embed_dim must be positive, got {embed_dim}.")

        self.in_features = in_features
        self.embed_dim = embed_dim
        self.use_bias = bias

        rng = random.Random(seed)
        std = math.sqrt(2.0 / float(in_features + embed_dim))
        self.weights: list[list[float]] = [
            [rng.gauss(0.0, std) for _ in range(embed_dim)] for _ in range(in_features)
        ]

        if bias:
            self.bias_weights: list[float] = [0.0 for _ in range(embed_dim)]
        else:
            self.bias_weights = []

        self.zero_grad()
        self._cached_x: list[list[list[float]]] | None = None

    def zero_grad(self) -> None:
        """Clear parameter gradient buffers."""
        self.grad_weights: list[list[float]] = [
            [0.0 for _ in range(self.embed_dim)] for _ in range(self.in_features)
        ]
        if self.use_bias:
            self.grad_bias: list[float] = [0.0 for _ in range(self.embed_dim)]
        else:
            self.grad_bias = []

    def get_parameters(self) -> dict[str, Any]:
        """Return trainable parameters mapping."""
        params: dict[str, Any] = {"weights": copy.deepcopy(self.weights)}
        if self.use_bias:
            params["bias"] = copy.deepcopy(self.bias_weights)
        return params

    def set_parameters(self, params: dict[str, Any]) -> None:
        """Load trainable parameter values."""
        if "weights" in params:
            w = params["weights"]
            if len(w) != self.in_features or len(w[0]) != self.embed_dim:
                raise ValidationError(
                    f"weights shape mismatch: expected ({self.in_features}, "
                    f"{self.embed_dim}), got ({len(w)}, {len(w[0]) if w else 0})."
                )
            self.weights = copy.deepcopy(w)
        if self.use_bias and "bias" in params:
            b = params["bias"]
            if len(b) != self.embed_dim:
                raise ValidationError(
                    f"bias mismatch: expected ({self.embed_dim},), got ({len(b)},)"
                )
            self.bias_weights = copy.deepcopy(b)

    def get_gradients(self) -> dict[str, Any]:
        """Return computed parameter gradients."""
        grads: dict[str, Any] = {"weights": copy.deepcopy(self.grad_weights)}
        if self.use_bias:
            grads["bias"] = copy.deepcopy(self.grad_bias)
        return grads

    def forward(self, inputs: Any) -> list[list[list[float]]]:
        """Project patch tokens [N, L, D_patch] into embedding space [N, L, D_embed]."""
        p_3d = ensure_3d_tensor(inputs)
        n_samples = len(p_3d)
        seq_len = len(p_3d[0])
        d_in = len(p_3d[0][0])

        if d_in != self.in_features:
            raise ValidationError(
                f"Input patch dimension ({d_in}) does not match in_features "
                f"({self.in_features})."
            )

        self._cached_x = p_3d

        out_3d: list[list[list[float]]] = []
        for n in range(n_samples):
            sample_out: list[list[float]] = []
            for token_idx in range(seq_len):
                token_p = p_3d[n][token_idx]
                token_out = [0.0] * self.embed_dim
                for d in range(self.embed_dim):
                    dot = sum(
                        token_p[k] * self.weights[k][d] for k in range(self.in_features)
                    )
                    if self.use_bias:
                        dot += self.bias_weights[d]
                    token_out[d] = dot
                sample_out.append(token_out)
            out_3d.append(sample_out)

        return out_3d

    def backward(self, d_out: Any) -> list[list[list[float]]]:
        """Compute analytical input gradient dP and accumulate parameter gradients."""
        if self._cached_x is None:
            raise ValidationError("Cannot run backward before forward pass.")

        de_3d = ensure_3d_tensor(d_out)
        n_samples = len(self._cached_x)
        seq_len = len(self._cached_x[0])

        if (
            len(de_3d) != n_samples
            or len(de_3d[0]) != seq_len
            or len(de_3d[0][0]) != self.embed_dim
        ):
            raise ValidationError(
                f"Upstream gradient shape does not match expected "
                f"({n_samples}, {seq_len}, {self.embed_dim})."
            )

        # Compute dP: [N, L, in_features]
        dp_3d: list[list[list[float]]] = []
        for n in range(n_samples):
            sample_dp: list[list[float]] = []
            for token_idx in range(seq_len):
                token_de = de_3d[n][token_idx]
                token_dp = [0.0] * self.in_features
                for k in range(self.in_features):
                    token_dp[k] = sum(
                        token_de[d] * self.weights[k][d] for d in range(self.embed_dim)
                    )
                sample_dp.append(token_dp)
            dp_3d.append(sample_dp)

        # Accumulate parameter gradients
        for n in range(n_samples):
            for token_idx in range(seq_len):
                token_x = self._cached_x[n][token_idx]
                token_de = de_3d[n][token_idx]
                for k in range(self.in_features):
                    x_k = token_x[k]
                    for d in range(self.embed_dim):
                        self.grad_weights[k][d] += x_k * token_de[d]
                if self.use_bias:
                    for d in range(self.embed_dim):
                        self.grad_bias[d] += token_de[d]

        return dp_3d


class ClassToken:
    """Learnable classification token [1, 1, D_embed] prepended to token sequences."""

    def __init__(
        self,
        embed_dim: int,
        seed: int = 42,
        init_std: float = 0.02,
    ) -> None:
        if embed_dim <= 0:
            raise ValidationError(f"embed_dim must be positive, got {embed_dim}.")

        self.embed_dim = embed_dim
        rng = random.Random(seed)
        self.token: list[list[list[float]]] = [
            [[rng.gauss(0.0, init_std) for _ in range(embed_dim)]]
        ]
        self.zero_grad()
        self._cached_batch_size: int | None = None
        self._cached_seq_len: int | None = None

    def zero_grad(self) -> None:
        """Clear CLS token gradient buffer."""
        self.grad_token: list[list[list[float]]] = [
            [[0.0 for _ in range(self.embed_dim)]]
        ]

    def get_parameters(self) -> dict[str, Any]:
        """Return trainable CLS parameter mapping."""
        return {"token": copy.deepcopy(self.token)}

    def set_parameters(self, params: dict[str, Any]) -> None:
        """Load trainable CLS parameter."""
        if "token" in params:
            tok = params["token"]
            if len(tok) != 1 or len(tok[0]) != 1 or len(tok[0][0]) != self.embed_dim:
                actual_d = len(tok[0][0]) if tok and tok[0] else 0
                raise ValidationError(
                    f"token shape mismatch: expected (1, 1, {self.embed_dim}), "
                    f"got ({len(tok)}, {len(tok[0]) if tok else 0}, {actual_d})."
                )
            self.token = copy.deepcopy(tok)

    def get_gradients(self) -> dict[str, Any]:
        """Return computed CLS token gradient."""
        return {"token": copy.deepcopy(self.grad_token)}

    def forward(self, inputs: Any) -> list[list[list[float]]]:
        """Prepend CLS token [1, 1, D] to sequence [N, L, D] -> [N, L+1, D]."""
        e_3d = ensure_3d_tensor(inputs)
        n_samples = len(e_3d)
        seq_len = len(e_3d[0])
        d_in = len(e_3d[0][0])

        if d_in != self.embed_dim:
            raise ValidationError(
                f"Input embedding dimension ({d_in}) does not match CLS embed_dim "
                f"({self.embed_dim})."
            )

        self._cached_batch_size = n_samples
        self._cached_seq_len = seq_len

        cls_vec = self.token[0][0]
        out_3d: list[list[list[float]]] = []
        for n in range(n_samples):
            sample_seq = [list(cls_vec)] + [list(token) for token in e_3d[n]]
            out_3d.append(sample_seq)

        return out_3d

    def backward(self, d_out: Any) -> list[list[list[float]]]:
        """Separate CLS gradient and patch gradients, accumulating CLS across batch."""
        if self._cached_batch_size is None or self._cached_seq_len is None:
            raise ValidationError("Cannot run backward before forward pass.")

        dz_3d = ensure_3d_tensor(d_out)
        n_samples = self._cached_batch_size
        expected_len = self._cached_seq_len + 1

        if (
            len(dz_3d) != n_samples
            or len(dz_3d[0]) != expected_len
            or len(dz_3d[0][0]) != self.embed_dim
        ):
            raise ValidationError(
                f"Upstream gradient shape does not match expected "
                f"({n_samples}, {expected_len}, {self.embed_dim})."
            )

        # Accumulate CLS token gradient from position 0 across all batch items
        for n in range(n_samples):
            cls_grad_n = dz_3d[n][0]
            for d in range(self.embed_dim):
                self.grad_token[0][0][d] += cls_grad_n[d]

        # Extract patch token gradients [N, L, D_embed]
        de_3d: list[list[list[float]]] = []
        for n in range(n_samples):
            sample_de = [
                list(dz_3d[n][token_idx + 1])
                for token_idx in range(self._cached_seq_len)
            ]
            de_3d.append(sample_de)

        return de_3d


class PositionalEmbedding:
    """Learnable 1D positional embeddings added to token sequences: Y = X + P_pos."""

    def __init__(
        self,
        num_positions: int,
        embed_dim: int,
        seed: int = 42,
        init_std: float = 0.02,
    ) -> None:
        if num_positions <= 0:
            raise ValidationError(
                f"num_positions must be positive, got {num_positions}."
            )
        if embed_dim <= 0:
            raise ValidationError(f"embed_dim must be positive, got {embed_dim}.")

        self.num_positions = num_positions
        self.embed_dim = embed_dim

        rng = random.Random(seed)
        self.embeddings: list[list[list[float]]] = [
            [
                [rng.gauss(0.0, init_std) for _ in range(embed_dim)]
                for _ in range(num_positions)
            ]
        ]
        self.zero_grad()
        self._cached_batch_size: int | None = None
        self._cached_seq_len: int | None = None

    def zero_grad(self) -> None:
        """Clear positional embedding gradient buffer."""
        self.grad_embeddings: list[list[list[float]]] = [
            [[0.0 for _ in range(self.embed_dim)] for _ in range(self.num_positions)]
        ]

    def get_parameters(self) -> dict[str, Any]:
        """Return trainable positional embedding parameter mapping."""
        return {"embeddings": copy.deepcopy(self.embeddings)}

    def set_parameters(self, params: dict[str, Any]) -> None:
        """Load trainable positional embedding values."""
        if "embeddings" in params:
            emb = params["embeddings"]
            if (
                len(emb) != 1
                or len(emb[0]) != self.num_positions
                or len(emb[0][0]) != self.embed_dim
            ):
                actual_d = len(emb[0][0]) if emb and emb[0] else 0
                raise ValidationError(
                    f"embeddings shape mismatch: expected (1, {self.num_positions}, "
                    f"{self.embed_dim}), got ({len(emb)}, {len(emb[0]) if emb else 0}, "
                    f"{actual_d})."
                )
            self.embeddings = copy.deepcopy(emb)

    def get_gradients(self) -> dict[str, Any]:
        """Return computed positional embedding gradients."""
        return {"embeddings": copy.deepcopy(self.grad_embeddings)}

    def forward(self, inputs: Any) -> list[list[list[float]]]:
        """Add learned positional embeddings to input sequence [N, S, D_embed]."""
        x_3d = ensure_3d_tensor(inputs)
        n_samples = len(x_3d)
        seq_len = len(x_3d[0])
        d_in = len(x_3d[0][0])

        if d_in != self.embed_dim:
            raise ValidationError(
                f"Input dimension ({d_in}) does not match PositionalEmbedding "
                f"embed_dim ({self.embed_dim})."
            )

        if seq_len > self.num_positions:
            raise ValidationError(
                f"Sequence length ({seq_len}) exceeds configured num_positions "
                f"({self.num_positions})."
            )

        self._cached_batch_size = n_samples
        self._cached_seq_len = seq_len

        out_3d: list[list[list[float]]] = []
        for n in range(n_samples):
            sample_out: list[list[float]] = []
            for s in range(seq_len):
                token_x = x_3d[n][s]
                pos_vec = self.embeddings[0][s]
                token_out = [token_x[d] + pos_vec[d] for d in range(self.embed_dim)]
                sample_out.append(token_out)
            out_3d.append(sample_out)

        return out_3d

    def backward(self, d_out: Any) -> list[list[list[float]]]:
        """Compute analytical input gradient dX = dY and accumulate dP_pos."""
        if self._cached_batch_size is None or self._cached_seq_len is None:
            raise ValidationError("Cannot run backward before forward pass.")

        dy_3d = ensure_3d_tensor(d_out)
        n_samples = self._cached_batch_size
        seq_len = self._cached_seq_len

        if (
            len(dy_3d) != n_samples
            or len(dy_3d[0]) != seq_len
            or len(dy_3d[0][0]) != self.embed_dim
        ):
            raise ValidationError(
                f"Upstream gradient shape does not match expected "
                f"({n_samples}, {seq_len}, {self.embed_dim})."
            )

        # Accumulate positional embedding gradients across all batch elements
        for n in range(n_samples):
            for s in range(seq_len):
                token_dy = dy_3d[n][s]
                for d in range(self.embed_dim):
                    self.grad_embeddings[0][s][d] += token_dy[d]

        # dX = dY (elementwise identity pass-through)
        return [[list(token) for token in sample] for sample in dy_3d]
