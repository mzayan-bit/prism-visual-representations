"""Explicit self-attention operations and multi-head attention."""

import copy
import math
import random
from typing import Any

from prism.core.errors import ValidationError
from prism.models.patches import ensure_3d_tensor


def ensure_4d_attention_tensor(data: Any) -> list[list[list[list[float]]]]:
    """Validate and normalize nested list structure into 4D tensor [N, H, L_q, L_k]."""
    if data is None:
        raise ValidationError("Input tensor cannot be None.")

    if not isinstance(data, (list, tuple)):
        raise ValidationError("Expected 4D nested list [N, H, L_q, L_k].")

    if not data:
        raise ValidationError("Tensor batch cannot be empty.")

    first_elem = data[0]
    if not isinstance(first_elem, (list, tuple)):
        raise ValidationError("Expected 4D nested list, got 1D sequence.")

    if not first_elem:
        raise ValidationError("Head dimension H cannot be 0.")

    second_elem = first_elem[0]
    if not isinstance(second_elem, (list, tuple)):
        raise ValidationError("Expected 4D nested list, got 2D structure.")

    third_elem = second_elem[0]
    if not isinstance(third_elem, (list, tuple)):
        # 3D tensor [H, L_q, L_k] -> wrap to [1, H, L_q, L_k]
        sample_4d: list[list[list[float]]] = []
        for h_idx, head_mat in enumerate(data):
            if not isinstance(head_mat, (list, tuple)):
                raise ValidationError(f"Invalid head matrix at index {h_idx}.")
            head_rows: list[list[float]] = []
            for row in head_mat:
                if not isinstance(row, (list, tuple)):
                    raise ValidationError(
                        "Invalid row structure in 3D attention tensor."
                    )
                row_floats: list[float] = []
                for val in row:
                    if (
                        not isinstance(val, (int, float))
                        or math.isnan(val)
                        or math.isinf(val)
                    ):
                        raise ValidationError(
                            f"Non-finite scalar in attention tensor: {val}"
                        )
                    row_floats.append(float(val))
                head_rows.append(row_floats)
            sample_4d.append(head_rows)
        return [sample_4d]

    # Already 4D: validate dimensions and all elements
    tensor_4d: list[list[list[list[float]]]] = []
    expected_h = len(first_elem)
    expected_lq = len(second_elem)
    expected_lk = len(third_elem)

    for n_idx, sample in enumerate(data):
        if not isinstance(sample, (list, tuple)) or len(sample) != expected_h:
            actual_h = len(sample) if isinstance(sample, (list, tuple)) else "invalid"
            raise ValidationError(
                f"Sample at batch {n_idx} has {actual_h} heads, expected {expected_h}."
            )
        sample_heads: list[list[list[float]]] = []
        for h_idx, head_mat in enumerate(sample):
            if not isinstance(head_mat, (list, tuple)) or len(head_mat) != expected_lq:
                actual_lq = (
                    len(head_mat) if isinstance(head_mat, (list, tuple)) else "invalid"
                )
                raise ValidationError(
                    f"Head ({n_idx}, {h_idx}) has L_q={actual_lq}, "
                    f"expected {expected_lq}."
                )
            head_rows = []
            for r_idx, row in enumerate(head_mat):
                if not isinstance(row, (list, tuple)) or len(row) != expected_lk:
                    actual_lk = (
                        len(row) if isinstance(row, (list, tuple)) else "invalid"
                    )
                    raise ValidationError(
                        f"Row ({n_idx}, {h_idx}, {r_idx}) has L_k={actual_lk}, "
                        f"expected {expected_lk}."
                    )
                row_floats = []
                for val in row:
                    if (
                        not isinstance(val, (int, float))
                        or math.isnan(val)
                        or math.isinf(val)
                    ):
                        raise ValidationError(
                            f"Non-finite scalar in attention tensor: {val}"
                        )
                    row_floats.append(float(val))
                head_rows.append(row_floats)
            sample_heads.append(head_rows)
        tensor_4d.append(sample_heads)

    return tensor_4d


def softmax_1d(x: list[float]) -> list[float]:
    """Compute stable 1D softmax: y = exp(x - max(x)) / sum(exp(x - max(x)))."""
    if not x:
        return []
    max_val = max(x)
    if math.isnan(max_val) or math.isinf(max_val):
        raise ValidationError(f"Non-finite scalar in softmax input: {x}")
    exp_shifted = [math.exp(val - max_val) for val in x]
    sum_exp = sum(exp_shifted)
    if sum_exp <= 0.0 or math.isnan(sum_exp) or math.isinf(sum_exp):
        raise ValidationError(f"Invalid sum of exponentials in softmax: {sum_exp}")
    return [e / sum_exp for e in exp_shifted]


def softmax_backward_1d(y: list[float], dy: list[float]) -> list[float]:
    """Compute analytical derivative through 1D softmax: dx = y * (dy - sum(dy * y))."""
    if len(y) != len(dy):
        raise ValidationError(
            f"Softmax backward dimension mismatch: y({len(y)}) vs dy({len(dy)})."
        )
    dot = sum(y_i * dy_i for y_i, dy_i in zip(y, dy, strict=True))
    return [y_i * (dy_i - dot) for y_i, dy_i in zip(y, dy, strict=True)]


class ScaledDotProductAttention:
    """Scaled Dot-Product Attention: Output = softmax(Q K^T / sqrt(D_h)) V."""

    def __init__(self) -> None:
        self._cached_q: list[list[list[list[float]]]] | None = None
        self._cached_k: list[list[list[list[float]]]] | None = None
        self._cached_v: list[list[list[list[float]]]] | None = None
        self._cached_weights: list[list[list[list[float]]]] | None = None
        self._cached_scale: float | None = None

    def forward(
        self,
        q: Any,
        k: Any,
        v: Any,
        mask: Any = None,
    ) -> tuple[list[list[list[list[float]]]], list[list[list[list[float]]]]]:
        """Compute scaled dot-product attention producing (output, attention_weights).

        Parameters
        ----------
        q : Tensor of shape [N, H, L_q, D_h]
        k : Tensor of shape [N, H, L_k, D_h]
        v : Tensor of shape [N, H, L_k, D_v]
        mask : Optional mask of shape [N, H, L_q, L_k] or [1, 1, L_q, L_k]

        Returns
        -------
        tuple[list[list[list[list[float]]]], list[list[list[list[float]]]]]
            (output [N, H, L_q, D_v], attention_weights [N, H, L_q, L_k])
        """
        q_4d = ensure_4d_attention_tensor(q)
        k_4d = ensure_4d_attention_tensor(k)
        v_4d = ensure_4d_attention_tensor(v)

        n_samples = len(q_4d)
        num_heads = len(q_4d[0])
        l_q = len(q_4d[0][0])
        d_h = len(q_4d[0][0][0])

        if len(k_4d) != n_samples or len(k_4d[0]) != num_heads:
            raise ValidationError(
                f"K batch/head shape ({len(k_4d)}, {len(k_4d[0]) if k_4d else 0}) "
                f"does not match Q ({n_samples}, {num_heads})."
            )
        l_k = len(k_4d[0][0])
        if len(k_4d[0][0][0]) != d_h:
            raise ValidationError(
                f"K head dimension ({len(k_4d[0][0][0])}) does not match Q ({d_h})."
            )

        if (
            len(v_4d) != n_samples
            or len(v_4d[0]) != num_heads
            or len(v_4d[0][0]) != l_k
        ):
            raise ValidationError(
                f"V sequence length ({len(v_4d[0][0]) if v_4d and v_4d[0] else 0}) "
                f"does not match K sequence length ({l_k})."
            )
        d_v = len(v_4d[0][0][0])

        scale = 1.0 / math.sqrt(float(d_h))

        # Compute Attention Weights: A = softmax(Q K^T / sqrt(D_h))
        weights_4d: list[list[list[list[float]]]] = []
        for n in range(n_samples):
            sample_weights: list[list[list[float]]] = []
            for h in range(num_heads):
                head_weights: list[list[float]] = []
                for i in range(l_q):
                    q_vec = q_4d[n][h][i]
                    raw_scores: list[float] = []
                    for j in range(l_k):
                        k_vec = k_4d[n][h][j]
                        score = scale * sum(q_vec[d] * k_vec[d] for d in range(d_h))
                        if mask is not None:
                            # Apply additive mask if provided
                            m_val = (
                                mask[n][h][i][j] if len(mask) > 1 else mask[0][0][i][j]
                            )
                            score += m_val
                        raw_scores.append(score)
                    attn_row = softmax_1d(raw_scores)
                    head_weights.append(attn_row)
                sample_weights.append(head_weights)
            weights_4d.append(sample_weights)

        # Compute Output: Out = A V
        out_4d: list[list[list[list[float]]]] = []
        for n in range(n_samples):
            sample_out: list[list[list[float]]] = []
            for h in range(num_heads):
                head_out: list[list[float]] = []
                for i in range(l_q):
                    attn_row = weights_4d[n][h][i]
                    out_row: list[float] = [0.0] * d_v
                    for d in range(d_v):
                        out_row[d] = sum(
                            attn_row[j] * v_4d[n][h][j][d] for j in range(l_k)
                        )
                    head_out.append(out_row)
                sample_out.append(head_out)
            out_4d.append(sample_out)

        # Cache forward state for analytical backward
        self._cached_q = q_4d
        self._cached_k = k_4d
        self._cached_v = v_4d
        self._cached_weights = weights_4d
        self._cached_scale = scale

        return out_4d, weights_4d

    def backward(
        self, d_out: Any
    ) -> tuple[
        list[list[list[list[float]]]],
        list[list[list[list[float]]]],
        list[list[list[list[float]]]],
    ]:
        """Compute analytical gradients (dQ, dK, dV) through attention and softmax.

        Parameters
        ----------
        d_out : Upstream gradient tensor [N, H, L_q, D_v]

        Returns
        -------
        tuple[Tensor, Tensor, Tensor]
            (dQ [N, H, L_q, D_h], dK [N, H, L_k, D_h], dV [N, H, L_k, D_v])
        """
        if (
            self._cached_q is None
            or self._cached_k is None
            or self._cached_v is None
            or self._cached_weights is None
            or self._cached_scale is None
        ):
            raise ValidationError("Cannot run backward before forward pass.")

        d_out_4d = ensure_4d_attention_tensor(d_out)
        q_4d = self._cached_q
        k_4d = self._cached_k
        v_4d = self._cached_v
        weights_4d = self._cached_weights
        scale = self._cached_scale

        n_samples = len(q_4d)
        num_heads = len(q_4d[0])
        l_q = len(q_4d[0][0])
        d_h = len(q_4d[0][0][0])
        l_k = len(k_4d[0][0])
        d_v = len(v_4d[0][0][0])

        if (
            len(d_out_4d) != n_samples
            or len(d_out_4d[0]) != num_heads
            or len(d_out_4d[0][0]) != l_q
            or len(d_out_4d[0][0][0]) != d_v
        ):
            raise ValidationError(
                f"d_out shape mismatch: expected ({n_samples}, {num_heads}, "
                f"{l_q}, {d_v})."
            )

        # 1. dV = A^T d_out: [N, H, L_k, D_v]
        dv_4d: list[list[list[list[float]]]] = []
        for n in range(n_samples):
            sample_dv: list[list[list[float]]] = []
            for h in range(num_heads):
                head_dv: list[list[float]] = []
                for j in range(l_k):
                    v_row: list[float] = [0.0] * d_v
                    for d in range(d_v):
                        v_row[d] = sum(
                            weights_4d[n][h][i][j] * d_out_4d[n][h][i][d]
                            for i in range(l_q)
                        )
                    head_dv.append(v_row)
                sample_dv.append(head_dv)
            dv_4d.append(sample_dv)

        # 2. dA = d_out V^T: [N, H, L_q, L_k]
        # 3. dScores = softmax_backward(A, dA): [N, H, L_q, L_k]
        dscores_4d: list[list[list[list[float]]]] = []
        for n in range(n_samples):
            sample_dscores: list[list[list[float]]] = []
            for h in range(num_heads):
                head_dscores: list[list[float]] = []
                for i in range(l_q):
                    d_out_row = d_out_4d[n][h][i]
                    da_row: list[float] = [0.0] * l_k
                    for j in range(l_k):
                        v_vec = v_4d[n][h][j]
                        da_row[j] = sum(d_out_row[d] * v_vec[d] for d in range(d_v))
                    a_row = weights_4d[n][h][i]
                    dscore_row = softmax_backward_1d(a_row, da_row)
                    head_dscores.append(dscore_row)
                sample_dscores.append(head_dscores)
            dscores_4d.append(sample_dscores)

        # 4. dQ = scale * (dScores K): [N, H, L_q, D_h]
        dq_4d: list[list[list[list[float]]]] = []
        for n in range(n_samples):
            sample_dq: list[list[list[float]]] = []
            for h in range(num_heads):
                head_dq: list[list[float]] = []
                for i in range(l_q):
                    dscore_row = dscores_4d[n][h][i]
                    q_row: list[float] = [0.0] * d_h
                    for d in range(d_h):
                        q_row[d] = scale * sum(
                            dscore_row[j] * k_4d[n][h][j][d] for j in range(l_k)
                        )
                    head_dq.append(q_row)
                sample_dq.append(head_dq)
            dq_4d.append(sample_dq)

        # 5. dK = scale * (dScores^T Q): [N, H, L_k, D_h]
        dk_4d: list[list[list[list[float]]]] = []
        for n in range(n_samples):
            sample_dk: list[list[list[float]]] = []
            for h in range(num_heads):
                head_dk: list[list[float]] = []
                for j in range(l_k):
                    k_row: list[float] = [0.0] * d_h
                    for d in range(d_h):
                        k_row[d] = scale * sum(
                            dscores_4d[n][h][i][j] * q_4d[n][h][i][d]
                            for i in range(l_q)
                        )
                    head_dk.append(k_row)
                sample_dk.append(head_dk)
            dk_4d.append(sample_dk)

        return dq_4d, dk_4d, dv_4d


class MultiHeadSelfAttention:
    """Multi-Head Self-Attention with analytical backpropagation through projections."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        bias: bool = True,
        seed: int = 42,
    ) -> None:
        if embed_dim <= 0:
            raise ValidationError(f"embed_dim must be positive, got {embed_dim}.")
        if num_heads <= 0:
            raise ValidationError(f"num_heads must be positive, got {num_heads}.")
        if embed_dim % num_heads != 0:
            raise ValidationError(
                f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})."
            )

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.d_head = embed_dim // num_heads
        self.use_bias = bias

        rng = random.Random(seed)
        std = math.sqrt(2.0 / float(embed_dim + embed_dim))

        def _init_w() -> list[list[float]]:
            return [
                [rng.gauss(0.0, std) for _ in range(embed_dim)]
                for _ in range(embed_dim)
            ]

        def _init_b() -> list[float]:
            return [0.0 for _ in range(embed_dim)] if bias else []

        self.w_q = _init_w()
        self.b_q = _init_b()
        self.w_k = _init_w()
        self.b_k = _init_b()
        self.w_v = _init_w()
        self.b_v = _init_b()
        self.w_o = _init_w()
        self.b_o = _init_b()

        self.attn = ScaledDotProductAttention()
        self.last_attention_weights: list[list[list[list[float]]]] | None = None
        self._cached_x: list[list[list[float]]] | None = None
        self._cached_concat: list[list[list[float]]] | None = None

        self.zero_grad()

    def zero_grad(self) -> None:
        """Clear all parameter gradient buffers."""
        self.grad_w_q: list[list[float]] = [
            [0.0 for _ in range(self.embed_dim)] for _ in range(self.embed_dim)
        ]
        self.grad_b_q: list[float] = (
            [0.0 for _ in range(self.embed_dim)] if self.use_bias else []
        )
        self.grad_w_k: list[list[float]] = [
            [0.0 for _ in range(self.embed_dim)] for _ in range(self.embed_dim)
        ]
        self.grad_b_k: list[float] = (
            [0.0 for _ in range(self.embed_dim)] if self.use_bias else []
        )
        self.grad_w_v: list[list[float]] = [
            [0.0 for _ in range(self.embed_dim)] for _ in range(self.embed_dim)
        ]
        self.grad_b_v: list[float] = (
            [0.0 for _ in range(self.embed_dim)] if self.use_bias else []
        )
        self.grad_w_o: list[list[float]] = [
            [0.0 for _ in range(self.embed_dim)] for _ in range(self.embed_dim)
        ]
        self.grad_b_o: list[float] = (
            [0.0 for _ in range(self.embed_dim)] if self.use_bias else []
        )

    def get_parameters(self) -> dict[str, Any]:
        """Return all trainable parameter tensors."""
        params: dict[str, Any] = {
            "w_q": copy.deepcopy(self.w_q),
            "w_k": copy.deepcopy(self.w_k),
            "w_v": copy.deepcopy(self.w_v),
            "w_o": copy.deepcopy(self.w_o),
        }
        if self.use_bias:
            params["b_q"] = copy.deepcopy(self.b_q)
            params["b_k"] = copy.deepcopy(self.b_k)
            params["b_v"] = copy.deepcopy(self.b_v)
            params["b_o"] = copy.deepcopy(self.b_o)
        return params

    def set_parameters(self, params: dict[str, Any]) -> None:
        """Load trainable parameters from mapping."""
        for name in ("w_q", "w_k", "w_v", "w_o"):
            if name in params:
                val = params[name]
                if len(val) != self.embed_dim or len(val[0]) != self.embed_dim:
                    actual_d = len(val[0]) if val else 0
                    raise ValidationError(
                        f"{name} shape mismatch: expected ({self.embed_dim}, "
                        f"{self.embed_dim}), got ({len(val)}, {actual_d})."
                    )
                setattr(self, name, copy.deepcopy(val))
        if self.use_bias:
            for name in ("b_q", "b_k", "b_v", "b_o"):
                if name in params:
                    val = params[name]
                    if len(val) != self.embed_dim:
                        raise ValidationError(
                            f"{name} shape mismatch: expected ({self.embed_dim},), "
                            f"got ({len(val)},)."
                        )
                    setattr(self, name, copy.deepcopy(val))

    def get_gradients(self) -> dict[str, Any]:
        """Return computed parameter gradients."""
        grads: dict[str, Any] = {
            "w_q": copy.deepcopy(self.grad_w_q),
            "w_k": copy.deepcopy(self.grad_w_k),
            "w_v": copy.deepcopy(self.grad_w_v),
            "w_o": copy.deepcopy(self.grad_w_o),
        }
        if self.use_bias:
            grads["b_q"] = copy.deepcopy(self.grad_b_q)
            grads["b_k"] = copy.deepcopy(self.grad_b_k)
            grads["b_v"] = copy.deepcopy(self.grad_b_v)
            grads["b_o"] = copy.deepcopy(self.grad_b_o)
        return grads

    def forward(self, inputs: Any, mask: Any = None) -> list[list[list[float]]]:
        """Compute multi-head self-attention producing [N, L, D_embed]."""
        x_3d = ensure_3d_tensor(inputs)
        n_samples = len(x_3d)
        seq_len = len(x_3d[0])
        d_in = len(x_3d[0][0])

        if d_in != self.embed_dim:
            raise ValidationError(
                f"Input embedding dimension ({d_in}) does not match MHSA "
                f"embed_dim ({self.embed_dim})."
            )

        self._cached_x = x_3d

        # 1. Project inputs: Q = X W_Q + b_Q, K = X W_K + b_K, V = X W_V + b_V
        q_4d: list[list[list[list[float]]]] = []
        k_4d: list[list[list[list[float]]]] = []
        v_4d: list[list[list[list[float]]]] = []

        for n in range(n_samples):
            sample_q: list[list[list[float]]] = [[] for _ in range(self.num_heads)]
            sample_k: list[list[list[float]]] = [[] for _ in range(self.num_heads)]
            sample_v: list[list[list[float]]] = [[] for _ in range(self.num_heads)]

            for seq_idx in range(seq_len):
                x_vec = x_3d[n][seq_idx]
                q_flat = [0.0] * self.embed_dim
                k_flat = [0.0] * self.embed_dim
                v_flat = [0.0] * self.embed_dim

                for d in range(self.embed_dim):
                    q_val = sum(
                        x_vec[k] * self.w_q[k][d] for k in range(self.embed_dim)
                    )
                    k_val = sum(
                        x_vec[k] * self.w_k[k][d] for k in range(self.embed_dim)
                    )
                    v_val = sum(
                        x_vec[k] * self.w_v[k][d] for k in range(self.embed_dim)
                    )
                    if self.use_bias:
                        q_val += self.b_q[d]
                        k_val += self.b_k[d]
                        v_val += self.b_v[d]
                    q_flat[d] = q_val
                    k_flat[d] = k_val
                    v_flat[d] = v_val

                # Split into heads [H, L, D_head]
                for h in range(self.num_heads):
                    h_start = h * self.d_head
                    h_end = h_start + self.d_head
                    sample_q[h].append(q_flat[h_start:h_end])
                    sample_k[h].append(k_flat[h_start:h_end])
                    sample_v[h].append(v_flat[h_start:h_end])

            q_4d.append(sample_q)
            k_4d.append(sample_k)
            v_4d.append(sample_v)

        # 2. Scaled Dot-Product Attention per head
        attn_out_4d, attn_weights_4d = self.attn.forward(
            q=q_4d, k=k_4d, v=v_4d, mask=mask
        )
        self.last_attention_weights = attn_weights_4d

        # 3. Concatenate heads: [N, H, L, D_head] -> [N, L, D_embed]
        concat_3d: list[list[list[float]]] = []
        for n in range(n_samples):
            sample_concat: list[list[float]] = []
            for seq_idx in range(seq_len):
                row: list[float] = []
                for h in range(self.num_heads):
                    row.extend(attn_out_4d[n][h][seq_idx])
                sample_concat.append(row)
            concat_3d.append(sample_concat)

        self._cached_concat = concat_3d

        # 4. Output projection: Out = Concat W_O + b_O
        out_3d: list[list[list[float]]] = []
        for n in range(n_samples):
            sample_out: list[list[float]] = []
            for seq_idx in range(seq_len):
                c_vec = concat_3d[n][seq_idx]
                row_out = [0.0] * self.embed_dim
                for d in range(self.embed_dim):
                    dot = sum(c_vec[k] * self.w_o[k][d] for k in range(self.embed_dim))
                    if self.use_bias:
                        dot += self.b_o[d]
                    row_out[d] = dot
                sample_out.append(row_out)
            out_3d.append(sample_out)

        return out_3d

    def backward(self, d_out: Any) -> list[list[list[float]]]:
        """Compute analytical backward through projections and sum input gradients."""
        if self._cached_x is None or self._cached_concat is None:
            raise ValidationError("Cannot run backward before forward pass.")

        d_out_3d = ensure_3d_tensor(d_out)
        n_samples = len(self._cached_x)
        seq_len = len(self._cached_x[0])

        if (
            len(d_out_3d) != n_samples
            or len(d_out_3d[0]) != seq_len
            or len(d_out_3d[0][0]) != self.embed_dim
        ):
            raise ValidationError(
                f"d_out shape mismatch: expected ({n_samples}, {seq_len}, "
                f"{self.embed_dim})."
            )

        # 1. Output projection backward: dConcat = dOut W_O^T, dW_O += Concat^T dOut
        dconcat_3d: list[list[list[float]]] = []
        for n in range(n_samples):
            sample_dconcat: list[list[float]] = []
            for seq_idx in range(seq_len):
                d_row = d_out_3d[n][seq_idx]
                c_row = self._cached_concat[n][seq_idx]
                dc_row = [0.0] * self.embed_dim
                for k in range(self.embed_dim):
                    dc_row[k] = sum(
                        d_row[d] * self.w_o[k][d] for d in range(self.embed_dim)
                    )
                    c_k = c_row[k]
                    for d in range(self.embed_dim):
                        self.grad_w_o[k][d] += c_k * d_row[d]
                if self.use_bias:
                    for d in range(self.embed_dim):
                        self.grad_b_o[d] += d_row[d]
                sample_dconcat.append(dc_row)
            dconcat_3d.append(sample_dconcat)

        # 2. Split dConcat into heads: [N, H, L, D_head]
        d_attn_out_4d: list[list[list[list[float]]]] = []
        for n in range(n_samples):
            sample_d_heads: list[list[list[float]]] = [
                [] for _ in range(self.num_heads)
            ]
            for seq_idx in range(seq_len):
                dc_row = dconcat_3d[n][seq_idx]
                for h in range(self.num_heads):
                    h_start = h * self.d_head
                    h_end = h_start + self.d_head
                    sample_d_heads[h].append(dc_row[h_start:h_end])
            d_attn_out_4d.append(sample_d_heads)

        # 3. Attention core backward: (dQ, dK, dV) [N, H, L, D_head]
        dq_4d, dk_4d, dv_4d = self.attn.backward(d_attn_out_4d)

        # 4. Merge heads for dQ, dK, dV -> [N, L, D_embed]
        dq_flat_3d: list[list[list[float]]] = []
        dk_flat_3d: list[list[list[float]]] = []
        dv_flat_3d: list[list[list[float]]] = []

        for n in range(n_samples):
            sample_dq_flat: list[list[float]] = []
            sample_dk_flat: list[list[float]] = []
            sample_dv_flat: list[list[float]] = []
            for seq_idx in range(seq_len):
                row_q: list[float] = []
                row_k: list[float] = []
                row_v: list[float] = []
                for h in range(self.num_heads):
                    row_q.extend(dq_4d[n][h][seq_idx])
                    row_k.extend(dk_4d[n][h][seq_idx])
                    row_v.extend(dv_4d[n][h][seq_idx])
                sample_dq_flat.append(row_q)
                sample_dk_flat.append(row_k)
                sample_dv_flat.append(row_v)
            dq_flat_3d.append(sample_dq_flat)
            dk_flat_3d.append(sample_dk_flat)
            dv_flat_3d.append(sample_dv_flat)

        # 5. Backward through Q, K, V projections and accumulate input gradients:
        # dX = dX_Q + dX_K + dX_V where dX_Q = dQ W_Q^T, etc.
        dx_3d: list[list[list[float]]] = []
        for n in range(n_samples):
            sample_dx: list[list[float]] = []
            for seq_idx in range(seq_len):
                x_vec = self._cached_x[n][seq_idx]
                dq_row = dq_flat_3d[n][seq_idx]
                dk_row = dk_flat_3d[n][seq_idx]
                dv_row = dv_flat_3d[n][seq_idx]

                dx_row = [0.0] * self.embed_dim
                for k in range(self.embed_dim):
                    dx_q_k = sum(
                        dq_row[d] * self.w_q[k][d] for d in range(self.embed_dim)
                    )
                    dx_k_k = sum(
                        dk_row[d] * self.w_k[k][d] for d in range(self.embed_dim)
                    )
                    dx_v_k = sum(
                        dv_row[d] * self.w_v[k][d] for d in range(self.embed_dim)
                    )
                    dx_row[k] = dx_q_k + dx_k_k + dx_v_k

                    # Accumulate parameter gradients
                    x_k = x_vec[k]
                    for d in range(self.embed_dim):
                        self.grad_w_q[k][d] += x_k * dq_row[d]
                        self.grad_w_k[k][d] += x_k * dk_row[d]
                        self.grad_w_v[k][d] += x_k * dv_row[d]

                if self.use_bias:
                    for d in range(self.embed_dim):
                        self.grad_b_q[d] += dq_row[d]
                        self.grad_b_k[d] += dk_row[d]
                        self.grad_b_v[d] += dv_row[d]

                sample_dx.append(dx_row)
            dx_3d.append(sample_dx)

        return dx_3d
