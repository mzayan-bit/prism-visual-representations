"""Transformer encoder building blocks, stacked encoders, and Vision Transformer."""

from __future__ import annotations

import copy
import math
import random
from typing import Any

from prism.core.errors import ValidationError
from prism.models.activations import BaseActivation, get_activation
from prism.models.attention import MultiHeadSelfAttention
from prism.models.base import BaseVisionModel
from prism.models.normalization import LayerNorm
from prism.models.patches import (
    ClassToken,
    ImagePatchExtractor,
    LearnablePositionalEmbedding,
    PatchEmbedding,
    PatchGeometry,
    ensure_3d_tensor,
)
from prism.models.spatial import ensure_4d_tensor
from prism.models.specifications import ModelSpecification


class TransformerFeedForward:
    """Token-wise Feed-Forward Network: FFN(x) = GELU(x W_1 + b_1) W_2 + b_2.

    Applies two linear projections with non-linear activation independently
    to each token vector.
    """

    def __init__(
        self,
        in_features: int,
        hidden_dim: int,
        bias: bool = True,
        activation: str | BaseActivation = "gelu",
        seed: int = 42,
    ) -> None:
        if in_features <= 0:
            raise ValidationError(f"in_features must be positive, got {in_features}.")
        if hidden_dim <= 0:
            raise ValidationError(f"hidden_dim must be positive, got {hidden_dim}.")

        self.in_features = in_features
        self.hidden_dim = hidden_dim
        self.use_bias = bias

        if isinstance(activation, str):
            self.activation: BaseActivation = get_activation(activation)
        else:
            self.activation = activation

        rng = random.Random(seed)
        std1 = math.sqrt(2.0 / float(in_features + hidden_dim))
        std2 = math.sqrt(2.0 / float(hidden_dim + in_features))

        self.w_1: list[list[float]] = [
            [rng.gauss(0.0, std1) for _ in range(hidden_dim)]
            for _ in range(in_features)
        ]
        self.b_1: list[float] = [0.0] * hidden_dim if bias else []

        self.w_2: list[list[float]] = [
            [rng.gauss(0.0, std2) for _ in range(in_features)]
            for _ in range(hidden_dim)
        ]
        self.b_2: list[float] = [0.0] * in_features if bias else []

        self.zero_grad()

        # Cache for analytical backward pass
        self._cached_x: list[list[list[float]]] | None = None
        self._cached_h1: list[list[list[float]]] | None = None
        self._cached_a1: list[list[list[float]]] | None = None

    def zero_grad(self) -> None:
        """Clear all parameter gradient buffers."""
        self.grad_w_1: list[list[float]] = [
            [0.0] * self.hidden_dim for _ in range(self.in_features)
        ]
        self.grad_b_1: list[float] = [0.0] * self.hidden_dim if self.use_bias else []
        self.grad_w_2: list[list[float]] = [
            [0.0] * self.in_features for _ in range(self.hidden_dim)
        ]
        self.grad_b_2: list[float] = [0.0] * self.in_features if self.use_bias else []

    def get_parameters(self) -> dict[str, Any]:
        """Return trainable parameter tensors."""
        params: dict[str, Any] = {
            "w_1": copy.deepcopy(self.w_1),
            "w_2": copy.deepcopy(self.w_2),
        }
        if self.use_bias:
            params["b_1"] = copy.deepcopy(self.b_1)
            params["b_2"] = copy.deepcopy(self.b_2)
        return params

    def set_parameters(self, params: dict[str, Any]) -> None:
        """Load trainable parameters from mapping."""
        if "w_1" in params:
            w1 = params["w_1"]
            if len(w1) != self.in_features or len(w1[0]) != self.hidden_dim:
                raise ValidationError(
                    f"w_1 shape mismatch: expected ({self.in_features}, "
                    f"{self.hidden_dim}), got ({len(w1)}, {len(w1[0]) if w1 else 0})."
                )
            self.w_1 = copy.deepcopy(w1)

        if "w_2" in params:
            w2 = params["w_2"]
            if len(w2) != self.hidden_dim or len(w2[0]) != self.in_features:
                raise ValidationError(
                    f"w_2 shape mismatch: expected ({self.hidden_dim}, "
                    f"{self.in_features}), got ({len(w2)}, {len(w2[0]) if w2 else 0})."
                )
            self.w_2 = copy.deepcopy(w2)

        if self.use_bias:
            if "b_1" in params:
                b1 = params["b_1"]
                if len(b1) != self.hidden_dim:
                    raise ValidationError(
                        f"b_1 shape mismatch: expected ({self.hidden_dim},), "
                        f"got ({len(b1)},)."
                    )
                self.b_1 = copy.deepcopy(b1)
            if "b_2" in params:
                b2 = params["b_2"]
                if len(b2) != self.in_features:
                    raise ValidationError(
                        f"b_2 shape mismatch: expected ({self.in_features},), "
                        f"got ({len(b2)},)."
                    )
                self.b_2 = copy.deepcopy(b2)

    def get_gradients(self) -> dict[str, Any]:
        """Return computed parameter gradients."""
        grads: dict[str, Any] = {
            "w_1": copy.deepcopy(self.grad_w_1),
            "w_2": copy.deepcopy(self.grad_w_2),
        }
        if self.use_bias:
            grads["b_1"] = copy.deepcopy(self.grad_b_1)
            grads["b_2"] = copy.deepcopy(self.grad_b_2)
        return grads

    def get_state(self) -> dict[str, Any]:
        """Return non-trainable state."""
        return {}

    def set_state(self, state: dict[str, Any]) -> None:
        """Restore non-trainable state."""
        pass

    def forward(self, inputs: Any) -> list[list[list[float]]]:
        """Compute token-wise forward pass producing [N, T, in_features]."""
        x_3d = ensure_3d_tensor(inputs)
        n_samples = len(x_3d)
        seq_len = len(x_3d[0])
        d_in = len(x_3d[0][0])

        if d_in != self.in_features:
            raise ValidationError(
                f"Input dimension ({d_in}) does not match FFN in_features "
                f"({self.in_features})."
            )

        self._cached_x = x_3d

        # 1. First projection: H1 = X W_1 + b_1
        h1_3d: list[list[list[float]]] = []
        for n in range(n_samples):
            sample_h1: list[list[float]] = []
            for t in range(seq_len):
                x_vec = x_3d[n][t]
                row_h1 = [0.0] * self.hidden_dim
                for h in range(self.hidden_dim):
                    dot = sum(
                        x_vec[k] * self.w_1[k][h] for k in range(self.in_features)
                    )
                    if self.use_bias:
                        dot += self.b_1[h]
                    row_h1[h] = dot
                sample_h1.append(row_h1)
            h1_3d.append(sample_h1)

        self._cached_h1 = h1_3d

        # 2. Activation: A1 = act(H1)
        a1_3d = self.activation.forward(h1_3d)
        self._cached_a1 = a1_3d

        # 3. Second projection: Y = A1 W_2 + b_2
        out_3d: list[list[list[float]]] = []
        for n in range(n_samples):
            sample_out: list[list[float]] = []
            for t in range(seq_len):
                a1_vec = a1_3d[n][t]
                row_out = [0.0] * self.in_features
                for d in range(self.in_features):
                    dot = sum(
                        a1_vec[h] * self.w_2[h][d] for h in range(self.hidden_dim)
                    )
                    if self.use_bias:
                        dot += self.b_2[d]
                    row_out[d] = dot
                sample_out.append(row_out)
            out_3d.append(sample_out)

        return out_3d

    def backward(self, d_out: Any) -> list[list[list[float]]]:
        """Compute analytical input gradient dX and accumulate parameter gradients."""
        if self._cached_x is None or self._cached_h1 is None or self._cached_a1 is None:
            raise ValidationError("Cannot run backward before forward pass.")

        dy_3d = ensure_3d_tensor(d_out)
        n_samples = len(self._cached_x)
        seq_len = len(self._cached_x[0])

        if (
            len(dy_3d) != n_samples
            or len(dy_3d[0]) != seq_len
            or len(dy_3d[0][0]) != self.in_features
        ):
            raise ValidationError(
                f"d_out shape mismatch: expected ({n_samples}, {seq_len}, "
                f"{self.in_features})."
            )

        # 1. Backward through second projection: dA1 = dY W_2^T, dW_2 += A1^T dY
        da1_3d: list[list[list[float]]] = []
        for n in range(n_samples):
            sample_da1: list[list[float]] = []
            for t in range(seq_len):
                dy_vec = dy_3d[n][t]
                a1_vec = self._cached_a1[n][t]
                da1_vec = [0.0] * self.hidden_dim

                for h in range(self.hidden_dim):
                    da1_vec[h] = sum(
                        dy_vec[d] * self.w_2[h][d] for d in range(self.in_features)
                    )
                    a1_h = a1_vec[h]
                    for d in range(self.in_features):
                        self.grad_w_2[h][d] += a1_h * dy_vec[d]

                if self.use_bias:
                    for d in range(self.in_features):
                        self.grad_b_2[d] += dy_vec[d]

                sample_da1.append(da1_vec)
            da1_3d.append(sample_da1)

        # 2. Backward through activation: dH1 = act_backward(H1, dA1)
        dh1_3d = self.activation.backward(self._cached_h1, da1_3d)

        # 3. Backward through first projection: dX = dH1 W_1^T, dW_1 += X^T dH1
        dx_3d: list[list[list[float]]] = []
        for n in range(n_samples):
            sample_dx: list[list[float]] = []
            for t in range(seq_len):
                dh1_vec = dh1_3d[n][t]
                x_vec = self._cached_x[n][t]
                dx_vec = [0.0] * self.in_features

                for k in range(self.in_features):
                    dx_vec[k] = sum(
                        dh1_vec[h] * self.w_1[k][h] for h in range(self.hidden_dim)
                    )
                    x_k = x_vec[k]
                    for h in range(self.hidden_dim):
                        self.grad_w_1[k][h] += x_k * dh1_vec[h]

                if self.use_bias:
                    for h in range(self.hidden_dim):
                        self.grad_b_1[h] += dh1_vec[h]

                sample_dx.append(dx_vec)
            dx_3d.append(sample_dx)

        return dx_3d


class TransformerEncoderBlock:
    """Pre-Norm Transformer Encoder Block with explicit dual residual pathways.

    Architecture:
        U = X + MHSA(LN_1(X))
        Y = U + FFN(LN_2(U))
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        hidden_dim: int | None = None,
        mlp_ratio: float = 4.0,
        activation: str | BaseActivation = "gelu",
        norm_eps: float = 1e-5,
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
        if hidden_dim is not None and hidden_dim > 0:
            self.hidden_dim = hidden_dim
        else:
            self.hidden_dim = max(1, round(embed_dim * mlp_ratio))

        self.norm_eps = norm_eps
        self.use_bias = bias

        # Block-specific seed derivation
        seed_attn = seed
        seed_ffn = seed + 1007

        self.ln1 = LayerNorm(normalized_shape=embed_dim, eps=norm_eps, affine=True)
        self.attn = MultiHeadSelfAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            bias=bias,
            seed=seed_attn,
        )
        self.ln2 = LayerNorm(normalized_shape=embed_dim, eps=norm_eps, affine=True)
        self.ffn = TransformerFeedForward(
            in_features=embed_dim,
            hidden_dim=self.hidden_dim,
            bias=bias,
            activation=activation,
            seed=seed_ffn,
        )

        # Intermediate representations for inspection
        self.last_attention_weights: list[list[list[list[float]]]] | None = None
        self.last_ln1_out: list[list[list[float]]] | None = None
        self.last_attn_out: list[list[list[float]]] | None = None
        self.last_u: list[list[list[float]]] | None = None
        self.last_ln2_out: list[list[list[float]]] | None = None
        self.last_ffn_out: list[list[list[float]]] | None = None
        self.last_output: list[list[list[float]]] | None = None

        # Cached tensors for backward pass
        self._cached_x: list[list[list[float]]] | None = None
        self._cached_u: list[list[list[float]]] | None = None

    def zero_grad(self) -> None:
        """Clear parameter gradients in all constituent sub-layers."""
        self.ln1.zero_grad()
        self.attn.zero_grad()
        self.ln2.zero_grad()
        self.ffn.zero_grad()

    def get_parameters(self) -> dict[str, Any]:
        """Return all trainable parameter tensors with hierarchical naming."""
        params: dict[str, Any] = {}
        for k, v in self.ln1.get_parameters().items():
            params[f"ln1.{k}"] = v
        for k, v in self.attn.get_parameters().items():
            params[f"attn.{k}"] = v
        for k, v in self.ln2.get_parameters().items():
            params[f"ln2.{k}"] = v
        for k, v in self.ffn.get_parameters().items():
            params[f"ffn.{k}"] = v
        return params

    def set_parameters(self, params: dict[str, Any]) -> None:
        """Load trainable parameter values from hierarchical mapping."""
        ln1_p = {k[4:]: v for k, v in params.items() if k.startswith("ln1.")}
        attn_p = {k[5:]: v for k, v in params.items() if k.startswith("attn.")}
        ln2_p = {k[4:]: v for k, v in params.items() if k.startswith("ln2.")}
        ffn_p = {k[4:]: v for k, v in params.items() if k.startswith("ffn.")}

        if ln1_p:
            self.ln1.set_parameters(ln1_p)
        if attn_p:
            self.attn.set_parameters(attn_p)
        if ln2_p:
            self.ln2.set_parameters(ln2_p)
        if ffn_p:
            self.ffn.set_parameters(ffn_p)

    def get_gradients(self) -> dict[str, Any]:
        """Return computed parameter gradients."""
        grads: dict[str, Any] = {}
        for k, v in self.ln1.get_gradients().items():
            grads[f"ln1.{k}"] = v
        for k, v in self.attn.get_gradients().items():
            grads[f"attn.{k}"] = v
        for k, v in self.ln2.get_gradients().items():
            grads[f"ln2.{k}"] = v
        for k, v in self.ffn.get_gradients().items():
            grads[f"ffn.{k}"] = v
        return grads

    def get_state(self) -> dict[str, Any]:
        """Return non-trainable persistent state."""
        return {}

    def set_state(self, state: dict[str, Any]) -> None:
        """Restore non-trainable state."""
        pass

    def forward(self, inputs: Any, mask: Any = None) -> list[list[list[float]]]:
        """Compute pre-norm transformer block forward pass producing [N, T, D]."""
        x_3d = ensure_3d_tensor(inputs)
        n_samples = len(x_3d)
        seq_len = len(x_3d[0])
        d_in = len(x_3d[0][0])

        if d_in != self.embed_dim:
            raise ValidationError(
                f"Input dimension ({d_in}) does not match block embed_dim "
                f"({self.embed_dim})."
            )

        self._cached_x = x_3d

        # 1. Branch 1: Pre-norm attention and residual addition
        ln1_out = self.ln1.forward(x_3d)
        self.last_ln1_out = ln1_out

        attn_out = self.attn.forward(ln1_out, mask=mask)
        self.last_attn_out = attn_out
        self.last_attention_weights = self.attn.last_attention_weights

        # U = X + Attn(LN1(X))
        u_3d: list[list[list[float]]] = []
        for n in range(n_samples):
            sample_u: list[list[float]] = []
            for t in range(seq_len):
                x_vec = x_3d[n][t]
                a_vec = attn_out[n][t]
                u_row = [x_vec[d] + a_vec[d] for d in range(self.embed_dim)]
                sample_u.append(u_row)
            u_3d.append(sample_u)

        self.last_u = u_3d
        self._cached_u = u_3d

        # 2. Branch 2: Pre-norm FFN and residual addition
        ln2_out = self.ln2.forward(u_3d)
        self.last_ln2_out = ln2_out

        ffn_out = self.ffn.forward(ln2_out)
        self.last_ffn_out = ffn_out

        # Y = U + FFN(LN2(U))
        y_3d: list[list[list[float]]] = []
        for n in range(n_samples):
            sample_y: list[list[float]] = []
            for t in range(seq_len):
                u_vec = u_3d[n][t]
                f_vec = ffn_out[n][t]
                y_row = [u_vec[d] + f_vec[d] for d in range(self.embed_dim)]
                sample_y.append(y_row)
            y_3d.append(sample_y)

        self.last_output = y_3d
        return y_3d

    def backward(self, d_out: Any) -> list[list[list[float]]]:
        """Compute exact analytical backpropagation through both residual pathways."""
        if self._cached_x is None or self._cached_u is None:
            raise ValidationError("Cannot run backward before forward pass.")

        dy_3d = ensure_3d_tensor(d_out)
        n_samples = len(self._cached_x)
        seq_len = len(self._cached_x[0])

        if (
            len(dy_3d) != n_samples
            or len(dy_3d[0]) != seq_len
            or len(dy_3d[0][0]) != self.embed_dim
        ):
            raise ValidationError(
                f"d_out shape mismatch: expected ({n_samples}, {seq_len}, "
                f"{self.embed_dim})."
            )

        # 1. Second residual split: Y = U + FFN(LN2(U))
        # dU_direct = dY
        # dU_ffn_path = LN2_backward(FFN_backward(dY))
        # dU = dU_direct + dU_ffn_path
        d_ffn_out = self.ffn.backward(dy_3d)
        d_ln2_out = self.ln2.backward(d_ffn_out)

        du_3d: list[list[list[float]]] = []
        for n in range(n_samples):
            sample_du: list[list[float]] = []
            for t in range(seq_len):
                dy_vec = dy_3d[n][t]
                dln2_vec = d_ln2_out[n][t]
                du_row = [dy_vec[d] + dln2_vec[d] for d in range(self.embed_dim)]
                sample_du.append(du_row)
            du_3d.append(sample_du)

        # 2. First residual split: U = X + MHSA(LN1(X))
        # dX_direct = dU
        # dX_attn_path = LN1_backward(MHSA_backward(dU))
        # dX = dX_direct + dX_attn_path
        d_attn_out = self.attn.backward(du_3d)
        d_ln1_out = self.ln1.backward(d_attn_out)

        dx_3d: list[list[list[float]]] = []
        for n in range(n_samples):
            sample_dx: list[list[float]] = []
            for t in range(seq_len):
                du_vec = du_3d[n][t]
                dln1_vec = d_ln1_out[n][t]
                dx_row = [du_vec[d] + dln1_vec[d] for d in range(self.embed_dim)]
                sample_dx.append(dx_row)
            dx_3d.append(sample_dx)

        return dx_3d


class TransformerEncoder:
    """Stacked Transformer Encoder composed of depth L TransformerEncoderBlocks."""

    def __init__(
        self,
        depth: int,
        embed_dim: int,
        num_heads: int,
        hidden_dim: int | None = None,
        mlp_ratio: float = 4.0,
        activation: str | BaseActivation = "gelu",
        norm_eps: float = 1e-5,
        bias: bool = True,
        seed: int = 42,
    ) -> None:
        if depth <= 0:
            raise ValidationError(f"depth must be positive, got {depth}.")

        self.depth = depth
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.mlp_ratio = mlp_ratio
        self.norm_eps = norm_eps
        self.use_bias = bias

        self.blocks: list[TransformerEncoderBlock] = []
        for idx in range(depth):
            block_seed = seed + idx * 1000 + 17
            block = TransformerEncoderBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                hidden_dim=hidden_dim,
                mlp_ratio=mlp_ratio,
                activation=activation,
                norm_eps=norm_eps,
                bias=bias,
                seed=block_seed,
            )
            self.blocks.append(block)

        self._cached_inputs: list[list[list[float]]] | None = None
        self._cached_block_outputs: list[list[list[list[float]]]] = []

    def zero_grad(self) -> None:
        """Clear parameter gradients in all stacked encoder blocks."""
        for block in self.blocks:
            block.zero_grad()

    def get_parameters(self) -> dict[str, Any]:
        """Return all trainable parameters prefixed with block index."""
        params: dict[str, Any] = {}
        for idx, block in enumerate(self.blocks):
            for k, v in block.get_parameters().items():
                params[f"blocks.{idx}.{k}"] = v
        return params

    def set_parameters(self, params: dict[str, Any]) -> None:
        """Load trainable parameters from hierarchical mapping."""
        for idx, block in enumerate(self.blocks):
            prefix = f"blocks.{idx}."
            block_p = {
                k[len(prefix) :]: v for k, v in params.items() if k.startswith(prefix)
            }
            if block_p:
                block.set_parameters(block_p)

    def get_gradients(self) -> dict[str, Any]:
        """Return computed parameter gradients across all blocks."""
        grads: dict[str, Any] = {}
        for idx, block in enumerate(self.blocks):
            for k, v in block.get_gradients().items():
                grads[f"blocks.{idx}.{k}"] = v
        return grads

    def get_state(self) -> dict[str, Any]:
        """Return non-trainable state."""
        return {}

    def set_state(self, state: dict[str, Any]) -> None:
        """Restore non-trainable state."""
        pass

    def forward(self, inputs: Any, mask: Any = None) -> list[list[list[float]]]:
        """Pass input sequence through all encoder blocks sequentially."""
        x = ensure_3d_tensor(inputs)
        self._cached_inputs = x
        self._cached_block_outputs = []

        current_x = x
        for block in self.blocks:
            current_x = block.forward(current_x, mask=mask)
            self._cached_block_outputs.append(current_x)

        return current_x

    def backward(self, d_out: Any) -> list[list[list[float]]]:
        """Propagate gradients through all encoder blocks in reverse order."""
        if self._cached_inputs is None:
            raise ValidationError("Cannot run backward before forward pass.")

        current_dout = ensure_3d_tensor(d_out)
        for block in reversed(self.blocks):
            current_dout = block.backward(current_dout)

        return current_dout

    def get_attention_weights(
        self,
    ) -> list[list[list[list[list[float]]]]]:
        """Return non-mutating copy of attention weights [L, N, H, T, T]."""
        weights_per_layer: list[list[list[list[list[float]]]]] = []
        for block in self.blocks:
            if block.last_attention_weights is not None:
                weights_per_layer.append(copy.deepcopy(block.last_attention_weights))
            else:
                weights_per_layer.append([])
        return weights_per_layer


class VisionTransformer(BaseVisionModel):
    """Complete Vision Transformer (ViT) model for visual representations.

    Pipeline:
        X (N, C, H, W)
        -> PatchExtractor (N, T, D_patch)
        -> PatchEmbedding (N, T, D_model)
        -> ClassToken (N, T+1, D_model)
        -> PositionalEmbedding (N, T+1, D_model)
        -> TransformerEncoder (N, T+1, D_model)
        -> Final LayerNorm (N, T+1, D_model)
        -> CLS Slice (N, D_model)
        -> Linear Classification Head (N, num_classes)
    """

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
        self.num_classes_val: int = spec.num_classes

        # 1. Input Spatial Dimensions
        if len(spec.input_shape) != 3:
            raise ValidationError(
                f"Expected input_shape (C, H, W), got {spec.input_shape}."
            )
        self.channels, self.image_height, self.image_width = spec.input_shape

        # 2. Hyperparameters
        patch_size_raw = spec.hyperparameters.get("patch_size", 4)
        if isinstance(patch_size_raw, (list, tuple)):
            self.patch_size: tuple[int, int] = (
                int(patch_size_raw[0]),
                int(patch_size_raw[1]),
            )
        else:
            self.patch_size = (int(patch_size_raw), int(patch_size_raw))

        self.embed_dim = int(spec.hyperparameters.get("embed_dim", 64))
        self.num_heads = int(spec.hyperparameters.get("num_heads", 4))
        self.depth = int(spec.hyperparameters.get("depth", 4))
        self.mlp_ratio = float(spec.hyperparameters.get("mlp_ratio", 4.0))
        self.norm_eps = float(spec.hyperparameters.get("norm_eps", 1e-5))
        self.act_name = str(spec.hyperparameters.get("activation", "gelu"))
        self.bias = bool(spec.hyperparameters.get("bias", True))
        self.seed = seed

        # 3. Component Instantiation
        self.geometry = PatchGeometry.create(
            image_size=(self.image_height, self.image_width),
            patch_size=self.patch_size,
            channels=self.channels,
        )

        self.patch_extractor = ImagePatchExtractor(geometry=self.geometry)

        self.patch_embed = PatchEmbedding(
            in_features=self.geometry.flattened_patch_dimension,
            embed_dim=self.embed_dim,
            bias=self.bias,
            seed=seed + 1,
        )

        self.cls_token = ClassToken(
            embed_dim=self.embed_dim,
            seed=seed + 2,
            init_std=0.02,
        )

        self.pos_embed = LearnablePositionalEmbedding(
            num_positions=self.geometry.total_patches + 1,
            embed_dim=self.embed_dim,
            seed=seed + 3,
            init_std=0.02,
        )

        self.encoder = TransformerEncoder(
            depth=self.depth,
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            mlp_ratio=self.mlp_ratio,
            activation=self.act_name,
            norm_eps=self.norm_eps,
            bias=self.bias,
            seed=seed + 10,
        )

        self.norm = LayerNorm(
            normalized_shape=self.embed_dim,
            eps=self.norm_eps,
            affine=True,
        )

        # 4. Classification Head: W_head [D_model, num_classes], b_head [num_classes]
        rng = random.Random(seed + 99)
        std_head = math.sqrt(2.0 / float(self.embed_dim + self.num_classes_val))
        self.classifier_w: list[list[float]] = [
            [rng.gauss(0.0, std_head) for _ in range(self.num_classes_val)]
            for _ in range(self.embed_dim)
        ]
        self.classifier_b: list[float] = [0.0] * self.num_classes_val

        self.zero_grad()

        # Cache for intermediate representations and backward pass
        self._cached_input: Any = None
        self._cached_patches: list[list[list[float]]] | None = None
        self._cached_embeddings: list[list[list[float]]] | None = None
        self._cached_cls_prepended: list[list[list[float]]] | None = None
        self._cached_pos_added: list[list[list[float]]] | None = None
        self._cached_encoder_out: list[list[list[float]]] | None = None
        self._cached_norm_out: list[list[list[float]]] | None = None
        self._cached_cls_repr: list[list[float]] | None = None

    def zero_grad(self) -> None:
        """Reset all parameter gradient buffers."""
        self.patch_embed.zero_grad()
        self.cls_token.zero_grad()
        self.pos_embed.zero_grad()
        self.encoder.zero_grad()
        self.norm.zero_grad()

        self.grad_classifier_w: list[list[float]] = [
            [0.0] * self.num_classes_val for _ in range(self.embed_dim)
        ]
        self.grad_classifier_b: list[float] = [0.0] * self.num_classes_val

    def get_parameters(self) -> dict[str, Any]:
        """Discover and return all trainable model parameters."""
        params: dict[str, Any] = {}

        for k, v in self.patch_embed.get_parameters().items():
            params[f"patch_embed.{k}"] = v

        for k, v in self.cls_token.get_parameters().items():
            params[f"cls_token.{k}"] = v

        for k, v in self.pos_embed.get_parameters().items():
            params[f"pos_embed.{k}"] = v

        for k, v in self.encoder.get_parameters().items():
            params[f"encoder.{k}"] = v

        for k, v in self.norm.get_parameters().items():
            params[f"norm.{k}"] = v

        params["classifier.weights"] = copy.deepcopy(self.classifier_w)
        params["classifier.bias"] = copy.deepcopy(self.classifier_b)

        return params

    def set_parameters(self, params: dict[str, Any]) -> None:
        """Load trainable parameters from hierarchical mapping."""
        pe_p = {k[12:]: v for k, v in params.items() if k.startswith("patch_embed.")}
        cls_p = {k[10:]: v for k, v in params.items() if k.startswith("cls_token.")}
        pos_p = {k[10:]: v for k, v in params.items() if k.startswith("pos_embed.")}
        enc_p = {k[8:]: v for k, v in params.items() if k.startswith("encoder.")}
        norm_p = {k[5:]: v for k, v in params.items() if k.startswith("norm.")}

        if pe_p:
            self.patch_embed.set_parameters(pe_p)
        if cls_p:
            self.cls_token.set_parameters(cls_p)
        if pos_p:
            self.pos_embed.set_parameters(pos_p)
        if enc_p:
            self.encoder.set_parameters(enc_p)
        if norm_p:
            self.norm.set_parameters(norm_p)

        if "classifier.weights" in params:
            cw = params["classifier.weights"]
            if len(cw) != self.embed_dim or len(cw[0]) != self.num_classes_val:
                raise ValidationError(
                    f"classifier.weights shape mismatch: expected "
                    f"({self.embed_dim}, {self.num_classes_val}), got ({len(cw)}, "
                    f"{len(cw[0]) if cw else 0})."
                )
            self.classifier_w = copy.deepcopy(cw)

        if "classifier.bias" in params:
            cb = params["classifier.bias"]
            if len(cb) != self.num_classes_val:
                raise ValidationError(
                    f"classifier.bias mismatch: expected ({self.num_classes_val},), "
                    f"got ({len(cb)},)."
                )
            self.classifier_b = copy.deepcopy(cb)

    def get_gradients(self) -> dict[str, Any]:
        """Return computed parameter gradients."""
        grads: dict[str, Any] = {}

        for k, v in self.patch_embed.get_gradients().items():
            grads[f"patch_embed.{k}"] = v

        for k, v in self.cls_token.get_gradients().items():
            grads[f"cls_token.{k}"] = v

        for k, v in self.pos_embed.get_gradients().items():
            grads[f"pos_embed.{k}"] = v

        for k, v in self.encoder.get_gradients().items():
            grads[f"encoder.{k}"] = v

        for k, v in self.norm.get_gradients().items():
            grads[f"norm.{k}"] = v

        grads["classifier.weights"] = copy.deepcopy(self.grad_classifier_w)
        grads["classifier.bias"] = copy.deepcopy(self.grad_classifier_b)

        return grads

    def forward(self, inputs: Any) -> list[list[float]]:
        """Compute ViT forward pass producing logits [N, num_classes]."""
        x_4d = ensure_4d_tensor(inputs)
        self._cached_input = x_4d

        # 1. Patch Extraction: [N, C, H, W] -> [N, T, D_patch]
        patches = self.patch_extractor.forward(x_4d)
        self._cached_patches = patches

        # 2. Patch Embedding: [N, T, D_patch] -> [N, T, D_model]
        embeddings = self.patch_embed.forward(patches)
        self._cached_embeddings = embeddings

        # 3. Class Token Prepending: [N, T, D_model] -> [N, T+1, D_model]
        tokens_with_cls = self.cls_token.forward(embeddings)
        self._cached_cls_prepended = tokens_with_cls

        # 4. Positional Encoding: [N, T+1, D_model] -> [N, T+1, D_model]
        tokens_with_pos = self.pos_embed.forward(tokens_with_cls)
        self._cached_pos_added = tokens_with_pos

        # 5. Transformer Encoder: [N, T+1, D_model] -> [N, T+1, D_model]
        encoder_out = self.encoder.forward(tokens_with_pos)
        self._cached_encoder_out = encoder_out

        # 6. Final LayerNorm: [N, T+1, D_model] -> [N, T+1, D_model]
        norm_out = self.norm.forward(encoder_out)
        self._cached_norm_out = norm_out

        # 7. Extract CLS representation at index 0: [N, D_model]
        cls_repr: list[list[float]] = [list(sample[0]) for sample in norm_out]
        self._cached_cls_repr = cls_repr

        # 8. Classification Head: [N, D_model] -> [N, num_classes]
        n_samples = len(cls_repr)
        logits: list[list[float]] = []
        for n in range(n_samples):
            cls_vec = cls_repr[n]
            row_logits: list[float] = [0.0] * self.num_classes_val
            for c in range(self.num_classes_val):
                dot = (
                    sum(
                        cls_vec[d] * self.classifier_w[d][c]
                        for d in range(self.embed_dim)
                    )
                    + self.classifier_b[c]
                )
                row_logits[c] = dot
            logits.append(row_logits)

        return logits

    def backward(self, d_logits: list[list[float]]) -> None:
        """Propagate gradients from output logits all the way to input pixels."""
        if (
            self._cached_cls_repr is None
            or self._cached_norm_out is None
            or self._cached_input is None
        ):
            raise ValidationError("Cannot run backward before forward pass.")

        n_samples = len(self._cached_cls_repr)
        seq_len = self.geometry.total_patches + 1

        if len(d_logits) != n_samples or len(d_logits[0]) != self.num_classes_val:
            raise ValidationError(
                f"d_logits shape mismatch: expected ({n_samples}, "
                f"{self.num_classes_val})."
            )

        # 1. Head backward: d_cls = d_logits W_head^T, dW_head += CLS^T d_logits
        d_cls: list[list[float]] = []
        for n in range(n_samples):
            dlog_vec = d_logits[n]
            cls_vec = self._cached_cls_repr[n]
            dcls_vec = [0.0] * self.embed_dim

            for d in range(self.embed_dim):
                dcls_vec[d] = sum(
                    dlog_vec[c] * self.classifier_w[d][c]
                    for c in range(self.num_classes_val)
                )
                cls_d = cls_vec[d]
                for c in range(self.num_classes_val):
                    self.grad_classifier_w[d][c] += cls_d * dlog_vec[c]

            for c in range(self.num_classes_val):
                self.grad_classifier_b[c] += dlog_vec[c]

            d_cls.append(dcls_vec)

        # 2. Embed d_cls into sequence gradient d_norm_out [N, T+1, D_model]
        d_norm_out: list[list[list[float]]] = []
        for n in range(n_samples):
            sample_d = [d_cls[n]] + [[0.0] * self.embed_dim for _ in range(seq_len - 1)]
            d_norm_out.append(sample_d)

        # 3. Final LayerNorm backward: d_encoder_out = LN_backward(d_norm_out)
        d_encoder_out = self.norm.backward(d_norm_out)

        # 4. Stacked Encoder backward: d_pos = Encoder_backward(d_encoder_out)
        d_pos_added = self.encoder.backward(d_encoder_out)

        # 5. Positional Embedding backward: d_cls_prepended = Pos_backward(d_pos)
        d_cls_prepended = self.pos_embed.backward(d_pos_added)

        # 6. Class Token backward: d_embeddings = CLS_backward(d_cls_prepended)
        d_embeddings = self.cls_token.backward(d_cls_prepended)

        # 7. Patch Embedding backward: d_patches = PatchEmbed_backward(d_embeddings)
        d_patches = self.patch_embed.backward(d_embeddings)

        # 8. Patch Extractor backward: dX = Extractor_backward(d_patches)
        _ = self.patch_extractor.backward(d_patches)

    def extract_representations(
        self, inputs: Any, layer: str = "cls_representation"
    ) -> Any:
        """Extract intermediate representations at specified architectural depth."""
        layer_norm_name = layer.strip().lower()

        # Run forward pass if not already computed for these exact inputs
        logits = self.forward(inputs)

        if layer_norm_name in ("input", "input_spatial"):
            return copy.deepcopy(self._cached_input)
        if layer_norm_name == "patches":
            return copy.deepcopy(self._cached_patches)
        if layer_norm_name in ("patch_embeddings", "embeddings"):
            return copy.deepcopy(self._cached_embeddings)
        if layer_norm_name == "tokens_pre_position":
            return copy.deepcopy(self._cached_cls_prepended)
        if layer_norm_name in ("tokens_post_position", "tokens_with_pos"):
            return copy.deepcopy(self._cached_pos_added)
        if layer_norm_name in ("final_tokens", "encoder_final"):
            return copy.deepcopy(self._cached_norm_out)
        if layer_norm_name in (
            "cls_representation",
            "cls",
            "final_hidden",
            "representation",
        ):
            return copy.deepcopy(self._cached_cls_repr)
        if layer_norm_name in ("patch_tokens", "patch_token_representations"):
            if self._cached_norm_out is not None:
                return [
                    [list(tok) for tok in sample[1:]]
                    for sample in self._cached_norm_out
                ]
            return []
        if layer_norm_name in ("logits", "output"):
            return copy.deepcopy(logits)
        if layer_norm_name in ("attention_weights", "attention"):
            return self.get_attention_weights()

        # Check block-specific output or attention: e.g. "encoder_0_output", "encoder_0"
        if layer_norm_name.startswith("encoder_"):
            parts = layer_norm_name.split("_")
            try:
                b_idx = int(parts[1])
                if 0 <= b_idx < len(self.encoder.blocks):
                    block = self.encoder.blocks[b_idx]
                    if "attn" in layer_norm_name or "attention" in layer_norm_name:
                        return copy.deepcopy(block.last_attention_weights)
                    return copy.deepcopy(block.last_output)
            except (ValueError, IndexError):
                pass

        raise ValidationError(
            f"Unsupported representation layer '{layer}'. Supported: 'input_spatial', "
            f"'patches', 'patch_embeddings', 'tokens_pre_position', "
            f"'tokens_post_position', 'encoder_<k>_output', 'final_tokens', "
            f"'cls_representation', 'patch_tokens', 'logits'."
        )

    def get_attention_weights(
        self,
    ) -> list[list[list[list[list[float]]]]]:
        """Return attention matrices [L, N, H, T+1, T+1] across all blocks."""
        return self.encoder.get_attention_weights()
