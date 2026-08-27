"""Convolutional Neural Network baseline model with spatial representations."""

import copy
import random
from typing import Any

from prism.core.enums import ModelFamily, TaskType
from prism.core.errors import ValidationError
from prism.models.activations import BaseActivation, get_activation
from prism.models.base import BaseVisionModel
from prism.models.convolution import Conv2D
from prism.models.initialization import (
    initialize_linear_parameters,
    initialize_mlp_parameters,
)
from prism.models.pooling import MaxPool2D
from prism.models.spatial import (
    compute_conv2d_output_shape,
    compute_pool2d_output_shape,
    compute_receptive_field,
    ensure_4d_tensor,
    normalize_spatial_pair,
)
from prism.models.specifications import ModelSpecification


class ConvolutionalNeuralNetwork(BaseVisionModel):
    """Configurable Convolutional Neural Network baseline architecture."""

    def __init__(self, spec: ModelSpecification, seed: int = 42) -> None:
        super().__init__(spec=spec)
        self.seed = seed

        if spec.family != ModelFamily.CNN:
            raise ValidationError(
                f"CNN model requires ModelFamily.CNN, got '{spec.family}'."
            )
        if TaskType.CLASSIFICATION not in spec.compatible_tasks:
            raise ValidationError(
                "CNN model requires classification task compatibility."
            )
        if spec.num_classes is None or spec.num_classes <= 0:
            raise ValidationError(
                f"num_classes must be positive for CNN, got {spec.num_classes}."
            )

        # 1. Parse Input Shape (C, H, W)
        if len(spec.input_shape) == 3:
            self.in_channels, self.in_height, self.in_width = spec.input_shape
        elif len(spec.input_shape) == 1:
            raise ValidationError(
                "CNN model requires a 3D input_shape (C, H, W), got 1D flat dimension."
            )
        else:
            raise ValidationError(
                f"Expected 3D input_shape (C, H, W), got {spec.input_shape}."
            )

        # 2. Extract Hyperparameters
        hp = spec.hyperparameters
        self.conv_channels: list[int] = hp.get("conv_channels", [16, 32])
        if not self.conv_channels:
            raise ValidationError("conv_channels cannot be empty for CNN.")
        for idx, ch in enumerate(self.conv_channels):
            if ch <= 0:
                raise ValidationError(
                    f"conv_channels[{idx}] must be positive, got {ch}."
                )

        num_blocks = len(self.conv_channels)
        self.activation_name: str = hp.get("activation", "relu").lower()
        self.dropout_rate: float = float(hp.get("dropout", 0.0))
        if self.dropout_rate < 0.0 or self.dropout_rate >= 1.0:
            raise ValidationError(
                f"Dropout probability must be in [0.0, 1.0), got {self.dropout_rate}."
            )

        self.classifier_hidden_dims: list[int] = hp.get("classifier_hidden_dims", [])

        raw_kernel_sizes = hp.get("kernel_sizes", 3)
        raw_strides = hp.get("strides", 1)
        raw_paddings = hp.get("paddings", 1)
        raw_pool_sizes = hp.get("pool_sizes", 2)
        raw_pool_strides = hp.get("pool_strides", 2)

        def _to_list(val: Any, length: int) -> list[Any]:
            if isinstance(val, (list, tuple)):
                if len(val) != length:
                    raise ValidationError(
                        f"Expected list of length {length}, got {len(val)}."
                    )
                return list(val)
            return [val] * length

        kernel_sizes = _to_list(raw_kernel_sizes, num_blocks)
        strides = _to_list(raw_strides, num_blocks)
        paddings = _to_list(raw_paddings, num_blocks)
        pool_sizes = _to_list(raw_pool_sizes, num_blocks)
        pool_strides = _to_list(raw_pool_strides, num_blocks)

        # 3. Construct Convolutional Blocks and Track Shapes
        self.conv_layers: list[Conv2D] = []
        self.activations: list[BaseActivation] = []
        self.pool_layers: list[MaxPool2D | None] = []
        self.stage_rf_tracking: list[tuple[int, int]] = []

        cur_c = self.in_channels
        cur_h = self.in_height
        cur_w = self.in_width

        for b_idx in range(num_blocks):
            out_c = self.conv_channels[b_idx]
            k_s = kernel_sizes[b_idx]
            s_s = strides[b_idx]
            p_s = paddings[b_idx]
            p_size = pool_sizes[b_idx]
            p_stride = pool_strides[b_idx]

            block_seed = (seed * 10007 + b_idx * 31) & 0x7FFFFFFF
            conv = Conv2D(
                in_channels=cur_c,
                out_channels=out_c,
                kernel_size=k_s,
                stride=s_s,
                padding=p_s,
                bias=True,
                seed=block_seed,
                activation=self.activation_name,
            )
            self.conv_layers.append(conv)
            self.activations.append(get_activation(self.activation_name))

            k_h, k_w = normalize_spatial_pair(k_s, "kernel_size")
            s_h, s_w = normalize_spatial_pair(s_s, "stride")
            self.stage_rf_tracking.append((max(k_h, k_w), max(s_h, s_w)))

            cur_h, cur_w = compute_conv2d_output_shape(
                input_height=cur_h,
                input_width=cur_w,
                kernel_size=k_s,
                stride=s_s,
                padding=p_s,
            )

            # Optional MaxPool
            if p_size is not None and p_size > 0:
                pool = MaxPool2D(
                    kernel_size=p_size,
                    stride=p_stride if p_stride is not None else p_size,
                    padding=0,
                )
                self.pool_layers.append(pool)
                pk_h, pk_w = normalize_spatial_pair(p_size, "pool_size")
                ps_h, ps_w = normalize_spatial_pair(
                    p_stride if p_stride is not None else p_size, "pool_stride"
                )
                self.stage_rf_tracking.append((max(pk_h, pk_w), max(ps_h, ps_w)))

                cur_h, cur_w = compute_pool2d_output_shape(
                    input_height=cur_h,
                    input_width=cur_w,
                    kernel_size=p_size,
                    stride=p_stride if p_stride is not None else p_size,
                    padding=0,
                )
            else:
                self.pool_layers.append(None)

            cur_c = out_c

        self.final_spatial_shape = (cur_c, cur_h, cur_w)
        self.flattened_dim = cur_c * cur_h * cur_w

        if self.flattened_dim <= 0:
            raise ValidationError(
                f"Calculated non-positive flattened dimension: {self.flattened_dim}"
            )

        # 4. Construct Classifier Head
        cls_seed = (seed * 1000003 + 99991) & 0x7FFFFFFF
        if not self.classifier_hidden_dims:
            w_cls, b_cls = initialize_linear_parameters(
                in_features=self.flattened_dim,
                num_classes=spec.num_classes,
                seed=cls_seed,
            )
            self.fc_weights: list[list[list[float]]] = [w_cls]
            self.fc_biases: list[list[float]] = [b_cls]
        else:
            w_mlp, b_mlp = initialize_mlp_parameters(
                in_features=self.flattened_dim,
                hidden_dims=self.classifier_hidden_dims,
                num_classes=spec.num_classes,
                seed=cls_seed,
                activation=self.activation_name,
            )
            self.fc_weights = w_mlp
            self.fc_biases = b_mlp

        self.zero_grad()

        # Step counter for deterministic training dropout
        self._step_counter = 0

        # Runtime Forward Caches for Backprop and Representations
        self._cached_inputs: Any = None
        self._cached_block_states: list[dict[str, Any]] = []
        self._cached_flat: list[list[float]] | None = None
        self._cached_fc_states: list[dict[str, Any]] = []

    @property
    def receptive_field(self) -> int:
        """Effective receptive field size across all conv and pooling stages."""
        rf, _ = compute_receptive_field(self.stage_rf_tracking)
        return rf

    def _flatten_spatial(self, x: list[list[list[list[float]]]]) -> list[list[float]]:
        """Flatten 4D spatial feature tensor [N, C, H, W] to 2D vector matrix [N, D]."""
        n_samples = len(x)
        c_channels = len(x[0])
        h_len = len(x[0][0])
        w_len = len(x[0][0][0])

        flat: list[list[float]] = []
        for n in range(n_samples):
            row: list[float] = []
            for c in range(c_channels):
                for h in range(h_len):
                    for w in range(w_len):
                        row.append(x[n][c][h][w])
            flat.append(row)
        return flat

    def _unflatten_spatial(
        self, flat: list[list[float]], c: int, h: int, w: int
    ) -> list[list[list[list[float]]]]:
        """Unflatten 2D gradient matrix [N, D] back to 4D tensor [N, C, H, W]."""
        n_samples = len(flat)
        out_4d: list[list[list[list[float]]]] = []

        for n in range(n_samples):
            sample_4d: list[list[list[float]]] = []
            idx = 0
            for _ in range(c):
                channel_2d: list[list[float]] = []
                for _ in range(h):
                    row: list[float] = []
                    for _ in range(w):
                        row.append(flat[n][idx])
                        idx += 1
                    channel_2d.append(row)
                sample_4d.append(channel_2d)
            out_4d.append(sample_4d)
        return out_4d

    def _convert_input_to_4d(self, inputs: Any) -> list[list[list[list[float]]]]:
        """Safely convert arbitrary input batches (nested 4D or flattened 2D) to 4D."""
        if not isinstance(inputs, (list, tuple)) or not inputs:
            raise ValidationError("Input batch cannot be empty.")

        # Check if already 4D or 3D
        first = inputs[0]
        if isinstance(first, (list, tuple)) and first:
            second = first[0]
            if isinstance(second, (list, tuple)) and second:
                third = second[0]
                if isinstance(third, (list, tuple)):
                    # 4D: [N, C, H, W]
                    return ensure_4d_tensor(inputs)
                else:
                    # 3D: Single sample [C, H, W]
                    return ensure_4d_tensor(inputs)
            else:
                # 2D matrix: [N, D] flattened samples
                # Reshape each sample to [C, H, W]
                n_samples = len(inputs)
                expected_dim = self.in_channels * self.in_height * self.in_width
                out_4d: list[list[list[list[float]]]] = []

                for n in range(n_samples):
                    flat_sample = inputs[n]
                    if len(flat_sample) != expected_dim:
                        raise ValidationError(
                            f"Flattened sample has {len(flat_sample)} features, "
                            f"expected {expected_dim} "
                            f"({self.in_channels}x{self.in_height}x{self.in_width})."
                        )
                    sample_3d = self._unflatten_spatial(
                        [flat_sample],
                        self.in_channels,
                        self.in_height,
                        self.in_width,
                    )[0]
                    out_4d.append(sample_3d)
                return out_4d

        raise ValidationError("Unsupported input data format for CNN.")

    def forward(self, inputs: Any) -> list[list[float]]:
        """Execute CNN forward pass producing raw logits [N, num_classes]."""
        x_4d = self._convert_input_to_4d(inputs)
        self._cached_inputs = x_4d
        self._cached_block_states = []

        if self.is_training:
            self._step_counter += 1

        cur_tensor = x_4d
        num_blocks = len(self.conv_layers)

        # Pass through convolutional blocks
        for b_idx in range(num_blocks):
            conv = self.conv_layers[b_idx]
            act = self.activations[b_idx]
            pool = self.pool_layers[b_idx]

            # 1. Conv2D
            conv_out = conv.forward(cur_tensor)

            # 2. Activation
            act_out = act.forward(conv_out)

            # 3. Optional Pooling
            if pool is not None:
                pool_out = pool.forward(act_out)
                block_out = pool_out
            else:
                pool_out = None
                block_out = act_out

            self._cached_block_states.append(
                {
                    "conv_pre": conv_out,
                    "conv_post": act_out,
                    "pool_post": pool_out,
                    "block_out": block_out,
                }
            )
            cur_tensor = block_out

        # Flatten final spatial representation
        final_flat = self._flatten_spatial(cur_tensor)
        self._cached_flat = final_flat

        # Pass through Fully Connected Classifier Head
        self._cached_fc_states = []
        cur_fc = final_flat
        num_fc_layers = len(self.fc_weights)

        for l_idx in range(num_fc_layers):
            is_output_layer = l_idx == (num_fc_layers - 1)
            w_mat = self.fc_weights[l_idx]
            b_vec = self.fc_biases[l_idx]

            n_samples = len(cur_fc)
            in_dim = len(w_mat)
            out_dim = len(b_vec)

            # Linear projection: Z = H @ W + b
            z_mat: list[list[float]] = []
            for n in range(n_samples):
                row_z: list[float] = []
                for j in range(out_dim):
                    val = b_vec[j]
                    for i in range(in_dim):
                        val += cur_fc[n][i] * w_mat[i][j]
                    row_z.append(val)
                z_mat.append(row_z)

            if is_output_layer:
                self._cached_fc_states.append(
                    {
                        "input": cur_fc,
                        "z": z_mat,
                        "a": z_mat,
                        "dropout_mask": None,
                    }
                )
                return z_mat

            # Hidden FC Layer: Activation
            act = get_activation(self.activation_name)
            a_mat = act.forward(z_mat)

            # Dropout in hidden FC layer
            dropout_mask: list[list[float]] | None = None
            if self.is_training and self.dropout_rate > 0.0:
                p_keep = 1.0 - self.dropout_rate
                scale = 1.0 / p_keep
                drop_seed = (
                    (self.seed * 1000003) ^ (l_idx * 10007) ^ (self._step_counter * 31)
                ) & 0x7FFFFFFF
                rng = random.Random(drop_seed)

                dropout_mask = []
                for n in range(n_samples):
                    mask_row: list[float] = []
                    for j in range(out_dim):
                        if rng.random() < p_keep:
                            mask_row.append(scale)
                            a_mat[n][j] *= scale
                        else:
                            mask_row.append(0.0)
                            a_mat[n][j] = 0.0
                    dropout_mask.append(mask_row)

            self._cached_fc_states.append(
                {
                    "input": cur_fc,
                    "z": z_mat,
                    "a": a_mat,
                    "dropout_mask": dropout_mask,
                }
            )
            cur_fc = a_mat

        raise ValidationError("Classifier head produced empty output.")

    def backward(self, d_logits: list[list[float]]) -> None:
        """Propagate loss derivatives backwards through classifier and blocks."""
        if (
            self._cached_inputs is None
            or self._cached_flat is None
            or not self._cached_block_states
            or not self._cached_fc_states
        ):
            raise ValidationError("Cannot perform backward pass before forward pass.")

        n_samples = len(d_logits)
        num_fc = len(self.fc_weights)

        # 1. Backprop through Fully Connected Classifier Head
        d_out = d_logits

        for l_idx in reversed(range(num_fc)):
            is_output_layer = l_idx == (num_fc - 1)
            fc_state = self._cached_fc_states[l_idx]
            h_in = fc_state["input"]
            w_mat = self.fc_weights[l_idx]

            in_dim = len(w_mat)
            out_dim = len(w_mat[0])

            if is_output_layer:
                d_z = d_out
            else:
                d_a = d_out
                # Apply cached dropout mask
                mask = fc_state["dropout_mask"]
                if mask is not None:
                    d_a = [
                        [d_a[n][j] * mask[n][j] for j in range(len(d_a[0]))]
                        for n in range(len(d_a))
                    ]
                act = get_activation(self.activation_name)
                d_z = act.backward(fc_state["z"], d_a)

            # Accumulate FC weight and bias gradients
            for n in range(n_samples):
                for j in range(out_dim):
                    dz_val = d_z[n][j]
                    self.grad_fc_biases[l_idx][j] += dz_val
                    for i in range(in_dim):
                        self.grad_fc_weights[l_idx][i][j] += dz_val * h_in[n][i]

            # Compute dH_in for previous FC layer
            d_h_prev: list[list[float]] = []
            for n in range(n_samples):
                row_dh: list[float] = []
                for i in range(in_dim):
                    accum = 0.0
                    for j in range(out_dim):
                        accum += d_z[n][j] * w_mat[i][j]
                    row_dh.append(accum)
                d_h_prev.append(row_dh)

            d_out = d_h_prev

        # d_out is now gradient w.r.t flattened final spatial representation: [N, D]
        # Reshape to 4D tensor [N, C_last, H_last, W_last]
        c_last, h_last, w_last = self.final_spatial_shape
        d_spatial = self._unflatten_spatial(d_out, c_last, h_last, w_last)

        # 2. Backprop through Convolutional Blocks in reverse
        num_blocks = len(self.conv_layers)
        cur_d_spatial = d_spatial

        for b_idx in reversed(range(num_blocks)):
            conv = self.conv_layers[b_idx]
            act = self.activations[b_idx]
            pool = self.pool_layers[b_idx]
            block_state = self._cached_block_states[b_idx]

            # Backward Pool
            d_act = pool.backward(cur_d_spatial) if pool is not None else cur_d_spatial

            # Backward Activation
            d_conv = act.backward(block_state["conv_pre"], d_act)

            # Backward Conv2D
            d_prev_block = conv.backward(d_conv)
            cur_d_spatial = d_prev_block

    def extract_representations(self, inputs: Any, layer: str = "final_hidden") -> Any:
        """Extract intermediate features or spatial feature maps in evaluation mode."""
        layer_norm = layer.strip().lower()

        # Temporarily switch to eval mode to prevent dropout and gradient updates
        was_training = self.is_training
        self.eval()

        try:
            # 1. Input layer
            if layer_norm in ("input", "input_spatial", "input_image"):
                x_4d = self._convert_input_to_4d(inputs)
                return x_4d

            if layer_norm in ("input_flat", "input_flattened"):
                x_4d = self._convert_input_to_4d(inputs)
                return self._flatten_spatial(x_4d)

            # Forward pass to populate caches
            logits = self.forward(inputs)

            # 2. Block representations
            for b_idx in range(len(self.conv_layers)):
                b_state = self._cached_block_states[b_idx]

                if layer_norm == f"conv_{b_idx}_pre":
                    return b_state["conv_pre"]
                if layer_norm in (f"conv_{b_idx}", f"conv_{b_idx}_post"):
                    return b_state["conv_post"]
                if layer_norm in (f"pool_{b_idx}", f"block_{b_idx}"):
                    return b_state["block_out"]

            # 3. Final spatial feature map
            if layer_norm in ("final_spatial", "spatial_features"):
                return self._cached_block_states[-1]["block_out"]

            # 4. Final vector representation (flattened spatial or penultimate FC)
            if layer_norm in (
                "final_hidden",
                "final_representation",
                "embedding",
            ):
                if len(self.fc_weights) == 1:
                    return self._cached_flat
                else:
                    return self._cached_fc_states[-2]["a"]

            # 5. Logits
            if layer_norm in ("logits", "output"):
                return logits

            valid_layers = (
                ["input", "input_flat", "final_spatial", "final_hidden", "logits"]
                + [f"conv_{i}_pre" for i in range(len(self.conv_layers))]
                + [f"conv_{i}" for i in range(len(self.conv_layers))]
                + [f"pool_{i}" for i in range(len(self.conv_layers))]
            )
            raise ValidationError(
                f"Unknown layer '{layer}' for ConvolutionalNeuralNetwork. "
                f"Valid layers: {valid_layers}"
            )
        finally:
            if was_training:
                self.train()

    def zero_grad(self) -> None:
        """Clear all gradients in conv layers and classifier head."""
        for conv in self.conv_layers:
            conv.zero_grad()

        self.grad_fc_weights: list[list[list[float]]] = [
            [[0.0 for _ in range(len(w[0]))] for _ in range(len(w))]
            for w in self.fc_weights
        ]
        self.grad_fc_biases: list[list[float]] = [
            [0.0 for _ in range(len(b))] for b in self.fc_biases
        ]

    def get_parameters(self) -> dict[str, Any]:
        """Return parameters dictionary for optimizer."""
        params: dict[str, Any] = {}
        for b_idx, conv in enumerate(self.conv_layers):
            params[f"conv_{b_idx}_weights"] = copy.deepcopy(conv.weights)
            if conv.use_bias:
                params[f"conv_{b_idx}_bias"] = list(conv.bias_weights)

        for l_idx, (w, b) in enumerate(
            zip(self.fc_weights, self.fc_biases, strict=True)
        ):
            if len(self.fc_weights) == 1:
                params["classifier_weights"] = copy.deepcopy(w)
                params["classifier_bias"] = list(b)
            else:
                params[f"fc_{l_idx}_weights"] = copy.deepcopy(w)
                params[f"fc_{l_idx}_bias"] = list(b)
        return params

    def set_parameters(self, params: dict[str, Any]) -> None:
        """Set parameters from dictionary."""
        for b_idx, conv in enumerate(self.conv_layers):
            w_key = f"conv_{b_idx}_weights"
            b_key = f"conv_{b_idx}_bias"
            if w_key in params:
                conv.weights = copy.deepcopy(params[w_key])
            if b_key in params and conv.use_bias:
                conv.bias_weights = list(params[b_key])

        for l_idx in range(len(self.fc_weights)):
            if len(self.fc_weights) == 1:
                w_key = "classifier_weights"
                b_key = "classifier_bias"
            else:
                w_key = f"fc_{l_idx}_weights"
                b_key = f"fc_{l_idx}_bias"

            if w_key in params:
                self.fc_weights[l_idx] = copy.deepcopy(params[w_key])
            if b_key in params:
                self.fc_biases[l_idx] = list(params[b_key])

    def get_gradients(self) -> dict[str, Any]:
        """Return computed gradients dictionary."""
        grads: dict[str, Any] = {}
        for b_idx, conv in enumerate(self.conv_layers):
            grads[f"grad_conv_{b_idx}_weights"] = copy.deepcopy(conv.grad_weights)
            if conv.use_bias:
                grads[f"grad_conv_{b_idx}_bias"] = list(conv.grad_bias_weights)

        for l_idx, (w, b) in enumerate(
            zip(self.grad_fc_weights, self.grad_fc_biases, strict=True)
        ):
            if len(self.fc_weights) == 1:
                grads["grad_classifier_weights"] = copy.deepcopy(w)
                grads["grad_classifier_bias"] = list(b)
            else:
                grads[f"grad_fc_{l_idx}_weights"] = copy.deepcopy(w)
                grads[f"grad_fc_{l_idx}_bias"] = list(b)
        return grads


# Friendly Alias
SimpleCNN = ConvolutionalNeuralNetwork
