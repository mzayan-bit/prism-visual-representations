"""Residual Neural Network (ResNet) architecture with explicit skip connections."""

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
from prism.models.normalization import BaseNormalization, get_normalization
from prism.models.pooling import MaxPool2D
from prism.models.residual import ResidualBlock
from prism.models.spatial import (
    compute_conv2d_output_shape,
    compute_pool2d_output_shape,
    compute_receptive_field,
    ensure_4d_tensor,
    normalize_spatial_pair,
)
from prism.models.specifications import ModelSpecification


class ResidualNeuralNetwork(BaseVisionModel):
    """Configurable Deep Residual Neural Network (ResNet) vision baseline."""

    def __init__(self, spec: ModelSpecification, seed: int = 42) -> None:
        super().__init__(spec=spec)
        self.seed = seed

        if spec.family not in (ModelFamily.RESNET, ModelFamily.CNN):
            raise ValidationError(
                f"ResNet requires ModelFamily.RESNET or CNN, got '{spec.family}'."
            )
        if TaskType.CLASSIFICATION not in spec.compatible_tasks:
            raise ValidationError(
                "ResidualNeuralNetwork requires classification task compatibility."
            )
        if spec.num_classes is None or spec.num_classes <= 0:
            raise ValidationError(
                f"num_classes must be positive, got {spec.num_classes}."
            )

        # 1. Parse Input Shape (C, H, W)
        if len(spec.input_shape) == 3:
            self.in_channels, self.in_height, self.in_width = spec.input_shape
        else:
            raise ValidationError(
                f"Expected 3D input_shape (C, H, W), got {spec.input_shape}."
            )

        # 2. Extract Hyperparameters
        hp = spec.hyperparameters
        self.stage_widths: list[int] = hp.get("stage_widths", [16, 32, 64])
        if not self.stage_widths:
            raise ValidationError("stage_widths cannot be empty for ResNet.")
        for idx, w in enumerate(self.stage_widths):
            if w <= 0:
                raise ValidationError(f"stage_widths[{idx}] must be positive, got {w}.")

        num_stages = len(self.stage_widths)
        raw_blocks = hp.get("blocks_per_stage", [2] * num_stages)
        if isinstance(raw_blocks, int):
            self.blocks_per_stage = [raw_blocks] * num_stages
        elif isinstance(raw_blocks, (list, tuple)):
            if len(raw_blocks) != num_stages:
                raise ValidationError(
                    f"blocks_per_stage ({len(raw_blocks)}) != stages ({num_stages})."
                )
            self.blocks_per_stage = [int(b) for b in raw_blocks]
        else:
            raise ValidationError("Invalid blocks_per_stage format.")

        for idx, b in enumerate(self.blocks_per_stage):
            if b <= 0:
                raise ValidationError(
                    f"blocks_per_stage[{idx}] must be positive, got {b}."
                )

        raw_strides = hp.get("strides", [1] + [2] * (num_stages - 1))
        if isinstance(raw_strides, int):
            self.stage_strides = [raw_strides] * num_stages
        elif isinstance(raw_strides, (list, tuple)):
            if len(raw_strides) != num_stages:
                raise ValidationError(
                    f"strides ({len(raw_strides)}) != stages ({num_stages})."
                )
            self.stage_strides = [int(s) for s in raw_strides]
        else:
            raise ValidationError("Invalid strides format.")

        self.stem_channels: int = int(hp.get("stem_channels", self.stage_widths[0]))
        self.stem_kernel_size: int = int(hp.get("stem_kernel_size", 3))
        self.stem_stride: int = int(hp.get("stem_stride", 1))
        self.stem_padding: int = int(hp.get("stem_padding", 1))
        self.stem_pool: bool = bool(hp.get("stem_pool", False))

        self.activation_name: str = hp.get("activation", "relu").lower()
        self.normalization_name: str = hp.get("normalization", "batch_norm").lower()
        self.norm_eps: float = float(hp.get("norm_eps", 1e-5))
        self.norm_momentum: float = float(hp.get("norm_momentum", 0.1))
        self.norm_affine: bool = bool(hp.get("norm_affine", True))

        self.dropout_rate: float = float(hp.get("dropout", 0.0))
        if self.dropout_rate < 0.0 or self.dropout_rate >= 1.0:
            raise ValidationError(
                f"Dropout rate must be in [0.0, 1.0), got {self.dropout_rate}."
            )

        self.classifier_hidden_dims: list[int] = hp.get("classifier_hidden_dims", [])

        # 3. Construct Stem
        self.stage_rf_tracking: list[tuple[int, int]] = []
        stem_seed = (seed * 100003 + 7) & 0x7FFFFFFF
        self.stem_conv = Conv2D(
            in_channels=self.in_channels,
            out_channels=self.stem_channels,
            kernel_size=self.stem_kernel_size,
            stride=self.stem_stride,
            padding=self.stem_padding,
            bias=True,
            seed=stem_seed,
            activation=self.activation_name,
        )
        self.stem_norm: BaseNormalization | None = get_normalization(
            norm_type=self.normalization_name,
            num_features=self.stem_channels,
            is_spatial=True,
            eps=self.norm_eps,
            momentum=self.norm_momentum,
            affine=self.norm_affine,
        )
        self.stem_act: BaseActivation = get_activation(self.activation_name)

        sk_h, sk_w = normalize_spatial_pair(self.stem_kernel_size, "stem_kernel")
        ss_h, ss_w = normalize_spatial_pair(self.stem_stride, "stem_stride")
        self.stage_rf_tracking.append((max(sk_h, sk_w), max(ss_h, ss_w)))

        cur_h, cur_w = compute_conv2d_output_shape(
            input_height=self.in_height,
            input_width=self.in_width,
            kernel_size=self.stem_kernel_size,
            stride=self.stem_stride,
            padding=self.stem_padding,
        )

        if self.stem_pool:
            self.stem_pool_layer: MaxPool2D | None = MaxPool2D(
                kernel_size=2, stride=2, padding=0
            )
            self.stage_rf_tracking.append((2, 2))
            cur_h, cur_w = compute_pool2d_output_shape(
                input_height=cur_h,
                input_width=cur_w,
                kernel_size=2,
                stride=2,
                padding=0,
            )
        else:
            self.stem_pool_layer = None

        cur_c = self.stem_channels

        # 4. Construct Residual Stages and Blocks
        self.stages: list[list[ResidualBlock]] = []

        for s_idx in range(num_stages):
            out_c = self.stage_widths[s_idx]
            num_b = self.blocks_per_stage[s_idx]
            s_stride = self.stage_strides[s_idx]
            stage_blocks: list[ResidualBlock] = []

            for b_idx in range(num_b):
                block_seed = (seed * 10007 + s_idx * 101 + b_idx * 31) & 0x7FFFFFFF
                b_stride = s_stride if b_idx == 0 else 1
                b_in = cur_c if b_idx == 0 else out_c

                block = ResidualBlock(
                    in_channels=b_in,
                    out_channels=out_c,
                    stride=b_stride,
                    normalization=self.normalization_name,
                    norm_eps=self.norm_eps,
                    norm_momentum=self.norm_momentum,
                    norm_affine=self.norm_affine,
                    activation=self.activation_name,
                    seed=block_seed,
                )
                stage_blocks.append(block)

                # Track RF for conv1 and conv2
                self.stage_rf_tracking.append((3, b_stride))
                self.stage_rf_tracking.append((3, 1))

                # Update spatial dimensions
                cur_h, cur_w = compute_conv2d_output_shape(
                    input_height=cur_h,
                    input_width=cur_w,
                    kernel_size=3,
                    stride=b_stride,
                    padding=1,
                )
                cur_h, cur_w = compute_conv2d_output_shape(
                    input_height=cur_h,
                    input_width=cur_w,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                )

            self.stages.append(stage_blocks)
            cur_c = out_c

        self.final_spatial_shape = (cur_c, cur_h, cur_w)
        self.flattened_dim = cur_c * cur_h * cur_w

        if self.flattened_dim <= 0:
            raise ValidationError(
                f"Calculated non-positive flattened dimension: {self.flattened_dim}."
            )

        # 5. Construct Classifier Head
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
        self._step_counter = 0

        # Runtime forward caches
        self._cached_inputs: Any = None
        self._cached_stem_states: dict[str, Any] = {}
        self._cached_stage_states: list[list[dict[str, Any]]] = []
        self._cached_flat: list[list[float]] | None = None
        self._cached_fc_states: list[dict[str, Any]] = []

    def train(self, mode: bool = True) -> "ResidualNeuralNetwork":
        super().train(mode)
        if self.stem_norm is not None:
            self.stem_norm.train(mode)
        for stage in self.stages:
            for block in stage:
                block.train(mode)
        return self

    def eval(self) -> "ResidualNeuralNetwork":
        super().eval()
        if self.stem_norm is not None:
            self.stem_norm.eval()
        for stage in self.stages:
            for block in stage:
                block.eval()
        return self

    @property
    def receptive_field(self) -> int:
        rf, _ = compute_receptive_field(self.stage_rf_tracking)
        return rf

    def _flatten_spatial(self, x: list[list[list[list[float]]]]) -> list[list[float]]:
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
        if not isinstance(inputs, (list, tuple)) or not inputs:
            raise ValidationError("Input batch cannot be empty.")

        first = inputs[0]
        if isinstance(first, (list, tuple)) and first:
            second = first[0]
            if isinstance(second, (list, tuple)) and second:
                third = second[0]
                if isinstance(third, (list, tuple)):
                    return ensure_4d_tensor(inputs)
                else:
                    return ensure_4d_tensor(inputs)
            else:
                n_samples = len(inputs)
                expected_dim = self.in_channels * self.in_height * self.in_width
                out_4d: list[list[list[list[float]]]] = []
                for n in range(n_samples):
                    flat_sample = inputs[n]
                    if len(flat_sample) != expected_dim:
                        raise ValidationError(
                            f"Flattened sample has {len(flat_sample)} features, "
                            f"expected {expected_dim}."
                        )
                    sample_3d = self._unflatten_spatial(
                        [flat_sample],
                        self.in_channels,
                        self.in_height,
                        self.in_width,
                    )[0]
                    out_4d.append(sample_3d)
                return out_4d

        raise ValidationError("Unsupported input data format for ResNet.")

    def forward(self, inputs: Any) -> list[list[float]]:
        """Compute ResNet forward pass producing raw logits [N, num_classes]."""
        x_4d = self._convert_input_to_4d(inputs)
        self._cached_inputs = x_4d

        if self.is_training:
            self._step_counter += 1

        # 1. Stem Forward
        stem_conv_out = self.stem_conv.forward(x_4d)
        stem_norm_out = (
            self.stem_norm.forward(stem_conv_out)
            if self.stem_norm is not None
            else stem_conv_out
        )
        stem_act_out = self.stem_act.forward(stem_norm_out)

        if self.stem_pool_layer is not None:
            stem_pool_out = self.stem_pool_layer.forward(stem_act_out)
            stem_out = stem_pool_out
        else:
            stem_pool_out = None
            stem_out = stem_act_out

        self._cached_stem_states = {
            "conv_pre": stem_conv_out,
            "conv_post_norm": stem_norm_out,
            "conv_post_act": stem_act_out,
            "pool_out": stem_pool_out,
            "stem_out": stem_out,
        }

        # 2. Residual Stages Forward
        self._cached_stage_states = []
        cur_tensor = stem_out

        for stage in self.stages:
            stage_block_states: list[dict[str, Any]] = []
            for block in stage:
                block_out = block.forward(cur_tensor)
                stage_block_states.append(
                    {
                        "input": cur_tensor,
                        "residual_branch": block._cached_norm2_out,
                        "shortcut_branch": block._cached_shortcut_out,
                        "post_add": block._cached_add_out,
                        "output": block_out,
                    }
                )
                cur_tensor = block_out
            self._cached_stage_states.append(stage_block_states)

        # 3. Flatten Final Spatial Representation
        final_flat = self._flatten_spatial(cur_tensor)
        self._cached_flat = final_flat

        # 4. Classifier Head Forward
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

            act = get_activation(self.activation_name)
            a_mat = act.forward(z_mat)

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
        """Propagate gradients backward through classifier, stages, and stem."""
        if (
            self._cached_inputs is None
            or self._cached_flat is None
            or not self._cached_stage_states
            or not self._cached_fc_states
        ):
            raise ValidationError("Cannot perform backward pass before forward pass.")

        n_samples = len(d_logits)
        num_fc = len(self.fc_weights)
        d_out = d_logits

        # 1. Backprop through Classifier Head
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
                mask = fc_state["dropout_mask"]
                if mask is not None:
                    d_a = [
                        [d_a[n][j] * mask[n][j] for j in range(len(d_a[0]))]
                        for n in range(len(d_a))
                    ]
                act = get_activation(self.activation_name)
                d_z = act.backward(fc_state["z"], d_a)

            for n in range(n_samples):
                for j in range(out_dim):
                    dz_val = d_z[n][j]
                    self.grad_fc_biases[l_idx][j] += dz_val
                    for i in range(in_dim):
                        self.grad_fc_weights[l_idx][i][j] += dz_val * h_in[n][i]

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

        # Unflatten to 4D tensor
        c_last, h_last, w_last = self.final_spatial_shape
        d_spatial = self._unflatten_spatial(d_out, c_last, h_last, w_last)

        # 2. Backprop through Residual Stages in reverse
        cur_d_spatial = d_spatial
        self._cached_stage_grad_states: dict[int, dict[int, dict[str, Any]]] = {}
        for s_idx in reversed(range(len(self.stages))):
            stage = self.stages[s_idx]
            self._cached_stage_grad_states[s_idx] = {}
            for b_idx in reversed(range(len(stage))):
                block = stage[b_idx]
                self._cached_stage_grad_states[s_idx][b_idx] = {
                    "output": cur_d_spatial,
                }
                cur_d_spatial = block.backward(cur_d_spatial)

        # 3. Backprop through Stem
        if self.stem_pool_layer is not None:
            d_stem_act = self.stem_pool_layer.backward(cur_d_spatial)
        else:
            d_stem_act = cur_d_spatial

        d_stem_norm = self.stem_act.backward(
            self._cached_stem_states["conv_post_norm"], d_stem_act
        )
        d_stem_conv = (
            self.stem_norm.backward(d_stem_norm)
            if self.stem_norm is not None
            else d_stem_norm
        )
        d_input = self.stem_conv.backward(d_stem_conv)
        self._cached_stem_grad_states = {
            "stem_out": cur_d_spatial,
            "conv_post_norm": d_stem_norm,
            "conv_pre": d_stem_conv,
            "input": d_input,
        }
        self._cached_input_grad = d_input

    def extract_spatial_activation_and_gradient(
        self, layer: str = "final_stage"
    ) -> tuple[list[list[list[list[float]]]], list[list[list[list[float]]]]]:
        """Extract spatial activation tensor A and gradient dS/dA for Grad-CAM."""
        layer_norm = layer.strip().lower()
        if not hasattr(self, "_cached_stage_states") or not self._cached_stage_states:
            raise ValidationError(
                "Must run forward pass before extracting spatial activations."
            )
        if (
            not hasattr(self, "_cached_stage_grad_states")
            or not self._cached_stage_grad_states
        ):
            raise ValidationError(
                "Must run backward pass before extracting spatial gradients."
            )

        num_stages = len(self.stages)
        if layer_norm in (
            "final_stage",
            "final_spatial",
            "final_conv",
            "spatial_features",
            "last_stage",
            "last_block",
            "default",
        ):
            s_idx = num_stages - 1
            b_idx = len(self.stages[s_idx]) - 1
            act = self._cached_stage_states[s_idx][b_idx]["output"]
            grad = self._cached_stage_grad_states[s_idx][b_idx]["output"]
            return act, grad

        for s_idx, stage in enumerate(self.stages):
            if layer_norm == f"stage_{s_idx}":
                b_idx = len(stage) - 1
                act = self._cached_stage_states[s_idx][b_idx]["output"]
                grad = self._cached_stage_grad_states[s_idx][b_idx]["output"]
                return act, grad
            for b_idx in range(len(stage)):
                if layer_norm in (
                    f"stage_{s_idx}_block_{b_idx}",
                    f"stage_{s_idx}_block_{b_idx}_output",
                ):
                    act = self._cached_stage_states[s_idx][b_idx]["output"]
                    grad = self._cached_stage_grad_states[s_idx][b_idx]["output"]
                    return act, grad

        if layer_norm in ("stem", "stem_out"):
            act = self._cached_stem_states["stem_out"]
            grad = self._cached_stem_grad_states["stem_out"]
            return act, grad

        valid = [f"stage_{i}" for i in range(num_stages)] + ["final_stage", "stem"]
        raise ValidationError(
            f"Layer '{layer}' is not a valid spatial residual layer for ResNet. "
            f"Available: {valid}"
        )

    def extract_representations(self, inputs: Any, layer: str = "final_hidden") -> Any:
        """Extract intermediate activations or spatial maps in evaluation mode."""
        layer_norm = layer.strip().lower()
        was_training = self.is_training
        self.eval()

        try:
            if layer_norm in ("input", "input_spatial", "input_image"):
                return self._convert_input_to_4d(inputs)
            if layer_norm in ("input_flat", "input_flattened"):
                return self._flatten_spatial(self._convert_input_to_4d(inputs))

            logits = self.forward(inputs)

            # Stem extraction
            if layer_norm == "stem_pre_norm":
                return self._cached_stem_states["conv_pre"]
            if layer_norm == "stem_post_norm":
                return self._cached_stem_states["conv_post_norm"]
            if layer_norm in ("stem", "stem_out"):
                return self._cached_stem_states["stem_out"]

            # Stage & Block extraction
            for s_idx, stage in enumerate(self.stages):
                for b_idx in range(len(stage)):
                    b_state = self._cached_stage_states[s_idx][b_idx]
                    if layer_norm == f"stage_{s_idx}_block_{b_idx}_residual":
                        return b_state["residual_branch"]
                    if layer_norm == f"stage_{s_idx}_block_{b_idx}_shortcut":
                        return b_state["shortcut_branch"]
                    if layer_norm == f"stage_{s_idx}_block_{b_idx}_post_add":
                        return b_state["post_add"]
                    if layer_norm in (
                        f"stage_{s_idx}_block_{b_idx}",
                        f"stage_{s_idx}_block_{b_idx}_output",
                    ):
                        return b_state["output"]

                if layer_norm == f"stage_{s_idx}":
                    return self._cached_stage_states[s_idx][-1]["output"]

            if layer_norm in ("final_spatial", "spatial_features"):
                return self._cached_stage_states[-1][-1]["output"]

            if layer_norm in (
                "final_hidden",
                "final_representation",
                "embedding",
            ):
                if len(self.fc_weights) == 1:
                    return self._cached_flat
                else:
                    return self._cached_fc_states[-2]["a"]

            if layer_norm in ("logits", "output"):
                return logits

            valid = [
                "input",
                "stem",
                "final_spatial",
                "final_hidden",
                "logits",
            ]
            raise ValidationError(
                f"Unknown layer '{layer}' for ResidualNeuralNetwork. "
                f"Valid layers include: {valid}."
            )
        finally:
            if was_training:
                self.train()

    def zero_grad(self) -> None:
        self.stem_conv.zero_grad()
        if self.stem_norm is not None:
            self.stem_norm.zero_grad()

        for stage in self.stages:
            for block in stage:
                block.zero_grad()

        self.grad_fc_weights: list[list[list[float]]] = [
            [[0.0 for _ in range(len(w[0]))] for _ in range(len(w))]
            for w in self.fc_weights
        ]
        self.grad_fc_biases: list[list[float]] = [
            [0.0 for _ in range(len(b))] for b in self.fc_biases
        ]

    def get_parameters(self) -> dict[str, Any]:
        params: dict[str, Any] = {}

        # Stem
        params["stem_conv_weights"] = copy.deepcopy(self.stem_conv.weights)
        if self.stem_conv.use_bias:
            params["stem_conv_bias"] = list(self.stem_conv.bias_weights)
        if self.stem_norm is not None:
            for k, v in self.stem_norm.get_parameters().items():
                params[f"stem_norm_{k}"] = copy.deepcopy(v)

        # Stages & Blocks
        for s_idx, stage in enumerate(self.stages):
            for b_idx, block in enumerate(stage):
                prefix = f"stage_{s_idx}_block_{b_idx}"
                block_params = block.get_parameters(prefix=prefix)
                params.update(block_params)

        # Classifier Head
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
        # Stem
        if "stem_conv_weights" in params:
            self.stem_conv.weights = copy.deepcopy(params["stem_conv_weights"])
        if "stem_conv_bias" in params and self.stem_conv.use_bias:
            self.stem_conv.bias_weights = list(params["stem_conv_bias"])
        if self.stem_norm is not None:
            stem_n = {}
            if "stem_norm_gamma" in params:
                stem_n["gamma"] = copy.deepcopy(params["stem_norm_gamma"])
            if "stem_norm_beta" in params:
                stem_n["beta"] = copy.deepcopy(params["stem_norm_beta"])
            self.stem_norm.set_parameters(stem_n)

        # Stages & Blocks
        for s_idx, stage in enumerate(self.stages):
            for b_idx, block in enumerate(stage):
                prefix = f"stage_{s_idx}_block_{b_idx}"
                block.set_parameters(params, prefix=prefix)

        # Classifier Head
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
        grads: dict[str, Any] = {}

        # Stem
        grads["grad_stem_conv_weights"] = copy.deepcopy(self.stem_conv.grad_weights)
        if self.stem_conv.use_bias:
            grads["grad_stem_conv_bias"] = list(self.stem_conv.grad_bias_weights)
        if self.stem_norm is not None:
            for k, v in self.stem_norm.get_gradients().items():
                grads[f"grad_stem_norm_{k.replace('grad_', '')}"] = copy.deepcopy(v)

        # Stages & Blocks
        for s_idx, stage in enumerate(self.stages):
            for b_idx, block in enumerate(stage):
                prefix = f"stage_{s_idx}_block_{b_idx}"
                block_grads = block.get_gradients(prefix=prefix)
                grads.update(block_grads)

        # Classifier Head
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

    def get_state(self) -> dict[str, Any]:
        state: dict[str, Any] = {}

        # Stem
        if self.stem_norm is not None:
            for k, v in self.stem_norm.get_state().items():
                state[f"stem_norm_{k}"] = copy.deepcopy(v)

        # Stages & Blocks
        for s_idx, stage in enumerate(self.stages):
            for b_idx, block in enumerate(stage):
                prefix = f"stage_{s_idx}_block_{b_idx}"
                block_state = block.get_state(prefix=prefix)
                state.update(block_state)

        return state

    def set_state(self, state: dict[str, Any]) -> None:
        # Stem
        if self.stem_norm is not None:
            stem_s = {}
            if "stem_norm_running_mean" in state:
                stem_s["running_mean"] = copy.deepcopy(state["stem_norm_running_mean"])
            if "stem_norm_running_var" in state:
                stem_s["running_var"] = copy.deepcopy(state["stem_norm_running_var"])
            if "stem_norm_num_batches_tracked" in state:
                stem_s["num_batches_tracked"] = state["stem_norm_num_batches_tracked"]
            self.stem_norm.set_state(stem_s)

        # Stages & Blocks
        for s_idx, stage in enumerate(self.stages):
            for b_idx, block in enumerate(stage):
                prefix = f"stage_{s_idx}_block_{b_idx}"
                block.set_state(state, prefix=prefix)


# Friendly Alias
SimpleResNet = ResidualNeuralNetwork
