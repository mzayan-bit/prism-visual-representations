"""Frame encoder adapter for extracting multi-frame representation sequences."""

from __future__ import annotations

from typing import Any

from prism.core.enums import ModelFamily
from prism.core.errors import ValidationError
from prism.models.base import BaseVisionModel
from prism.models.cnn import ConvolutionalNeuralNetwork
from prism.models.resnet import ResidualNeuralNetwork
from prism.models.transformer import VisionTransformer


def get_available_temporal_layers(model: BaseVisionModel) -> list[str]:
    """Discover all valid layer names for temporal sequence feature extraction."""
    family = model.spec.family

    if family == ModelFamily.CNN:
        cnn = model if isinstance(model, ConvolutionalNeuralNetwork) else None
        num_blocks = len(cnn.conv_layers) if cnn is not None else 2
        layers = ["input"]
        layers.extend([f"conv_{i}" for i in range(num_blocks)])
        layers.extend([f"block_{i}" for i in range(num_blocks)])
        layers.extend(["final_spatial", "final_hidden"])
        return layers

    if family == ModelFamily.RESNET:
        resnet = model if isinstance(model, ResidualNeuralNetwork) else None
        layers = ["input", "stem"]
        if resnet is not None:
            for s_idx, stage in enumerate(resnet.stages):
                for b_idx in range(len(stage)):
                    layers.append(f"stage_{s_idx}_block_{b_idx}_post_add")
                layers.append(f"stage_{s_idx}")
        layers.extend(["final_spatial", "final_hidden"])
        return layers

    if family == ModelFamily.VISION_TRANSFORMER:
        vit = model if isinstance(model, VisionTransformer) else None
        layers = ["input", "patch_embeddings"]
        num_blocks = len(vit.encoder.blocks) if vit is not None else 2
        for b_idx in range(num_blocks):
            layers.append(f"encoder_{b_idx}")
        layers.extend(["final_tokens", "cls", "final_hidden"])
        return layers

    return ["final_hidden"]


class TemporalFrameEncoder:
    """Encodes video frames using a shared 2D image encoder into [N, T, D] tensors."""

    def __init__(
        self,
        model: BaseVisionModel,
        layer_name: str = "final_hidden",
    ) -> None:
        self.model = model
        self.layer_name = layer_name.strip().lower()

        valid_layers = get_available_temporal_layers(model)
        normalized_valid = [lay.lower() for lay in valid_layers]
        if self.layer_name not in normalized_valid and self.layer_name not in (
            "final_hidden",
            "final_representation",
            "cls",
            "cls_representation",
        ):
            raise ValidationError(
                f"Invalid temporal layer '{layer_name}' for {model.spec.family}. "
                f"Available layers: {valid_layers}."
            )

    @property
    def is_training(self) -> bool:
        """Return training mode status."""
        return self.model.is_training

    def train(self, mode: bool = True) -> TemporalFrameEncoder:
        """Set training mode for the underlying frame encoder."""
        self.model.train(mode)
        return self

    def eval(self) -> TemporalFrameEncoder:
        """Set evaluation mode for the underlying frame encoder."""
        self.model.eval()
        return self

    def _pool_spatial_or_sequence(self, feature: Any) -> list[float]:
        """Convert multi-dimensional layer outputs to 1D vector."""
        # 1D vector already: [D]
        if (
            isinstance(feature, list)
            and feature
            and isinstance(feature[0], (int, float))
        ):
            return [float(x) for x in feature]

        # 3D token tensor for one frame: [L, D] (e.g. ViT patch tokens)
        if (
            isinstance(feature, list)
            and feature
            and isinstance(feature[0], list)
            and feature[0]
            and isinstance(feature[0][0], (int, float))
        ):
            num_tokens = len(feature)
            dim = len(feature[0])
            pooled = [0.0] * dim
            for token in feature:
                for d_i, val in enumerate(token):
                    pooled[d_i] += val
            return [p / max(1, num_tokens) for p in pooled]

        # 3D spatial map for one frame: [C, H, W]
        if (
            isinstance(feature, list)
            and feature
            and isinstance(feature[0], list)
            and feature[0]
            and isinstance(feature[0][0], list)
            and feature[0][0]
            and isinstance(feature[0][0][0], (int, float))
        ):
            channels = len(feature)
            h = len(feature[0])
            w = len(feature[0][0])
            spatial_size = max(1, h * w)
            pooled_c = [0.0] * channels
            for c_i in range(channels):
                c_sum = 0.0
                for y in range(h):
                    for x in range(w):
                        c_sum += feature[c_i][y][x]
                pooled_c[c_i] = c_sum / spatial_size
            return pooled_c

        raise ValidationError(
            f"Unexpected feature representation shape: {type(feature)}."
        )

    def forward(
        self,
        videos: list[list[list[list[list[float]]]]],  # N x T x C x H x W
    ) -> list[list[list[float]]]:  # N x T x D
        """Extract frame representations for all video sequences in batch."""
        if not videos or not videos[0]:
            return []

        n_videos = len(videos)
        t_frames = len(videos[0])

        flat_frames: list[list[list[list[float]]]] = []
        for v in videos:
            for f in v:
                flat_frames.append(f)

        raw_reprs = self.model.extract_representations(
            flat_frames, layer=self.layer_name
        )

        if not isinstance(raw_reprs, list) or len(raw_reprs) != (n_videos * t_frames):
            raise ValidationError(
                f"Expected {n_videos * t_frames} frames, got {len(raw_reprs)}."
            )

        output_sequences: list[list[list[float]]] = []
        idx = 0
        for _ in range(n_videos):
            seq_reprs: list[list[float]] = []
            for _ in range(t_frames):
                frame_feat = raw_reprs[idx]
                pooled_vec = self._pool_spatial_or_sequence(frame_feat)
                seq_reprs.append(pooled_vec)
                idx += 1
            output_sequences.append(seq_reprs)

        return output_sequences
