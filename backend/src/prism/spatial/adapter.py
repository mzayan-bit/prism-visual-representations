"""Spatial representation adapter for extracting 4D feature maps from encoders."""

from __future__ import annotations

from typing import Any

from prism.core.enums import ModelFamily
from prism.core.errors import ValidationError
from prism.models.base import BaseVisionModel
from prism.models.cnn import ConvolutionalNeuralNetwork
from prism.models.patches import PatchGeometry
from prism.models.resnet import ResidualNeuralNetwork
from prism.models.transformer import VisionTransformer


def get_available_spatial_layers(model: BaseVisionModel) -> list[str]:
    """Discover all valid spatial feature extraction layers for a vision model."""
    family = model.spec.family

    if family == ModelFamily.CNN:
        cnn = model if isinstance(model, ConvolutionalNeuralNetwork) else None
        num_blocks = len(cnn.conv_layers) if cnn is not None else 2
        layers = [f"conv_{i}" for i in range(num_blocks)]
        layers.extend([f"block_{i}" for i in range(num_blocks)])
        layers.append("final_spatial")
        return layers

    if family == ModelFamily.RESNET:
        resnet = model if isinstance(model, ResidualNeuralNetwork) else None
        layers = ["stem"]
        if resnet is not None:
            for s_idx, stage in enumerate(resnet.stages):
                for b_idx in range(len(stage)):
                    layers.append(f"stage_{s_idx}_block_{b_idx}_post_add")
                layers.append(f"stage_{s_idx}")
        layers.append("final_spatial")
        return layers

    if family == ModelFamily.VISION_TRANSFORMER:
        vit = model if isinstance(model, VisionTransformer) else None
        layers = ["patch_embeddings"]
        num_blocks = len(vit.encoder.blocks) if vit is not None else 2
        for b_idx in range(num_blocks):
            layers.append(f"encoder_{b_idx}")
        layers.extend(["final_tokens", "patch_tokens", "final_spatial"])
        return layers

    raise ValidationError(
        f"Spatial representation adapter does not support model family '{family}'."
    )


class SpatialRepresentationAdapter:
    """Adapter exposing uniform 4D spatial feature maps [N, C_f, H_f, W_f]."""

    def __init__(
        self,
        model: BaseVisionModel,
        layer_name: str = "final_spatial",
    ) -> None:
        self.model = model
        self.layer_name = layer_name.strip().lower()

        disallowed = [
            "logits",
            "output",
            "cls",
            "cls_representation",
            "final_hidden",
            "final_representation",
            "embedding",
            "input_flat",
            "input_flattened",
        ]
        if self.layer_name in disallowed:
            raise ValidationError(
                f"Cannot use non-spatial layer '{layer_name}' for spatial transfer. "
                f"Spatial heads require 2D/3D feature grid representations."
            )

        valid_layers = get_available_spatial_layers(model)
        normalized_valid = [lay.lower() for lay in valid_layers]
        if self.layer_name not in normalized_valid and self.layer_name not in (
            "final_spatial",
            "spatial_features",
        ):
            raise ValidationError(
                f"Invalid spatial layer '{layer_name}' for {model.spec.family}. "
                f"Available spatial layers: {valid_layers}."
            )

    def extract_spatial_features(self, inputs: Any) -> list[list[list[list[float]]]]:
        """Extract spatial feature tensor [N, C_f, H_f, W_f] from the model."""
        family = self.model.spec.family

        if family in (ModelFamily.CNN, ModelFamily.RESNET):
            raw = self.model.extract_representations(inputs, layer=self.layer_name)
            if not isinstance(raw, (list, tuple)) or not raw:
                raise ValidationError("Extracted spatial features are empty.")
            if not isinstance(raw[0], (list, tuple)) or not raw[0]:
                raise ValidationError(
                    "Extracted spatial features must have channel dimension."
                )
            if not isinstance(raw[0][0], (list, tuple)) or not raw[0][0]:
                raise ValidationError(
                    "Extracted spatial features must have 2D spatial dimensions."
                )
            return [
                [[[float(val) for val in row] for row in ch] for ch in sample]
                for sample in raw
            ]

        if family == ModelFamily.VISION_TRANSFORMER:
            if not isinstance(self.model, VisionTransformer):
                raise ValidationError("Expected model instance of VisionTransformer.")

            eff_layer = self.layer_name
            if eff_layer in (
                "final_spatial",
                "spatial_features",
                "final_tokens",
                "patch_tokens",
            ):
                eff_layer = "final_tokens"

            raw_tokens = self.model.extract_representations(inputs, layer=eff_layer)

            if not isinstance(raw_tokens, (list, tuple)) or not raw_tokens:
                raise ValidationError("ViT extracted tokens are empty.")

            n_samples = len(raw_tokens)
            p_geom: PatchGeometry | None = getattr(
                self.model, "geometry", getattr(self.model, "patch_geometry", None)
            )
            if p_geom is None:
                raise ValidationError(
                    "VisionTransformer model missing geometry descriptor."
                )
            t_expected = p_geom.total_patches
            h_patches = p_geom.patches_per_column
            w_patches = p_geom.patches_per_row

            first_seq_len = len(raw_tokens[0])
            if first_seq_len == t_expected + 1:
                patch_tokens = [
                    [list(tok) for tok in sample[1:]] for sample in raw_tokens
                ]
            elif first_seq_len == t_expected:
                patch_tokens = [[list(tok) for tok in sample] for sample in raw_tokens]
            else:
                raise ValidationError(
                    f"Unexpected ViT token sequence length {first_seq_len}, "
                    f"expected {t_expected} patches (or {t_expected + 1} with CLS)."
                )

            d_dim = len(patch_tokens[0][0])

            reshaped_features: list[list[list[list[float]]]] = []
            for n in range(n_samples):
                sample_features: list[list[list[float]]] = []
                for d in range(d_dim):
                    channel_grid: list[list[float]] = []
                    for r in range(h_patches):
                        row: list[float] = []
                        for c in range(w_patches):
                            patch_idx = r * w_patches + c
                            val = float(patch_tokens[n][patch_idx][d])
                            row.append(val)
                        channel_grid.append(row)
                    sample_features.append(channel_grid)
                reshaped_features.append(sample_features)

            return reshaped_features

        raise ValidationError(
            f"Unsupported model family for spatial adapter: {family}."
        )

    def compute_feature_shape(
        self, input_shape: tuple[int, int, int] = (3, 32, 32)
    ) -> tuple[int, int, int]:
        """Compute (channels, height, width) of extracted spatial feature map."""
        c, h, w = input_shape
        dummy_input = [[[[0.0 for _ in range(w)] for _ in range(h)] for _ in range(c)]]
        features = self.extract_spatial_features(dummy_input)
        return (len(features[0]), len(features[0][0]), len(features[0][0][0]))
