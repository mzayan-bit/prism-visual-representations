"""Representation encoder adapter wrapping existing PRISM vision models."""

from __future__ import annotations

from typing import Any

from prism.models.base import BaseVisionModel
from prism.transfer.head import replace_classifier_head


class RepresentationEncoder:
    """Adapts existing BaseVisionModel (CNN, ResNet, ViT) as representation encoder.

    Guarantees:
    - Extracts feature representations h without applying task-specific class heads.
    - Accurately backpropagates upstream representation gradients d_h into backbone.
    - Exposes trainable parameters and gradients for optimizer updating.
    """

    def __init__(
        self,
        backbone: BaseVisionModel,
        representation_dim: int | None = None,
        seed: int = 42,
    ) -> None:
        self.backbone = backbone
        self.seed = seed

        # 1. Determine representation dimension by dry forward if not provided
        if representation_dim is None:
            # Infer from model structure or test sample
            c, h, w = backbone.spec.input_shape
            dummy_sample = [
                [[0.0 for _ in range(w)] for _ in range(h)] for _ in range(c)
            ]
            layer_name = (
                "final_hidden"
                if hasattr(backbone, "conv_layers") or hasattr(backbone, "stages")
                else "cls_representation"
            )
            feat = backbone.extract_representations([dummy_sample], layer=layer_name)
            if isinstance(feat, list) and isinstance(feat[0], list):
                self.representation_dim = len(feat[0])
            else:
                self.representation_dim = len(feat)
        else:
            self.representation_dim = representation_dim

        # 2. Configure identity head so forward(x) returns representation
        replace_classifier_head(
            model=self.backbone,
            num_classes=self.representation_dim,
            seed=seed,
        )

        # Set identity matrix and zero bias
        self._set_identity_head()

    def _set_identity_head(self) -> None:
        """Set classifier head weights to identity and bias to zero."""
        dim = self.representation_dim
        params = self.backbone.get_parameters()

        # ResNet and CNN head parameters
        if "classifier_weights" in params:
            params["classifier_weights"] = [
                [1.0 if i == j else 0.0 for j in range(dim)] for i in range(dim)
            ]
            params["classifier_bias"] = [0.0 for _ in range(dim)]
        elif (
            "fc_0_weights" in params
            and len(
                [k for k in params if k.endswith("_weights") and k.startswith("fc_")]
            )
            == 1
        ):
            params["fc_0_weights"] = [
                [1.0 if i == j else 0.0 for j in range(dim)] for i in range(dim)
            ]
            params["fc_0_bias"] = [0.0 for _ in range(dim)]
        elif "classifier.weights" in params:
            # ViT head
            params["classifier.weights"] = [
                [1.0 if i == j else 0.0 for j in range(dim)] for i in range(dim)
            ]
            params["classifier.bias"] = [0.0 for _ in range(dim)]

        self.backbone.set_parameters(params)

    def train(self, mode: bool = True) -> RepresentationEncoder:
        """Set encoder to training mode."""
        self.backbone.train(mode)
        return self

    def eval(self) -> RepresentationEncoder:
        """Set encoder to evaluation mode."""
        self.backbone.eval()
        return self

    def forward(self, inputs: list[list[list[list[float]]]]) -> list[list[float]]:
        """Forward pass extracting representation batch [N x D]."""
        # Ensure identity head is preserved
        self._set_identity_head()
        return self.backbone.forward(inputs)

    def backward(self, d_representations: list[list[float]]) -> None:
        """Backpropagate representation gradients into encoder parameters."""
        self._set_identity_head()
        self.backbone.backward(d_representations)

    def zero_grad(self) -> None:
        """Clear all stored parameter gradients."""
        self.backbone.zero_grad()

    def get_parameters(self) -> dict[str, Any]:
        """Return trainable encoder parameters (excluding dummy identity head)."""
        all_params = self.backbone.get_parameters()
        # Filter out classifier head params
        return {
            k: v
            for k, v in all_params.items()
            if not k.startswith("classifier")
            and not (k.startswith("fc_") and "weights" in k)
        }

    def set_parameters(self, params: dict[str, Any]) -> None:
        """Set encoder parameters."""
        self.backbone.set_parameters(params)
        self._set_identity_head()

    def get_gradients(self) -> dict[str, Any]:
        """Return encoder parameter gradients."""
        all_grads = self.backbone.get_gradients()
        return {
            k: v
            for k, v in all_grads.items()
            if not k.startswith("classifier")
            and not (k.startswith("fc_") and "weights" in k)
        }
