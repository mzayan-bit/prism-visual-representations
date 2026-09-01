"""Classification head replacement routines across PRISM model architectures."""

from __future__ import annotations

import math
import random

from prism.core.errors import ValidationError
from prism.models.base import BaseVisionModel
from prism.models.cnn import ConvolutionalNeuralNetwork
from prism.models.initialization import (
    initialize_linear_parameters,
    initialize_mlp_parameters,
)
from prism.models.linear import LinearSoftmaxClassifier
from prism.models.mlp import MultiLayerPerceptron
from prism.models.resnet import ResidualNeuralNetwork
from prism.models.transformer import VisionTransformer


def replace_classifier_head(
    model: BaseVisionModel,
    num_classes: int,
    seed: int = 42,
) -> BaseVisionModel:
    """Replace and re-initialize the classification head for a new target class count.

    Preserves all feature extraction backbone parameters and normalization states while
    constructing a freshly initialized head for downstream target transfer.

    Args:
        model: Vision model instance whose classifier head will be replaced.
        num_classes: New number of target classes (must be >= 2).
        seed: Random seed for deterministic head initialization.

    Returns:
        The mutated model instance with updated classification head and specification.
    """
    if num_classes < 2:
        raise ValidationError(f"Target num_classes must be >= 2, got {num_classes}.")

    cls_seed = (seed * 1000003 + 99991) & 0x7FFFFFFF

    if isinstance(model, VisionTransformer):
        std_head = math.sqrt(2.0 / float(model.embed_dim + num_classes))
        rng = random.Random(cls_seed)
        model.classifier_w = [
            [rng.gauss(0.0, std_head) for _ in range(num_classes)]
            for _ in range(model.embed_dim)
        ]
        model.classifier_b = [0.0] * num_classes
        model.num_classes_val = num_classes
        model.spec = model.spec.model_copy(update={"num_classes": num_classes})
        model.zero_grad()

    elif isinstance(model, (ConvolutionalNeuralNetwork, ResidualNeuralNetwork)):
        if not model.classifier_hidden_dims:
            w_cls, b_cls = initialize_linear_parameters(
                in_features=model.flattened_dim,
                num_classes=num_classes,
                seed=cls_seed,
            )
            model.fc_weights = [w_cls]
            model.fc_biases = [b_cls]
        else:
            w_mlp, b_mlp = initialize_mlp_parameters(
                in_features=model.flattened_dim,
                hidden_dims=model.classifier_hidden_dims,
                num_classes=num_classes,
                seed=cls_seed,
                activation=model.activation_name,
            )
            model.fc_weights = w_mlp
            model.fc_biases = b_mlp

        model.spec = model.spec.model_copy(update={"num_classes": num_classes})
        model.zero_grad()

    elif isinstance(model, MultiLayerPerceptron):
        w_last, b_last = initialize_linear_parameters(
            in_features=model.hidden_dims[-1],
            num_classes=num_classes,
            seed=cls_seed,
        )
        model.layer_weights[-1] = w_last
        model.layer_biases[-1] = b_last
        model.num_classes_val = num_classes
        model.spec = model.spec.model_copy(update={"num_classes": num_classes})
        model.zero_grad()

    elif isinstance(model, LinearSoftmaxClassifier):
        w_lin, b_lin = initialize_linear_parameters(
            in_features=model.in_features,
            num_classes=num_classes,
            seed=cls_seed,
        )
        model.weights = w_lin
        model.bias = b_lin
        model.num_classes_val = num_classes
        model.spec = model.spec.model_copy(update={"num_classes": num_classes})
        model.zero_grad()

        raise ValidationError(
            f"Unsupported model type for classifier replacement: "
            f"{type(model).__name__}."
        )

    return model
