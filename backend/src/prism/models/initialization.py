"""Deterministic parameter initialization routines for PRISM models."""

import math
import random
from collections.abc import Sequence

from prism.core.enums import InitializationStrategy
from prism.core.errors import ValidationError


def initialize_linear_parameters(
    in_features: int,
    num_classes: int,
    seed: int = 42,
    strategy: InitializationStrategy = InitializationStrategy.RANDOM,
    scale: float | None = None,
) -> tuple[list[list[float]], list[float]]:
    """Deterministically initialize weight matrix and bias vector for a linear layer.

    Parameters
    ----------
    in_features : int
        Flattened input feature dimensionality (D > 0).
    num_classes : int
        Number of output category logits (C > 0).
    seed : int
        Pseudo-random seed for local RNG.
    strategy : InitializationStrategy
        Initialization scheme.
    scale : float | None
        Optional std scaling factor. Defaults to sqrt(2 / (D + C)).

    Returns
    -------
    tuple[list[list[float]], list[float]]
        (weights [D, C], bias [C]).
    """
    if in_features <= 0:
        raise ValidationError(f"in_features must be positive, got {in_features}.")
    if num_classes <= 0:
        raise ValidationError(f"num_classes must be positive, got {num_classes}.")

    rng = random.Random(seed)
    std = (
        scale
        if scale is not None
        else math.sqrt(2.0 / float(in_features + num_classes))
    )

    # Weights shape: [in_features, num_classes]
    weights: list[list[float]] = []
    for _ in range(in_features):
        row = [rng.gauss(0.0, std) for _ in range(num_classes)]
        weights.append(row)

    # Zero initialization for bias vector [num_classes]
    bias: list[float] = [0.0 for _ in range(num_classes)]

    return weights, bias


def initialize_mlp_parameters(
    in_features: int,
    hidden_dims: Sequence[int],
    num_classes: int,
    seed: int = 42,
    activation: str = "relu",
) -> tuple[list[list[list[float]]], list[list[float]]]:
    """Deterministically initialize weights and biases for all layers in an MLP.

    Parameters
    ----------
    in_features : int
        Flattened input dimension (D > 0).
    hidden_dims : Sequence[int]
        Dimensionalities of successive hidden layers.
    num_classes : int
        Number of output classes (C > 0).
    seed : int
        Random seed for deterministic initialization.
    activation : str
        Activation type ("relu", "gelu") for He/Xavier scaling.

    Returns
    -------
    tuple[list[list[list[float]]], list[list[float]]]
        (layer_weights, layer_biases) for all layers including the output layer.
    """
    if in_features <= 0:
        raise ValidationError(f"in_features must be positive, got {in_features}.")
    if num_classes <= 0:
        raise ValidationError(f"num_classes must be positive, got {num_classes}.")
    if not hidden_dims:
        raise ValidationError("hidden_dims cannot be empty for an MLP.")
    for idx, dim in enumerate(hidden_dims):
        if dim <= 0:
            raise ValidationError(f"hidden_dims[{idx}] must be positive, got {dim}.")

    rng = random.Random(seed)
    all_dims = [in_features, *list(hidden_dims), num_classes]
    num_layers = len(all_dims) - 1

    layer_weights: list[list[list[float]]] = []
    layer_biases: list[list[float]] = []

    for l_idx in range(num_layers):
        fan_in = all_dims[l_idx]
        fan_out = all_dims[l_idx + 1]

        # Use He/Kaiming initialization for ReLU hidden layers, Xavier for output
        is_hidden = l_idx < (num_layers - 1)
        if is_hidden and activation.lower() == "relu":
            std = math.sqrt(2.0 / float(fan_in))
        else:
            std = math.sqrt(2.0 / float(fan_in + fan_out))

        w_mat: list[list[float]] = []
        for _ in range(fan_in):
            row = [rng.gauss(0.0, std) for _ in range(fan_out)]
            w_mat.append(row)

        b_vec: list[float] = [0.0 for _ in range(fan_out)]

        layer_weights.append(w_mat)
        layer_biases.append(b_vec)

    return layer_weights, layer_biases
