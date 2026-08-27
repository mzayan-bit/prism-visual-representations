"""Deterministic parameter initialization routines for PRISM models."""

import math
import random

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
