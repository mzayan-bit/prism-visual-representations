"""Self-Supervised Representation Learning (SimCLR Contrastive Learning) module."""

from prism.ssl.adapter import RepresentationEncoder
from prism.ssl.context import AugmentationContext, DeterministicFloatRNG
from prism.ssl.loss import ContrastiveNTXentLoss
from prism.ssl.projection import (
    SimCLRProjectionHead,
    backward_normalize_embeddings,
    normalize_embeddings,
)
from prism.ssl.transforms import (
    AugmentationPolicy,
    AugmentationTrace,
    BaseAugmentation,
    ColorJitter,
    Grayscale,
    RandomCropWithPadding,
    RandomHorizontalFlip,
)
from prism.ssl.views import (
    ContrastiveBatch,
    ContrastiveBatchLoader,
    ContrastiveSamplePair,
    ContrastiveViewGenerator,
)

__all__ = [
    "AugmentationContext",
    "AugmentationPolicy",
    "AugmentationTrace",
    "BaseAugmentation",
    "ColorJitter",
    "ContrastiveBatch",
    "ContrastiveBatchLoader",
    "ContrastiveNTXentLoss",
    "ContrastiveSamplePair",
    "ContrastiveViewGenerator",
    "DeterministicFloatRNG",
    "Grayscale",
    "RandomCropWithPadding",
    "RandomHorizontalFlip",
    "RepresentationEncoder",
    "SimCLRProjectionHead",
    "backward_normalize_embeddings",
    "normalize_embeddings",
]
