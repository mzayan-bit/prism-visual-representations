"""Model architectures, spatial layers, CNNs, normalization, and residual models."""

from prism.models.activations import (
    BaseActivation,
    GELUActivation,
    ReLUActivation,
    get_activation,
)
from prism.models.base import BaseVisionModel
from prism.models.cnn import ConvolutionalNeuralNetwork, SimpleCNN
from prism.models.convolution import Conv2D
from prism.models.initialization import (
    initialize_conv2d_parameters,
    initialize_linear_parameters,
    initialize_mlp_parameters,
)
from prism.models.linear import LinearSoftmaxClassifier
from prism.models.mlp import MultiLayerPerceptron
from prism.models.normalization import (
    BaseNormalization,
    BatchNorm1D,
    BatchNorm2D,
    get_normalization,
)
from prism.models.patches import (
    ClassToken,
    PatchEmbedding,
    PatchExtractor,
    PositionalEmbedding,
    ensure_3d_tensor,
)
from prism.models.pooling import AvgPool2D, MaxPool2D
from prism.models.residual import (
    IdentityShortcut,
    ProjectionShortcut,
    ResidualAdd,
    ResidualBlock,
)
from prism.models.resnet import ResidualNeuralNetwork, SimpleResNet
from prism.models.spatial import (
    compute_conv2d_output_shape,
    compute_pool2d_output_shape,
    compute_receptive_field,
    ensure_4d_tensor,
)
from prism.models.specifications import ModelSpecification

__all__ = [
    "AvgPool2D",
    "BaseActivation",
    "BaseNormalization",
    "BaseVisionModel",
    "BatchNorm1D",
    "BatchNorm2D",
    "ClassToken",
    "Conv2D",
    "ConvolutionalNeuralNetwork",
    "GELUActivation",
    "IdentityShortcut",
    "LinearSoftmaxClassifier",
    "MaxPool2D",
    "ModelSpecification",
    "MultiLayerPerceptron",
    "PatchEmbedding",
    "PatchExtractor",
    "PositionalEmbedding",
    "ProjectionShortcut",
    "ReLUActivation",
    "ResidualAdd",
    "ResidualBlock",
    "ResidualNeuralNetwork",
    "SimpleCNN",
    "SimpleResNet",
    "compute_conv2d_output_shape",
    "compute_pool2d_output_shape",
    "compute_receptive_field",
    "ensure_3d_tensor",
    "ensure_4d_tensor",
    "get_activation",
    "get_normalization",
    "initialize_conv2d_parameters",
    "initialize_linear_parameters",
    "initialize_mlp_parameters",
]
