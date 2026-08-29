"""Model architectures, spatial layers, CNNs, normalization, and residual models."""

from prism.models.activations import (
    BaseActivation,
    GELUActivation,
    ReLUActivation,
    get_activation,
)
from prism.models.attention import (
    MultiHeadSelfAttention,
    ScaledDotProductAttention,
    ensure_4d_attention_tensor,
    softmax_1d,
    softmax_backward_1d,
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
    LayerNorm,
    get_normalization,
)
from prism.models.patches import (
    ClassToken,
    ImagePatchExtractor,
    LearnablePositionalEmbedding,
    PatchEmbedding,
    PatchExtractor,
    PatchGeometry,
    PositionalEmbedding,
    ensure_3d_tensor,
    patches_to_image,
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
from prism.models.transformer import (
    TransformerEncoder,
    TransformerEncoderBlock,
    TransformerFeedForward,
    VisionTransformer,
)

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
    "ImagePatchExtractor",
    "LayerNorm",
    "LearnablePositionalEmbedding",
    "LinearSoftmaxClassifier",
    "MaxPool2D",
    "ModelSpecification",
    "MultiHeadSelfAttention",
    "MultiLayerPerceptron",
    "PatchEmbedding",
    "PatchExtractor",
    "PatchGeometry",
    "PositionalEmbedding",
    "ProjectionShortcut",
    "ReLUActivation",
    "ResidualAdd",
    "ResidualBlock",
    "ResidualNeuralNetwork",
    "ScaledDotProductAttention",
    "SimpleCNN",
    "SimpleResNet",
    "TransformerEncoder",
    "TransformerEncoderBlock",
    "TransformerFeedForward",
    "VisionTransformer",
    "compute_conv2d_output_shape",
    "compute_pool2d_output_shape",
    "compute_receptive_field",
    "ensure_3d_tensor",
    "ensure_4d_attention_tensor",
    "ensure_4d_tensor",
    "get_activation",
    "get_normalization",
    "initialize_conv2d_parameters",
    "initialize_linear_parameters",
    "initialize_mlp_parameters",
    "patches_to_image",
    "softmax_1d",
    "softmax_backward_1d",
]
