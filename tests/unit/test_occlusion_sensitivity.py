"""Unit tests for sliding-window occlusion sensitivity attribution."""

from prism.core.enums import ModelFamily
from prism.explainability.attribution import (
    OcclusionFillPolicy,
    TargetClassMode,
)
from prism.explainability.occlusion import (
    compute_occlusion_sensitivity,
)
from prism.models.cnn import ConvolutionalNeuralNetwork
from prism.models.specifications import ModelSpecification


def test_occlusion_sensitivity_execution() -> None:
    """Test occlusion sensitivity on small CNN with zero and mean fill policies."""
    c, h, w, num_classes = 3, 8, 8, 2
    spec = ModelSpecification(
        model_id="cnn_occ",
        name="CNN Occ",
        family=ModelFamily.CNN,
        architecture="cnn_simple",
        input_shape=(c, h, w),
        num_classes=num_classes,
        hyperparameters={
            "conv_channels": [4],
            "kernel_sizes": [3],
            "activation": "relu",
            "use_batch_norm": False,
            "pooling": "none",
            "hidden_dims": [4],
        },
    )
    model = ConvolutionalNeuralNetwork(spec, seed=42)
    model.eval()

    image = [[[0.5 for _ in range(w)] for _ in range(h)] for _ in range(c)]

    # 1. Zero fill
    res_zero = compute_occlusion_sensitivity(
        model=model,
        image=image,
        target_mode=TargetClassMode.PREDICTED_CLASS,
        window_size=(2, 2),
        stride=(2, 2),
        fill_policy=OcclusionFillPolicy.ZERO,
    )
    assert res_zero.attribution_shape == [h, w]
    assert res_zero.statistics.is_finite is True

    # 2. Image mean fill
    res_mean = compute_occlusion_sensitivity(
        model=model,
        image=image,
        target_mode=TargetClassMode.PREDICTED_CLASS,
        window_size=(2, 2),
        stride=(2, 2),
        fill_policy=OcclusionFillPolicy.IMAGE_MEAN,
    )
    assert res_mean.attribution_shape == [h, w]
    assert res_mean.method_metadata["fill_policy"] == "image_mean"
