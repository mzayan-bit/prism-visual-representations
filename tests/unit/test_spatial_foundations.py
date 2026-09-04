"""Unit tests for spatial transfer foundations: annotations, adapter, heads, losses."""

import pytest

from prism.core.enums import ModelFamily, TaskType
from prism.core.errors import ValidationError
from prism.models.cnn import ConvolutionalNeuralNetwork
from prism.models.resnet import ResidualNeuralNetwork
from prism.models.specifications import ModelSpecification
from prism.models.transformer import VisionTransformer
from prism.spatial import (
    BoundingBox,
    DetectionAnnotation,
    DetectionSample,
    GridDetectionHead,
    GridDetectionLoss,
    PixelCrossEntropyLoss,
    SegmentationConfusionMatrix,
    SegmentationHead,
    SegmentationResizePolicy,
    SpatialRepresentationAdapter,
    compute_iou_xyxy,
    generate_synthetic_spatial_dataset,
    get_available_spatial_layers,
)


def test_bounding_box_validation():
    """Test valid and invalid bounding box configurations."""
    box = BoundingBox(x_min=0.1, y_min=0.2, x_max=0.5, y_max=0.8)
    assert box.width == pytest.approx(0.4)
    assert box.height == pytest.approx(0.6)
    assert box.area == pytest.approx(0.24)
    assert box.center[0] == pytest.approx(0.3)
    assert box.center[1] == pytest.approx(0.5)

    with pytest.raises(ValidationError):
        BoundingBox(x_min=0.6, y_min=0.2, x_max=0.5, y_max=0.8)

    with pytest.raises(ValidationError):
        BoundingBox(x_min=0.1, y_min=0.9, x_max=0.5, y_max=0.8)

    with pytest.raises(ValidationError):
        BoundingBox(x_min=-0.1, y_min=0.2, x_max=0.5, y_max=0.8)

    with pytest.raises(ValidationError):
        BoundingBox(x_min=0.1, y_min=0.2, x_max=1.5, y_max=0.8)


def test_iou_calculation():
    """Test exact bounding box IoU calculation."""
    box1 = BoundingBox(x_min=0.0, y_min=0.0, x_max=0.5, y_max=0.5)
    box2 = BoundingBox(x_min=0.0, y_min=0.0, x_max=0.5, y_max=0.5)
    assert compute_iou_xyxy(box1, box2) == pytest.approx(1.0)

    box3 = BoundingBox(x_min=0.6, y_min=0.6, x_max=0.9, y_max=0.9)
    assert compute_iou_xyxy(box1, box3) == pytest.approx(0.0)

    box4 = BoundingBox(x_min=0.25, y_min=0.0, x_max=0.75, y_max=0.5)
    assert compute_iou_xyxy(box1, box4) == pytest.approx(1.0 / 3.0)


def test_synthetic_spatial_dataset():
    """Test deterministic synthetic dataset generation and alignment."""
    det_samples, seg_samples = generate_synthetic_spatial_dataset(
        num_samples=10,
        image_shape=(3, 16, 16),
        num_classes=3,
        seed=123,
    )
    assert len(det_samples) == 10
    assert len(seg_samples) == 10

    for det, seg in zip(det_samples, seg_samples, strict=True):
        assert det.sample_id == seg.sample_id
        assert len(det.image) == 3
        assert len(det.image[0]) == 16
        assert len(det.image[0][0]) == 16
        assert len(seg.mask) == 16
        assert len(seg.mask[0]) == 16

        for row in seg.mask:
            for val in row:
                assert 0 <= val < 3


def test_spatial_representation_adapter_cnn():
    """Test spatial representation adapter on CNN."""
    spec = ModelSpecification(
        model_id="test_cnn_spatial",
        name="Test CNN",
        architecture="cnn_toy",
        family=ModelFamily.CNN,
        input_shape=(3, 16, 16),
        num_classes=4,
        compatible_tasks=[TaskType.CLASSIFICATION],
        hyperparameters={
            "conv_channels": [8, 16],
            "kernel_sizes": [3, 3],
            "strides": [1, 1],
            "paddings": [1, 1],
            "pool_sizes": [2, 2],
            "pool_strides": [2, 2],
        },
    )
    model = ConvolutionalNeuralNetwork(spec=spec, seed=42)
    layers = get_available_spatial_layers(model)
    assert "conv_0" in layers
    assert "final_spatial" in layers

    adapter = SpatialRepresentationAdapter(model=model, layer_name="conv_0")
    dummy_input = [[[[0.1 for _ in range(16)] for _ in range(16)] for _ in range(3)]]
    features = adapter.extract_spatial_features(dummy_input)

    assert len(features) == 1
    assert len(features[0]) == 8
    assert len(features[0][0]) == 16
    assert len(features[0][0][0]) == 16


def test_spatial_representation_adapter_resnet():
    """Test spatial representation adapter on ResNet."""
    spec = ModelSpecification(
        model_id="test_resnet_spatial",
        name="Test ResNet",
        architecture="resnet_toy",
        family=ModelFamily.RESNET,
        input_shape=(3, 16, 16),
        num_classes=4,
        compatible_tasks=[TaskType.CLASSIFICATION],
        hyperparameters={
            "stem_channels": 8,
            "stages": [
                {"channels": 8, "num_blocks": 1, "stride": 1},
                {"channels": 16, "num_blocks": 1, "stride": 2},
            ],
        },
    )
    model = ResidualNeuralNetwork(spec=spec, seed=42)
    layers = get_available_spatial_layers(model)
    assert "stem" in layers
    assert "final_spatial" in layers

    adapter = SpatialRepresentationAdapter(model=model, layer_name="stem")
    dummy_input = [[[[0.2 for _ in range(16)] for _ in range(16)] for _ in range(3)]]
    features = adapter.extract_spatial_features(dummy_input)

    assert len(features) == 1
    assert len(features[0]) == 8
    assert len(features[0][0]) == 16
    assert len(features[0][0][0]) == 16


def test_spatial_representation_adapter_vit():
    """Test spatial representation adapter on VisionTransformer."""
    spec = ModelSpecification(
        model_id="test_vit_spatial",
        name="Test ViT",
        architecture="vit_toy",
        family=ModelFamily.VISION_TRANSFORMER,
        input_shape=(3, 16, 16),
        num_classes=4,
        compatible_tasks=[TaskType.CLASSIFICATION],
        hyperparameters={
            "patch_size": 4,
            "embed_dim": 16,
            "depth": 2,
            "num_heads": 2,
            "mlp_dim": 32,
        },
    )
    model = VisionTransformer(spec=spec, seed=42)
    layers = get_available_spatial_layers(model)
    assert "patch_embeddings" in layers
    assert "encoder_0" in layers
    assert "final_spatial" in layers

    adapter = SpatialRepresentationAdapter(model=model, layer_name="encoder_0")
    dummy_input = [[[[0.5 for _ in range(16)] for _ in range(16)] for _ in range(3)]]
    features = adapter.extract_spatial_features(dummy_input)

    assert len(features) == 1
    assert len(features[0]) == 16
    assert len(features[0][0]) == 4
    assert len(features[0][0][0]) == 4


def test_grid_detection_head_and_loss():
    """Test GridDetectionHead forward, backward, loss, and decoding."""
    head = GridDetectionHead(in_channels=8, num_classes=3, seed=42)
    loss_fn = GridDetectionLoss(lambda_obj=1.0, lambda_cls=1.0, lambda_box=1.0)

    dummy_features = [[[[0.1 for _ in range(4)] for _ in range(4)] for _ in range(8)]]
    outputs = head.forward(dummy_features)

    assert len(outputs) == 1
    assert len(outputs[0]) == 8
    assert len(outputs[0][0]) == 4
    assert len(outputs[0][0][0]) == 4

    target_box = BoundingBox(x_min=0.2, y_min=0.2, x_max=0.6, y_max=0.6)
    ann = DetectionAnnotation(
        class_id=1, box=target_box, image_width=16, image_height=16
    )
    target_sample = DetectionSample(
        sample_id="test_sample_0",
        image=[[[0.0 for _ in range(16)] for _ in range(16)] for _ in range(3)],
        annotations=[ann],
        dataset_fingerprint="fp123",
    )

    loss_dict, d_preds = loss_fn.compute_loss_and_gradients(
        outputs, [target_sample], num_classes=3
    )
    assert "loss" in loss_dict
    assert loss_dict["loss"] > 0.0
    assert loss_dict["num_positive_cells"] == 1

    d_features = head.backward(d_preds)
    assert len(d_features) == 1
    assert len(d_features[0]) == 8
    assert len(d_features[0][0]) == 4
    assert len(d_features[0][0][0]) == 4

    preds = head.decode_predictions(outputs, objectness_threshold=0.0)
    assert len(preds) == 1
    assert len(preds[0].boxes) == 16


def test_segmentation_head_and_loss():
    """Test SegmentationHead forward, backward, and pixel cross-entropy loss."""
    head = SegmentationHead(
        in_channels=8,
        num_classes=3,
        target_spatial_shape=(16, 16),
        resize_policy=SegmentationResizePolicy.BILINEAR,
        seed=42,
    )
    loss_fn = PixelCrossEntropyLoss()

    dummy_features = [[[[0.2 for _ in range(4)] for _ in range(4)] for _ in range(8)]]
    logits = head.forward(dummy_features)

    assert len(logits) == 1
    assert len(logits[0]) == 3
    assert len(logits[0][0]) == 16
    assert len(logits[0][0][0]) == 16

    target_mask = [[1 for _ in range(16)] for _ in range(16)]
    loss_dict, d_logits = loss_fn.compute_loss_and_gradients(logits, [target_mask])

    assert "loss" in loss_dict
    assert loss_dict["loss"] > 0.0
    assert loss_dict["valid_pixels"] == 256

    d_features = head.backward(d_logits)
    assert len(d_features) == 1
    assert len(d_features[0]) == 8
    assert len(d_features[0][0]) == 4
    assert len(d_features[0][0][0]) == 4

    masks = head.predict_masks(logits)
    assert len(masks) == 1
    assert len(masks[0]) == 16
    assert len(masks[0][0]) == 16


def test_segmentation_confusion_matrix_and_metrics():
    """Test SegmentationConfusionMatrix accumulation and metric calculation."""
    cm = SegmentationConfusionMatrix(num_classes=3)
    gt = [[[0, 1], [2, 1]]]
    pred = [[[0, 1], [1, 1]]]

    cm.update(pred, gt)
    metrics = cm.compute_metrics()

    assert metrics.total_pixels == 4
    assert metrics.pixel_accuracy == pytest.approx(0.75)
    assert metrics.per_class_iou[0] == pytest.approx(1.0)
    assert metrics.per_class_iou[1] == pytest.approx(2.0 / 3.0)
    assert metrics.per_class_iou[2] == pytest.approx(0.0)
