"""Unit tests for spatial heads and analytical spatial loss functions."""

import copy

import pytest

from prism.spatial.annotations import BoundingBox, DetectionAnnotation, DetectionSample
from prism.spatial.enums import SegmentationResizePolicy
from prism.spatial.heads import GridDetectionHead, SegmentationHead
from prism.spatial.losses import GridDetectionLoss, PixelCrossEntropyLoss


def test_grid_detection_head_forward_and_shapes():
    """Test GridDetectionHead output shapes and dimension decomposition."""
    in_channels = 8
    num_classes = 3
    head = GridDetectionHead(in_channels=in_channels, num_classes=num_classes, seed=42)

    # Input: [N=2, C=8, H=4, W=4]
    features = [
        [[[0.1 for _ in range(4)] for _ in range(4)] for _ in range(in_channels)]
        for _ in range(2)
    ]

    out = head.forward(features)
    # Output channels: 1 (obj) + 3 (classes) + 4 (box) = 8
    assert len(out) == 2
    assert len(out[0]) == 8
    assert len(out[0][0]) == 4
    assert len(out[0][0][0]) == 4


def test_grid_detection_head_decode_predictions():
    """Test decoding detection predictions from raw head output."""
    in_channels = 4
    num_classes = 2
    head = GridDetectionHead(in_channels=in_channels, num_classes=num_classes, seed=42)

    # Synthetic grid logits: initialize all to negative (no object)
    num_out = 1 + num_classes + 4
    grid_logits = [
        [[[-10.0 for _ in range(3)] for _ in range(3)] for _ in range(num_out)]
    ]
    # Set high objectness logit at (1, 1)
    grid_logits[0][0][1][1] = 5.0
    # Set class 1 logit high
    grid_logits[0][2][1][1] = 3.0
    # Set box offsets: cx=0.5, cy=0.5, w=0.5, h=0.5
    grid_logits[0][3][1][1] = 0.5  # cx
    grid_logits[0][4][1][1] = 0.5  # cy
    grid_logits[0][5][1][1] = 0.5  # w
    grid_logits[0][6][1][1] = 0.5  # h

    preds = head.decode_predictions(grid_logits, objectness_threshold=0.5)
    assert len(preds) == 1
    assert len(preds[0].boxes) == 1
    assert preds[0].class_ids[0] == 1
    assert preds[0].confidences[0] > 0.5


def test_grid_detection_loss_forward_and_backward():
    """Test GridDetectionLoss computation and gradient backpropagation."""
    loss_fn = GridDetectionLoss(lambda_obj=1.0, lambda_cls=1.0, lambda_box=1.0)
    in_channels = 4
    num_classes = 2
    head = GridDetectionHead(in_channels=in_channels, num_classes=num_classes, seed=42)

    features = [
        [[[0.2 for _ in range(4)] for _ in range(4)] for _ in range(in_channels)]
    ]
    logits = head.forward(features)

    box = BoundingBox(x_min=0.25, y_min=0.25, x_max=0.5, y_max=0.5)
    targets = [
        DetectionSample(
            sample_id="det_01",
            image=[[[0.0 for _ in range(4)] for _ in range(4)]],
            annotations=[DetectionAnnotation(class_id=1, box=box)],
        )
    ]

    loss_metrics, grad_logits = loss_fn.compute_loss_and_gradients(
        logits, targets, num_classes=num_classes
    )
    assert loss_metrics["loss"] > 0.0
    assert "obj_loss" in loss_metrics
    assert "cls_loss" in loss_metrics
    assert "box_loss" in loss_metrics

    # Test backward gradient shapes
    assert len(grad_logits) == len(logits)
    assert len(grad_logits[0]) == len(logits[0])
    assert len(grad_logits[0][0]) == len(logits[0][0])


def test_grid_detection_loss_numerical_gradient():
    """Verify GridDetectionLoss backward gradients via finite differences."""
    loss_fn = GridDetectionLoss(lambda_obj=1.0, lambda_cls=1.0, lambda_box=1.0)
    num_classes = 2
    c_out = 1 + num_classes + 4  # 7
    h, w = 2, 2

    # Deterministic logits
    logits = [
        [
            [[0.1 * (c + r + ch) for c in range(w)] for r in range(h)]
            for ch in range(c_out)
        ]
    ]
    box = BoundingBox(x_min=0.1, y_min=0.1, x_max=0.4, y_max=0.4)
    targets = [
        DetectionSample(
            sample_id="grad_check",
            image=[[[0.0 for _ in range(w)] for _ in range(h)]],
            annotations=[DetectionAnnotation(class_id=0, box=box)],
        )
    ]

    _, analytical_grad = loss_fn.compute_loss_and_gradients(
        logits, targets, num_classes=num_classes
    )

    # Numerical finite differences
    eps = 1e-5
    for ch in range(c_out):
        for r in range(h):
            for c in range(w):
                logits_pos = copy.deepcopy(logits)
                logits_neg = copy.deepcopy(logits)
                logits_pos[0][ch][r][c] += eps
                logits_neg[0][ch][r][c] -= eps

                loss_pos, _ = loss_fn.compute_loss_and_gradients(
                    logits_pos, targets, num_classes=num_classes
                )
                loss_neg, _ = loss_fn.compute_loss_and_gradients(
                    logits_neg, targets, num_classes=num_classes
                )

                num_grad = (loss_pos["loss"] - loss_neg["loss"]) / (2 * eps)
                ana_grad = analytical_grad[0][ch][r][c]
                assert ana_grad == pytest.approx(num_grad, abs=1e-3, rel=1e-2)


def test_segmentation_head_forward_and_resizing():
    """Test SegmentationHead projection and bilinear/nearest upsampling."""
    in_channels = 8
    num_classes = 3
    target_spatial_shape = (8, 8)
    head_nearest = SegmentationHead(
        in_channels=in_channels,
        num_classes=num_classes,
        target_spatial_shape=target_spatial_shape,
        resize_policy=SegmentationResizePolicy.NEAREST,
        seed=42,
    )

    # Feature input [N=1, C=8, H=4, W=4]
    features = [
        [[[0.5 for _ in range(4)] for _ in range(4)] for _ in range(in_channels)]
    ]

    out_nearest = head_nearest.forward(features)
    assert len(out_nearest) == 1
    assert len(out_nearest[0]) == num_classes
    assert len(out_nearest[0][0]) == 8
    assert len(out_nearest[0][0][0]) == 8

    # Test predict_masks
    pred_masks = head_nearest.predict_masks(out_nearest)
    assert len(pred_masks) == 1
    assert len(pred_masks[0]) == 8
    assert len(pred_masks[0][0]) == 8

    # Bilinear policy
    head_bilinear = SegmentationHead(
        in_channels=in_channels,
        num_classes=num_classes,
        target_spatial_shape=target_spatial_shape,
        resize_policy=SegmentationResizePolicy.BILINEAR,
        seed=42,
    )
    out_bilinear = head_bilinear.forward(features)
    assert len(out_bilinear[0][0]) == 8


def test_pixel_cross_entropy_loss_forward_and_backward():
    """Test PixelCrossEntropyLoss forward evaluation and analytical backward."""
    loss_fn = PixelCrossEntropyLoss()
    num_classes = 3
    h, w = 4, 4

    logits = [
        [
            [[1.0 for _ in range(w)] for _ in range(h)],
            [[0.0 for _ in range(w)] for _ in range(h)],
            [[0.0 for _ in range(w)] for _ in range(h)],
        ]
    ]
    target_mask = [[[0 for _ in range(w)] for _ in range(h)]]

    loss_metrics, grad = loss_fn.compute_loss_and_gradients(logits, target_mask)
    assert loss_metrics["loss"] > 0.0

    assert len(grad) == 1
    assert len(grad[0]) == num_classes
    assert len(grad[0][0]) == h
    assert len(grad[0][0][0]) == w


def test_pixel_cross_entropy_numerical_gradient():
    """Verify PixelCrossEntropyLoss backward via numerical finite differences."""
    loss_fn = PixelCrossEntropyLoss()
    num_classes = 2
    h, w = 2, 2

    logits = [[[[0.2, 0.4], [0.6, 0.8]], [[0.1, 0.3], [0.5, 0.7]]]]
    target_mask = [[[0, 1], [1, 0]]]

    _, analytical_grad = loss_fn.compute_loss_and_gradients(logits, target_mask)

    eps = 1e-5
    for c in range(num_classes):
        for r in range(h):
            for col in range(w):
                pos = copy.deepcopy(logits)
                neg = copy.deepcopy(logits)
                pos[0][c][r][col] += eps
                neg[0][c][r][col] -= eps

                loss_pos, _ = loss_fn.compute_loss_and_gradients(pos, target_mask)
                loss_neg, _ = loss_fn.compute_loss_and_gradients(neg, target_mask)

                num_grad = (loss_pos["loss"] - loss_neg["loss"]) / (2 * eps)
                ana_grad = analytical_grad[0][c][r][col]
                assert ana_grad == pytest.approx(num_grad, abs=1e-3, rel=1e-2)
