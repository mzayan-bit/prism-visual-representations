"""Unit tests for reconstruction decoders and analytical gradients."""

import pytest

from prism.data.materialized import MaterializedSample
from prism.models.patches import PatchGeometry
from prism.reconstruction.batch import (
    prepare_denoising_batch,
    prepare_masked_patch_batch,
)
from prism.reconstruction.decoders import (
    PatchReconstructionDecoder,
    SpatialReconstructionDecoder,
)
from prism.reconstruction.loss import MaskedMSELoss
from prism.reconstruction.mask import PatchMask
from prism.robustness.corruptions import CorruptionSpecification, CorruptionType


def test_patch_reconstruction_decoder_gradient_check() -> None:
    """Validate analytical gradients with finite differences."""
    in_dim = 4
    patch_dim = 3
    decoder = PatchReconstructionDecoder(
        in_features=in_dim, patch_dim=patch_dim, bias=True, seed=42
    )

    # Input: 1 sample, 2 tokens, in_dim=4
    x = [[[0.5, -0.2, 0.8, 0.1], [0.1, 0.4, -0.3, 0.6]]]
    out = decoder.forward(x)
    assert len(out) == 1 and len(out[0]) == 2 and len(out[0][0]) == patch_dim

    # Upstream gradient
    d_out = [[[1.2, -0.5, 0.3], [-0.4, 0.8, -0.1]]]
    _d_x = decoder.backward(d_out)
    grads = decoder.get_gradients()

    # Numerical gradient check on weights[0][0]
    eps = 1e-5
    params = decoder.get_parameters()
    orig_w = params["weights"][0][0]

    params["weights"][0][0] = orig_w + eps
    decoder.set_parameters(params)
    out_pos = decoder.forward(x)

    params["weights"][0][0] = orig_w - eps
    decoder.set_parameters(params)
    out_neg = decoder.forward(x)

    # Numerical dot product: sum(d_out * (out_pos - out_neg) / 2eps)
    num_grad = 0.0
    for s in range(len(x)):
        for t in range(len(x[0])):
            for p in range(patch_dim):
                num_grad += (
                    d_out[s][t][p] * (out_pos[s][t][p] - out_neg[s][t][p]) / (2.0 * eps)
                )

    analytic_grad = grads["weights"][0][0]
    assert pytest.approx(analytic_grad, abs=1e-4) == num_grad


def test_spatial_reconstruction_decoder() -> None:
    """Test SpatialReconstructionDecoder output shapes and backpropagation."""
    in_dim = 8
    output_shape = (3, 4, 4)
    decoder = SpatialReconstructionDecoder(
        in_features=in_dim, output_shape=output_shape, bias=True, seed=42
    )

    # 2 samples
    latents = [
        [0.1 * i for i in range(in_dim)],
        [-0.1 * i for i in range(in_dim)],
    ]
    imgs = decoder.forward(latents)
    assert len(imgs) == 2
    assert len(imgs[0]) == 3
    assert len(imgs[0][0]) == 4
    assert len(imgs[0][0][0]) == 4

    # Backward
    d_imgs = [
        [[[1.0 for _ in range(4)] for _ in range(4)] for _ in range(3)]
        for _ in range(2)
    ]
    d_latents = decoder.backward(d_imgs)
    assert len(d_latents) == 2
    assert len(d_latents[0]) == in_dim

    grads = decoder.get_gradients()
    assert len(grads["weights"]) == in_dim
    assert len(grads["weights"][0]) == 3 * 4 * 4


def test_masked_mse_loss_hand_calculated() -> None:
    """Verify MaskedMSELoss on hand-calculated numbers and visible exclusion."""
    loss_fn = MaskedMSELoss(reduction="mean")

    # 1 sample, 3 patches, patch_dim = 2
    preds = [[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]]
    targets = [[[1.0, 2.0], [0.0, 0.0], [1.0, 2.0]]]
    # Mask patch 1 only (patches 0 and 2 are visible)
    mask = PatchMask(
        total_patches=3,
        masked_indices=[1],
        visible_indices=[0, 2],
        mask_ratio=0.33,
        sample_id="s1",
        seed_identity="seed1",
    )

    # For patch 1: diffs are (3-0)=3, (4-0)=4. Sum of sq = 9 + 16 = 25.
    # Total scored elements = 1 patch * 2 dim = 2.
    # Mean masked MSE = 25 / 2 = 12.5.
    loss, d_preds, metrics = loss_fn(preds, targets, masks=[mask])
    assert pytest.approx(loss, abs=1e-6) == 12.5
    assert pytest.approx(metrics["masked_mse"], abs=1e-6) == 12.5

    # Gradients on visible patches (0 and 2) MUST be exactly 0.0
    assert d_preds[0][0] == [0.0, 0.0]
    assert d_preds[0][2] == [0.0, 0.0]

    # Gradients on masked patch 1: dL/dp = 2 * diff / 2 = diff
    # For pred[0][1][0]=3: grad = 2*(3)/2 = 3.0
    # For pred[0][1][1]=4: grad = 2*(4)/2 = 4.0
    assert pytest.approx(d_preds[0][1][0], abs=1e-6) == 3.0
    assert pytest.approx(d_preds[0][1][1], abs=1e-6) == 4.0


def test_visible_values_exclusion_invariant() -> None:
    """Visible target values do NOT alter masked loss or gradients."""
    loss_fn = MaskedMSELoss(reduction="mean")

    preds = [[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]]
    targets_a = [[[1.0, 2.0], [0.0, 0.0], [1.0, 2.0]]]
    # Identical masked target (patch 1), different visible targets (0 and 2)
    targets_b = [[[999.0, -888.0], [0.0, 0.0], [123.0, 456.0]]]

    mask = PatchMask(
        total_patches=3,
        masked_indices=[1],
        visible_indices=[0, 2],
        mask_ratio=0.33,
        sample_id="s1",
        seed_identity="seed1",
    )

    loss_a, d_preds_a, metrics_a = loss_fn(preds, targets_a, masks=[mask])
    loss_b, d_preds_b, metrics_b = loss_fn(preds, targets_b, masks=[mask])

    assert pytest.approx(loss_a, abs=1e-7) == loss_b
    assert pytest.approx(metrics_a["masked_mse"], abs=1e-7) == metrics_b["masked_mse"]
    assert d_preds_a[0][1] == d_preds_b[0][1]


def test_masked_mse_numerical_gradient() -> None:
    """Verify analytical gradients of MaskedMSELoss with finite differences."""
    loss_fn = MaskedMSELoss(reduction="mean")

    preds = [[[0.4, -0.7], [1.1, 0.5], [-0.3, 0.9]]]
    targets = [[[0.2, 0.1], [0.8, -0.2], [0.0, 0.5]]]
    mask = PatchMask(
        total_patches=3,
        masked_indices=[0, 2],
        visible_indices=[1],
        mask_ratio=0.67,
        sample_id="s1",
        seed_identity="seed1",
    )

    _loss, d_preds, _ = loss_fn(preds, targets, masks=[mask])

    eps = 1e-5
    # Check gradient on masked element preds[0][0][1]
    orig = preds[0][0][1]

    preds[0][0][1] = orig + eps
    loss_pos, _, _ = loss_fn(preds, targets, masks=[mask])

    preds[0][0][1] = orig - eps
    loss_neg, _, _ = loss_fn(preds, targets, masks=[mask])

    preds[0][0][1] = orig

    num_grad = (loss_pos - loss_neg) / (2.0 * eps)
    analytic_grad = d_preds[0][0][1]
    assert pytest.approx(analytic_grad, abs=1e-4) == num_grad


def test_reconstruction_batch_preparation() -> None:
    """Test masked and denoising batch preparation functions."""
    c, h, w = 3, 8, 8
    samples = [
        MaterializedSample(
            sample_id=f"s_{i}",
            source_split="train",
            source_index=i,
            data=[[[float(i) for _ in range(w)] for _ in range(h)] for _ in range(c)],
            target=i % 2,
        )
        for i in range(2)
    ]

    geom = PatchGeometry.create((h, w), (4, 4), channels=c)
    batch_masked = prepare_masked_patch_batch(
        samples=samples, geometry=geom, epoch=0, mask_ratio=0.5, seed=42
    )

    assert batch_masked.batch_size == 2
    assert batch_masked.masks is not None
    assert len(batch_masked.masks) == 2
    assert batch_masked.masks[0].total_patches == geom.total_patches

    # Denoising batch
    corr_spec = CorruptionSpecification(
        corruption_type=CorruptionType.GAUSSIAN_NOISE, severity=2
    )
    batch_denoised = prepare_denoising_batch(
        samples=samples, corruption_spec=corr_spec, epoch=0, seed=42
    )
    assert batch_denoised.batch_size == 2
    assert batch_denoised.masks is None
    assert len(batch_denoised.inputs) == 2
