"""Unit tests for ReconstructionTrainingEngine and linear probing."""

import pytest

from prism.core.enums import ModelFamily
from prism.data.materialized import MaterializedDataset, MaterializedSample
from prism.models.specifications import ModelSpecification
from prism.reconstruction.engine import ReconstructionTrainingEngine
from prism.reconstruction.enums import ReconstructionMethod
from prism.reconstruction.specification import ReconstructionLearningSpecification


def create_synthetic_dataset(
    n_samples: int = 4, label_offset: int = 0
) -> MaterializedDataset:
    """Create tiny synthetic image dataset [3 x 8 x 8]."""
    c, h, w = 3, 8, 8
    samples: list[MaterializedSample] = []
    for i in range(n_samples):
        img = [
            [
                [float((i * 10 + r * 2 + col) % 255) / 255.0 for col in range(w)]
                for r in range(h)
            ]
            for _ in range(c)
        ]
        samples.append(
            MaterializedSample(
                sample_id=f"synth_{i}",
                source_split="train",
                source_index=i,
                data=img,
                target=(i + label_offset) % 2,
            )
        )
    return MaterializedDataset(
        dataset_id=f"synth_ds_{label_offset}",
        split_name="train",
        samples=samples,
    )


def test_vit_masked_patch_reconstruction_training() -> None:
    """Test ViT masked patch reconstruction pretraining loop and parameter updates."""
    vit_spec = ModelSpecification(
        model_id="vit_test_recon",
        name="ViT Test Recon",
        family=ModelFamily.VISION_TRANSFORMER,
        architecture="vit_tiny",
        input_shape=(3, 8, 8),
        num_classes=2,
        hyperparameters={
            "patch_size": 4,
            "embed_dim": 8,
            "num_layers": 1,
            "num_heads": 2,
            "mlp_dim": 16,
        },
    )
    recon_spec = ReconstructionLearningSpecification(
        reconstruction_id="recon_vit_test_1",
        method=ReconstructionMethod.MASKED_PATCH_RECONSTRUCTION,
        encoder_family=ModelFamily.VISION_TRANSFORMER,
        encoder_spec=vit_spec,
        input_shape=(3, 8, 8),
        patch_size=4,
        mask_ratio=0.5,
        epochs=3,
        batch_size=2,
        learning_rate=0.05,
        seed=42,
        dataset_id="synth_ds",
    )

    dataset = create_synthetic_dataset(n_samples=4)
    engine = ReconstructionTrainingEngine()
    report = engine.train(dataset=dataset, spec=recon_spec)

    assert report.epochs_trained == 3
    assert len(report.loss_history) == 3
    assert len(report.masked_mse_history) == 3
    assert report.encoder_snapshot_id == "recon_vit_test_1"
    assert report.diagnostics.mean_reconstruction_error >= 0.0


def test_cnn_denoising_reconstruction_training() -> None:
    """Test CNN spatial denoising autoencoder pretraining loop."""
    cnn_spec = ModelSpecification(
        model_id="cnn_test_recon",
        name="CNN Test Recon",
        family=ModelFamily.CNN,
        architecture="cnn_simple",
        input_shape=(3, 8, 8),
        num_classes=2,
        hyperparameters={
            "conv_channels": [4, 8],
            "kernel_size": 3,
            "fc_dims": [8],
        },
    )
    recon_spec = ReconstructionLearningSpecification(
        reconstruction_id="recon_cnn_test_1",
        method=ReconstructionMethod.DENOISING_AUTOENCODER,
        encoder_family=ModelFamily.CNN,
        encoder_spec=cnn_spec,
        input_shape=(3, 8, 8),
        epochs=2,
        batch_size=2,
        learning_rate=0.05,
        seed=42,
        dataset_id="synth_ds",
    )

    dataset = create_synthetic_dataset(n_samples=4)
    engine = ReconstructionTrainingEngine()
    report = engine.train(dataset=dataset, spec=recon_spec)

    assert report.epochs_trained == 2
    assert len(report.loss_history) == 2
    assert report.diagnostics.latent_variance >= 0.0


def test_reconstruction_label_independence() -> None:
    """Prove that changing ground truth labels does NOT change reconstruction loss."""
    vit_spec = ModelSpecification(
        model_id="vit_label_indep",
        name="ViT Label Indep",
        family=ModelFamily.VISION_TRANSFORMER,
        architecture="vit_tiny",
        input_shape=(3, 8, 8),
        num_classes=2,
        hyperparameters={
            "patch_size": 4,
            "embed_dim": 8,
            "num_layers": 1,
            "num_heads": 2,
            "mlp_dim": 16,
        },
    )
    recon_spec = ReconstructionLearningSpecification(
        reconstruction_id="recon_label_indep",
        method=ReconstructionMethod.MASKED_PATCH_RECONSTRUCTION,
        encoder_family=ModelFamily.VISION_TRANSFORMER,
        encoder_spec=vit_spec,
        input_shape=(3, 8, 8),
        patch_size=4,
        mask_ratio=0.5,
        epochs=2,
        batch_size=2,
        learning_rate=0.05,
        seed=42,
        dataset_id="synth_ds",
    )

    ds_labels_0 = create_synthetic_dataset(n_samples=4, label_offset=0)
    # Inverted labels to test label independence
    ds_labels_1 = create_synthetic_dataset(n_samples=4, label_offset=1)

    engine_a = ReconstructionTrainingEngine()
    report_a = engine_a.train(dataset=ds_labels_0, spec=recon_spec)

    engine_b = ReconstructionTrainingEngine()
    report_b = engine_b.train(dataset=ds_labels_1, spec=recon_spec)

    for loss_a, loss_b in zip(
        report_a.loss_history, report_b.loss_history, strict=True
    ):
        assert pytest.approx(loss_a, abs=1e-7) == loss_b
    assert report_a.parameter_checksum == report_b.parameter_checksum


def test_downstream_linear_probe_evaluation() -> None:
    """Test that downstream linear probe evaluates on frozen representations."""
    vit_spec = ModelSpecification(
        model_id="vit_probe_eval",
        name="ViT Probe Eval",
        family=ModelFamily.VISION_TRANSFORMER,
        architecture="vit_tiny",
        input_shape=(3, 8, 8),
        num_classes=2,
        hyperparameters={
            "patch_size": 4,
            "embed_dim": 8,
            "num_layers": 1,
            "num_heads": 2,
            "mlp_dim": 16,
        },
    )
    recon_spec = ReconstructionLearningSpecification(
        reconstruction_id="recon_vit_probe",
        method=ReconstructionMethod.MASKED_PATCH_RECONSTRUCTION,
        encoder_family=ModelFamily.VISION_TRANSFORMER,
        encoder_spec=vit_spec,
        input_shape=(3, 8, 8),
        patch_size=4,
        mask_ratio=0.5,
        epochs=2,
        batch_size=2,
        learning_rate=0.05,
        seed=42,
        dataset_id="synth_ds",
    )

    train_ds = create_synthetic_dataset(n_samples=4)
    target_ds = create_synthetic_dataset(n_samples=4)

    engine = ReconstructionTrainingEngine()
    report = engine.train(
        dataset=train_ds, spec=recon_spec, downstream_target_dataset=target_ds
    )

    assert report.downstream_linear_probe_accuracy is not None
    assert 0.0 <= report.downstream_linear_probe_accuracy <= 1.0
