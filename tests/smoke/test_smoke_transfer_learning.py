"""End-to-end smoke test validating Phase 17 transfer learning capabilities."""

import tempfile
from pathlib import Path

from prism.core.enums import ModelFamily
from prism.data.batching import DeterministicBatchLoader
from prism.data.materialized import MaterializedDataset, MaterializedSample
from prism.models.cnn import ConvolutionalNeuralNetwork
from prism.models.resnet import ResidualNeuralNetwork
from prism.models.specifications import ModelSpecification
from prism.models.transformer import VisionTransformer
from prism.transfer.runner import TransferTrainingRunner
from prism.transfer.snapshot import (
    ModelStateSnapshot,
    create_model_state_snapshot,
)
from prism.transfer.specification import (
    TransferLearningSpecification,
    TransferStrategy,
)


def _make_smoke_dataset(
    name: str, num_samples: int = 12, num_classes: int = 2
) -> MaterializedDataset:
    samples = []
    c, h, w = 3, 8, 8
    for i in range(num_samples):
        target = i % num_classes
        img = [
            [[0.1 * (target + 1) for _ in range(w)] for _ in range(h)] for _ in range(c)
        ]
        samples.append(
            MaterializedSample(
                sample_id=f"{name}_{i}",
                source_split="train",
                source_index=i,
                data=img,
                target=target,
            )
        )
    return MaterializedDataset(
        dataset_id=name,
        split_name="train",
        samples=samples,
    )


def test_smoke_transfer_learning_lifecycle() -> None:
    """Comprehensive smoke test for Phase 17 Transfer Learning lifecycle."""
    c, h, w = 3, 8, 8

    # 1. Instantiate CNN, ResNet, and ViT
    specs = {
        "cnn": ModelSpecification(
            model_id="cnn_smoke",
            name="CNN Smoke",
            family=ModelFamily.CNN,
            architecture="cnn_simple",
            input_shape=(c, h, w),
            num_classes=3,
            hyperparameters={
                "conv_channels": [4],
                "kernel_sizes": [3],
                "activation": "relu",
                "use_batch_norm": False,
            },
        ),
        "resnet": ModelSpecification(
            model_id="resnet_smoke",
            name="ResNet Smoke",
            family=ModelFamily.RESNET,
            architecture="resnet_tiny",
            input_shape=(c, h, w),
            num_classes=3,
            hyperparameters={
                "stem_channels": 4,
                "stage_widths": [4],
                "blocks_per_stage": [1],
                "activation": "relu",
                "use_batch_norm": False,
            },
        ),
        "vit": ModelSpecification(
            model_id="vit_smoke",
            name="ViT Smoke",
            family=ModelFamily.VISION_TRANSFORMER,
            architecture="vit_tiny",
            input_shape=(c, h, w),
            num_classes=3,
            hyperparameters={
                "patch_size": 2,
                "embed_dim": 8,
                "depth": 1,
                "num_heads": 2,
                "mlp_ratio": 2.0,
                "activation": "gelu",
            },
        ),
    }

    models = {
        "cnn": ConvolutionalNeuralNetwork(specs["cnn"], seed=42),
        "resnet": ResidualNeuralNetwork(specs["resnet"], seed=42),
        "vit": VisionTransformer(specs["vit"], seed=42),
    }

    tgt_train = _make_smoke_dataset("smoke_tgt_train", num_samples=8, num_classes=2)
    tgt_val = _make_smoke_dataset("smoke_tgt_val", num_samples=4, num_classes=2)
    train_loader = DeterministicBatchLoader(tgt_train, batch_size=4, seed=42)
    val_loader = DeterministicBatchLoader(tgt_val, batch_size=4, seed=42)

    runner = TransferTrainingRunner()

    for arch_name, model in models.items():
        # A. Snapshot source model state
        snapshot = create_model_state_snapshot(
            model, source_experiment_id=f"exp_{arch_name}_smoke"
        )
        assert len(snapshot.parameters) > 0
        assert snapshot.verify_integrity() is True

        # Snapshot file persistence test
        with tempfile.TemporaryDirectory() as tmp_dir:
            snap_path = Path(tmp_dir) / f"{arch_name}_snapshot.json"
            snap_path.write_text(snapshot.to_json())
            loaded_snap = ModelStateSnapshot.from_json(snap_path.read_text())
            assert loaded_snap.source_experiment_id == snapshot.source_experiment_id
            assert loaded_snap.verify_integrity() is True

        # B. Linear probe transfer test
        spec_lp = TransferLearningSpecification(
            transfer_id=f"trans_{arch_name}_lp",
            source_experiment_id=f"exp_{arch_name}_smoke",
            source_model_id=model.model_id,
            source_dataset_id="smoke_src_ds",
            target_dataset_id=tgt_train.dataset_id,
            target_num_classes=2,
            strategy=TransferStrategy.LINEAR_PROBE,
            target_epochs=2,
        )

        rep_lp = runner.run_transfer(
            specification=spec_lp,
            source_snapshot=snapshot,
            target_train_dataset=tgt_train,
            target_train_loader=train_loader,
            target_val_dataset=tgt_val,
            target_val_loader=val_loader,
            run_scratch_comparison=True,
        )

        assert rep_lp.strategy == TransferStrategy.LINEAR_PROBE
        assert rep_lp.freeze_plan.frozen_tensors > 0
        assert rep_lp.scratch_comparison is not None
        assert rep_lp.representation_drift is not None
        assert rep_lp.representation_drift.is_frozen_backbone is True

        # C. Full fine-tune transfer test
        spec_fft = TransferLearningSpecification(
            transfer_id=f"trans_{arch_name}_fft",
            source_experiment_id=f"exp_{arch_name}_smoke",
            source_model_id=model.model_id,
            source_dataset_id="smoke_src_ds",
            target_dataset_id=tgt_train.dataset_id,
            target_num_classes=2,
            strategy=TransferStrategy.FULL_FINE_TUNE,
            target_epochs=2,
        )

        rep_fft = runner.run_transfer(
            specification=spec_fft,
            source_snapshot=snapshot,
            target_train_dataset=tgt_train,
            target_train_loader=train_loader,
            target_val_dataset=tgt_val,
            target_val_loader=val_loader,
            run_scratch_comparison=True,
        )

        assert rep_fft.strategy == TransferStrategy.FULL_FINE_TUNE
        assert rep_fft.freeze_plan.frozen_tensors == 0
        assert rep_fft.representation_drift is not None
