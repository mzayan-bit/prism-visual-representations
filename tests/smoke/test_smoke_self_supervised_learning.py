"""End-to-end smoke test validating Phase 18 Self-Supervised Learning lifecycle."""

import tempfile
from pathlib import Path

from prism.core.enums import ModelFamily
from prism.data.batching import DeterministicBatchLoader
from prism.data.materialized import MaterializedDataset, MaterializedSample
from prism.models.specifications import ModelSpecification
from prism.ssl.engine import SelfSupervisedTrainingEngine
from prism.ssl.specification import SelfSupervisedTrainingSpecification
from prism.transfer.runner import TransferTrainingRunner
from prism.transfer.snapshot import ModelStateSnapshot
from prism.transfer.specification import (
    TransferLearningSpecification,
    TransferStrategy,
)


def _make_smoke_dataset(
    name: str, num_samples: int = 12, num_classes: int = 2
) -> MaterializedDataset:
    c, h, w = 3, 8, 8
    samples: list[MaterializedSample] = []
    for i in range(num_samples):
        cls_id = i % num_classes
        # Deterministic synthetic image pattern
        img = [
            [
                [float(cls_id * 30 + r * w + col) / 255.0 for col in range(w)]
                for r in range(h)
            ]
            for _ in range(c)
        ]
        samples.append(
            MaterializedSample(
                sample_id=f"{name}_sample_{i}",
                source_split="train",
                source_index=i,
                data=img,
                target=cls_id,
            )
        )
    return MaterializedDataset(
        dataset_id=name,
        split_name="train",
        samples=samples,
    )


def test_smoke_self_supervised_learning_lifecycle() -> None:
    """Comprehensive smoke test for Phase 18 Self-Supervised Learning lifecycle."""
    c, h, w = 3, 8, 8

    # 1. Unlabeled Pretraining Dataset
    ssl_dataset = _make_smoke_dataset(
        "ssl_pretrain_data", num_samples=12, num_classes=3
    )

    # 2. Target Evaluation Dataset for Downstream Linear Probing
    tgt_train = _make_smoke_dataset("target_train_data", num_samples=8, num_classes=2)
    tgt_val = _make_smoke_dataset("target_val_data", num_samples=4, num_classes=2)

    train_loader = DeterministicBatchLoader(dataset=tgt_train, batch_size=4, seed=42)
    val_loader = DeterministicBatchLoader(dataset=tgt_val, batch_size=4, seed=42)

    # 3. Model Specifications (CNN, ResNet, ViT)
    specs = {
        "cnn": ModelSpecification(
            model_id="cnn_smoke",
            name="CNN Smoke",
            family=ModelFamily.CNN,
            architecture="cnn_simple",
            input_shape=(c, h, w),
            num_classes=4,
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
            num_classes=4,
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
            num_classes=4,
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

    ssl_engine = SelfSupervisedTrainingEngine()
    transfer_runner = TransferTrainingRunner()

    for arch_name, spec in specs.items():
        # A. SimCLR Pretraining
        ssl_spec = SelfSupervisedTrainingSpecification(
            ssl_id=f"ssl_{arch_name}_smoke",
            encoder_family=spec.family,
            encoder_spec=spec,
            dataset_id=ssl_dataset.dataset_id,
            projection_hidden_dim=8,
            projection_out_dim=4,
            temperature=0.5,
            epochs=2,
            batch_size=4,
            learning_rate=0.05,
            seed=42,
        )

        _encoder, snapshot, ssl_report = ssl_engine.train_ssl(
            specification=ssl_spec, dataset=ssl_dataset
        )

        assert len(ssl_report.loss_trajectory) == 2
        assert ssl_report.collapse_summary.total_dimensions > 0
        assert snapshot.verify_integrity() is True

        # Snapshot file persistence test
        with tempfile.TemporaryDirectory() as tmp_dir:
            snap_path = Path(tmp_dir) / f"{arch_name}_ssl_snapshot.json"
            snap_path.write_text(snapshot.to_json())
            loaded_snap = ModelStateSnapshot.from_json(snap_path.read_text())
            assert loaded_snap.source_experiment_id == snapshot.source_experiment_id
            assert loaded_snap.verify_integrity() is True

        # B. Downstream Linear Probe Evaluation
        transfer_spec = TransferLearningSpecification(
            transfer_id=f"trans_{arch_name}_ssl_lp",
            source_experiment_id=ssl_spec.ssl_id,
            source_model_id=spec.model_id,
            source_dataset_id=ssl_dataset.dataset_id,
            target_dataset_id=tgt_train.dataset_id,
            target_num_classes=2,
            strategy=TransferStrategy.LINEAR_PROBE,
            target_epochs=2,
        )

        rep_lp = transfer_runner.run_transfer(
            specification=transfer_spec,
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
