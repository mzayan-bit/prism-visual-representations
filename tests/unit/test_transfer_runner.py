"""Unit tests for TransferTrainingRunner end-to-end orchestration."""

from prism.core.enums import ModelFamily
from prism.data.batching import DeterministicBatchLoader
from prism.data.materialized import MaterializedDataset, MaterializedSample
from prism.models.cnn import ConvolutionalNeuralNetwork
from prism.models.specifications import ModelSpecification
from prism.transfer.runner import TransferTrainingRunner
from prism.transfer.snapshot import create_model_state_snapshot
from prism.transfer.specification import (
    NormalizationTransferPolicy,
    TransferLearningSpecification,
    TransferStrategy,
)


def _make_dataset(
    name: str, num_samples: int = 12, num_classes: int = 2
) -> MaterializedDataset:
    samples = []
    c, h, w = 3, 8, 8
    for i in range(num_samples):
        target = i % num_classes
        img = [
            [[0.05 * (i + 1) * (c_idx + 1) for _ in range(w)] for _ in range(h)]
            for c_idx in range(c)
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


def test_transfer_runner_linear_probe() -> None:
    """Test transfer runner execution with Linear Probe strategy."""
    c, h, w = 3, 8, 8
    spec = ModelSpecification(
        model_id="cnn_src",
        name="CNN Source",
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
    )
    src_model = ConvolutionalNeuralNetwork(spec, seed=42)
    snapshot = create_model_state_snapshot(src_model, source_experiment_id="exp_src")

    tgt_train = _make_dataset("tgt_train", num_samples=8, num_classes=2)
    tgt_val = _make_dataset("tgt_val", num_samples=6, num_classes=2)
    train_loader = DeterministicBatchLoader(tgt_train, batch_size=4, seed=42)
    val_loader = DeterministicBatchLoader(tgt_val, batch_size=4, seed=42)

    transfer_spec = TransferLearningSpecification(
        transfer_id="transfer_lp_test",
        source_experiment_id="exp_src",
        source_model_id="cnn_src",
        source_dataset_id="src_ds",
        target_dataset_id="tgt_train",
        target_num_classes=2,
        strategy=TransferStrategy.LINEAR_PROBE,
        normalization_policy=NormalizationTransferPolicy.FREEZE_SOURCE_STATS,
        target_epochs=2,
    )

    runner = TransferTrainingRunner()
    report = runner.run_transfer(
        specification=transfer_spec,
        source_snapshot=snapshot,
        target_train_dataset=tgt_train,
        target_train_loader=train_loader,
        target_val_dataset=tgt_val,
        target_val_loader=val_loader,
        probe_layers=["final_hidden"],
        run_scratch_comparison=True,
    )

    assert report.transfer_id == "transfer_lp_test"
    assert report.strategy == TransferStrategy.LINEAR_PROBE
    assert 0.0 <= report.train_accuracy <= 1.0
    assert 0.0 <= report.val_accuracy <= 1.0
    assert report.epochs_trained == 2
    assert report.scratch_comparison is not None
    assert len(report.layer_probes) == 1
    assert report.representation_drift is not None
    assert report.representation_drift.is_frozen_backbone is True
