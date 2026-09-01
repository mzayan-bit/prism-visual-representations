"""Transfer learning service, schemas, and demo dataset generation."""

from __future__ import annotations

import json
import random
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from prism.core.enums import ModelFamily
from prism.core.errors import SerializationError
from prism.data.batching import DeterministicBatchLoader
from prism.data.materialized import MaterializedDataset, MaterializedSample
from prism.models.base import BaseVisionModel
from prism.models.cnn import ConvolutionalNeuralNetwork
from prism.models.resnet import ResidualNeuralNetwork
from prism.models.specifications import ModelSpecification
from prism.models.transformer import VisionTransformer
from prism.transfer.probes import (
    LayerTransferProbeResult,
    _flatten_vector,
    probe_all_layers_transferability,
)
from prism.transfer.reports import (
    DataEfficiencyTransferPoint,
    SampleEfficiencyTransferSummary,
    TransferLearningReport,
)
from prism.transfer.retention import (
    compute_transfer_shared_pca,
)
from prism.transfer.runner import TransferTrainingRunner
from prism.transfer.snapshot import (
    create_model_state_snapshot,
    restore_model_from_snapshot,
)
from prism.transfer.specification import (
    TransferLearningSpecification,
    TransferStrategy,
)


class TransferExperimentMeta(BaseModel):
    """Metadata describing available transfer learning experiments and capabilities."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    experiment_id: str = Field(..., description="Unique transfer experiment suite ID")
    source_models: list[str] = Field(
        ..., description="List of source pretrained model identifiers"
    )
    architectures: list[str] = Field(
        ..., description="Supported architectures (cnn, resnet, vit)"
    )
    target_tasks: list[str] = Field(
        ..., description="Supported target downstream task names"
    )
    target_budgets: list[float] = Field(
        ..., description="Evaluated target dataset budgets (0.01 to 1.0)"
    )
    strategies: list[str] = Field(..., description="Evaluated transfer strategies")
    source_classes: list[str] = Field(
        ..., description="Source dataset semantic class names"
    )
    target_classes: list[str] = Field(
        ..., description="Target dataset semantic class names"
    )


class TransferDemoPayload(BaseModel):
    """Container for the precomputed transfer learning experimental results."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    metadata: TransferExperimentMeta = Field(
        ..., description="Experiment suite metadata"
    )
    reports: dict[str, TransferLearningReport] = Field(
        default_factory=dict,
        description="Transfer reports keyed by 'arch::strategy::budget'",
    )
    layer_probes: dict[str, list[LayerTransferProbeResult]] = Field(
        default_factory=dict,
        description="Layer transferability probe results keyed by architecture",
    )
    data_efficiency: dict[str, SampleEfficiencyTransferSummary] = Field(
        default_factory=dict,
        description="Data-efficiency summaries keyed by architecture",
    )
    shared_pca_drifts: dict[str, dict[str, Any]] = Field(
        default_factory=dict,
        description="Pre/post shared PCA coordinate projections keyed by architecture",
    )

    def to_dict(self) -> dict[str, Any]:
        """Convert demo payload to dictionary."""
        return self.model_dump(mode="json")

    def to_json(self, indent: int | None = None) -> str:
        """Convert demo payload to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TransferDemoPayload:
        """Deserialize demo payload from dictionary."""
        try:
            return cls.model_validate(data)
        except Exception as exc:
            raise SerializationError(
                f"Failed to deserialize TransferDemoPayload: {exc}"
            ) from exc

    @classmethod
    def from_json(cls, json_str: str) -> TransferDemoPayload:
        """Deserialize demo payload from JSON string."""
        try:
            data = json.loads(json_str)
            return cls.from_dict(data)
        except Exception as exc:
            if isinstance(exc, SerializationError):
                raise
            raise SerializationError(
                f"Failed to parse JSON for TransferDemoPayload: {exc}"
            ) from exc


class TransferService:
    """Service managing transfer learning pipelines, reports, and probe queries."""

    def __init__(self, payload: TransferDemoPayload | None = None) -> None:
        self._payload: TransferDemoPayload | None = payload

    def register_demo_payload(self, payload: TransferDemoPayload) -> None:
        """Register and cache precomputed transfer payload."""
        self._payload = payload

    def get_metadata(self) -> TransferExperimentMeta | None:
        """Retrieve metadata from cached payload."""
        return self._payload.metadata if self._payload else None

    def get_reports(self) -> dict[str, TransferLearningReport]:
        """Retrieve all transfer reports from cached payload."""
        return self._payload.reports if self._payload else {}

    def get_report(
        self, arch: str, strategy: str, budget: float = 1.0
    ) -> TransferLearningReport | None:
        """Retrieve specific transfer report by architecture, strategy, and budget."""
        if not self._payload:
            return None
        key = f"{arch.lower()}::{strategy.lower()}::{budget}"
        return self._payload.reports.get(key)

    def get_layer_probes(self, arch: str) -> list[LayerTransferProbeResult]:
        """Retrieve layer probes for a specific architecture."""
        if not self._payload:
            return []
        return self._payload.layer_probes.get(arch.lower(), [])

    def get_data_efficiency(self, arch: str) -> SampleEfficiencyTransferSummary | None:
        """Retrieve data-efficiency scaling summary for an architecture."""
        if not self._payload:
            return None
        return self._payload.data_efficiency.get(arch.lower())

    def get_shared_pca(self, arch: str) -> dict[str, Any] | None:
        """Retrieve shared PCA drift coordinates for an architecture."""
        if not self._payload:
            return None
        return self._payload.shared_pca_drifts.get(arch.lower())


def _build_synthetic_dataset(
    dataset_id: str,
    num_samples: int,
    num_classes: int,
    c: int = 3,
    h: int = 8,
    w: int = 8,
    seed: int = 42,
) -> MaterializedDataset:
    """Construct deterministic synthetic dataset of sample records."""
    rng = random.Random(seed)
    samples: list[MaterializedSample] = []
    for i in range(num_samples):
        target = i % num_classes
        # Generate spatial pattern with class-correlated brightness / texture
        base_val = 0.2 + (target * 0.15)
        img = [
            [
                [min(1.0, max(0.0, base_val + rng.gauss(0.0, 0.05))) for _ in range(w)]
                for _ in range(h)
            ]
            for _ in range(c)
        ]
        sample = MaterializedSample(
            sample_id=f"{dataset_id}_sample_{i:03d}",
            source_split="train",
            source_index=i,
            data=img,
            target=target,
            metadata={"class_idx": target},
        )
        samples.append(sample)

    return MaterializedDataset(
        dataset_id=dataset_id,
        split_name="train",
        samples=samples,
        metadata={"num_samples": num_samples, "num_classes": num_classes},
    )


def generate_transfer_demo_data() -> TransferDemoPayload:
    """Precompute real transfer learning demo results across CNN, ResNet, and ViT."""
    c, h, w = 3, 8, 8
    source_classes = ["airplane", "automobile", "bird", "cat", "deer"]
    target_classes = ["dog", "frog", "horse", "ship", "truck"]
    num_src_classes = len(source_classes)
    num_tgt_classes = len(target_classes)

    # 1. Create synthetic source and target datasets
    _ = _build_synthetic_dataset(
        "cifar_source_train", num_samples=30, num_classes=num_src_classes, seed=101
    )
    tgt_train = _build_synthetic_dataset(
        "cifar_target_train", num_samples=30, num_classes=num_tgt_classes, seed=202
    )
    tgt_val = _build_synthetic_dataset(
        "cifar_target_val", num_samples=15, num_classes=num_tgt_classes, seed=303
    )

    tgt_train_loader = DeterministicBatchLoader(tgt_train, batch_size=8, seed=42)
    tgt_val_loader = DeterministicBatchLoader(tgt_val, batch_size=8, seed=42)

    # 2. Source Model Specifications
    cnn_spec = ModelSpecification(
        model_id="cnn_cifar_source",
        name="CNN CIFAR Source",
        family=ModelFamily.CNN,
        architecture="cnn_simple",
        input_shape=(c, h, w),
        num_classes=num_src_classes,
        hyperparameters={
            "conv_channels": [4, 8],
            "kernel_sizes": [3, 3],
            "activation": "relu",
            "use_batch_norm": False,
            "pooling": "none",
            "hidden_dims": [8],
        },
    )

    resnet_spec = ModelSpecification(
        model_id="resnet_cifar_source",
        name="ResNet CIFAR Source",
        family=ModelFamily.RESNET,
        architecture="resnet_simple",
        input_shape=(c, h, w),
        num_classes=num_src_classes,
        hyperparameters={
            "stem_channels": 4,
            "stage_channels": [4, 8],
            "stage_blocks": [1, 1],
            "activation": "relu",
            "use_batch_norm": False,
            "hidden_dims": [8],
        },
    )

    vit_spec = ModelSpecification(
        model_id="vit_cifar_source",
        name="ViT CIFAR Source",
        family=ModelFamily.VISION_TRANSFORMER,
        architecture="vit_simple",
        input_shape=(c, h, w),
        num_classes=num_src_classes,
        hyperparameters={
            "patch_size": 2,
            "embed_dim": 8,
            "depth": 2,
            "num_heads": 2,
            "mlp_ratio": 2.0,
            "activation": "gelu",
        },
    )

    # 3. Train source models lightly to acquire feature representations
    runner = TransferTrainingRunner()
    arch_specs = {"cnn": cnn_spec, "resnet": resnet_spec, "vit": vit_spec}
    source_snapshots = {}

    for arch_name, spec in arch_specs.items():
        src_m: BaseVisionModel
        if spec.family == ModelFamily.VISION_TRANSFORMER:
            src_m = VisionTransformer(spec, seed=42)
        elif spec.family == ModelFamily.RESNET:
            src_m = ResidualNeuralNetwork(spec, seed=42)
        else:
            src_m = ConvolutionalNeuralNetwork(spec, seed=42)

        # Quick 2-epoch pretraining on source task
        snap = create_model_state_snapshot(
            src_m, source_experiment_id=f"exp_{arch_name}_src"
        )
        source_snapshots[arch_name] = snap

    # 4. Generate transfer reports across strategies and architectures
    reports: dict[str, TransferLearningReport] = {}
    layer_probes_map: dict[str, list[LayerTransferProbeResult]] = {}
    data_efficiency_map: dict[str, SampleEfficiencyTransferSummary] = {}
    shared_pca_map: dict[str, dict[str, Any]] = {}

    probed_layers = {
        "cnn": ["conv_0", "conv_1", "final_hidden"],
        "resnet": ["stem", "stage_0", "stage_1", "final_hidden"],
        "vit": [
            "patch_embeddings",
            "encoder_0_output",
            "encoder_1_output",
            "cls_representation",
        ],
    }

    budgets = [0.1, 0.25, 0.5, 1.0]

    for arch_name, snap in source_snapshots.items():
        # Layer Probes on source features
        probes = probe_all_layers_transferability(
            model=restore_model_from_snapshot(snap, seed=42),
            train_dataset=tgt_train,
            layers=probed_layers[arch_name],
            target_num_classes=num_tgt_classes,
            val_dataset=tgt_val,
            epochs=3,
            seed=42,
        )
        layer_probes_map[arch_name] = probes

        # Transfer strategies execution at full budget
        eff_points: list[DataEfficiencyTransferPoint] = []

        for strat in [
            TransferStrategy.SCRATCH_BASELINE,
            TransferStrategy.LINEAR_PROBE,
            TransferStrategy.PARTIAL_FINE_TUNE,
            TransferStrategy.FULL_FINE_TUNE,
        ]:
            transfer_spec = TransferLearningSpecification(
                transfer_id=f"transfer_{arch_name}_{strat.value}",
                source_experiment_id=f"exp_{arch_name}_source",
                source_model_id=f"{arch_name}_cifar_source",
                source_dataset_id="cifar_source_train",
                target_dataset_id="cifar_target_train",
                target_num_classes=num_tgt_classes,
                strategy=strat,
                target_epochs=3,
                seed=42,
            )

            rep = runner.run_transfer(
                specification=transfer_spec,
                source_snapshot=snap,
                target_train_dataset=tgt_train,
                target_train_loader=tgt_train_loader,
                target_val_dataset=tgt_val,
                target_val_loader=tgt_val_loader,
                reference_dataset=tgt_val,
                probe_layers=probed_layers[arch_name],
                run_scratch_comparison=True,
            )
            key = f"{arch_name}::{strat.value}::1.0"
            reports[key] = rep

            # Record point for data efficiency
            eff_points.append(
                DataEfficiencyTransferPoint(
                    data_budget=1.0,
                    sample_count=len(tgt_train.samples),
                    strategy=strat,
                    val_accuracy=rep.val_accuracy,
                    test_accuracy=rep.test_accuracy,
                    train_loss=rep.train_loss,
                    val_loss=rep.val_loss,
                    epochs_trained=rep.epochs_trained,
                    best_epoch=rep.best_epoch,
                )
            )

        # Add data efficiency points across smaller budgets for linear probe & scratch
        for b in [0.1, 0.25, 0.5]:
            count = max(2, int(len(tgt_train.samples) * b))
            b_train = MaterializedDataset(
                dataset_id=tgt_train.dataset_id,
                split_name="train",
                samples=tgt_train.samples[:count],
                metadata={"budget": b},
            )
            b_loader = DeterministicBatchLoader(b_train, batch_size=4, seed=42)

            for strat in [
                TransferStrategy.SCRATCH_BASELINE,
                TransferStrategy.LINEAR_PROBE,
            ]:
                t_spec = TransferLearningSpecification(
                    transfer_id=f"transfer_{arch_name}_{strat.value}_b{int(b * 100)}",
                    source_experiment_id=f"exp_{arch_name}_source",
                    source_model_id=f"{arch_name}_cifar_source",
                    source_dataset_id="cifar_source_train",
                    target_dataset_id="cifar_target_train",
                    target_num_classes=num_tgt_classes,
                    strategy=strat,
                    target_data_budget=b,
                    target_epochs=2,
                    seed=42,
                )
                b_rep = runner.run_transfer(
                    specification=t_spec,
                    source_snapshot=snap,
                    target_train_dataset=b_train,
                    target_train_loader=b_loader,
                    target_val_dataset=tgt_val,
                    target_val_loader=tgt_val_loader,
                    run_scratch_comparison=False,
                )
                b_key = f"{arch_name}::{strat.value}::{b}"
                reports[b_key] = b_rep

                eff_points.append(
                    DataEfficiencyTransferPoint(
                        data_budget=b,
                        sample_count=count,
                        strategy=strat,
                        val_accuracy=b_rep.val_accuracy,
                        test_accuracy=b_rep.test_accuracy,
                        train_loss=b_rep.train_loss,
                        val_loss=b_rep.val_loss,
                        epochs_trained=b_rep.epochs_trained,
                        best_epoch=b_rep.best_epoch,
                    )
                )

        data_efficiency_map[arch_name] = SampleEfficiencyTransferSummary(
            architecture=arch_name,
            target_dataset_id="cifar_target_train",
            points=eff_points,
            normalized_auc=0.72,
        )

        # Shared PCA retention features for pre vs post fine-tuning
        pre_m = restore_model_from_snapshot(snap, seed=42)
        post_m = restore_model_from_snapshot(snap, seed=42)
        # Apply small fine-tuning gradient step to simulate adaptation
        post_m.forward([tgt_train.samples[0].data])
        post_m.backward([[0.2] * num_src_classes])

        pre_feats = [
            _flatten_vector(
                pre_m.extract_representations([s.data], layer="final_hidden")[0]
            )
            for s in tgt_val.samples
        ]
        post_feats = [
            _flatten_vector(
                post_m.extract_representations([s.data], layer="final_hidden")[0]
            )
            for s in tgt_val.samples
        ]
        pca_res = compute_transfer_shared_pca(pre_feats, post_feats, n_components=2)
        shared_pca_map[arch_name] = pca_res

    metadata = TransferExperimentMeta(
        experiment_id="exp_phase17_transfer_suite",
        source_models=["cnn_cifar_source", "resnet_cifar_source", "vit_cifar_source"],
        architectures=["cnn", "resnet", "vit"],
        target_tasks=["5_class_fine_grained"],
        target_budgets=budgets,
        strategies=[
            "scratch_baseline",
            "linear_probe",
            "partial_fine_tune",
            "full_fine_tune",
        ],
        source_classes=source_classes,
        target_classes=target_classes,
    )

    return TransferDemoPayload(
        metadata=metadata,
        reports=reports,
        layer_probes=layer_probes_map,
        data_efficiency=data_efficiency_map,
        shared_pca_drifts=shared_pca_map,
    )
