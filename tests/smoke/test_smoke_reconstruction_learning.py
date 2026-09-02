"""End-to-End CPU smoke test for reconstruction-based representation learning."""

import json

from prism.core.enums import ModelFamily
from prism.data.materialized import MaterializedDataset, MaterializedSample
from prism.models.specifications import ModelSpecification
from prism.reconstruction.engine import ReconstructionTrainingEngine
from prism.reconstruction.enums import ReconstructionMethod
from prism.reconstruction.specification import ReconstructionLearningSpecification


def test_smoke_reconstruction_learning_pipeline() -> None:
    """Execute end-to-end smoke test verifying masked reconstruction.

    Validates:
    - Synthetic images -> Masking -> ViT Encoder -> Latent Tokens -> Decoder
    - Masked MSE Loss -> Backward -> Optimizer Step
    - Snapshot -> Linear Probing -> Report Serialization
    """
    # 1. Create tiny synthetic dataset
    c, h, w = 3, 8, 8
    samples: list[MaterializedSample] = []
    for i in range(4):
        img = [
            [
                [float((i * 15 + r * 3 + col * 2) % 255) / 255.0 for col in range(w)]
                for r in range(h)
            ]
            for _ in range(c)
        ]
        samples.append(
            MaterializedSample(
                sample_id=f"smoke_sample_{i}",
                source_split="train",
                source_index=i,
                data=img,
                target=i % 2,
            )
        )
    dataset = MaterializedDataset(
        dataset_id="smoke_reconstruction_dataset",
        split_name="train",
        samples=samples,
    )

    # 2. Configure ViT and Reconstruction Specifications
    vit_spec = ModelSpecification(
        model_id="vit_smoke_recon",
        name="ViT Smoke Recon",
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
        reconstruction_id="smoke_recon_exp_1",
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
        dataset_id="smoke_reconstruction_dataset",
    )

    # 3. Execute pretraining and downstream linear probing
    engine = ReconstructionTrainingEngine()
    report = engine.train(
        dataset=dataset, spec=recon_spec, downstream_target_dataset=dataset
    )

    # 4. Assert report correctness and metrics
    assert report.reconstruction_id == "smoke_recon_exp_1"
    assert report.method == ReconstructionMethod.MASKED_PATCH_RECONSTRUCTION
    assert report.epochs_trained == 2
    assert len(report.loss_history) == 2
    assert len(report.masked_mse_history) == 2
    assert report.downstream_linear_probe_accuracy is not None
    assert 0.0 <= report.downstream_linear_probe_accuracy <= 1.0

    # 5. Assert report JSON serialization and deserialization
    serialized = json.dumps(report.model_dump(), indent=2)
    deserialized = json.loads(serialized)
    assert deserialized["reconstruction_id"] == "smoke_recon_exp_1"
    assert "diagnostics" in deserialized
    assert "parameter_checksum" in deserialized
