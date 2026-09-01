"""Unit tests for layer transferability linear probes across architectures."""

from prism.core.enums import ModelFamily
from prism.data.materialized import MaterializedDataset, MaterializedSample
from prism.models.cnn import ConvolutionalNeuralNetwork
from prism.models.specifications import ModelSpecification
from prism.models.transformer import VisionTransformer
from prism.transfer.probes import (
    probe_all_layers_transferability,
    probe_layer_transferability,
)


def _make_dataset(num_samples: int = 16, num_classes: int = 3) -> MaterializedDataset:
    samples = []
    c, h, w = 3, 8, 8
    for i in range(num_samples):
        target = i % num_classes
        img = [[[0.2 * target for _ in range(w)] for _ in range(h)] for _ in range(c)]
        samples.append(
            MaterializedSample(
                sample_id=f"probe_sample_{i}",
                source_split="train",
                source_index=i,
                data=img,
                target=target,
            )
        )
    return MaterializedDataset(
        dataset_id="test_probe_ds",
        split_name="train",
        samples=samples,
    )


def test_probe_layer_transferability_cnn() -> None:
    """Test linear probe evaluation on intermediate CNN layers."""
    c, h, w, num_classes = 3, 8, 8, 3
    spec = ModelSpecification(
        model_id="cnn_probe_test",
        name="CNN Probe Test",
        family=ModelFamily.CNN,
        architecture="cnn_simple",
        input_shape=(c, h, w),
        num_classes=num_classes,
        hyperparameters={
            "conv_channels": [4, 8],
            "kernel_sizes": [3, 3],
            "activation": "relu",
            "use_batch_norm": False,
        },
    )
    model = ConvolutionalNeuralNetwork(spec, seed=42)
    dataset = _make_dataset(num_samples=12, num_classes=3)

    result = probe_layer_transferability(
        model=model,
        train_dataset=dataset,
        layer="final_hidden",
        target_num_classes=3,
        epochs=3,
        seed=42,
    )

    assert result.layer_name == "final_hidden"
    assert result.representation_dim > 0
    assert 0.0 <= result.train_accuracy <= 1.0
    assert 0.0 <= result.val_accuracy <= 1.0
    assert result.epochs_trained == 3


def test_probe_all_layers_transferability_vit() -> None:
    """Test multi-layer probing on Vision Transformer."""
    c, h, w, num_classes = 3, 8, 8, 2
    spec = ModelSpecification(
        model_id="vit_probe_test",
        name="ViT Probe Test",
        family=ModelFamily.VISION_TRANSFORMER,
        architecture="vit_simple",
        input_shape=(c, h, w),
        num_classes=num_classes,
        hyperparameters={
            "patch_size": 2,
            "embed_dim": 8,
            "depth": 1,
            "num_heads": 2,
            "mlp_ratio": 2.0,
            "activation": "gelu",
        },
    )
    model = VisionTransformer(spec, seed=42)
    dataset = _make_dataset(num_samples=10, num_classes=2)

    layers = ["patch_embeddings", "encoder_0_output", "cls_representation"]
    results = probe_all_layers_transferability(
        model=model,
        train_dataset=dataset,
        layers=layers,
        target_num_classes=2,
        epochs=2,
        seed=42,
    )

    assert len(results) == len(layers)
    for res, expected_layer in zip(results, layers, strict=True):
        assert res.layer_name == expected_layer
        assert res.representation_dim > 0
