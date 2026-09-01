"""Unit tests for representation retention metrics and shared PCA drift."""

from prism.core.enums import ModelFamily
from prism.data.materialized import MaterializedDataset, MaterializedSample
from prism.models.cnn import ConvolutionalNeuralNetwork
from prism.models.specifications import ModelSpecification
from prism.transfer.retention import (
    compute_representation_retention,
    compute_transfer_shared_pca,
)


def _make_dataset(num_samples: int = 10, num_classes: int = 2) -> MaterializedDataset:
    samples = []
    c, h, w = 3, 8, 8
    for i in range(num_samples):
        target = i % num_classes
        img = [[[0.1 * (i + 1) for _ in range(w)] for _ in range(h)] for _ in range(c)]
        samples.append(
            MaterializedSample(
                sample_id=f"ret_sample_{i}",
                source_split="train",
                source_index=i,
                data=img,
                target=target,
            )
        )
    return MaterializedDataset(
        dataset_id="test_retention_ds",
        split_name="train",
        samples=samples,
    )


def test_compute_representation_retention_frozen_vs_finetuned() -> None:
    """Test retention drift is 0 for identical models and positive when updated."""
    c, h, w, num_classes = 3, 8, 8, 2
    spec = ModelSpecification(
        model_id="cnn_ret_test",
        name="CNN Retention Test",
        family=ModelFamily.CNN,
        architecture="cnn_simple",
        input_shape=(c, h, w),
        num_classes=num_classes,
        hyperparameters={
            "conv_channels": [4],
            "kernel_sizes": [3],
            "activation": "relu",
            "use_batch_norm": False,
        },
    )
    model1 = ConvolutionalNeuralNetwork(spec, seed=42)
    model2 = ConvolutionalNeuralNetwork(spec, seed=42)
    dataset = _make_dataset(num_samples=8, num_classes=2)

    # 1. Identical models (linear probe)
    summary_identical = compute_representation_retention(
        pre_model=model1,
        post_model=model2,
        reference_dataset=dataset,
        layer="final_hidden",
        transfer_strategy="linear_probe",
    )
    assert summary_identical.mean_euclidean_drift < 1e-5
    assert summary_identical.mean_cosine_similarity > 0.9999
    assert summary_identical.is_frozen_backbone is True

    # 2. Mutated model (fine-tuned)
    model2.forward([dataset.samples[0].data])
    model2.backward([[1.0, 0.0]])
    params = model2.get_parameters()
    params["conv_0_weights"][0][0][0][0] += 0.5
    model2.set_parameters(params)

    summary_mutated = compute_representation_retention(
        pre_model=model1,
        post_model=model2,
        reference_dataset=dataset,
        layer="final_hidden",
        transfer_strategy="full_fine_tune",
    )
    assert summary_mutated.mean_euclidean_drift > 0.0
    assert summary_mutated.is_frozen_backbone is False


def test_compute_transfer_shared_pca() -> None:
    """Test shared PCA coordinate projection for pre and post vectors."""
    pre_vecs = [
        [1.0, 2.0, 3.0],
        [2.0, 3.0, 4.0],
        [3.0, 4.0, 5.0],
        [4.0, 5.0, 6.0],
    ]
    post_vecs = [
        [1.1, 2.1, 3.1],
        [2.2, 3.2, 4.2],
        [3.1, 4.1, 5.1],
        [4.2, 5.2, 6.2],
    ]

    result = compute_transfer_shared_pca(pre_vecs, post_vecs, n_components=2)
    assert len(result["pre_coordinates"]) == 4
    assert len(result["post_coordinates"]) == 4
    assert len(result["displacement_vectors"]) == 4
    assert len(result["explained_variance_ratio"]) == 2
    assert result["mean_displacement"] > 0.0
