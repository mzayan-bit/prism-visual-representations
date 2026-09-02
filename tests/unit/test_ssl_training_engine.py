"""Unit tests for SimCLR contrastive training engine and label independence."""

from prism.core.enums import ModelFamily
from prism.data.materialized import MaterializedDataset, MaterializedSample
from prism.models.specifications import ModelSpecification
from prism.ssl.engine import SelfSupervisedTrainingEngine
from prism.ssl.specification import SelfSupervisedTrainingSpecification


def _make_ssl_test_dataset(num_samples: int = 12) -> MaterializedDataset:
    c, h, w = 3, 8, 8
    samples: list[MaterializedSample] = []
    for i in range(num_samples):
        # Deterministic distinct gradient pattern per sample
        img = [
            [
                [float(i * 10 + r * w + col) / 255.0 for col in range(w)]
                for r in range(h)
            ]
            for _ in range(c)
        ]
        samples.append(
            MaterializedSample(
                sample_id=f"ssl_s_{i}",
                source_split="train",
                source_index=i,
                data=img,
                target=i % 2,
            )
        )
    return MaterializedDataset(
        dataset_id="ds_ssl_test",
        split_name="train",
        samples=samples,
    )


def test_ssl_training_engine_step_and_update() -> None:
    """Test full SSL training loop on CNN backbone."""
    c, h, w = 3, 8, 8
    encoder_spec = ModelSpecification(
        model_id="cnn_ssl_test",
        name="CNN SSL Test",
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
    )

    spec = SelfSupervisedTrainingSpecification(
        ssl_id="ssl_test_run",
        encoder_family=ModelFamily.CNN,
        encoder_spec=encoder_spec,
        dataset_id="ds_ssl_test",
        projection_hidden_dim=8,
        projection_out_dim=4,
        temperature=0.5,
        epochs=2,
        batch_size=4,
        learning_rate=0.05,
        seed=42,
    )

    dataset = _make_ssl_test_dataset(num_samples=8)
    engine = SelfSupervisedTrainingEngine()

    _encoder, snapshot, report = engine.train_ssl(specification=spec, dataset=dataset)

    assert report.ssl_id == "ssl_test_run"
    assert len(report.loss_trajectory) == 2
    assert report.total_encoder_parameters > 0
    assert report.projection_head_parameters > 0
    assert snapshot.verify_integrity() is True


def test_ssl_label_independence_invariant() -> None:
    """Changing target labels does NOT alter SSL loss or parameter updates."""
    c, h, w = 3, 8, 8
    encoder_spec = ModelSpecification(
        model_id="cnn_label_indep",
        name="CNN Label Indep",
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
    )

    spec = SelfSupervisedTrainingSpecification(
        ssl_id="ssl_indep_run",
        encoder_family=ModelFamily.CNN,
        encoder_spec=encoder_spec,
        dataset_id="ds_ssl_test",
        projection_hidden_dim=8,
        projection_out_dim=4,
        temperature=0.5,
        epochs=1,
        batch_size=4,
        learning_rate=0.05,
        seed=100,
    )

    # Dataset A: labels 0 and 1
    dataset_a = _make_ssl_test_dataset(num_samples=8)

    # Dataset B: exactly identical image tensors, but inverted labels (99 and 88)
    samples_b = [
        MaterializedSample(
            sample_id=s.sample_id,
            source_split=s.source_split,
            source_index=s.source_index,
            data=s.data,
            target=99 if s.target == 0 else 88,
        )
        for s in dataset_a.samples
    ]
    dataset_b = MaterializedDataset(
        dataset_id="ds_ssl_test_b",
        split_name="train",
        samples=samples_b,
    )

    engine_a = SelfSupervisedTrainingEngine()
    encoder_a, _, report_a = engine_a.train_ssl(specification=spec, dataset=dataset_a)

    engine_b = SelfSupervisedTrainingEngine()
    encoder_b, _, report_b = engine_b.train_ssl(specification=spec, dataset=dataset_b)

    # Losses must be bitwise identical
    assert report_a.loss_trajectory == report_b.loss_trajectory

    # Parameter values must be bitwise identical
    params_a = encoder_a.get_parameters()
    params_b = encoder_b.get_parameters()
    for k in params_a:
        assert params_a[k] == params_b[k]
