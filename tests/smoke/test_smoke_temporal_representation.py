"""Smoke test for PRISM Phase 21: Video & Temporal Representation Learning pipeline."""

import json

from prism.core.enums import ModelFamily, SplitName, TaskType
from prism.models.cnn import ConvolutionalNeuralNetwork
from prism.models.specifications import ModelSpecification
from prism.temporal.adapter import TemporalFrameEncoder
from prism.temporal.aggregators import (
    LearnedTemporalPooling,
    MeanTemporalPooling,
    SimpleRNN,
)
from prism.temporal.enums import (
    PretrainingObjective,
    TemporalAggregationType,
    TemporalCorruptionType,
    TemporalTransferStrategy,
)
from prism.temporal.heads import TemporalClassificationHead, TemporalRepresentationModel
from prism.temporal.metrics import (
    compute_motion_sensitivity,
    compute_temporal_consistency,
    compute_temporal_drift_curve,
)
from prism.temporal.runner import TemporalTrainingRunner
from prism.temporal.specification import TemporalTransferSpecification
from prism.temporal.synthetic import SyntheticVideoGenerator


def _create_smoke_cnn() -> ConvolutionalNeuralNetwork:
    spec = ModelSpecification(
        model_id="smoke_cnn",
        name="Smoke CNN",
        architecture="cnn_smoke",
        family=ModelFamily.CNN,
        input_shape=(3, 16, 16),
        num_classes=4,
        compatible_tasks=[TaskType.CLASSIFICATION],
        hyperparameters={
            "conv_channels": [4, 8],
            "kernel_sizes": [3, 3],
            "strides": [1, 1],
            "paddings": [1, 1],
            "use_batch_norm": False,
            "hidden_dims": [16],
        },
    )
    return ConvolutionalNeuralNetwork(spec=spec, seed=42)


def test_smoke_temporal_pipeline_end_to_end() -> None:
    # 1. Generate synthetic short videos
    gen = SyntheticVideoGenerator(num_frames=4, height=16, width=16, seed=42)
    train_samples = gen.generate_dataset(num_samples=4, split=SplitName.TRAIN)
    val_samples = gen.generate_dataset(num_samples=4, split=SplitName.VAL)

    assert len(train_samples) == 4
    assert len(val_samples) == 4

    # 2. Shared image encoder
    cnn = _create_smoke_cnn()
    encoder = TemporalFrameEncoder(model=cnn, layer_name="final_hidden")

    # 3. Frame representation extraction: [N, T, D]
    raw_videos = [s.frame_tensors for s in train_samples]
    frame_feats = encoder.forward(raw_videos)
    assert len(frame_feats) == 4
    assert len(frame_feats[0]) == 4
    feat_dim = len(frame_feats[0][0])
    assert feat_dim > 0

    # 4. Aggregators
    # 4a. Mean pooling
    mean_pool = MeanTemporalPooling()
    z_mean = mean_pool.forward(frame_feats)
    assert len(z_mean) == 4
    assert len(z_mean[0]) == feat_dim

    # 4b. Learned temporal pooling
    learned_pool = LearnedTemporalPooling(input_dim=feat_dim, seed=42)
    z_learned = learned_pool.forward(frame_feats)
    assert len(z_learned) == 4
    assert len(z_learned[0]) == feat_dim

    # 4c. SimpleRNN
    rnn = SimpleRNN(input_dim=feat_dim, hidden_dim=8, seed=42)
    z_rnn = rnn.forward(frame_feats)
    assert len(z_rnn) == 4
    assert len(z_rnn[0]) == 8

    # 5. Classifier head & backward pass
    head = TemporalClassificationHead(input_dim=8, num_classes=4, seed=42)
    logits = head.forward(z_rnn)
    assert len(logits) == 4
    assert len(logits[0]) == 4

    targets = [s.label for s in train_samples]
    loss, d_logits = head.compute_loss_and_grad(logits, targets)
    assert loss > 0.0

    d_z = head.backward(d_logits)
    d_frames = rnn.backward(d_z)
    assert len(d_frames) == 4
    assert len(d_frames[0]) == 4
    assert len(d_frames[0][0]) == feat_dim

    # 6. Unified Temporal Representation Model
    model = TemporalRepresentationModel(
        frame_encoder=encoder,
        aggregator=rnn,
        classifier=head,
        train_encoder=False,
    )
    m_logits = model.forward(raw_videos)
    assert len(m_logits) == 4

    # 7. Metrics & Dynamics
    sample_0 = train_samples[0]
    sample_0_feats = frame_feats[0]

    consistency = compute_temporal_consistency(sample_0_feats)
    assert consistency.mean_adjacent_distance >= 0.0
    assert consistency.mean_adjacent_cosine_similarity <= 1.0

    drift_curve = compute_temporal_drift_curve(sample_0_feats)
    assert len(drift_curve) == 4
    assert drift_curve[0]["euclidean_drift"] == 0.0

    if sample_0.motion_trajectory:
        motion_res = compute_motion_sensitivity(
            sample_0_feats,
            sample_0.motion_trajectory.per_frame_positions,
        )
        assert len(motion_res["paired_deltas"]) == 3

    # 8. Full Runner & Report Serialization
    spec = TemporalTransferSpecification(
        source_objective=PretrainingObjective.RECONSTRUCTION,
        architecture=ModelFamily.CNN,
        selected_layer="final_hidden",
        temporal_aggregator=TemporalAggregationType.SIMPLE_RNN,
        transfer_strategy=TemporalTransferStrategy.FROZEN_FRAME_ENCODER,
        rnn_hidden_dim=8,
        epochs=2,
        seed=42,
    )

    runner = TemporalTrainingRunner(
        spec=spec,
        model=_create_smoke_cnn(),
        train_samples=train_samples,
        val_samples=val_samples,
    )

    report = runner.run_transfer()
    assert report.video_accuracy >= 0.0
    assert TemporalCorruptionType.FRAME_DROP.value in report.robustness_summaries

    # 9. Verify JSON roundtrip
    report_dict = report.to_dict()
    json_str = json.dumps(report_dict)
    assert len(json_str) > 0
    reloaded_dict = json.loads(json_str)
    assert reloaded_dict["spec"]["source_objective"] == "reconstruction"
