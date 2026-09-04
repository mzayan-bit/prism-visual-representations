"""Training runner and orchestration engine for temporal representation transfer."""

from __future__ import annotations

import math
from typing import Any

from prism.models.base import BaseVisionModel
from prism.temporal.adapter import TemporalFrameEncoder
from prism.temporal.aggregators import (
    BaseTemporalAggregator,
    LastFramePooling,
    LearnedTemporalPooling,
    MaxTemporalPooling,
    MeanTemporalPooling,
    SimpleRNN,
)
from prism.temporal.contracts import VideoSample
from prism.temporal.corruptions import apply_temporal_corruption
from prism.temporal.enums import (
    TemporalAggregationType,
    TemporalCorruptionType,
    TemporalFailureType,
    TemporalTransferStrategy,
)
from prism.temporal.heads import (
    TemporalClassificationHead,
    TemporalRepresentationModel,
)
from prism.temporal.metrics import (
    compute_motion_sensitivity,
    compute_temporal_consistency,
    compute_temporal_drift_curve,
    compute_video_classification_metrics,
)
from prism.temporal.reports import (
    TemporalRepresentationReport,
    TemporalRepresentationRetentionRecord,
    TemporalRobustnessSummary,
)
from prism.temporal.specification import TemporalTransferSpecification


class TemporalTrainingRunner:
    """Orchestrates temporal adaptation experiments and metric generation."""

    def __init__(
        self,
        spec: TemporalTransferSpecification,
        model: BaseVisionModel,
        train_samples: list[VideoSample],
        val_samples: list[VideoSample],
    ) -> None:
        self.spec = spec
        self.base_model = model
        self.train_samples = train_samples
        self.val_samples = val_samples

        self.frame_encoder = TemporalFrameEncoder(
            model=self.base_model,
            layer_name=self.spec.selected_layer,
        )

        dummy_video = [[self.train_samples[0].frame_tensors[0]]]
        probe_out = self.frame_encoder.forward(dummy_video)
        self.feature_dim = len(probe_out[0][0])

        self.aggregator: BaseTemporalAggregator
        if self.spec.temporal_aggregator == TemporalAggregationType.MEAN_POOL:
            self.aggregator = MeanTemporalPooling()
            self.seq_dim = self.feature_dim
        elif self.spec.temporal_aggregator == TemporalAggregationType.MAX_POOL:
            self.aggregator = MaxTemporalPooling()
            self.seq_dim = self.feature_dim
        elif self.spec.temporal_aggregator == TemporalAggregationType.LAST_FRAME:
            self.aggregator = LastFramePooling()
            self.seq_dim = self.feature_dim
        elif (
            self.spec.temporal_aggregator
            == TemporalAggregationType.LEARNED_TEMPORAL_POOLING
        ):
            self.aggregator = LearnedTemporalPooling(
                input_dim=self.feature_dim,
                seed=self.spec.seed,
            )
            self.seq_dim = self.feature_dim
        elif self.spec.temporal_aggregator == TemporalAggregationType.SIMPLE_RNN:
            self.aggregator = SimpleRNN(
                input_dim=self.feature_dim,
                hidden_dim=self.spec.rnn_hidden_dim,
                mode=self.spec.rnn_mode,
                seed=self.spec.seed,
            )
            self.seq_dim = self.spec.rnn_hidden_dim
        else:
            self.aggregator = MeanTemporalPooling()
            self.seq_dim = self.feature_dim

        self.classifier = TemporalClassificationHead(
            input_dim=self.seq_dim,
            num_classes=self.spec.num_classes,
            seed=self.spec.seed,
        )

        train_encoder = self.spec.transfer_strategy in (
            TemporalTransferStrategy.PARTIAL_FINE_TUNE,
            TemporalTransferStrategy.FULL_FINE_TUNE,
        )

        self.temporal_model = TemporalRepresentationModel(
            frame_encoder=self.frame_encoder,
            aggregator=self.aggregator,
            classifier=self.classifier,
            train_encoder=train_encoder,
        )

    def _count_parameters(self) -> tuple[int, int]:
        """Count total and trainable parameter scalars."""
        total_p = 0
        trainable_p = 0

        agg_params = self.aggregator.get_parameters()
        for val in agg_params.values():
            if isinstance(val, list):
                if val and isinstance(val[0], list):
                    count = sum(len(row) for row in val)
                else:
                    count = len(val)
                total_p += count
                trainable_p += count
            elif isinstance(val, (int, float)):
                total_p += 1
                trainable_p += 1

        cls_params = self.classifier.get_parameters()
        for val in cls_params.values():
            if isinstance(val, list):
                if val and isinstance(val[0], list):
                    count = sum(len(row) for row in val)
                else:
                    count = len(val)
                total_p += count
                trainable_p += count

        if hasattr(self.base_model, "get_parameters"):
            enc_params = self.base_model.get_parameters()
            enc_count = 0
            for v in enc_params.values():
                if isinstance(v, list):
                    if v and isinstance(v[0], list):
                        enc_count += sum(len(row) for row in v)
                    else:
                        enc_count += len(v)
            total_p += enc_count
            if self.temporal_model.train_encoder:
                trainable_p += enc_count

        return total_p, trainable_p

    def _sgd_step(
        self, params: dict[str, Any], grads: dict[str, Any], lr: float
    ) -> None:
        """Apply vanilla SGD in-place parameter update."""
        for key in params:
            if key not in grads:
                continue
            p_val = params[key]
            g_val = grads[key]

            if isinstance(p_val, list) and isinstance(g_val, list):
                if p_val and isinstance(p_val[0], list):
                    for r in range(len(p_val)):
                        for c in range(len(p_val[r])):
                            p_val[r][c] -= lr * g_val[r][c]
                else:
                    for i in range(len(p_val)):
                        p_val[i] -= lr * g_val[i]
            elif isinstance(p_val, float) and isinstance(g_val, float):
                params[key] = p_val - lr * g_val

    def _evaluate_model(
        self,
        samples: list[VideoSample],
    ) -> tuple[float, list[int], list[int]]:
        """Compute video classification accuracy on given samples."""
        self.temporal_model.eval()
        predictions: list[int] = []
        targets: list[int] = []

        for s in samples:
            logits = self.temporal_model.forward([s.frame_tensors])
            row = logits[0]
            pred = max(range(len(row)), key=lambda k: row[k])
            predictions.append(pred)
            targets.append(s.label)

        metrics = compute_video_classification_metrics(predictions, targets)
        return metrics["accuracy"], predictions, targets

    def _evaluate_frame_independent_baseline(
        self,
        samples: list[VideoSample],
    ) -> float:
        """Evaluate frame-independent classification baseline."""
        self.temporal_model.eval()
        predictions: list[int] = []
        targets: list[int] = []

        for s in samples:
            frame_logits_list: list[list[float]] = []
            for frame in s.frame_tensors:
                dummy_v = [[frame]]
                f_repr = self.frame_encoder.forward(dummy_v)[0][0]
                if len(f_repr) == self.classifier.input_dim:
                    l_row = self.classifier.forward([f_repr])[0]
                else:
                    l_row = [0.0] * self.classifier.num_classes
                frame_logits_list.append(l_row)

            num_classes = self.classifier.num_classes
            mean_logits = [0.0] * num_classes
            for f_logits in frame_logits_list:
                for c_i in range(num_classes):
                    mean_logits[c_i] += f_logits[c_i]
            pred = max(range(num_classes), key=lambda k: mean_logits[k])
            predictions.append(pred)
            targets.append(s.label)

        metrics = compute_video_classification_metrics(predictions, targets)
        return metrics["accuracy"]

    def run_transfer(self) -> TemporalRepresentationReport:
        """Execute temporal training, drift tracking, and comprehensive evaluation."""
        self.temporal_model.eval()
        pre_seq_reprs: list[list[float]] = []
        pre_frame_reprs: list[list[list[float]]] = []

        for s in self.val_samples:
            f_repr = self.temporal_model.extract_frame_representations(
                [s.frame_tensors]
            )[0]
            s_repr = self.temporal_model.extract_sequence_representations(
                [s.frame_tensors]
            )[0]
            pre_frame_reprs.append(f_repr)
            pre_seq_reprs.append(s_repr)

        lr = self.spec.learning_rate
        for _ in range(self.spec.epochs):
            self.temporal_model.train(True)
            for s in self.train_samples:
                self.temporal_model.zero_grad()

                logits = self.temporal_model.forward([s.frame_tensors])
                _, d_logits = self.classifier.compute_loss_and_grad(logits, [s.label])

                self.temporal_model.backward(d_logits)

                agg_params = self.aggregator.get_parameters()
                agg_grads = self.aggregator.get_gradients()
                self._sgd_step(agg_params, agg_grads, lr)
                self.aggregator.set_parameters(agg_params)

                cls_params = self.classifier.get_parameters()
                cls_grads = self.classifier.get_gradients()
                self._sgd_step(cls_params, cls_grads, lr)
                self.classifier.set_parameters(cls_params)

                if self.temporal_model.train_encoder and hasattr(
                    self.base_model, "get_parameters"
                ):
                    enc_params = self.base_model.get_parameters()
                    enc_grads = self.base_model.get_gradients()
                    self._sgd_step(enc_params, enc_grads, lr)
                    self.base_model.set_parameters(enc_params)

        clean_acc, _, _ = self._evaluate_model(self.val_samples)
        frame_baseline_acc = self._evaluate_frame_independent_baseline(self.val_samples)

        self.temporal_model.eval()
        post_seq_reprs: list[list[float]] = []
        post_frame_reprs: list[list[list[float]]] = []

        for s in self.val_samples:
            f_repr = self.temporal_model.extract_frame_representations(
                [s.frame_tensors]
            )[0]
            s_repr = self.temporal_model.extract_sequence_representations(
                [s.frame_tensors]
            )[0]
            post_frame_reprs.append(f_repr)
            post_seq_reprs.append(s_repr)

        overall_consistency = compute_temporal_consistency(post_frame_reprs[0])

        seq_drifts: list[float] = []
        for i in range(len(self.val_samples)):
            diff = sum(
                (a - b) ** 2
                for a, b in zip(pre_seq_reprs[i], post_seq_reprs[i], strict=True)
            )
            seq_drifts.append(math.sqrt(diff))
        mean_seq_drift = sum(seq_drifts) / max(1, len(seq_drifts))

        t_count = len(post_frame_reprs[0])
        per_timestep_drift: list[float] = [0.0] * t_count
        for i in range(len(self.val_samples)):
            for t_i in range(t_count):
                f_diff = sum(
                    (a - b) ** 2
                    for a, b in zip(
                        pre_frame_reprs[i][t_i],
                        post_frame_reprs[i][t_i],
                        strict=True,
                    )
                )
                per_timestep_drift[t_i] += math.sqrt(f_diff)
        per_timestep_drift = [
            d / max(1, len(self.val_samples)) for d in per_timestep_drift
        ]
        mean_frame_drift = sum(per_timestep_drift) / max(1, len(per_timestep_drift))

        retention_record = TemporalRepresentationRetentionRecord(
            mean_frame_drift=float(mean_frame_drift),
            sequence_drift=float(mean_seq_drift),
            per_timestep_drift=[float(d) for d in per_timestep_drift],
        )

        sample_0 = self.val_samples[0]
        drift_curve = compute_temporal_drift_curve(post_frame_reprs[0])
        motion_sens = (
            compute_motion_sensitivity(
                post_frame_reprs[0],
                sample_0.motion_trajectory.per_frame_positions,
            )
            if sample_0.motion_trajectory
            else {}
        )

        weight_sum = None
        if isinstance(self.aggregator, LearnedTemporalPooling):
            weight_sum = self.aggregator.get_weight_summary(0)

        rnn_dyn = None
        if isinstance(self.aggregator, SimpleRNN):
            rnn_dyn = self.aggregator.get_dynamics_summary(0)

        robustness_map: dict[str, TemporalRobustnessSummary] = {}
        candidate_failures: list[dict[str, Any]] = []

        corruptions_to_test: list[tuple[TemporalCorruptionType, dict[str, Any]]] = [
            (TemporalCorruptionType.FRAME_DROP, {"drop_fraction": 0.5}),
            (
                TemporalCorruptionType.FRAME_DUPLICATION,
                {"dup_index": 0, "dup_count": 1},
            ),
            (TemporalCorruptionType.FRAME_SHUFFLE, {}),
            (TemporalCorruptionType.TEMPORAL_SUBSAMPLING, {"stride": 2}),
            (TemporalCorruptionType.SPATIAL_COMPOSITE, {"noise_level": 0.15}),
        ]

        for c_type, c_kwargs in corruptions_to_test:
            corrupted_samples: list[VideoSample] = []
            lineage_list = []
            for s in self.val_samples:
                c_sample, c_meta = apply_temporal_corruption(s, c_type, **c_kwargs)
                corrupted_samples.append(c_sample)
                lineage_list.append(c_meta)

            corr_acc, _, _ = self._evaluate_model(corrupted_samples)
            acc_delta = corr_acc - clean_acc

            corr_seq_drifts = []
            for i, c_s in enumerate(corrupted_samples):
                c_seq = self.temporal_model.extract_sequence_representations(
                    [c_s.frame_tensors]
                )[0]
                diff = sum(
                    (a - b) ** 2 for a, b in zip(post_seq_reprs[i], c_seq, strict=True)
                )
                corr_seq_drifts.append(math.sqrt(diff))
            mean_corr_drift = sum(corr_seq_drifts) / max(1, len(corr_seq_drifts))

            summary = TemporalRobustnessSummary(
                corruption_type=c_type.value,
                clean_accuracy=float(clean_acc),
                perturbed_accuracy=float(corr_acc),
                accuracy_delta=float(acc_delta),
                sequence_representation_drift=float(mean_corr_drift),
                temporal_consistency_change=float(abs(acc_delta)),
                lineage=lineage_list[0] if lineage_list else {},
            )
            robustness_map[c_type.value] = summary

            if c_type == TemporalCorruptionType.FRAME_DROP and acc_delta < -0.15:
                candidate_failures.append(
                    {
                        "failure_type": TemporalFailureType.FRAME_DROP_FAILURE.value,
                        "sample_id": sample_0.video_id,
                        "description": (
                            f"Accuracy dropped by {abs(acc_delta):.1%} on frame drop."
                        ),
                    }
                )

            if (
                c_type == TemporalCorruptionType.FRAME_SHUFFLE
                and isinstance(self.aggregator, SimpleRNN)
                and abs(acc_delta) < 0.05
            ):
                candidate_failures.append(
                    {
                        "failure_type": (
                            TemporalFailureType.ORDER_SENSITIVITY_FAILURE.value
                        ),
                        "sample_id": sample_0.video_id,
                        "description": "RNN model exhibited low order sensitivity.",
                    }
                )

        if motion_sens and motion_sens.get("motion_drift_correlation", 0.0) < 0.10:
            candidate_failures.append(
                {
                    "failure_type": TemporalFailureType.MOTION_INSENSITIVITY.value,
                    "sample_id": sample_0.video_id,
                    "description": "Low motion-drift correlation on sequence.",
                }
            )

        tot_p, train_p = self._count_parameters()
        trainable_fraction = train_p / max(1, tot_p)

        warnings = [
            "Synthetic geometric video dataset used for verification.",
            "Post-hoc temporal adaptation without full video pretraining.",
        ]
        if self.spec.temporal_aggregator in (
            TemporalAggregationType.MEAN_POOL,
            TemporalAggregationType.MAX_POOL,
            TemporalAggregationType.LEARNED_TEMPORAL_POOLING,
        ):
            warnings.append("Temporal aggregator is order-invariant by construction.")

        return TemporalRepresentationReport(
            spec=self.spec,
            video_accuracy=float(clean_acc),
            frame_baseline_accuracy=float(frame_baseline_acc),
            temporal_consistency=overall_consistency,
            mean_sequence_drift=float(mean_seq_drift),
            trainable_fraction=float(trainable_fraction),
            drift_curve=drift_curve,
            motion_sensitivity=motion_sens,
            weight_summary=weight_sum,
            rnn_dynamics=rnn_dyn,
            robustness_summaries=robustness_map,
            retention_record=retention_record,
            candidate_failures=candidate_failures,
            warnings=warnings,
        )
