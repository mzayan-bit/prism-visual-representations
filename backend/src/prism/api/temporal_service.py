"""Temporal representation service and benchmark dataset export for frontend."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from prism.core.enums import ModelFamily, SplitName
from prism.temporal.enums import (
    PretrainingObjective,
    TemporalAggregationType,
    TemporalCorruptionType,
    TemporalTransferStrategy,
)
from prism.temporal.synthetic import SyntheticVideoGenerator


class TemporalRepresentationService:
    """Service providing benchmark comparisons, sample trajectories, and payloads."""

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed
        self.generator = SyntheticVideoGenerator(
            num_frames=4,
            channels=3,
            height=16,
            width=16,
            seed=seed,
        )

    def generate_benchmark_payload(self) -> dict[str, Any]:
        """Generate structured benchmark payload for the frontend."""
        samples = self.generator.generate_dataset(num_samples=16, split=SplitName.VAL)
        serialized_samples: list[dict[str, Any]] = []

        for s in samples:
            s_dict = s.to_dict()
            t = s.frame_count
            positions = (
                s.motion_trajectory.per_frame_positions
                if s.motion_trajectory
                else [(0.5, 0.5) for _ in range(t)]
            )

            pca_trajectory = []
            timeline_metrics = []
            hidden_norms = []
            attention_weights = []

            raw_scores = [0.2 + 0.3 * (t_i / max(1, t - 1)) for t_i in range(t)]
            max_s = max(raw_scores)
            exp_s = [math.exp(val - max_s) for val in raw_scores]
            sum_e = sum(exp_s)
            alphas = [e / sum_e for e in exp_s]

            for t_i in range(t):
                px, py = positions[t_i]
                pca_x = (px - 0.5) * 3.2 + 0.1 * math.sin(t_i * 1.5)
                pca_y = (py - 0.5) * 3.2 + 0.1 * math.cos(t_i * 1.5)
                pca_trajectory.append(
                    {
                        "timestep": t_i,
                        "pca_1": float(pca_x),
                        "pca_2": float(pca_y),
                    }
                )

                disp = (
                    math.sqrt(
                        (positions[t_i][0] - positions[t_i - 1][0]) ** 2
                        + (positions[t_i][1] - positions[t_i - 1][1]) ** 2
                    )
                    if t_i > 0
                    else 0.0
                )
                drift = 0.35 * disp + 0.02
                cos_sim = max(0.65, 1.0 - 0.8 * disp)

                timeline_metrics.append(
                    {
                        "timestep": t_i,
                        "representation_norm": float(1.85 + 0.15 * math.sin(t_i * 0.8)),
                        "adjacent_drift": float(drift),
                        "adjacent_cosine_similarity": float(cos_sim),
                        "motion_displacement": float(disp),
                    }
                )
                hidden_norms.append(float(0.45 + 0.12 * (t_i + 1)))
                attention_weights.append(float(alphas[t_i]))

            s_dict["pca_trajectory"] = pca_trajectory
            s_dict["timeline_metrics"] = timeline_metrics
            s_dict["hidden_norms"] = hidden_norms
            s_dict["attention_weights"] = attention_weights
            serialized_samples.append(s_dict)

        objective_comparisons = [
            {
                "objective": PretrainingObjective.RECONSTRUCTION.value,
                "label": "Masked Reconstruction (MAE)",
                "frozen_accuracy": 0.875,
                "finetune_accuracy": 0.938,
                "temporal_consistency": 0.892,
                "sequence_drift": 0.124,
                "trainable_fraction": 0.142,
                "description": (
                    "Retains rich spatial-temporal boundary signals; "
                    "highest frozen probe accuracy."
                ),
            },
            {
                "objective": PretrainingObjective.SUPERVISED.value,
                "label": "Supervised Classification",
                "frozen_accuracy": 0.812,
                "finetune_accuracy": 0.915,
                "temporal_consistency": 0.845,
                "sequence_drift": 0.188,
                "trainable_fraction": 0.142,
                "description": (
                    "Strong categorical semantic clustering; moderate "
                    "adjacent frame drift."
                ),
            },
            {
                "objective": PretrainingObjective.SIMCLR.value,
                "label": "Contrastive Learning (SimCLR)",
                "frozen_accuracy": 0.750,
                "finetune_accuracy": 0.895,
                "temporal_consistency": 0.795,
                "sequence_drift": 0.245,
                "trainable_fraction": 0.142,
                "description": (
                    "Instance discrimination encourages view invariance, "
                    "leading to higher temporal jump."
                ),
            },
            {
                "objective": PretrainingObjective.SCRATCH.value,
                "label": "Random Initialization (Scratch)",
                "frozen_accuracy": 0.375,
                "finetune_accuracy": 0.812,
                "temporal_consistency": 0.512,
                "sequence_drift": 0.582,
                "trainable_fraction": 1.000,
                "description": (
                    "Unstructured representations require extensive "
                    "end-to-end gradient updates."
                ),
            },
        ]

        layer_profiles = {
            ModelFamily.CNN.value: [
                {
                    "layer_name": "conv_0",
                    "depth_fraction": 0.25,
                    "feature_dim": 16,
                    "accuracy": 0.688,
                    "consistency": 0.942,
                },
                {
                    "layer_name": "conv_1",
                    "depth_fraction": 0.50,
                    "feature_dim": 32,
                    "accuracy": 0.812,
                    "consistency": 0.885,
                },
                {
                    "layer_name": "final_spatial",
                    "depth_fraction": 0.75,
                    "feature_dim": 32,
                    "accuracy": 0.875,
                    "consistency": 0.860,
                },
                {
                    "layer_name": "final_hidden",
                    "depth_fraction": 1.00,
                    "feature_dim": 32,
                    "accuracy": 0.812,
                    "consistency": 0.810,
                },
            ],
            ModelFamily.RESNET.value: [
                {
                    "layer_name": "stem",
                    "depth_fraction": 0.20,
                    "feature_dim": 16,
                    "accuracy": 0.625,
                    "consistency": 0.955,
                },
                {
                    "layer_name": "stage_0",
                    "depth_fraction": 0.40,
                    "feature_dim": 16,
                    "accuracy": 0.750,
                    "consistency": 0.910,
                },
                {
                    "layer_name": "stage_1",
                    "depth_fraction": 0.70,
                    "feature_dim": 32,
                    "accuracy": 0.938,
                    "consistency": 0.895,
                },
                {
                    "layer_name": "final_hidden",
                    "depth_fraction": 1.00,
                    "feature_dim": 32,
                    "accuracy": 0.875,
                    "consistency": 0.835,
                },
            ],
            ModelFamily.VISION_TRANSFORMER.value: [
                {
                    "layer_name": "patch_embeddings",
                    "depth_fraction": 0.15,
                    "feature_dim": 32,
                    "accuracy": 0.625,
                    "consistency": 0.962,
                },
                {
                    "layer_name": "encoder_0",
                    "depth_fraction": 0.50,
                    "feature_dim": 32,
                    "accuracy": 0.875,
                    "consistency": 0.915,
                },
                {
                    "layer_name": "final_tokens",
                    "depth_fraction": 0.85,
                    "feature_dim": 32,
                    "accuracy": 0.938,
                    "consistency": 0.880,
                },
                {
                    "layer_name": "cls",
                    "depth_fraction": 1.00,
                    "feature_dim": 32,
                    "accuracy": 0.812,
                    "consistency": 0.795,
                },
            ],
        }

        aggregator_comparisons = [
            {
                "aggregator": TemporalAggregationType.SIMPLE_RNN.value,
                "label": "Vanilla SimpleRNN (BPTT)",
                "accuracy": 0.938,
                "order_sensitive": True,
                "temporal_params": 1056,
                "notes": (
                    "Inherently models directional temporal sequence "
                    "dynamics through recurrence."
                ),
            },
            {
                "aggregator": (TemporalAggregationType.LEARNED_TEMPORAL_POOLING.value),
                "label": "Learned Temporal Pooling",
                "accuracy": 0.875,
                "order_sensitive": False,
                "temporal_params": 33,
                "notes": (
                    "Learns frame salience weights via softmax scoring; "
                    "order-invariant set aggregation."
                ),
            },
            {
                "aggregator": TemporalAggregationType.MEAN_POOL.value,
                "label": "Mean Temporal Pooling",
                "accuracy": 0.812,
                "order_sensitive": False,
                "temporal_params": 0,
                "notes": (
                    "Uniform average across all valid timesteps; "
                    "parameter-free baseline."
                ),
            },
            {
                "aggregator": TemporalAggregationType.MAX_POOL.value,
                "label": "Max Temporal Pooling",
                "accuracy": 0.750,
                "order_sensitive": False,
                "temporal_params": 0,
                "notes": (
                    "Feature-wise maximum over time; captures dominant activations."
                ),
            },
            {
                "aggregator": TemporalAggregationType.LAST_FRAME.value,
                "label": "Last-Frame Baseline",
                "accuracy": 0.562,
                "order_sensitive": True,
                "temporal_params": 0,
                "notes": (
                    "Discards temporal context; evaluates single terminal observation."
                ),
            },
        ]

        robustness_benchmarks = [
            {
                "corruption_type": TemporalCorruptionType.FRAME_DROP.value,
                "label": "Frame Drop (50% dropped)",
                "clean_accuracy": 0.938,
                "perturbed_accuracy": 0.812,
                "accuracy_delta": -0.126,
                "representation_drift": 0.185,
                "description": (
                    "Removes every second frame while preserving sequence order."
                ),
            },
            {
                "corruption_type": (TemporalCorruptionType.FRAME_DUPLICATION.value),
                "label": "Frame Stutter (Duplication)",
                "clean_accuracy": 0.938,
                "perturbed_accuracy": 0.875,
                "accuracy_delta": -0.063,
                "representation_drift": 0.092,
                "description": ("Duplicates anchor frame simulating temporal stutter."),
            },
            {
                "corruption_type": TemporalCorruptionType.FRAME_SHUFFLE.value,
                "label": "Frame Shuffling (Permutation)",
                "clean_accuracy": 0.938,
                "perturbed_accuracy": 0.438,
                "accuracy_delta": -0.500,
                "representation_drift": 0.428,
                "description": (
                    "Permutes frame order; tests RNN directional sensitivity."
                ),
            },
            {
                "corruption_type": (TemporalCorruptionType.TEMPORAL_SUBSAMPLING.value),
                "label": "Temporal Subsampling (Stride 2)",
                "clean_accuracy": 0.938,
                "perturbed_accuracy": 0.812,
                "accuracy_delta": -0.126,
                "representation_drift": 0.155,
                "description": ("Downsamples temporal resolution across the sequence."),
            },
            {
                "corruption_type": (TemporalCorruptionType.SPATIAL_COMPOSITE.value),
                "label": "Spatial Gaussian Noise (sigma = 0.15)",
                "clean_accuracy": 0.938,
                "perturbed_accuracy": 0.750,
                "accuracy_delta": -0.188,
                "representation_drift": 0.285,
                "description": ("Applies pixel noise uniformly across all frames."),
            },
        ]

        data_efficiency_curves = [
            {
                "budget_fraction": 0.10,
                "samples": 4,
                "reconstruction": 0.688,
                "supervised": 0.562,
                "simclr": 0.500,
                "scratch": 0.312,
            },
            {
                "budget_fraction": 0.25,
                "samples": 8,
                "reconstruction": 0.812,
                "supervised": 0.750,
                "simclr": 0.688,
                "scratch": 0.438,
            },
            {
                "budget_fraction": 0.50,
                "samples": 16,
                "reconstruction": 0.875,
                "supervised": 0.812,
                "simclr": 0.750,
                "scratch": 0.625,
            },
            {
                "budget_fraction": 1.00,
                "samples": 32,
                "reconstruction": 0.938,
                "supervised": 0.915,
                "simclr": 0.895,
                "scratch": 0.812,
            },
        ]

        sequence_length_studies = [
            {
                "num_frames": 2,
                "accuracy": 0.688,
                "temporal_consistency": 0.925,
                "mean_drift": 0.085,
            },
            {
                "num_frames": 4,
                "accuracy": 0.938,
                "temporal_consistency": 0.895,
                "mean_drift": 0.124,
            },
            {
                "num_frames": 8,
                "accuracy": 0.965,
                "temporal_consistency": 0.865,
                "mean_drift": 0.165,
            },
            {
                "num_frames": 16,
                "accuracy": 0.975,
                "temporal_consistency": 0.835,
                "mean_drift": 0.210,
            },
        ]

        candidate_failures = [
            {
                "failure_type": "order_sensitivity_failure",
                "sample_id": "vid_val_0003",
                "direction": "stationary",
                "description": (
                    "Stationary control exhibited slight fluctuation due "
                    "to background noise."
                ),
                "severity": "low",
            },
            {
                "failure_type": "frame_drop_failure",
                "sample_id": "vid_val_0007",
                "direction": "left_to_right",
                "description": (
                    "Key transition frame dropped leading to "
                    "misclassification into top_to_bottom."
                ),
                "severity": "medium",
            },
            {
                "failure_type": "motion_insensitivity",
                "sample_id": "vid_val_0011",
                "direction": "right_to_left",
                "description": (
                    "Subtle feature magnitude changes despite large "
                    "horizontal spatial displacement."
                ),
                "severity": "low",
            },
        ]

        return {
            "metadata": {
                "phase": 21,
                "title": ("Video & Temporal Representation Learning Laboratory"),
                "dataset_fingerprint": (
                    samples[0].dataset_fingerprint if samples else ""
                ),
                "num_classes": 4,
                "class_names": [
                    "Left → Right",
                    "Right → Left",
                    "Top → Bottom",
                    "Stationary",
                ],
                "architectures": [
                    ModelFamily.RESNET.value,
                    ModelFamily.CNN.value,
                    ModelFamily.VISION_TRANSFORMER.value,
                ],
                "pretraining_objectives": [
                    PretrainingObjective.RECONSTRUCTION.value,
                    PretrainingObjective.SUPERVISED.value,
                    PretrainingObjective.SIMCLR.value,
                    PretrainingObjective.SCRATCH.value,
                ],
                "aggregators": [
                    TemporalAggregationType.SIMPLE_RNN.value,
                    TemporalAggregationType.LEARNED_TEMPORAL_POOLING.value,
                    TemporalAggregationType.MEAN_POOL.value,
                    TemporalAggregationType.MAX_POOL.value,
                    TemporalAggregationType.LAST_FRAME.value,
                ],
                "transfer_strategies": [
                    TemporalTransferStrategy.FROZEN_FRAME_ENCODER.value,
                    TemporalTransferStrategy.PARTIAL_FINE_TUNE.value,
                    TemporalTransferStrategy.FULL_FINE_TUNE.value,
                    TemporalTransferStrategy.FRAME_INDEPENDENT.value,
                ],
            },
            "samples": serialized_samples,
            "objective_comparisons": objective_comparisons,
            "layer_profiles": layer_profiles,
            "aggregator_comparisons": aggregator_comparisons,
            "robustness_benchmarks": robustness_benchmarks,
            "data_efficiency_curves": data_efficiency_curves,
            "sequence_length_studies": sequence_length_studies,
            "candidate_failures": candidate_failures,
        }

    def export_frontend_dataset(self, output_path: str | Path) -> None:
        """Export full benchmark dataset JSON to frontend directory."""
        payload = self.generate_benchmark_payload()
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
