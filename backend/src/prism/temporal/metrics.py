"""Temporal representation metrics, consistency dynamics, and motion sensitivity."""

from __future__ import annotations

import math
from typing import Any

from prism.temporal.contracts import TemporalConsistencySummary


def _euclidean_distance(u: list[float], v: list[float]) -> float:
    """Compute Euclidean distance between two vectors."""
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(u, v, strict=True)))


def _cosine_similarity(u: list[float], v: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    norm_u = math.sqrt(sum(a * a for a in u))
    norm_v = math.sqrt(sum(b * b for b in v))
    if norm_u < 1e-12 or norm_v < 1e-12:
        return 1.0 if (norm_u < 1e-12 and norm_v < 1e-12) else 0.0
    dot = sum(a * b for a, b in zip(u, v, strict=True))
    cos = dot / (norm_u * norm_v)
    return max(-1.0, min(1.0, cos))


def compute_temporal_consistency(
    frame_features: list[list[float]],
) -> TemporalConsistencySummary:
    """Compute representation stability and consistency across adjacent frames."""
    t_steps = len(frame_features)
    if t_steps <= 1:
        return TemporalConsistencySummary(
            mean_adjacent_distance=0.0,
            median_adjacent_distance=0.0,
            std_adjacent_distance=0.0,
            mean_adjacent_cosine_similarity=1.0,
            max_temporal_jump=0.0,
            timestep_of_max_jump=0,
            adjacent_distances=[],
            adjacent_cosine_similarities=[],
        )

    distances: list[float] = []
    cosines: list[float] = []

    for t in range(1, t_steps):
        dist = _euclidean_distance(frame_features[t], frame_features[t - 1])
        cos = _cosine_similarity(frame_features[t], frame_features[t - 1])
        distances.append(dist)
        cosines.append(cos)

    mean_dist = sum(distances) / len(distances)
    sorted_dist = sorted(distances)
    median_dist = sorted_dist[len(sorted_dist) // 2]
    var_dist = sum((d - mean_dist) ** 2 for d in distances) / len(distances)
    std_dist = math.sqrt(var_dist)

    mean_cos = sum(cosines) / len(cosines)
    max_jump = -1.0
    max_jump_t = 1
    for t_idx, d_val in enumerate(distances, start=1):
        if d_val > max_jump:
            max_jump = d_val
            max_jump_t = t_idx

    return TemporalConsistencySummary(
        mean_adjacent_distance=float(mean_dist),
        median_adjacent_distance=float(median_dist),
        std_adjacent_distance=float(std_dist),
        mean_adjacent_cosine_similarity=float(mean_cos),
        max_temporal_jump=float(max_jump),
        timestep_of_max_jump=int(max_jump_t),
        adjacent_distances=[float(d) for d in distances],
        adjacent_cosine_similarities=[float(c) for c in cosines],
    )


def compute_temporal_drift_curve(
    frame_features: list[list[float]],
) -> list[dict[str, float]]:
    """Compute representation drift relative to anchor frame 0 across all timesteps."""
    if not frame_features:
        return []

    h_0 = frame_features[0]
    norm_0 = math.sqrt(sum(a * a for a in h_0))
    curve: list[dict[str, float]] = []

    for t, h_t in enumerate(frame_features):
        norm_t = math.sqrt(sum(b * b for b in h_t))
        dist = _euclidean_distance(h_0, h_t)
        cos_sim = _cosine_similarity(h_0, h_t)
        cos_dist = 1.0 - cos_sim
        norm_ratio = norm_t / max(1e-12, norm_0)

        curve.append(
            {
                "timestep": float(t),
                "euclidean_drift": float(dist),
                "cosine_distance": float(cos_dist),
                "cosine_similarity": float(cos_sim),
                "norm_ratio": float(norm_ratio),
                "feature_norm": float(norm_t),
            }
        )

    return curve


def compute_motion_sensitivity(
    frame_features: list[list[float]],
    per_frame_positions: list[tuple[float, float]],
) -> dict[str, Any]:
    """Analyze correspondence between physical motion and representation drift."""
    t_steps = min(len(frame_features), len(per_frame_positions))
    if t_steps <= 1:
        return {
            "paired_deltas": [],
            "motion_drift_correlation": 0.0,
            "mean_motion_displacement": 0.0,
            "mean_representation_drift": 0.0,
            "sensitivity_ratio": 0.0,
        }

    motion_displacements: list[float] = []
    feature_drifts: list[float] = []
    paired_deltas: list[dict[str, float]] = []

    for t in range(1, t_steps):
        p_prev = per_frame_positions[t - 1]
        p_curr = per_frame_positions[t]
        disp = math.sqrt((p_curr[0] - p_prev[0]) ** 2 + (p_curr[1] - p_prev[1]) ** 2)

        drift = _euclidean_distance(frame_features[t], frame_features[t - 1])

        motion_displacements.append(disp)
        feature_drifts.append(drift)
        paired_deltas.append(
            {
                "timestep": float(t),
                "motion_displacement": float(disp),
                "representation_drift": float(drift),
            }
        )

    mean_disp = sum(motion_displacements) / len(motion_displacements)
    mean_drift = sum(feature_drifts) / len(feature_drifts)

    var_disp = sum((d - mean_disp) ** 2 for d in motion_displacements)
    var_drift = sum((d - mean_drift) ** 2 for d in feature_drifts)

    if var_disp > 1e-12 and var_drift > 1e-12:
        cov = sum(
            (motion_displacements[i] - mean_disp) * (feature_drifts[i] - mean_drift)
            for i in range(len(motion_displacements))
        )
        corr = cov / (math.sqrt(var_disp) * math.sqrt(var_drift))
        corr = max(-1.0, min(1.0, corr))
    else:
        corr = 0.0

    ratio = mean_drift / (mean_disp + 1e-6)

    return {
        "paired_deltas": paired_deltas,
        "motion_drift_correlation": float(corr),
        "mean_motion_displacement": float(mean_disp),
        "mean_representation_drift": float(mean_drift),
        "sensitivity_ratio": float(ratio),
    }


def compute_video_classification_metrics(
    predictions: list[int],
    targets: list[int],
) -> dict[str, float]:
    """Compute video classification accuracy and summary metrics."""
    if not predictions or len(predictions) != len(targets):
        return {"accuracy": 0.0, "total_samples": 0.0, "correct_samples": 0.0}

    total = len(predictions)
    correct = sum(1 for p, y in zip(predictions, targets, strict=True) if p == y)
    accuracy = correct / total

    return {
        "accuracy": float(accuracy),
        "total_samples": float(total),
        "correct_samples": float(correct),
    }
