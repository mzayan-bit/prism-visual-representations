"""Cross-paradigm synthesis, Pareto analysis, profiles, and tradeoffs."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from prism.benchmarking.aggregation import group_and_aggregate
from prism.benchmarking.contracts import (
    AggregatedBenchmarkResult,
    ParetoAnalysisResult,
    RepresentationProfile,
    TradeoffPoint,
)
from prism.benchmarking.enums import MetricDirection, ResultStatus
from prism.benchmarking.registry import canonical_metric_registry
from prism.benchmarking.store import BenchmarkResultStore


def extract_representation_profile(
    store: BenchmarkResultStore,
    architecture: str,
    pretraining_objective: str | None = None,
    objective: str | None = None,
) -> RepresentationProfile:
    """Extract multi-dimensional representation profile across independent axes."""
    obj = objective or pretraining_objective or "supervised"
    query_cells = store.query(
        factors={
            "architecture": architecture,
            "pretraining_objective": obj,
        }
    )

    # Group by metric category
    cat_map: dict[str, dict[str, float]] = {
        "semantic_performance": {},
        "geometry": {},
        "label_efficiency": {},
        "transferability": {},
        "robustness": {},
        "spatial_transfer": {},
        "temporal_transfer": {},
        "calibration": {},
        "ood_separation": {},
        "multimodal_alignment": {},
    }

    for c in query_cells:
        if c.value is None or c.status != ResultStatus.OBSERVED:
            continue
        mid = c.metric_id
        val = float(c.value)

        if mid in ("accuracy", "loss"):
            cat_map["semantic_performance"][mid] = val
        elif mid in (
            "neighbor_consistency",
            "centroid_separation",
            "intra_class_compactness",
        ):
            cat_map["geometry"][mid] = val
        elif mid == "transfer_gain" or (
            mid == "linear_probe_accuracy" and c.factors.get("data_budget", 1.0) < 1.0
        ):
            cat_map["label_efficiency"][mid] = val
        elif mid == "linear_probe_accuracy":
            cat_map["transferability"][mid] = val
        elif mid in ("robustness_accuracy_drop", "representation_drift"):
            cat_map["robustness"][mid] = val
        elif mid in ("detection_mean_iou", "segmentation_miou", "pixel_accuracy"):
            cat_map["spatial_transfer"][mid] = val
        elif mid in ("video_accuracy", "temporal_consistency"):
            cat_map["temporal_transfer"][mid] = val
        elif mid in ("ece", "brier", "nll"):
            cat_map["calibration"][mid] = val
        elif mid == "ood_auroc":
            cat_map["ood_separation"][mid] = val
        elif mid in ("retrieval_r1", "retrieval_r5", "zero_shot_accuracy"):
            cat_map["multimodal_alignment"][mid] = val

    def _mean_or_none(d: dict[str, float]) -> float | None:
        return sum(d.values()) / float(len(d)) if d else None

    # Specifically for calibration: 1.0 - ECE if ECE is present
    cal_score: float | None = None
    if "ece" in cat_map["calibration"]:
        cal_score = max(0.0, 1.0 - cat_map["calibration"]["ece"])
    elif cat_map["calibration"]:
        cal_score = _mean_or_none(cat_map["calibration"])

    # For robustness: 1.0 - mean_drop if drop is present
    rob_score: float | None = None
    if "robustness_accuracy_drop" in cat_map["robustness"]:
        rob_score = max(0.0, 1.0 - cat_map["robustness"]["robustness_accuracy_drop"])
    elif cat_map["robustness"]:
        rob_score = _mean_or_none(cat_map["robustness"])

    profile_id = f"prof_{architecture}_{obj}"
    return RepresentationProfile(
        profile_id=profile_id,
        architecture=architecture,
        objective=obj,
        semantic_performance=cat_map["semantic_performance"].get(
            "accuracy", _mean_or_none(cat_map["semantic_performance"])
        ),
        geometry=_mean_or_none(cat_map["geometry"]),
        label_efficiency=_mean_or_none(cat_map["label_efficiency"]),
        transferability=cat_map["transferability"].get(
            "linear_probe_accuracy", _mean_or_none(cat_map["transferability"])
        ),
        robustness=rob_score,
        spatial_transfer=cat_map["spatial_transfer"].get(
            "segmentation_miou", _mean_or_none(cat_map["spatial_transfer"])
        ),
        temporal_transfer=cat_map["temporal_transfer"].get(
            "video_accuracy", _mean_or_none(cat_map["temporal_transfer"])
        ),
        calibration=cal_score,
        ood_separation=cat_map["ood_separation"].get(
            "ood_auroc", _mean_or_none(cat_map["ood_separation"])
        ),
        multimodal_alignment=cat_map["multimodal_alignment"].get(
            "zero_shot_accuracy", _mean_or_none(cat_map["multimodal_alignment"])
        ),
        metadata={
            "raw_measurements": cat_map,
        },
    )


def compute_pareto_front(
    candidates_or_store: BenchmarkResultStore | Sequence[dict[str, Any]],
    metric_ids: Sequence[str],
    metric_registry: Any = canonical_metric_registry,
    metric_directions: dict[str, MetricDirection] | None = None,
) -> ParetoAnalysisResult:
    """Compute non-dominated Pareto frontier across multiple target metrics."""
    analysis_id = f"pareto_{'_'.join(metric_ids) if metric_ids else 'empty'}"
    if not metric_ids:
        return ParetoAnalysisResult(
            analysis_id=analysis_id,
            metric_ids=[],
            candidate_experiment_ids=[],
            non_dominated_experiment_ids=[],
            dominated_relationships={},
            exclusions=[],
            missing_metric_warnings=["No metric IDs provided for Pareto optimization."],
        )

    # Convert BenchmarkResultStore if provided
    if isinstance(candidates_or_store, BenchmarkResultStore):
        exp_groups: dict[str, dict[str, Any]] = {}
        for c in candidates_or_store.all_cells():
            if c.value is not None and c.status == ResultStatus.OBSERVED:
                exp_id = c.experiment_id
                if exp_id not in exp_groups:
                    exp_groups[exp_id] = {
                        "experiment_id": exp_id,
                        "factors": dict(c.factors),
                        "metrics": {},
                    }
                exp_groups[exp_id]["metrics"][c.metric_id] = float(c.value)
        candidates = list(exp_groups.values())
    else:
        candidates = list(candidates_or_store)

    # Resolve metric directions
    directions: dict[str, MetricDirection] = {}
    for mid in metric_ids:
        if metric_directions and mid in metric_directions:
            directions[mid] = metric_directions[mid]
        elif metric_registry.has(mid):
            directions[mid] = metric_registry.get(mid).direction
        else:
            directions[mid] = MetricDirection.HIGHER_IS_BETTER

    valid_candidates: list[dict[str, Any]] = []
    exclusions: list[str] = []
    missing_warnings: list[str] = []

    for cand in candidates:
        metrics = cand.get("metrics", {})
        cand_id = cand.get("experiment_id", "unknown")
        has_all = all(mid in metrics and metrics[mid] is not None for mid in metric_ids)
        if has_all:
            valid_candidates.append(cand)
        else:
            exclusions.append(cand_id)
            missing_m = [
                m for m in metric_ids if m not in metrics or metrics[m] is None
            ]
            missing_warnings.append(
                f"Candidate '{cand_id}' missing metrics: {missing_m}"
            )

    non_dominated: list[str] = []
    dominated_rel: dict[str, list[str]] = {}

    def strictly_dominates(cand_a: dict[str, Any], cand_b: dict[str, Any]) -> bool:
        metrics_a = cand_a.get("metrics", {})
        metrics_b = cand_b.get("metrics", {})

        better_in_at_least_one = False
        for mid in metric_ids:
            val_a = float(metrics_a[mid])
            val_b = float(metrics_b[mid])
            direction = directions[mid]

            if direction == MetricDirection.HIGHER_IS_BETTER:
                if val_a < val_b:
                    return False
                if val_a > val_b:
                    better_in_at_least_one = True
            elif direction == MetricDirection.LOWER_IS_BETTER:
                if val_a > val_b:
                    return False
                if val_a < val_b:
                    better_in_at_least_one = True
            else:
                pass

        return better_in_at_least_one

    candidate_ids = [
        c.get("experiment_id", f"exp_{i}") for i, c in enumerate(valid_candidates)
    ]

    for i, cand_a in enumerate(valid_candidates):
        cand_a_id = candidate_ids[i]
        is_dom = False
        dominators: list[str] = []

        for j, cand_b in enumerate(valid_candidates):
            if i == j:
                continue
            cand_b_id = candidate_ids[j]
            if strictly_dominates(cand_b, cand_a):
                is_dom = True
                dominators.append(cand_b_id)

        if not is_dom:
            non_dominated.append(cand_a_id)
        else:
            dominated_rel[cand_a_id] = dominators

    return ParetoAnalysisResult(
        analysis_id=analysis_id,
        metric_ids=list(metric_ids),
        candidate_experiment_ids=candidate_ids,
        non_dominated_experiment_ids=non_dominated,
        dominated_relationships=dominated_rel,
        exclusions=exclusions,
        missing_metric_warnings=missing_warnings,
    )


def extract_tradeoff_pairs(
    store: BenchmarkResultStore,
    metric_x: str | None = None,
    metric_y: str | None = None,
    x_metric_id: str | None = None,
    y_metric_id: str | None = None,
    group_factors: Sequence[str] = ("architecture", "pretraining_objective"),
) -> list[TradeoffPoint]:
    """Extract paired observation points for scatter-ready tradeoff analysis."""
    mx = metric_x or x_metric_id or "accuracy"
    my = metric_y or y_metric_id or "loss"

    cells_x = store.query(metric_id=mx)
    cells_y = store.query(metric_id=my)

    map_x: dict[tuple[str, ...], float] = {}
    map_y: dict[tuple[str, ...], float] = {}
    factor_meta: dict[tuple[str, ...], dict[str, Any]] = {}
    exp_ids: dict[tuple[str, ...], str] = {}

    for c in cells_x:
        if c.value is not None and c.status == ResultStatus.OBSERVED:
            key = tuple(str(c.factors.get(f, "unknown")) for f in group_factors)
            map_x[key] = float(c.value)
            factor_meta[key] = dict(c.factors)
            exp_ids[key] = c.experiment_id

    for c in cells_y:
        if c.value is not None and c.status == ResultStatus.OBSERVED:
            key = tuple(str(c.factors.get(f, "unknown")) for f in group_factors)
            map_y[key] = float(c.value)
            if key not in factor_meta:
                factor_meta[key] = dict(c.factors)
                exp_ids[key] = c.experiment_id

    tradeoff_points: list[TradeoffPoint] = []
    common_keys = sorted(set(map_x.keys()).intersection(set(map_y.keys())))

    for k in common_keys:
        tradeoff_points.append(
            TradeoffPoint(
                experiment_id=exp_ids[k],
                factors=factor_meta[k],
                x_metric=mx,
                x_value=map_x[k],
                y_metric=my,
                y_value=map_y[k],
                note="Descriptive tradeoff pair; does not imply causal relationship.",
            )
        )

    return tradeoff_points


def synthesize_cross_architecture(
    store: BenchmarkResultStore,
    architectures: Sequence[str] = ("cnn", "resnet", "vit"),
) -> dict[str, dict[str, AggregatedBenchmarkResult]]:
    """Synthesize metrics across architectures grouped by metric identifier."""
    res: dict[str, dict[str, AggregatedBenchmarkResult]] = {}
    for arch in architectures:
        arch_cells = store.query(factors={"architecture": arch})
        by_metric = group_and_aggregate(arch_cells, group_by_factors=["architecture"])
        res[arch] = {agg.metric_id: agg for agg in by_metric}
    return res


def synthesize_cross_objective(
    store: BenchmarkResultStore,
    objectives: Sequence[str] = (
        "supervised",
        "simclr",
        "reconstruction",
        "vision_language",
        "scratch",
    ),
) -> dict[str, dict[str, AggregatedBenchmarkResult]]:
    """Synthesize metrics across pretraining objectives grouped by metric identifier."""
    res: dict[str, dict[str, AggregatedBenchmarkResult]] = {}
    for obj in objectives:
        obj_cells = store.query(factors={"pretraining_objective": obj})
        by_metric = group_and_aggregate(
            obj_cells, group_by_factors=["pretraining_objective"]
        )
        res[obj] = {agg.metric_id: agg for agg in by_metric}
    return res
