"""Adapters transforming heterogeneous PRISM reports into canonical result cells."""

from __future__ import annotations

from typing import Any

from prism.benchmarking.contracts import BenchmarkResultCell
from prism.benchmarking.enums import ResultStatus
from prism.core.errors import ValidationError


def adapt_architecture_comparison_report(report: Any) -> list[BenchmarkResultCell]:
    """Adapt ArchitectureComparisonReport into canonical benchmark cells."""
    cells: list[BenchmarkResultCell] = []
    suite_id = getattr(report, "suite_id", "arch_suite")

    # 1. From metric_summaries list if present
    metric_summaries = getattr(report, "metric_summaries", [])
    if metric_summaries:
        for m_sum in metric_summaries:
            exp_id = getattr(m_sum, "experiment_id", suite_id)
            model_fam = getattr(
                m_sum,
                "model_family",
                getattr(m_sum, "family", getattr(m_sum, "model_id", exp_id)),
            )
            arch_str = getattr(
                m_sum,
                "architecture",
                str(model_fam.value if hasattr(model_fam, "value") else model_fam),
            )

            factors = {
                "architecture": str(arch_str),
                "pretraining_objective": "supervised",
                "task": "classification",
                "seed": 42,
            }

            acc = getattr(m_sum, "test_accuracy", None)
            if acc is not None:
                cells.append(
                    BenchmarkResultCell(
                        result_id=f"{exp_id}_accuracy",
                        experiment_id=exp_id,
                        experiment_fingerprint=f"fp_{exp_id}",
                        metric_id="accuracy",
                        value=float(acc),
                        status=ResultStatus.OBSERVED,
                        source_report_type="ArchitectureComparisonReport",
                        source_run_id=exp_id,
                        factors=factors,
                    )
                )

            loss_val = getattr(m_sum, "final_validation_loss", None)
            if loss_val is not None:
                cells.append(
                    BenchmarkResultCell(
                        result_id=f"{exp_id}_loss",
                        experiment_id=exp_id,
                        experiment_fingerprint=f"fp_{exp_id}",
                        metric_id="loss",
                        value=float(loss_val),
                        status=ResultStatus.OBSERVED,
                        source_report_type="ArchitectureComparisonReport",
                        source_run_id=exp_id,
                        factors=factors,
                    )
                )

            param_count = getattr(m_sum, "parameter_count", None)
            if param_count is not None:
                cells.append(
                    BenchmarkResultCell(
                        result_id=f"{exp_id}_parameter_count",
                        experiment_id=exp_id,
                        experiment_fingerprint=f"fp_{exp_id}",
                        metric_id="parameter_count",
                        value=float(param_count),
                        status=ResultStatus.OBSERVED,
                        source_report_type="ArchitectureComparisonReport",
                        source_run_id=exp_id,
                        factors=factors,
                    )
                )
        return cells

    # 2. From model_results dictionary if present
    model_results = getattr(report, "model_results", {})
    if isinstance(model_results, dict):
        for arch_key, run_res in model_results.items():
            exp_id = getattr(run_res, "experiment_id", f"{suite_id}_{arch_key}")
            fingerprint = getattr(run_res, "fingerprint", f"fp_{exp_id}")
            family = getattr(run_res, "family", arch_key)
            seed = getattr(run_res, "seed", 42)

            family_val = (
                getattr(family, "value", family) if family is not None else arch_key
            )
            factors = {
                "architecture": str(family_val),
                "pretraining_objective": "supervised",
                "task": "classification",
                "seed": seed,
            }

            acc = getattr(run_res, "test_accuracy", None)
            if acc is not None:
                cells.append(
                    BenchmarkResultCell(
                        result_id=f"{exp_id}_accuracy",
                        experiment_id=exp_id,
                        experiment_fingerprint=fingerprint,
                        metric_id="accuracy",
                        value=float(acc),
                        status=ResultStatus.OBSERVED,
                        seed=seed,
                        source_report_type="ArchitectureComparisonReport",
                        source_run_id=getattr(run_res, "run_id", exp_id),
                        factors=factors,
                        provenance={"suite_id": suite_id, "arch_key": arch_key},
                    )
                )

            loss_val = getattr(run_res, "test_loss", None)
            if loss_val is not None:
                cells.append(
                    BenchmarkResultCell(
                        result_id=f"{exp_id}_loss",
                        experiment_id=exp_id,
                        experiment_fingerprint=fingerprint,
                        metric_id="loss",
                        value=float(loss_val),
                        status=ResultStatus.OBSERVED,
                        seed=seed,
                        source_report_type="ArchitectureComparisonReport",
                        source_run_id=getattr(run_res, "run_id", exp_id),
                        factors=factors,
                        provenance={"suite_id": suite_id, "arch_key": arch_key},
                    )
                )

            param_count = getattr(run_res, "parameter_count", None)
            if param_count is not None:
                cells.append(
                    BenchmarkResultCell(
                        result_id=f"{exp_id}_parameter_count",
                        experiment_id=exp_id,
                        experiment_fingerprint=fingerprint,
                        metric_id="parameter_count",
                        value=float(param_count),
                        status=ResultStatus.OBSERVED,
                        seed=seed,
                        source_report_type="ArchitectureComparisonReport",
                        source_run_id=getattr(run_res, "run_id", exp_id),
                        provenance={"suite_id": suite_id, "arch_key": arch_key},
                    )
                )

    return cells


def adapt_representation_geometry_report(report: Any) -> list[BenchmarkResultCell]:
    """Adapt RepresentationGeometryReport into canonical benchmark cells."""
    cells: list[BenchmarkResultCell] = []
    exp_id = getattr(report, "experiment_id", "rep_exp")
    fingerprint = getattr(report, "fingerprint", f"fp_{exp_id}")
    model_id = getattr(report, "model_id", "model")
    layer = getattr(report, "layer_name", "final_hidden")

    factors = {
        "architecture": model_id,
        "representation_layer": layer,
        "pretraining_objective": "supervised",
    }

    # Centroid separation & compactness
    centroid_geom = getattr(report, "centroid_geometry", None)
    if centroid_geom is not None:
        inter_dist = getattr(centroid_geom, "mean_inter_class_distance", None)
        if inter_dist is not None:
            cells.append(
                BenchmarkResultCell(
                    result_id=f"{exp_id}_centroid_separation",
                    experiment_id=exp_id,
                    experiment_fingerprint=fingerprint,
                    metric_id="centroid_separation",
                    value=float(inter_dist),
                    status=ResultStatus.OBSERVED,
                    source_report_type="RepresentationGeometryReport",
                    source_run_id=exp_id,
                    factors=factors,
                    provenance={"layer": layer},
                )
            )

        intra_comp = getattr(centroid_geom, "mean_intra_class_compactness", None)
        if intra_comp is not None:
            cells.append(
                BenchmarkResultCell(
                    result_id=f"{exp_id}_intra_class_compactness",
                    experiment_id=exp_id,
                    experiment_fingerprint=fingerprint,
                    metric_id="intra_class_compactness",
                    value=float(intra_comp),
                    status=ResultStatus.OBSERVED,
                    source_report_type="RepresentationGeometryReport",
                    source_run_id=exp_id,
                    factors=factors,
                    provenance={"layer": layer},
                )
            )

    # Neighborhood consistency
    neigh_geom = getattr(report, "neighborhood_geometry", None)
    if neigh_geom is not None:
        consistency = getattr(neigh_geom, "mean_neighborhood_consistency", None)
        if consistency is not None:
            cells.append(
                BenchmarkResultCell(
                    result_id=f"{exp_id}_neighbor_consistency",
                    experiment_id=exp_id,
                    experiment_fingerprint=fingerprint,
                    metric_id="neighbor_consistency",
                    value=float(consistency),
                    status=ResultStatus.OBSERVED,
                    source_report_type="RepresentationGeometryReport",
                    source_run_id=exp_id,
                    factors=factors,
                    provenance={"layer": layer},
                )
            )

    return cells


def adapt_robustness_report(report: Any) -> list[BenchmarkResultCell]:
    """Adapt robustness experiment reports into canonical benchmark cells."""
    cells: list[BenchmarkResultCell] = []

    # Case A: CrossArchitectureRobustnessReport
    arch_summaries = getattr(report, "architecture_summaries", None)
    if arch_summaries and isinstance(arch_summaries, dict):
        suite_id = getattr(report, "suite_id", "rob_suite")
        for arch_name, arch_sum in arch_summaries.items():
            exp_id = f"{suite_id}_{arch_name}"
            clean_acc = getattr(arch_sum, "clean_accuracy", None)
            drop = getattr(arch_sum, "mean_accuracy_drop", None)
            drift = getattr(arch_sum, "mean_representation_drift", None)

            factors = {
                "architecture": arch_name,
                "task": "robustness",
            }

            if clean_acc is not None:
                cells.append(
                    BenchmarkResultCell(
                        result_id=f"{exp_id}_clean_accuracy",
                        experiment_id=exp_id,
                        experiment_fingerprint=f"fp_{exp_id}",
                        metric_id="accuracy",
                        value=float(clean_acc),
                        status=ResultStatus.OBSERVED,
                        source_report_type="CrossArchitectureRobustnessReport",
                        source_run_id=exp_id,
                        factors=factors,
                    )
                )

            if drop is not None:
                cells.append(
                    BenchmarkResultCell(
                        result_id=f"{exp_id}_robustness_accuracy_drop",
                        experiment_id=exp_id,
                        experiment_fingerprint=f"fp_{exp_id}",
                        metric_id="robustness_accuracy_drop",
                        value=float(drop),
                        status=ResultStatus.OBSERVED,
                        source_report_type="CrossArchitectureRobustnessReport",
                        source_run_id=exp_id,
                        factors=factors,
                    )
                )

            if drift is not None:
                cells.append(
                    BenchmarkResultCell(
                        result_id=f"{exp_id}_representation_drift",
                        experiment_id=exp_id,
                        experiment_fingerprint=f"fp_{exp_id}",
                        metric_id="representation_drift",
                        value=float(drift),
                        status=ResultStatus.OBSERVED,
                        source_report_type="CrossArchitectureRobustnessReport",
                        source_run_id=exp_id,
                        factors=factors,
                    )
                )
        return cells

    # Case B: RobustnessExperimentReport
    exp_id = getattr(report, "experiment_id", "rob_exp")
    arch = getattr(report, "architecture", getattr(report, "model_id", "unknown"))
    drop = getattr(report, "mean_accuracy_drop", None)
    drift = getattr(report, "mean_representation_drift", None)

    factors = {"architecture": str(arch), "task": "robustness"}

    if drop is not None:
        cells.append(
            BenchmarkResultCell(
                result_id=f"{exp_id}_robustness_accuracy_drop",
                experiment_id=exp_id,
                experiment_fingerprint=f"fp_{exp_id}",
                metric_id="robustness_accuracy_drop",
                value=float(drop),
                status=ResultStatus.OBSERVED,
                source_report_type="RobustnessExperimentReport",
                source_run_id=exp_id,
                factors=factors,
            )
        )

    if drift is not None:
        cells.append(
            BenchmarkResultCell(
                result_id=f"{exp_id}_representation_drift",
                experiment_id=exp_id,
                experiment_fingerprint=f"fp_{exp_id}",
                metric_id="representation_drift",
                value=float(drift),
                status=ResultStatus.OBSERVED,
                source_report_type="RobustnessExperimentReport",
                source_run_id=exp_id,
                factors=factors,
            )
        )

    return cells


def adapt_explainability_report(report: Any) -> list[BenchmarkResultCell]:
    """Adapt AttributionComparisonReport into canonical benchmark cells."""
    cells: list[BenchmarkResultCell] = []
    exp_id = getattr(report, "experiment_id", "exp_report")
    arch = getattr(report, "architecture", getattr(report, "model_id", "model"))
    mean_agree = getattr(report, "mean_method_agreement", None)

    factors = {"architecture": str(arch), "task": "explainability"}

    if mean_agree is not None:
        cells.append(
            BenchmarkResultCell(
                result_id=f"{exp_id}_attribution_agreement",
                experiment_id=exp_id,
                experiment_fingerprint=f"fp_{exp_id}",
                metric_id="attribution_agreement",
                value=float(mean_agree),
                status=ResultStatus.OBSERVED,
                source_report_type="AttributionComparisonReport",
                source_run_id=exp_id,
                factors=factors,
            )
        )

    return cells


def adapt_transfer_report(report: Any) -> list[BenchmarkResultCell]:
    """Adapt TransferLearningReport into canonical benchmark cells."""
    cells: list[BenchmarkResultCell] = []
    spec_id = getattr(
        report, "spec_id", getattr(report, "report_id", "transfer_report")
    )
    arch = getattr(report, "source_architecture", "resnet")
    target_ds = getattr(report, "target_dataset", "target")

    # Linear probe accuracy
    lp_acc = getattr(report, "linear_probe_accuracy", None)
    if lp_acc is not None:
        cells.append(
            BenchmarkResultCell(
                result_id=f"{spec_id}_linear_probe_accuracy",
                experiment_id=spec_id,
                experiment_fingerprint=f"fp_{spec_id}",
                metric_id="linear_probe_accuracy",
                value=float(lp_acc),
                status=ResultStatus.OBSERVED,
                source_report_type="TransferLearningReport",
                source_run_id=spec_id,
                factors={
                    "architecture": str(arch),
                    "task": "transfer",
                    "dataset": str(target_ds),
                    "transfer_strategy": "linear_probe",
                },
            )
        )

    # Transfer gain
    gain = getattr(report, "transfer_gain", None)
    if gain is not None:
        cells.append(
            BenchmarkResultCell(
                result_id=f"{spec_id}_transfer_gain",
                experiment_id=spec_id,
                experiment_fingerprint=f"fp_{spec_id}",
                metric_id="transfer_gain",
                value=float(gain),
                status=ResultStatus.OBSERVED,
                source_report_type="TransferLearningReport",
                source_run_id=spec_id,
                factors={
                    "architecture": str(arch),
                    "task": "transfer",
                    "dataset": str(target_ds),
                },
            )
        )

    return cells


def adapt_ssl_report(report: Any) -> list[BenchmarkResultCell]:
    """Adapt SelfSupervisedLearningReport into canonical benchmark cells."""
    cells: list[BenchmarkResultCell] = []
    spec_id = getattr(report, "spec_id", getattr(report, "report_id", "ssl_report"))
    arch = getattr(report, "architecture", "resnet")

    factors = {
        "architecture": str(arch),
        "pretraining_objective": "simclr",
        "task": "self_supervised",
    }

    # Contrastive Loss
    loss_val = getattr(report, "final_loss", None)
    if loss_val is not None:
        cells.append(
            BenchmarkResultCell(
                result_id=f"{spec_id}_contrastive_loss",
                experiment_id=spec_id,
                experiment_fingerprint=f"fp_{spec_id}",
                metric_id="contrastive_loss",
                value=float(loss_val),
                status=ResultStatus.OBSERVED,
                source_report_type="SelfSupervisedLearningReport",
                source_run_id=spec_id,
                factors=factors,
            )
        )

    # Downstream linear probe accuracy if evaluated
    lp_acc = getattr(report, "linear_probe_accuracy", None)
    if lp_acc is not None:
        cells.append(
            BenchmarkResultCell(
                result_id=f"{spec_id}_linear_probe_accuracy",
                experiment_id=spec_id,
                experiment_fingerprint=f"fp_{spec_id}",
                metric_id="linear_probe_accuracy",
                value=float(lp_acc),
                status=ResultStatus.OBSERVED,
                source_report_type="SelfSupervisedLearningReport",
                source_run_id=spec_id,
                factors=factors,
            )
        )

    return cells


def adapt_reconstruction_report(report: Any) -> list[BenchmarkResultCell]:
    """Adapt ReconstructionLearningReport into canonical benchmark cells."""
    cells: list[BenchmarkResultCell] = []
    spec_id = getattr(report, "spec_id", getattr(report, "report_id", "recon_report"))
    arch = getattr(report, "architecture", "resnet")

    factors = {
        "architecture": str(arch),
        "pretraining_objective": "reconstruction",
        "task": "reconstruction",
    }

    # Reconstruction MSE
    mse = getattr(report, "reconstruction_mse", getattr(report, "final_loss", None))
    if mse is not None:
        cells.append(
            BenchmarkResultCell(
                result_id=f"{spec_id}_reconstruction_mse",
                experiment_id=spec_id,
                experiment_fingerprint=f"fp_{spec_id}",
                metric_id="reconstruction_mse",
                value=float(mse),
                status=ResultStatus.OBSERVED,
                source_report_type="ReconstructionLearningReport",
                source_run_id=spec_id,
                factors=factors,
            )
        )

    # Linear probe
    lp_acc = getattr(report, "linear_probe_accuracy", None)
    if lp_acc is not None:
        cells.append(
            BenchmarkResultCell(
                result_id=f"{spec_id}_linear_probe_accuracy",
                experiment_id=spec_id,
                experiment_fingerprint=f"fp_{spec_id}",
                metric_id="linear_probe_accuracy",
                value=float(lp_acc),
                status=ResultStatus.OBSERVED,
                source_report_type="ReconstructionLearningReport",
                source_run_id=spec_id,
                factors=factors,
            )
        )

    return cells


def adapt_spatial_transfer_report(report: Any) -> list[BenchmarkResultCell]:
    """Adapt SpatialTransferReport into canonical benchmark cells."""
    cells: list[BenchmarkResultCell] = []
    spec_id = getattr(report, "spec_id", getattr(report, "report_id", "spatial_report"))
    arch = getattr(report, "architecture", "resnet")
    obj = getattr(report, "source_objective", "supervised")

    factors = {
        "architecture": str(arch),
        "pretraining_objective": str(obj),
        "task": "spatial",
    }

    # Detection mean IoU
    det_metrics = getattr(report, "detection_metrics", None)
    if det_metrics is not None:
        miou = getattr(det_metrics, "mean_iou", None)
        if miou is not None:
            cells.append(
                BenchmarkResultCell(
                    result_id=f"{spec_id}_detection_mean_iou",
                    experiment_id=spec_id,
                    experiment_fingerprint=f"fp_{spec_id}",
                    metric_id="detection_mean_iou",
                    value=float(miou),
                    status=ResultStatus.OBSERVED,
                    source_report_type="SpatialTransferReport",
                    source_run_id=spec_id,
                    factors=factors,
                )
            )

    # Segmentation mIoU and pixel accuracy
    seg_metrics = getattr(report, "segmentation_metrics", None)
    if seg_metrics is not None:
        seg_miou = getattr(seg_metrics, "mean_iou", None)
        if seg_miou is not None:
            cells.append(
                BenchmarkResultCell(
                    result_id=f"{spec_id}_segmentation_miou",
                    experiment_id=spec_id,
                    experiment_fingerprint=f"fp_{spec_id}",
                    metric_id="segmentation_miou",
                    value=float(seg_miou),
                    status=ResultStatus.OBSERVED,
                    source_report_type="SpatialTransferReport",
                    source_run_id=spec_id,
                    factors=factors,
                )
            )

        pix_acc = getattr(seg_metrics, "pixel_accuracy", None)
        if pix_acc is not None:
            cells.append(
                BenchmarkResultCell(
                    result_id=f"{spec_id}_pixel_accuracy",
                    experiment_id=spec_id,
                    experiment_fingerprint=f"fp_{spec_id}",
                    metric_id="pixel_accuracy",
                    value=float(pix_acc),
                    status=ResultStatus.OBSERVED,
                    source_report_type="SpatialTransferReport",
                    source_run_id=spec_id,
                    factors=factors,
                )
            )

    return cells


def adapt_temporal_report(report: Any) -> list[BenchmarkResultCell]:
    """Adapt TemporalRepresentationReport into canonical benchmark cells."""
    cells: list[BenchmarkResultCell] = []
    spec_id = getattr(
        report, "spec_id", getattr(report, "report_id", "temporal_report")
    )
    arch = getattr(report, "architecture", "cnn")
    obj = getattr(report, "source_objective", "supervised")

    factors = {
        "architecture": str(arch),
        "pretraining_objective": str(obj),
        "task": "temporal",
    }

    # Video accuracy
    vid_acc = getattr(report, "final_accuracy", None)
    if vid_acc is not None:
        cells.append(
            BenchmarkResultCell(
                result_id=f"{spec_id}_video_accuracy",
                experiment_id=spec_id,
                experiment_fingerprint=f"fp_{spec_id}",
                metric_id="video_accuracy",
                value=float(vid_acc),
                status=ResultStatus.OBSERVED,
                source_report_type="TemporalRepresentationReport",
                source_run_id=spec_id,
                factors=factors,
            )
        )

    # Temporal consistency
    t_const = getattr(report, "temporal_consistency", None)
    if t_const is not None:
        adj_cos = getattr(t_const, "mean_adjacent_cosine", None)
        if adj_cos is not None:
            cells.append(
                BenchmarkResultCell(
                    result_id=f"{spec_id}_temporal_consistency",
                    experiment_id=spec_id,
                    experiment_fingerprint=f"fp_{spec_id}",
                    metric_id="temporal_consistency",
                    value=float(adj_cos),
                    status=ResultStatus.OBSERVED,
                    source_report_type="TemporalRepresentationReport",
                    source_run_id=spec_id,
                    factors=factors,
                )
            )

    return cells


def adapt_multimodal_report(report: Any) -> list[BenchmarkResultCell]:
    """Adapt VisionLanguageRepresentationReport into canonical benchmark cells."""
    cells: list[BenchmarkResultCell] = []
    spec_id = getattr(
        report, "spec_id", getattr(report, "report_id", "multimodal_report")
    )
    arch = getattr(report, "visual_architecture", "cnn")

    factors = {
        "architecture": str(arch),
        "pretraining_objective": "vision_language",
        "task": "multimodal",
    }

    # Retrieval R@1 and R@5
    ret_sum = getattr(report, "retrieval_summary", None)
    if ret_sum is not None:
        i2t_r1 = getattr(ret_sum, "image_to_text_r1", None)
        t2i_r1 = getattr(ret_sum, "text_to_image_r1", None)
        if i2t_r1 is not None and t2i_r1 is not None:
            mean_r1 = (float(i2t_r1) + float(t2i_r1)) / 2.0
            cells.append(
                BenchmarkResultCell(
                    result_id=f"{spec_id}_retrieval_r1",
                    experiment_id=spec_id,
                    experiment_fingerprint=f"fp_{spec_id}",
                    metric_id="retrieval_r1",
                    value=mean_r1,
                    status=ResultStatus.OBSERVED,
                    source_report_type="VisionLanguageRepresentationReport",
                    source_run_id=spec_id,
                    factors=factors,
                )
            )

        i2t_r5 = getattr(ret_sum, "image_to_text_r5", None)
        t2i_r5 = getattr(ret_sum, "text_to_image_r5", None)
        if i2t_r5 is not None and t2i_r5 is not None:
            mean_r5 = (float(i2t_r5) + float(t2i_r5)) / 2.0
            cells.append(
                BenchmarkResultCell(
                    result_id=f"{spec_id}_retrieval_r5",
                    experiment_id=spec_id,
                    experiment_fingerprint=f"fp_{spec_id}",
                    metric_id="retrieval_r5",
                    value=mean_r5,
                    status=ResultStatus.OBSERVED,
                    source_report_type="VisionLanguageRepresentationReport",
                    source_run_id=spec_id,
                    factors=factors,
                )
            )

    # Zero-shot classification
    zs_sum = getattr(report, "zero_shot_summary", None)
    if zs_sum is not None:
        zs_acc = getattr(zs_sum, "top1_accuracy", None)
        if zs_acc is not None:
            cells.append(
                BenchmarkResultCell(
                    result_id=f"{spec_id}_zero_shot_accuracy",
                    experiment_id=spec_id,
                    experiment_fingerprint=f"fp_{spec_id}",
                    metric_id="zero_shot_accuracy",
                    value=float(zs_acc),
                    status=ResultStatus.OBSERVED,
                    source_report_type="VisionLanguageRepresentationReport",
                    source_run_id=spec_id,
                    factors=factors,
                )
            )

    return cells


def adapt_uncertainty_report(report: Any) -> list[BenchmarkResultCell]:
    """Adapt UncertaintyAnalysisReport into canonical benchmark cells."""
    cells: list[BenchmarkResultCell] = []
    exp_id = getattr(report, "model_id", "uncertainty_model")
    arch = getattr(report, "architecture", "resnet")
    obj = getattr(report, "source_objective", "supervised")

    factors = {
        "architecture": str(arch),
        "pretraining_objective": str(obj),
        "task": "uncertainty",
    }

    # Calibration report
    cal_rep = getattr(report, "calibration_report", None)
    if cal_rep is not None:
        ece_val = getattr(cal_rep, "ece", None)
        if ece_val is not None:
            cells.append(
                BenchmarkResultCell(
                    result_id=f"{exp_id}_ece",
                    experiment_id=exp_id,
                    experiment_fingerprint=f"fp_{exp_id}",
                    metric_id="ece",
                    value=float(ece_val),
                    status=ResultStatus.OBSERVED,
                    source_report_type="UncertaintyAnalysisReport",
                    source_run_id=exp_id,
                    factors=factors,
                )
            )

        brier_val = getattr(cal_rep, "brier_score", None)
        if brier_val is not None:
            cells.append(
                BenchmarkResultCell(
                    result_id=f"{exp_id}_brier",
                    experiment_id=exp_id,
                    experiment_fingerprint=f"fp_{exp_id}",
                    metric_id="brier",
                    value=float(brier_val),
                    status=ResultStatus.OBSERVED,
                    source_report_type="UncertaintyAnalysisReport",
                    source_run_id=exp_id,
                    factors=factors,
                )
            )

        nll_val = getattr(cal_rep, "nll", None)
        if nll_val is not None:
            cells.append(
                BenchmarkResultCell(
                    result_id=f"{exp_id}_nll",
                    experiment_id=exp_id,
                    experiment_fingerprint=f"fp_{exp_id}",
                    metric_id="nll",
                    value=float(nll_val),
                    status=ResultStatus.OBSERVED,
                    source_report_type="UncertaintyAnalysisReport",
                    source_run_id=exp_id,
                    factors=factors,
                )
            )

    # OOD evaluations
    ood_evals = getattr(report, "ood_evaluations", {})
    if ood_evals and isinstance(ood_evals, dict):
        # Prefer nearest class centroid distance or MSP
        centroid_eval = ood_evals.get(
            "nearest_class_centroid_distance",
            ood_evals.get("max_softmax_probability", next(iter(ood_evals.values()))),
        )
        if centroid_eval is not None:
            auroc_val = getattr(centroid_eval, "auroc", None)
            if auroc_val is not None:
                cells.append(
                    BenchmarkResultCell(
                        result_id=f"{exp_id}_ood_auroc",
                        experiment_id=exp_id,
                        experiment_fingerprint=f"fp_{exp_id}",
                        metric_id="ood_auroc",
                        value=float(auroc_val),
                        status=ResultStatus.OBSERVED,
                        source_report_type="UncertaintyAnalysisReport",
                        source_run_id=exp_id,
                        factors=factors,
                    )
                )

    return cells


def adapt_any_report(report: Any) -> list[BenchmarkResultCell]:
    """Polymorphic adapter converting any supported PRISM report into cells."""
    cls_name = report.__class__.__name__

    if cls_name == "ArchitectureComparisonReport":
        return adapt_architecture_comparison_report(report)
    elif cls_name == "RepresentationGeometryReport":
        return adapt_representation_geometry_report(report)
    elif cls_name in (
        "CrossArchitectureRobustnessReport",
        "RobustnessExperimentReport",
    ):
        return adapt_robustness_report(report)
    elif cls_name == "AttributionComparisonReport":
        return adapt_explainability_report(report)
    elif cls_name == "TransferLearningReport":
        return adapt_transfer_report(report)
    elif cls_name == "SelfSupervisedLearningReport":
        return adapt_ssl_report(report)
    elif cls_name == "ReconstructionLearningReport":
        return adapt_reconstruction_report(report)
    elif cls_name == "SpatialTransferReport":
        return adapt_spatial_transfer_report(report)
    elif cls_name == "TemporalRepresentationReport":
        return adapt_temporal_report(report)
    elif cls_name == "VisionLanguageRepresentationReport":
        return adapt_multimodal_report(report)
    elif cls_name == "UncertaintyAnalysisReport":
        return adapt_uncertainty_report(report)
    else:
        raise ValidationError(
            f"Unsupported report type for benchmarking adapter: {cls_name}"
        )
