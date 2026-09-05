"""End-to-End Smoke Test for Uncertainty, Calibration & OOD Representation Analysis."""

from __future__ import annotations

import json
import math

from prism.uncertainty.calibration import (
    compute_brier_score,
    compute_calibration_report,
    compute_expected_calibration_error,
    compute_negative_log_likelihood,
    compute_reliability_bins,
)
from prism.uncertainty.contracts import (
    CalibrationSample,
    UncertaintyAnalysisReport,
)
from prism.uncertainty.corruptions import evaluate_corruption_uncertainty
from prism.uncertainty.enums import (
    BinningStrategy,
    OODCategory,
    OODScoreMethod,
    ThresholdPolicy,
)
from prism.uncertainty.failures import detect_uncertainty_failures
from prism.uncertainty.metrics import (
    evaluate_ood_binary_classification,
)
from prism.uncertainty.ood_scores import score_ood_sample
from prism.uncertainty.probabilities import (
    batch_predictive_distributions,
    compute_predictive_distribution,
    compute_predictive_entropy,
    compute_stable_softmax,
)
from prism.uncertainty.reference_set import (
    build_ood_reference_set,
)
from prism.uncertainty.relationships import (
    compute_representation_confidence_relationships,
)
from prism.uncertainty.reports import compile_uncertainty_analysis_report
from prism.uncertainty.synthetic import (
    SyntheticOODSpec,
    generate_synthetic_ood_dataset,
)
from prism.uncertainty.temperature import (
    evaluate_calibrated_predictions,
    fit_temperature_scaling,
)


def test_smoke_uncertainty_analysis_pipeline() -> None:
    """Execute complete Phase 23 uncertainty analysis pipeline end-to-end."""
    seed = 42

    # 1. Deterministic Synthetic OOD Dataset Generation
    spec = SyntheticOODSpec(
        dataset_name="smoke-ood-v1",
        num_samples=30,
        image_shape=(3, 16, 16),
        seed=seed,
    )
    ood_samples, ood_meta = generate_synthetic_ood_dataset(spec)
    assert len(ood_samples) == 30

    # 2. Probability Foundations & Numerical Softmax
    logits_sample = [4.2, 1.1, -0.5]
    probs = compute_stable_softmax(logits_sample)
    assert abs(sum(probs) - 1.0) < 1e-5
    entropy = compute_predictive_entropy(probs)
    assert 0.0 <= entropy <= math.log(3)

    # 3. Batch Predictive Distributions
    num_id = 20
    sample_ids = [f"id_sample_{i}" for i in range(num_id)]
    true_labels = [i % 3 for i in range(num_id)]
    test_logits = []
    test_reps = []
    for i, lbl in enumerate(true_labels):
        z = [0.2, 0.2, 0.2]
        z[lbl] = 3.5 if i % 5 != 0 else 0.5  # introduce ~80% accuracy
        test_logits.append(z)
        rep = [0.0, 0.0, 0.0]
        rep[lbl] = 2.0
        test_reps.append(rep)

    dists = batch_predictive_distributions(
        sample_ids=sample_ids,
        logits_matrix=test_logits,
        true_classes=true_labels,
    )
    assert len(dists) == 20

    # 4. Calibration & Reliability Analysis
    cal_samples = [
        CalibrationSample(
            sample_id=d.sample_id,
            confidence=d.max_probability,
            is_correct=(d.is_correct if d.is_correct is not None else False),
            predicted_class=d.predicted_class,
            true_class=d.true_class if d.true_class is not None else 0,
            probabilities=d.probabilities,
        )
        for d in dists
    ]

    rel_bins = compute_reliability_bins(
        cal_samples, bin_count=5, strategy=BinningStrategy.EQUAL_WIDTH
    )
    ece = compute_expected_calibration_error(rel_bins, total_sample_count=20)
    brier = compute_brier_score(dists)
    nll = compute_negative_log_likelihood(dists)
    assert 0.0 <= ece <= 1.0
    assert brier >= 0.0
    assert nll >= 0.0

    cal_report = compute_calibration_report(
        distributions=dists,
        bin_count=5,
        class_names={0: "Square", 1: "Circle", 2: "Triangle"},
    )
    assert cal_report.sample_count == 20

    # 5. Temperature Scaling Optimization
    val_logits = test_logits[:10]
    val_labels = true_labels[:10]
    temp_result = fit_temperature_scaling(
        val_logits,
        val_labels,
        search_range=(0.05, 10.0),
        coarse_steps=50,
        fine_steps=20,
    )
    assert temp_result.fitted_temperature > 0.0

    cal_eval = evaluate_calibrated_predictions(
        dists,
        fitted_temperature=temp_result.fitted_temperature,
        bin_count=5,
    )
    assert cal_report.accuracy == cal_eval.accuracy  # Strict invariant

    # 6. Reference Set & Feature Space Centroids
    ref_set = build_ood_reference_set(
        sample_ids=sample_ids,
        vectors=test_reps,
        labels=true_labels,
        source_experiment="smoke_exp",
        representation_layer="backbone.encoder",
    )
    assert len(ref_set.class_centroids) == 3

    # 7. OOD Scoring across Methods & Discrimination Metrics
    ood_logits = [[0.5, 0.5, 0.5] for _ in ood_samples]
    ood_reps = [[5.0, 5.0, 5.0] for _ in ood_samples]
    ood_dists = [
        compute_predictive_distribution(
            sample_id=ood_samples[i].sample_id,
            logits=ood_logits[i],
            true_class=None,
        )
        for i in range(len(ood_samples))
    ]

    ood_evals = {}
    for method in [
        OODScoreMethod.MAX_SOFTMAX_PROBABILITY,
        OODScoreMethod.PREDICTIVE_ENTROPY,
        OODScoreMethod.NEAREST_CLASS_CENTROID_DISTANCE,
        OODScoreMethod.KNN_REPRESENTATION_DISTANCE,
        OODScoreMethod.ENERGY_SCORE,
    ]:
        id_scores = [
            score_ood_sample(
                sample_id=sample_ids[i],
                category=OODCategory.IN_DISTRIBUTION,
                distribution=dists[i],
                score_method=method,
                representation=test_reps[i],
                reference_set=ref_set,
                reference_vectors=test_reps,
            ).normalized_ood_score
            for i in range(len(test_logits))
        ]
        ood_score_vals = [
            score_ood_sample(
                sample_id=ood_samples[i].sample_id,
                category=ood_samples[i].category,
                distribution=ood_dists[i],
                score_method=method,
                representation=ood_reps[i],
                reference_set=ref_set,
                reference_vectors=test_reps,
            ).normalized_ood_score
            for i in range(len(ood_samples))
        ]

        summary = evaluate_ood_binary_classification(
            id_scores=id_scores,
            ood_scores=ood_score_vals,
            score_method=method,
            threshold_policy=ThresholdPolicy.TARGET_ID_TPR,
            target_id_tpr=0.95,
        )
        assert 0.0 <= summary.auroc <= 1.0
        ood_evals[method.value] = summary

    # 8. Representation Novelty vs Confidence Relationships
    relationships = compute_representation_confidence_relationships(
        distributions=dists,
        representations=test_reps,
        reference_set=ref_set,
    )

    # 9. Corruption Trajectories
    corr_dists = {
        1: dists,
        2: dists,
    }
    corr_reps = {1: test_reps, 2: test_reps}
    corr_curve, flips = evaluate_corruption_uncertainty(
        clean_distributions=dists,
        clean_representations=test_reps,
        corrupted_distributions_by_severity=corr_dists,
        corrupted_representations_by_severity=corr_reps,
        corruption_name="gaussian_noise",
    )

    # 10. Failure Detection
    failures = detect_uncertainty_failures(
        distributions=dists,
        representations=test_reps,
        reference_set=ref_set,
        prediction_flips=flips,
    )

    # 11. Full Report Compilation and Serialization Roundtrip
    report = compile_uncertainty_analysis_report(
        model_name="smoke_test_model",
        architecture="ResNet",
        source_objective="supervised",
        dataset_name=ood_meta["dataset_name"],
        split="test",
        representation_layer="backbone.encoder",
        seed=seed,
        uncalibrated_report=cal_report,
        ood_evaluations=ood_evals,
        representation_relationships=relationships,
        temperature_scaling=temp_result,
        calibrated_report=cal_eval,
        corruption_curve=corr_curve,
        prediction_flips=flips,
        failure_records=failures,
    )

    assert isinstance(report, UncertaintyAnalysisReport)
    serialized = report.to_dict()
    assert isinstance(serialized, dict)
    json_str = json.dumps(serialized)
    assert len(json_str) > 500
