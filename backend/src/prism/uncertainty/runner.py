"""End-to-end uncertainty and OOD representation analysis runner."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from prism.uncertainty.calibration import compute_calibration_report
from prism.uncertainty.contracts import (
    CalibrationReport,
    CorruptionUncertaintyCurve,
    OODBinaryEvaluationSummary,
    OODReferenceSet,
    OODSample,
    PredictionFlipUncertainty,
    PredictiveDistribution,
    TemperatureScalingResult,
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
from prism.uncertainty.metrics import evaluate_ood_binary_classification
from prism.uncertainty.ood_scores import score_ood_sample
from prism.uncertainty.probabilities import batch_predictive_distributions
from prism.uncertainty.relationships import (
    compute_representation_confidence_relationships,
)
from prism.uncertainty.reports import compile_uncertainty_analysis_report
from prism.uncertainty.temperature import (
    evaluate_calibrated_predictions,
    fit_temperature_scaling,
)


@dataclass
class UncertaintyAnalysisConfig:
    """Configuration options for Uncertainty Analysis Pipeline."""

    model_name: str = "classifier_model"
    architecture: str = "ResNet"
    source_objective: str = "supervised"
    dataset_name: str = "synthetic_shapes"
    split: str = "test"
    representation_layer: str = "backbone.encoder"
    seed: int = 42
    bin_count: int = 10
    binning_strategy: BinningStrategy = BinningStrategy.EQUAL_WIDTH
    temperature_search_range: tuple[float, float] = (0.05, 10.0)
    target_id_tpr: float = 0.95
    k_neighbors: int = 5
    high_conf_threshold: float = 0.80
    low_conf_threshold: float = 0.40


class UncertaintyAnalysisRunner:
    """Pure-Python runner for uncertainty, calibration, and OOD analysis."""

    def __init__(self, config: UncertaintyAnalysisConfig | None = None) -> None:
        self.config = config or UncertaintyAnalysisConfig()

    def run_analysis(
        self,
        test_sample_ids: Sequence[str],
        test_logits: Sequence[Sequence[float]],
        test_targets: Sequence[int],
        test_representations: Sequence[Sequence[float]],
        reference_set: OODReferenceSet,
        val_logits: Sequence[Sequence[float]] | None = None,
        val_targets: Sequence[int] | None = None,
        ood_samples: Sequence[OODSample] | None = None,
        ood_logits: Sequence[Sequence[float]] | None = None,
        ood_representations: Sequence[Sequence[float]] | None = None,
        corrupted_distributions_by_severity: (
            Mapping[int, Sequence[PredictiveDistribution]] | None
        ) = None,
        corrupted_representations_by_severity: (
            Mapping[int, Sequence[Sequence[float]]] | None
        ) = None,
        corruption_name: str = "gaussian_noise",
    ) -> UncertaintyAnalysisReport:
        """Execute the uncertainty and OOD evaluation workflow.

        Parameters
        ----------
        test_sample_ids : Sequence[str]
            IDs for in-distribution test samples.
        test_logits : Sequence[Sequence[float]]
            Logit matrix (N, K) for ID test samples.
        test_targets : Sequence[int]
            Ground truth class labels (N,) for ID test samples.
        test_representations : Sequence[Sequence[float]]
            Backbone feature vectors (N, D) for ID test samples.
        reference_set : OODReferenceSet
            In-distribution training/reference set with class centroids.
        val_logits : Sequence[Sequence[float]] | None
            Optional validation logits for fitting temperature scaling.
        val_targets : Sequence[int] | None
            Optional validation targets for fitting temperature scaling.
        ood_samples : Sequence[OODSample] | None
            Out-of-distribution sample definitions.
        ood_logits : Sequence[Sequence[float]] | None
            Model logit outputs on OOD samples.
        ood_representations : Sequence[Sequence[float]] | None
            Backbone feature vectors on OOD samples.
        corrupted_distributions_by_severity : (
            dict[int, Sequence[PredictiveDistribution]] | None
        )
            Predictions on corrupted inputs across severity levels 1..5.
        corrupted_representations_by_severity : (
            dict[int, Sequence[Sequence[float]]] | None
        )
            Features on corrupted inputs across severity levels 1..5.
        corruption_name : str
            Name of corruption studied.

        Returns
        -------
        UncertaintyAnalysisReport
            Full empirical report covering calibration, OOD, and failures.
        """
        # 1. Compute in-distribution test distributions (uncalibrated)
        id_test_dists = batch_predictive_distributions(
            sample_ids=list(test_sample_ids),
            logits_matrix=[list(z) for z in test_logits],
            true_classes=list(test_targets),
            temperature=1.0,
        )

        uncal_report = compute_calibration_report(
            distributions=id_test_dists,
            bin_count=self.config.bin_count,
            strategy=self.config.binning_strategy,
        )

        # 2. Temperature scaling on validation set (if available)
        temp_result: TemperatureScalingResult | None = None
        cal_report: CalibrationReport | None = None

        if val_logits is not None and val_targets is not None and len(val_logits) > 0:
            temp_result = fit_temperature_scaling(
                val_logits=[list(z) for z in val_logits],
                val_targets=list(val_targets),
                search_range=self.config.temperature_search_range,
            )
            # Evaluate test set with fitted temperature
            cal_report = evaluate_calibrated_predictions(
                test_distributions=id_test_dists,
                fitted_temperature=temp_result.fitted_temperature,
                bin_count=self.config.bin_count,
            )

        # 3. OOD evaluations across scoring methods
        ood_evals: dict[str, OODBinaryEvaluationSummary] = {}
        ood_dists: list[PredictiveDistribution] = []

        if ood_samples and ood_logits and ood_representations:
            ood_sample_ids = [s.sample_id for s in ood_samples]
            ood_dists = batch_predictive_distributions(
                sample_ids=ood_sample_ids,
                logits_matrix=[list(z) for z in ood_logits],
                true_classes=None,
                temperature=1.0,
            )

            score_methods = [
                OODScoreMethod.MAX_SOFTMAX_PROBABILITY,
                OODScoreMethod.PREDICTIVE_ENTROPY,
                OODScoreMethod.NEAREST_CLASS_CENTROID_DISTANCE,
                OODScoreMethod.KNN_REPRESENTATION_DISTANCE,
                OODScoreMethod.ENERGY_SCORE,
            ]

            for method in score_methods:
                id_scores = [
                    score_ood_sample(
                        sample_id=d.sample_id,
                        category=OODCategory.IN_DISTRIBUTION,
                        distribution=d,
                        score_method=method,
                        representation=list(rep),
                        reference_set=reference_set,
                        k=self.config.k_neighbors,
                    )
                    for d, rep in zip(id_test_dists, test_representations, strict=False)
                ]

                ood_score_results = [
                    score_ood_sample(
                        sample_id=s.sample_id,
                        category=s.category,
                        distribution=d,
                        score_method=method,
                        representation=list(rep),
                        reference_set=reference_set,
                        k=self.config.k_neighbors,
                    )
                    for s, d, rep in zip(
                        ood_samples,
                        ood_dists,
                        ood_representations,
                        strict=False,
                    )
                ]

                eval_summary = evaluate_ood_binary_classification(
                    id_scores=[s.normalized_ood_score for s in id_scores],
                    ood_scores=[s.normalized_ood_score for s in ood_score_results],
                    score_method=method,
                    threshold_policy=ThresholdPolicy.TARGET_ID_TPR,
                    target_id_tpr=self.config.target_id_tpr,
                )
                ood_evals[method.value] = eval_summary

        # 4. Representation vs confidence relationships
        rep_rel = compute_representation_confidence_relationships(
            distributions=id_test_dists,
            representations=test_representations,
            reference_set=reference_set,
            k_neighbors=self.config.k_neighbors,
        )

        # 5. Corruption uncertainty curves & prediction flips
        corr_curve: CorruptionUncertaintyCurve | None = None
        prediction_flips: list[PredictionFlipUncertainty] = []
        if corrupted_distributions_by_severity:
            corr_curve, prediction_flips = evaluate_corruption_uncertainty(
                clean_distributions=id_test_dists,
                clean_representations=test_representations,
                corrupted_distributions_by_severity=corrupted_distributions_by_severity,
                corrupted_representations_by_severity=corrupted_representations_by_severity,
                reference_set=reference_set,
                corruption_name=corruption_name,
                bin_count=self.config.bin_count,
            )

        # 6. Failure taxonomy detection
        failures = detect_uncertainty_failures(
            distributions=id_test_dists,
            representations=test_representations,
            reference_set=reference_set,
            ood_samples=ood_samples,
            ood_distributions=ood_dists if ood_dists else None,
            ood_representations=ood_representations,
            prediction_flips=prediction_flips,
            high_conf_threshold=self.config.high_conf_threshold,
            low_conf_threshold=self.config.low_conf_threshold,
        )

        # 7. Compile report
        return compile_uncertainty_analysis_report(
            model_name=self.config.model_name,
            architecture=self.config.architecture,
            source_objective=self.config.source_objective,
            dataset_name=self.config.dataset_name,
            split=self.config.split,
            representation_layer=self.config.representation_layer,
            seed=self.config.seed,
            uncalibrated_report=uncal_report,
            ood_evaluations=ood_evals,
            representation_relationships=rep_rel,
            temperature_scaling=temp_result,
            calibrated_report=cal_report,
            corruption_curve=corr_curve,
            prediction_flips=prediction_flips,
            failure_records=failures,
        )
