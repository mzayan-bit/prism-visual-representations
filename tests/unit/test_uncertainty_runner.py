"""Unit tests for the end-to-end UncertaintyAnalysisRunner pipeline."""

from __future__ import annotations

from prism.uncertainty.contracts import (
    OODReferenceSet,
    OODSample,
    UncertaintyAnalysisReport,
)
from prism.uncertainty.enums import OODCategory
from prism.uncertainty.runner import (
    UncertaintyAnalysisConfig,
    UncertaintyAnalysisRunner,
)


def test_uncertainty_runner_execution() -> None:
    """Verify UncertaintyAnalysisRunner pipeline execution on test data."""
    config = UncertaintyAnalysisConfig(
        model_name="test_model",
        architecture="ResNet",
        bin_count=5,
    )
    runner = UncertaintyAnalysisRunner(config=config)

    test_ids = [f"test_{i}" for i in range(10)]
    test_logits = [[5.0, 1.0, 0.0] if i < 5 else [0.5, 4.0, 1.0] for i in range(10)]
    test_targets = [0 if i < 5 else 1 for i in range(10)]
    test_reps = [[1.0, 0.0] if i < 5 else [0.0, 1.0] for i in range(10)]

    ref_set = OODReferenceSet(
        source_experiment="test_exp",
        representation_layer="backbone.encoder",
        sample_ids=[f"ref_{i}" for i in range(6)],
        labels=[0, 0, 0, 1, 1, 1],
        class_centroids={
            "0": [1.0, 0.0],
            "1": [0.0, 1.0],
            "2": [-1.0, -1.0],
        },
        intra_class_radii={"0": 0.2, "1": 0.2, "2": 0.2},
        normalization_policy="none",
        distance_metric="euclidean",
        fingerprint="fp123456",
    )

    ood_samples = [
        OODSample(
            sample_id="ood_1",
            source_dataset_identity="synth_ood",
            category=OODCategory.OUT_OF_DISTRIBUTION,
            image=[[[0.0]]],
        ),
        OODSample(
            sample_id="ood_2",
            source_dataset_identity="synth_ood",
            category=OODCategory.NEAR_OOD,
            image=[[[0.5]]],
        ),
    ]
    ood_logits = [[1.0, 1.0, 1.0], [0.0, 0.5, 0.0]]
    ood_reps = [[10.0, 10.0], [5.0, 5.0]]

    report = runner.run_analysis(
        test_sample_ids=test_ids,
        test_logits=test_logits,
        test_targets=test_targets,
        test_representations=test_reps,
        reference_set=ref_set,
        ood_samples=ood_samples,
        ood_logits=ood_logits,
        ood_representations=ood_reps,
    )

    assert isinstance(report, UncertaintyAnalysisReport)
    assert report.model_id == "test_model"
    assert report.calibration_report.sample_count == 10
    assert len(report.ood_evaluations) > 0
    assert report.to_dict() is not None
