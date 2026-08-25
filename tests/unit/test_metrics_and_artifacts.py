"""Unit tests for MetricRecord, ArtifactReference, and EvaluationReport."""

import pytest
from pydantic import ValidationError as PydanticValidationError

from prism.artifacts.contracts import ArtifactReference
from prism.core.enums import ArtifactType, MetricDirection
from prism.evaluation.configuration import (
    EvaluationConfiguration,
    MetricSpecification,
)
from prism.evaluation.reports import EvaluationReport
from prism.experiments.metrics import MetricRecord


@pytest.mark.unit
def test_valid_metric_record() -> None:
    """Verify construction of valid metric records."""
    record = MetricRecord(
        metric_name="top1_accuracy",
        value=0.9125,
        split="test",
        step=5000,
        epoch=20,
        direction=MetricDirection.MAXIMIZE,
        metadata={"num_eval_samples": 10000},
    )
    assert record.metric_name == "top1_accuracy"
    assert record.value == 0.9125
    assert record.split == "test"
    assert record.epoch == 20
    assert record.direction == MetricDirection.MAXIMIZE


@pytest.mark.unit
@pytest.mark.parametrize("invalid_val", [float("nan"), float("inf"), float("-inf")])
def test_reject_non_finite_metric_values(invalid_val: float) -> None:
    """Verify NaN and Inf metric values are rejected."""
    with pytest.raises((PydanticValidationError, ValueError)):
        MetricRecord(metric_name="loss", value=invalid_val)


@pytest.mark.unit
def test_valid_artifact_reference() -> None:
    """Verify construction of valid artifact reference."""
    artifact = ArtifactReference(
        artifact_id="art-umap-embedding",
        artifact_type=ArtifactType.EMBEDDING_PROJECTION,
        logical_name="test_embedding_umap_2d",
        uri="artifacts/figures/umap_test.json",
        checksum="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        producing_run_id="run-exp01-trial01",
        size_bytes=1048576,
        metadata={"projection_method": "UMAP", "n_neighbors": 15},
    )
    assert artifact.artifact_id == "art-umap-embedding"
    assert artifact.artifact_type == ArtifactType.EMBEDDING_PROJECTION
    assert artifact.size_bytes == 1048576


@pytest.mark.unit
def test_valid_evaluation_report() -> None:
    """Verify construction and serialization of EvaluationReport."""
    eval_config = EvaluationConfiguration(
        target_splits=["test"],
        metrics=[MetricSpecification(name="top1_accuracy")],
    )
    metric_record = MetricRecord(
        metric_name="top1_accuracy",
        value=0.88,
        split="test",
    )
    artifact_ref = ArtifactReference(
        artifact_id="art-conf-matrix",
        artifact_type=ArtifactType.CONFUSION_MATRIX,
        logical_name="test_confusion_matrix",
        uri="artifacts/figures/conf_matrix.png",
        producing_run_id="run-exp01-trial01",
    )

    report = EvaluationReport(
        report_id="rep-cifar10-resnet18-test",
        experiment_id="exp-cifar10-resnet18",
        run_id="run-exp01-trial01",
        evaluation_config=eval_config,
        metric_records=[metric_record],
        artifact_references=[artifact_ref],
        summary_metrics={"top1_accuracy": 0.88},
    )

    assert report.report_id == "rep-cifar10-resnet18-test"
    assert report.summary_metrics["top1_accuracy"] == 0.88
    assert len(report.metric_records) == 1
    assert len(report.artifact_references) == 1

    # Round trip test
    dumped = report.to_dict()
    restored = EvaluationReport.from_dict(dumped)
    assert restored.report_id == report.report_id
    assert restored.summary_metrics == report.summary_metrics
