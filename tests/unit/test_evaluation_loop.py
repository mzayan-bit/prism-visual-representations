"""Unit tests for EvaluationEngine and evaluation loop guarantees."""

import pytest

from prism.core.enums import InitializationStrategy, ModelFamily, TaskType
from prism.data.batching import DeterministicBatchLoader
from prism.data.materialized import MaterializedDataset, MaterializedSample
from prism.evaluation.engine import EvaluationEngine
from prism.evaluation.reports import EvaluationReport
from prism.models.linear import LinearSoftmaxClassifier
from prism.models.specifications import ModelSpecification


@pytest.fixture
def eval_dataset() -> MaterializedDataset:
    samples = [
        MaterializedSample(
            sample_id=f"synth/test/{i:04d}",
            source_split="test",
            source_index=i,
            data=[1.0, 0.0] if i % 2 == 0 else [0.0, 1.0],
            target=i % 2,
        )
        for i in range(10)
    ]
    return MaterializedDataset(dataset_id="ds-eval-synth", samples=samples)


@pytest.fixture
def trained_linear_model() -> LinearSoftmaxClassifier:
    spec = ModelSpecification(
        model_id="model-eval-linear",
        name="Eval Linear Model",
        family=ModelFamily.LINEAR,
        architecture="linear_softmax",
        compatible_tasks=[TaskType.CLASSIFICATION],
        input_shape=(2,),
        num_classes=2,
        initialization=InitializationStrategy.RANDOM,
    )
    model = LinearSoftmaxClassifier(spec=spec, seed=42)
    # Set weights to perfectly separate [1, 0] -> 0 and [0, 1] -> 1
    model.weights = [[5.0, -5.0], [-5.0, 5.0]]
    model.bias = [0.0, 0.0]
    return model


@pytest.mark.unit
def test_evaluation_engine_produces_report(
    trained_linear_model: LinearSoftmaxClassifier,
    eval_dataset: MaterializedDataset,
) -> None:
    """Verify EvaluationEngine produces report without mutating model parameters."""
    initial_weights = [row[:] for row in trained_linear_model.weights]
    initial_bias = trained_linear_model.bias[:]

    loader = DeterministicBatchLoader(dataset=eval_dataset, batch_size=5)
    engine = EvaluationEngine()

    report = engine.evaluate(
        model=trained_linear_model,
        loader=loader,
        split_name="test",
        experiment_id="exp-eval-001",
        run_id="run-eval-001",
    )

    assert isinstance(report, EvaluationReport)
    assert report.experiment_id == "exp-eval-001"
    assert report.run_id == "run-eval-001"
    assert "test_top1_accuracy" in report.summary_metrics
    assert report.summary_metrics["test_top1_accuracy"] == 1.0
    assert report.summary_metrics["test_loss"] < 0.1

    # Parameters must NOT have mutated
    assert trained_linear_model.weights == initial_weights
    assert trained_linear_model.bias == initial_bias
