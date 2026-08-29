"""CPU-only synthetic smoke test for the Phase 13 reporting workflow."""

import pytest

from prism.core.enums import ModelFamily, RunStatus, TaskType
from prism.data.manifests import DatasetManifest, SplitSpecification
from prism.evaluation.configuration import EvaluationConfiguration, MetricSpecification
from prism.experiments.architecture import create_architecture_comparison_suite
from prism.experiments.definitions import ExperimentDefinition
from prism.experiments.reporting import ArchitectureRunResult, ExperimentSuiteRunner
from prism.models.specifications import ModelSpecification
from prism.training.configuration import OptimizerSpecification, TrainingConfiguration
from prism.training.results import TrainingResult


@pytest.mark.smoke
def test_smoke_synthetic_architecture_comparison_report() -> None:
    dataset = DatasetManifest(
        dataset_id="ds-smoke-architecture",
        name="Synthetic architecture data",
        splits=[
            SplitSpecification(split_name="train"),
            SplitSpecification(split_name="test"),
        ],
        num_classes=2,
    )
    definitions = []
    for family in (ModelFamily.CNN, ModelFamily.RESNET, ModelFamily.VISION_TRANSFORMER):
        definitions.append(
            ExperimentDefinition(
                experiment_id=f"exp-{family.value}",
                name=family.value,
                task_type=TaskType.CLASSIFICATION,
                dataset=dataset,
                model=ModelSpecification(
                    model_id=f"model-{family.value}",
                    name=family.value,
                    family=family,
                    architecture=family.value,
                    input_shape=(1, 4, 4),
                    num_classes=2,
                    hyperparameters={
                        "patch_size": 2,
                        "embed_dim": 4,
                        "num_heads": 2,
                        "depth": 1,
                    },
                ),
                training=TrainingConfiguration(
                    epochs=1,
                    batch_size=2,
                    optimizer=OptimizerSpecification(type="sgd", lr=0.01),
                ),
                evaluation=EvaluationConfiguration(
                    target_splits=["test"],
                    metrics=[MetricSpecification(name="top1_accuracy")],
                ),
            )
        )
    suite = create_architecture_comparison_suite(
        "suite-smoke-architecture",
        "Synthetic CNN ResNet ViT",
        "Measure architecture behavior under shared conditions.",
        definitions,
        intentionally_varied_factors=[
            "model_family",
            "model_architecture",
            "model.hyperparameters",
        ],
    )

    def execute(experiment: ExperimentDefinition) -> ArchitectureRunResult:
        return ArchitectureRunResult(
            experiment_id=experiment.experiment_id,
            training_result=TrainingResult(
                run_id=f"run-{experiment.model.family.value}",
                experiment_id=experiment.experiment_id,
                status=RunStatus.COMPLETED,
                epochs_completed=1,
                total_batches=1,
                total_examples=2,
                final_train_loss=0.69,
                final_train_accuracy=0.5,
            ),
        )

    report = ExperimentSuiteRunner().run(suite, execute)
    assert len(report.completed_experiment_ids) == 3
    assert report.failed_experiment_ids == []
    assert report.factor_audit.is_strictly_controlled
    assert (
        report.from_json(report.to_json()).suite_fingerprint == report.suite_fingerprint
    )
