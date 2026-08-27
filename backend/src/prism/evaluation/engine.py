"""Deterministic evaluation engine for assessing trained vision models."""

from typing import Any

from prism.core.enums import MetricDirection
from prism.core.errors import EvaluationError, ValidationError
from prism.core.identifiers import generate_report_id
from prism.data.batching import DeterministicBatchLoader
from prism.evaluation.configuration import (
    EvaluationConfiguration,
    MetricSpecification,
)
from prism.evaluation.reports import EvaluationReport
from prism.experiments.metrics import MetricRecord
from prism.models.base import BaseVisionModel
from prism.training.loss import SoftmaxCrossEntropyLoss, compute_accuracy


class EvaluationEngine:
    """Orchestrates model evaluation without parameter updates or gradient tracking."""

    def __init__(self) -> None:
        self.loss_fn = SoftmaxCrossEntropyLoss()

    def evaluate(
        self,
        model: BaseVisionModel,
        loader: DeterministicBatchLoader,
        split_name: str = "test",
        evaluation_config: EvaluationConfiguration | None = None,
        experiment_id: str = "exp-default",
        run_id: str = "run-default",
        step: int | None = None,
        epoch: int | None = None,
    ) -> EvaluationReport:
        """Evaluate a model on a dataset split and return an EvaluationReport.

        Guarantees:
        - Does not mutate model weights or biases.
        - Does not clear or accumulate gradients.
        - Computes finite quantitative metrics from raw predictions.
        """
        if model is None:
            raise ValidationError("Model cannot be None for evaluation.")
        if loader is None:
            raise ValidationError("Batch loader cannot be None for evaluation.")

        # Save initial parameters to guarantee no parameter mutation occurred
        initial_params = model.get_parameters()

        config = evaluation_config or EvaluationConfiguration(
            target_splits=[split_name],
            metrics=[
                MetricSpecification(
                    name="top1_accuracy",
                    direction=MetricDirection.MAXIMIZE,
                    target_split=split_name,
                ),
                MetricSpecification(
                    name="loss",
                    direction=MetricDirection.MINIMIZE,
                    target_split=split_name,
                ),
            ],
        )

        all_logits: list[list[float]] = []
        all_targets: list[Any] = []
        total_loss = 0.0
        total_samples = 0

        try:
            for batch in loader:
                logits = model.forward(batch.data)
                batch_loss, _ = self.loss_fn(logits, batch.targets)

                batch_size = len(batch.data)
                total_loss += batch_loss * float(batch_size)
                total_samples += batch_size

                all_logits.extend(logits)
                all_targets.extend(batch.targets)

            if total_samples == 0:
                raise EvaluationError(
                    f"No samples were evaluated for split '{split_name}'."
                )

            mean_loss = total_loss / float(total_samples)
            top1_acc = compute_accuracy(all_logits, all_targets)

            # Build MetricRecords
            metric_records: list[MetricRecord] = [
                MetricRecord(
                    metric_name=f"{split_name}_top1_accuracy",
                    value=top1_acc,
                    split=split_name,
                    step=step,
                    epoch=epoch,
                    direction=MetricDirection.MAXIMIZE,
                ),
                MetricRecord(
                    metric_name=f"{split_name}_loss",
                    value=mean_loss,
                    split=split_name,
                    step=step,
                    epoch=epoch,
                    direction=MetricDirection.MINIMIZE,
                ),
            ]

            summary_metrics = {
                f"{split_name}_top1_accuracy": top1_acc,
                f"{split_name}_loss": mean_loss,
            }

            report = EvaluationReport(
                report_id=generate_report_id(),
                experiment_id=experiment_id,
                run_id=run_id,
                evaluation_config=config,
                metric_records=metric_records,
                summary_metrics=summary_metrics,
                metadata={
                    "total_samples": total_samples,
                    "split_name": split_name,
                },
            )

            # Ensure model parameters remained unchanged
            current_params = model.get_parameters()
            if current_params != initial_params:
                raise EvaluationError(
                    "Model parameters were unexpectedly mutated during evaluation."
                )

            return report

        except Exception as exc:
            if isinstance(exc, EvaluationError):
                raise
            raise EvaluationError(
                f"Evaluation failed on split '{split_name}': {exc}"
            ) from exc
