"""Training engine orchestrating deterministic model training lifecycles."""

from __future__ import annotations

import time
import traceback
from typing import TYPE_CHECKING, Any

from prism.core.enums import ModelFamily, RunStatus
from prism.core.errors import (
    TrainingError,
    ValidationError,
)
from prism.core.identifiers import generate_run_id
from prism.data.batching import DeterministicBatchLoader
from prism.data.materialized import MaterializedDataset
from prism.evaluation.reports import EvaluationReport
from prism.experiments.metrics import MetricRecord
from prism.experiments.runs import ExperimentRun
from prism.models.base import BaseVisionModel
from prism.models.cnn import ConvolutionalNeuralNetwork
from prism.models.linear import LinearSoftmaxClassifier
from prism.models.mlp import MultiLayerPerceptron
from prism.models.resnet import ResidualNeuralNetwork
from prism.models.transformer import VisionTransformer
from prism.training.loss import SoftmaxCrossEntropyLoss, compute_accuracy
from prism.training.optimizers import BaseOptimizer, create_optimizer
from prism.training.results import TrainingResult
from prism.training.schedulers import create_scheduler

if TYPE_CHECKING:
    from prism.evaluation.engine import EvaluationEngine
    from prism.experiments.context import PreparedExecution
    from prism.experiments.definitions import ExperimentDefinition


class TrainingEngine:
    """Orchestrates end-to-end training and evaluation loops for PRISM experiments."""

    def __init__(self) -> None:
        self.loss_fn = SoftmaxCrossEntropyLoss()
        self._eval_engine: EvaluationEngine | None = None

    @property
    def eval_engine(self) -> EvaluationEngine:
        """Lazy access to EvaluationEngine to avoid circular module dependencies."""
        if self._eval_engine is None:
            from prism.evaluation.engine import EvaluationEngine

            self._eval_engine = EvaluationEngine()
        return self._eval_engine

    def train(
        self,
        experiment: ExperimentDefinition,
        prepared_execution: PreparedExecution,
        train_dataset: MaterializedDataset,
        train_loader: DeterministicBatchLoader,
        val_dataset: MaterializedDataset | None = None,
        val_loader: DeterministicBatchLoader | None = None,
        test_dataset: MaterializedDataset | None = None,
        test_loader: DeterministicBatchLoader | None = None,
        run: ExperimentRun | None = None,
        model: BaseVisionModel | None = None,
    ) -> TrainingResult:
        """Execute full training and evaluation lifecycle.

        Guarantees:
        - Validates compatibility across experiment definition and prepared context.
        - Respects ExperimentRun lifecycle state machine.
        - Employs deterministic epoch-aware batch ordering.
        - Supports Linear and MLP model architectures with activations and dropout.
        - Evaluates learning rate schedules and logs actual LR telemetry.
        - Strictly separates training mode (dropout active) from evaluation mode.
        - Transitions to COMPLETED on success, or FAILED on failure.
        """
        # 1. Compatibility Validation
        if experiment.experiment_id != prepared_execution.experiment_id:
            raise ValidationError(
                f"Experiment ID mismatch: '{experiment.experiment_id}' vs "
                f"prepared '{prepared_execution.experiment_id}'."
            )

        run_inst = run or ExperimentRun(
            run_id=prepared_execution.run_id or generate_run_id(),
            experiment_id=experiment.experiment_id,
            status=RunStatus.PLANNED,
            configuration_fingerprint=prepared_execution.configuration_fingerprint,
            reproducibility=experiment.reproducibility,
            environment=prepared_execution.environment,
            code_revision=prepared_execution.code_revision,
        )

        if run_inst.status not in (RunStatus.PLANNED, RunStatus.QUEUED):
            raise ValidationError(
                f"Cannot start training run in status '{run_inst.status}'."
            )

        # 2. Model, Optimizer, and Scheduler Initialization
        seed = experiment.reproducibility.seed or 42
        if model is not None:
            model_inst = model
        elif experiment.model.family == ModelFamily.VISION_TRANSFORMER:
            model_inst = VisionTransformer(spec=experiment.model, seed=seed)
        elif experiment.model.family == ModelFamily.RESNET:
            model_inst = ResidualNeuralNetwork(spec=experiment.model, seed=seed)
        elif experiment.model.family == ModelFamily.CNN:
            model_inst = ConvolutionalNeuralNetwork(spec=experiment.model, seed=seed)
        elif experiment.model.family == ModelFamily.MLP:
            model_inst = MultiLayerPerceptron(spec=experiment.model, seed=seed)
        else:
            model_inst = LinearSoftmaxClassifier(spec=experiment.model, seed=seed)

        optimizer: BaseOptimizer = create_optimizer(
            config=experiment.training.optimizer,
            model=model_inst,
        )

        total_steps_est = experiment.training.epochs * max(1, len(train_loader))
        scheduler = create_scheduler(
            spec=experiment.training.scheduler,
            base_lr=experiment.training.optimizer.lr,
            total_epochs=experiment.training.epochs,
            total_steps=total_steps_est,
        )

        # 3. Transition to RUNNING
        run_inst.start()
        start_time = time.perf_counter()

        total_batches = 0
        total_examples = 0
        final_train_loss = 0.0
        final_train_accuracy = 0.0
        evaluation_reports: list[EvaluationReport] = []
        summary_metrics: dict[str, float] = {}

        try:
            epochs = experiment.training.epochs
            for epoch in range(epochs):
                train_loader.set_epoch(epoch)

                # Update learning rate via scheduler if stepping per epoch
                if scheduler.step_unit == "epoch":
                    current_lr = scheduler.step(epoch)
                    optimizer.lr = current_lr

                    run_inst.add_metric(
                        MetricRecord(
                            metric_name="learning_rate",
                            value=current_lr,
                            split="train",
                            epoch=epoch,
                            step=total_batches,
                        )
                    )

                # Ensure model is in training mode for dropout / stochastic behavior
                model_inst.train()

                epoch_loss = 0.0
                epoch_samples = 0
                epoch_logits: list[list[float]] = []
                epoch_targets: list[Any] = []

                for batch in train_loader:
                    # Update learning rate via scheduler if stepping per batch step
                    if scheduler.step_unit == "step":
                        current_lr = scheduler.step(epoch)
                        optimizer.lr = current_lr

                        run_inst.add_metric(
                            MetricRecord(
                                metric_name="learning_rate",
                                value=current_lr,
                                split="train",
                                epoch=epoch,
                                step=total_batches,
                            )
                        )

                    logits = model_inst.forward(batch.data)

                    # Optimizer directly applies weight decay during step()
                    batch_loss, d_logits = self.loss_fn(
                        logits=logits,
                        targets=batch.targets,
                        weight_decay=0.0,
                    )

                    model_inst.zero_grad()
                    model_inst.backward(d_logits)
                    optimizer.step()

                    batch_size = len(batch.data)
                    epoch_loss += batch_loss * float(batch_size)
                    epoch_samples += batch_size
                    total_batches += 1
                    total_examples += batch_size

                    epoch_logits.extend(logits)
                    epoch_targets.extend(batch.targets)

                if epoch_samples == 0:
                    raise TrainingError(
                        "No samples were processed during training epoch."
                    )

                mean_epoch_loss = epoch_loss / float(epoch_samples)
                epoch_acc = compute_accuracy(epoch_logits, epoch_targets)

                final_train_loss = mean_epoch_loss
                final_train_accuracy = epoch_acc

                # Log training epoch metrics
                run_inst.add_metric(
                    MetricRecord(
                        metric_name="train_loss",
                        value=mean_epoch_loss,
                        split="train",
                        epoch=epoch,
                        step=total_batches,
                    )
                )
                run_inst.add_metric(
                    MetricRecord(
                        metric_name="train_top1_accuracy",
                        value=epoch_acc,
                        split="train",
                        epoch=epoch,
                        step=total_batches,
                    )
                )

                # Optional validation evaluation during training
                if val_loader is not None:
                    model_inst.eval()
                    val_report = self.eval_engine.evaluate(
                        model=model_inst,
                        loader=val_loader,
                        split_name="val",
                        experiment_id=experiment.experiment_id,
                        run_id=run_inst.run_id,
                        epoch=epoch,
                        step=total_batches,
                    )
                    for rec in val_report.metric_records:
                        run_inst.add_metric(rec)
                    evaluation_reports.append(val_report)
                    model_inst.train()

            # Post-training test evaluation
            if test_loader is not None:
                model_inst.eval()
                test_report = self.eval_engine.evaluate(
                    model=model_inst,
                    loader=test_loader,
                    split_name="test",
                    evaluation_config=experiment.evaluation,
                    experiment_id=experiment.experiment_id,
                    run_id=run_inst.run_id,
                    epoch=epochs - 1,
                    step=total_batches,
                )
                for rec in test_report.metric_records:
                    run_inst.add_metric(rec)
                evaluation_reports.append(test_report)
                summary_metrics.update(test_report.summary_metrics)

            model_inst.eval()
            duration = time.perf_counter() - start_time
            summary_metrics["final_train_loss"] = final_train_loss
            summary_metrics["final_train_accuracy"] = final_train_accuracy
            summary_metrics["final_learning_rate"] = optimizer.lr

            # Complete Run Lifecycle
            run_inst.complete(summary_metrics=summary_metrics)

            return TrainingResult(
                run_id=run_inst.run_id,
                experiment_id=experiment.experiment_id,
                status=run_inst.status,
                epochs_completed=epochs,
                total_batches=total_batches,
                total_examples=total_examples,
                final_train_loss=final_train_loss,
                final_train_accuracy=final_train_accuracy,
                evaluation_reports=evaluation_reports,
                summary_metrics=summary_metrics,
                duration_seconds=duration,
                metadata={
                    "backend": prepared_execution.hardware.compute_backend,
                },
            )

        except Exception as exc:
            tb = traceback.format_exc()
            run_inst.fail(
                error_type=exc.__class__.__name__,
                error_message=str(exc),
                traceback=tb,
            )
            if isinstance(exc, TrainingError):
                raise
            raise TrainingError(f"Training failed: {exc}") from exc
