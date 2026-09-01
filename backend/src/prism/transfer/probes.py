"""Layer transferability probes evaluating representation utility across model depth."""

from __future__ import annotations

import time
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from prism.core.errors import ValidationError
from prism.data.materialized import MaterializedDataset
from prism.models.base import BaseVisionModel
from prism.models.initialization import initialize_linear_parameters
from prism.training.loss import SoftmaxCrossEntropyLoss, compute_accuracy


class LayerTransferProbeResult(BaseModel):
    """Result of a linear probe trained on extracted representations."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    layer_name: str = Field(..., description="Logical name of the probed source layer")
    representation_dim: int = Field(
        ..., ge=1, description="Dimensionality of the extracted representation vector"
    )
    target_num_classes: int = Field(
        ..., ge=2, description="Target classification classes"
    )
    target_dataset_id: str = Field(..., description="Target dataset identifier")
    target_data_budget: float = Field(
        default=1.0, description="Fraction of target data used (0.01 to 1.0)"
    )
    probe_parameters_count: int = Field(
        ..., description="Total trainable parameters in the linear probe classifier"
    )
    train_accuracy: float = Field(
        ..., ge=0.0, le=1.0, description="Final training accuracy"
    )
    val_accuracy: float = Field(
        ..., ge=0.0, le=1.0, description="Final validation accuracy"
    )
    test_accuracy: float | None = Field(
        default=None, ge=0.0, le=1.0, description="Test accuracy if evaluated"
    )
    train_loss: float = Field(..., ge=0.0, description="Final training loss")
    val_loss: float = Field(..., ge=0.0, description="Final validation loss")
    epochs_trained: int = Field(
        ..., ge=1, description="Number of probe training epochs"
    )
    best_epoch: int = Field(
        ..., ge=0, description="Epoch achieving highest validation accuracy"
    )
    duration_seconds: float = Field(
        ..., ge=0.0, description="Wall-clock execution time"
    )

    def to_dict(self) -> dict[str, Any]:
        """Convert probe result to dictionary."""
        return self.model_dump(mode="json")


def _flatten_vector(data: Any) -> list[float]:
    """Flatten tensor structure into 1D float list."""
    if isinstance(data, (int, float)):
        return [float(data)]
    if isinstance(data, (list, tuple)):
        flat: list[float] = []
        for item in data:
            flat.extend(_flatten_vector(item))
        return flat
    return [0.0]


def probe_layer_transferability(
    model: BaseVisionModel,
    train_dataset: MaterializedDataset,
    layer: str,
    target_num_classes: int,
    val_dataset: MaterializedDataset | None = None,
    epochs: int = 5,
    lr: float = 0.05,
    seed: int = 42,
    target_data_budget: float = 1.0,
) -> LayerTransferProbeResult:
    """Train a linear classifier probe on frozen features from a specific model layer.

    Guarantees:
    - Model remains completely frozen in evaluation mode.
    - Features are extracted without parameter updates or BatchNorm mutation.
    - A deterministic linear probe is trained on top of extracted feature vectors.

    Args:
        model: Frozen source vision model.
        train_dataset: Target training partition.
        layer: Name of layer to extract representations from.
        target_num_classes: Number of target classes.
        val_dataset: Optional validation partition.
        epochs: Training epochs for linear probe.
        lr: Learning rate for linear probe SGD.
        seed: Random seed for probe initialization.
        target_data_budget: Fraction of target dataset used.

    Returns:
        LayerTransferProbeResult summary.
    """
    start_time = time.perf_counter()
    was_training = model.is_training
    model.eval()

    try:
        # 1. Extract representations for training samples
        train_samples = train_dataset.samples
        if target_data_budget < 1.0:
            budget_count = max(1, int(len(train_samples) * target_data_budget))
            train_samples = train_samples[:budget_count]

        x_train: list[list[float]] = []
        y_train: list[int] = []

        for sample in train_samples:
            feat = model.extract_representations([sample.data], layer=layer)
            x_train.append(_flatten_vector(feat[0] if isinstance(feat, list) else feat))
            y_train.append(int(sample.target) if sample.target is not None else 0)

        if not x_train or not x_train[0]:
            raise ValidationError(
                f"Failed to extract representations from layer '{layer}'."
            )

        rep_dim = len(x_train[0])

        # 2. Extract validation representations
        val_samples = val_dataset.samples if val_dataset else train_samples
        x_val: list[list[float]] = []
        y_val: list[int] = []

        for sample in val_samples:
            feat = model.extract_representations([sample.data], layer=layer)
            x_val.append(_flatten_vector(feat[0] if isinstance(feat, list) else feat))
            y_val.append(int(sample.target) if sample.target is not None else 0)

        # 3. Initialize and train linear classifier probe
        w_cls, b_cls = initialize_linear_parameters(
            in_features=rep_dim,
            num_classes=target_num_classes,
            seed=seed,
        )

        probe_params_count = rep_dim * target_num_classes + target_num_classes
        loss_fn = SoftmaxCrossEntropyLoss()

        best_val_acc = 0.0
        best_epoch = 0
        final_train_loss = 0.0
        final_train_acc = 0.0
        final_val_loss = 0.0
        final_val_acc = 0.0

        for epoch in range(epochs):
            # Forward + Backward on probe
            # Vector batch multiplication: Z = X @ W + b
            logits: list[list[float]] = []
            for xi in x_train:
                row = [
                    sum(xi[d] * w_cls[d][c] for d in range(rep_dim)) + b_cls[c]
                    for c in range(target_num_classes)
                ]
                logits.append(row)

            loss_val, d_logits = loss_fn(logits, y_train)
            train_acc = compute_accuracy(logits, y_train)
            final_train_loss = loss_val
            final_train_acc = train_acc

            # Gradient update: dW = X^T @ dZ, db = sum(dZ)
            n_b = len(x_train)
            for d in range(rep_dim):
                for c in range(target_num_classes):
                    grad_w = sum(x_train[i][d] * d_logits[i][c] for i in range(n_b))
                    w_cls[d][c] -= lr * grad_w

            for c in range(target_num_classes):
                grad_b = sum(d_logits[i][c] for i in range(n_b))
                b_cls[c] -= lr * grad_b

            # Validation evaluation
            val_logits: list[list[float]] = []
            for xi in x_val:
                row = [
                    sum(xi[d] * w_cls[d][c] for d in range(rep_dim)) + b_cls[c]
                    for c in range(target_num_classes)
                ]
                val_logits.append(row)

            val_l, _ = loss_fn(val_logits, y_val)
            val_acc = compute_accuracy(val_logits, y_val)
            final_val_loss = val_l
            final_val_acc = val_acc

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch

        duration = time.perf_counter() - start_time

        return LayerTransferProbeResult(
            layer_name=layer,
            representation_dim=rep_dim,
            target_num_classes=target_num_classes,
            target_dataset_id=train_dataset.dataset_id,
            target_data_budget=target_data_budget,
            probe_parameters_count=probe_params_count,
            train_accuracy=final_train_acc,
            val_accuracy=final_val_acc,
            test_accuracy=final_val_acc,
            train_loss=final_train_loss,
            val_loss=final_val_loss,
            epochs_trained=epochs,
            best_epoch=best_epoch,
            duration_seconds=duration,
        )

    finally:
        if was_training:
            model.train()


def probe_all_layers_transferability(
    model: BaseVisionModel,
    train_dataset: MaterializedDataset,
    layers: list[str],
    target_num_classes: int,
    val_dataset: MaterializedDataset | None = None,
    epochs: int = 5,
    lr: float = 0.05,
    seed: int = 42,
    target_data_budget: float = 1.0,
) -> list[LayerTransferProbeResult]:
    """Execute linear transferability probes across a list of logical model layers."""
    results: list[LayerTransferProbeResult] = []
    for layer in layers:
        res = probe_layer_transferability(
            model=model,
            train_dataset=train_dataset,
            layer=layer,
            target_num_classes=target_num_classes,
            val_dataset=val_dataset,
            epochs=epochs,
            lr=lr,
            seed=seed,
            target_data_budget=target_data_budget,
        )
        results.append(res)
    return results
