"""Numerically stable softmax cross-entropy loss and classification metrics."""

import math
from collections.abc import Sequence

from prism.core.errors import NumericalInstabilityError, ValidationError


class SoftmaxCrossEntropyLoss:
    """Numerically stable multiclass cross-entropy loss with analytic gradients."""

    def __call__(
        self,
        logits: list[list[float]],
        targets: Sequence[int | str | None],
        weight_decay: float = 0.0,
        weights: list[list[float]] | None = None,
    ) -> tuple[float, list[list[float]]]:
        """Compute scalar loss and upstream gradient dZ.

        Parameters
        ----------
        logits : list[list[float]]
            Raw model logits of shape [B, C].
        targets : Sequence[int | str | None]
            Ground-truth integer category labels of length B.
        weight_decay : float
            L2 regularization coefficient lambda >= 0.0.
        weights : list[list[float]] | None
            Weight matrix for L2 regularization.

        Returns
        -------
        tuple[float, list[list[float]]]
            (scalar_loss, d_logits [B, C])
        """
        if not logits:
            raise ValidationError("Logits list cannot be empty.")
        if len(logits) != len(targets):
            raise ValidationError(
                f"Batch size mismatch: {len(logits)} logits vs {len(targets)} targets."
            )

        batch_size = len(logits)
        num_classes = len(logits[0])

        if num_classes == 0:
            raise ValidationError("Number of classes must be greater than zero.")

        total_data_loss = 0.0
        d_logits: list[list[float]] = []

        for i in range(batch_size):
            row = logits[i]
            target_val = targets[i]

            if target_val is None:
                raise ValidationError(f"Target at batch index {i} is None.")
            if not isinstance(target_val, int):
                try:
                    target_idx = int(target_val)
                except ValueError as exc:
                    raise ValidationError(
                        f"Target at index {i} not convertible to int: {target_val}"
                    ) from exc
            else:
                target_idx = target_val

            if target_idx < 0 or target_idx >= num_classes:
                raise ValidationError(
                    f"Target index {target_idx} out of range [0, {num_classes - 1}] "
                    f"at sample {i}."
                )

            # 1. Numerical stabilization: subtract max logit
            max_logit = max(row)
            if math.isnan(max_logit) or math.isinf(max_logit):
                raise NumericalInstabilityError(
                    f"Non-finite logit detected at batch index {i}: {row}"
                )

            exps = [math.exp(z - max_logit) for z in row]
            sum_exps = sum(exps)
            if sum_exps <= 0.0 or math.isnan(sum_exps) or math.isinf(sum_exps):
                raise NumericalInstabilityError(
                    f"Numerical overflow/underflow in softmax at batch index {i}."
                )

            # 2. Compute probabilities P = exp(z_c) / sum_k exp(z_k)
            probs = [e / sum_exps for e in exps]
            target_prob = max(probs[target_idx], 1e-15)
            sample_loss = -math.log(target_prob)
            total_data_loss += sample_loss

            # 3. Compute gradient: dZ_c = (P_c - 1(y == c)) / B
            d_row = []
            for c in range(num_classes):
                indicator = 1.0 if c == target_idx else 0.0
                grad_val = (probs[c] - indicator) / float(batch_size)
                d_row.append(grad_val)
            d_logits.append(d_row)

        mean_loss = total_data_loss / float(batch_size)

        # 4. Optional L2 weight decay regularization
        if weight_decay > 0.0 and weights is not None:
            l2_reg = 0.0
            for w_row in weights:
                for w_val in w_row:
                    l2_reg += w_val * w_val
            mean_loss += 0.5 * weight_decay * l2_reg

        if math.isnan(mean_loss) or math.isinf(mean_loss):
            raise NumericalInstabilityError(
                f"Calculated loss is non-finite: {mean_loss}"
            )

        return mean_loss, d_logits


def compute_accuracy(
    logits: list[list[float]],
    targets: Sequence[int | str | None],
) -> float:
    """Compute Top-1 classification accuracy for predictions against ground truth."""
    if not logits or not targets:
        return 0.0
    if len(logits) != len(targets):
        raise ValidationError(
            f"Length mismatch: {len(logits)} logits vs {len(targets)} targets."
        )

    correct = 0
    total = len(targets)

    for i in range(total):
        row = logits[i]
        target_val = targets[i]
        if target_val is None:
            continue

        target_idx = int(target_val)
        pred_idx = max(range(len(row)), key=lambda c: row[c])
        if pred_idx == target_idx:
            correct += 1

    return float(correct) / float(total)
