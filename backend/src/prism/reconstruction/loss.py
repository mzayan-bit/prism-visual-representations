"""Mean Squared Error reconstruction losses and analytical gradients."""

from __future__ import annotations

from prism.core.errors import ValidationError
from prism.reconstruction.mask import PatchMask


class MaskedMSELoss:
    """Mean Squared Error loss computed strictly over masked regions.

    Supports both patch sequences [N x T x D] and 4D image tensors [N x C x H x W].

    Guarantees:
    - Visible (unmasked) elements receive exactly zero gradient.
    - Altering unscored target values does not alter the masked reconstruction loss.
    """

    def __init__(self, reduction: str = "mean") -> None:
        if reduction not in ("mean", "sum"):
            raise ValidationError(
                f"Unsupported reduction: {reduction}. Use 'mean' or 'sum'."
            )
        self.reduction = reduction

    def __call__(
        self,
        predictions: list[list[list[float]]],
        targets: list[list[list[float]]],
        masks: list[PatchMask] | list[list[int]] | None = None,
    ) -> tuple[float, list[list[list[float]]], dict[str, float]]:
        """Compute masked MSE loss and gradient for patch sequences [N x T x D].

        Parameters
        ----------
        predictions : list[list[list[float]]]
            Predicted patch vectors of shape [N x T x D_patch].
        targets : list[list[list[float]]]
            Clean ground-truth patch vectors of shape [N x T x D_patch].
        masks : list[PatchMask] | list[list[int]] | None
            Mask definitions specifying which patch indices are scored.
            If None, all patches are scored (full-image MSE).

        Returns
        -------
        tuple[float, list[list[list[float]]], dict[str, float]]
            (loss_value, d_predictions, metrics_dict)
        """
        n_samples = len(predictions)
        if n_samples == 0:
            raise ValidationError("Empty prediction batch.")
        if len(targets) != n_samples:
            raise ValidationError(
                f"Batch size mismatch: pred ({n_samples}) vs target ({len(targets)})."
            )

        n_patches = len(predictions[0])
        patch_dim = len(predictions[0][0])

        # Resolve masked indices set per sample
        masked_sets: list[set[int]] = []
        if masks is None:
            # Full sequence scored
            full_set = set(range(n_patches))
            masked_sets = [full_set for _ in range(n_samples)]
        else:
            for m in masks:
                if isinstance(m, PatchMask):
                    masked_sets.append(set(m.masked_indices))
                elif isinstance(m, (list, set)):
                    masked_sets.append(set(m))
                else:
                    raise ValidationError(f"Unsupported mask format: {type(m)}.")

        total_masked_elements = sum(len(s) * patch_dim for s in masked_sets)
        total_visible_elements = sum(
            (n_patches - len(s)) * patch_dim for s in masked_sets
        )

        if total_masked_elements == 0:
            raise ValidationError("No elements to score: masked element count is 0.")

        norm_factor = float(total_masked_elements) if self.reduction == "mean" else 1.0

        masked_sq_err = 0.0
        visible_sq_err = 0.0
        d_preds: list[list[list[float]]] = []

        for n in range(n_samples):
            pred_sample = predictions[n]
            tgt_sample = targets[n]
            m_set = masked_sets[n]
            sample_grads: list[list[float]] = []

            for t in range(n_patches):
                p_vec = pred_sample[t]
                t_vec = tgt_sample[t]
                grad_vec: list[float] = [0.0] * patch_dim

                is_masked = t in m_set

                for d in range(patch_dim):
                    diff = p_vec[d] - t_vec[d]
                    sq = diff * diff

                    if is_masked:
                        masked_sq_err += sq
                        # Analytical gradient: dL / d(p_hat) = 2 * diff / normalizer
                        grad_vec[d] = (2.0 * diff) / norm_factor
                    else:
                        visible_sq_err += sq
                        # Visible patch receives EXACTLY ZERO gradient
                        grad_vec[d] = 0.0

                sample_grads.append(grad_vec)
            d_preds.append(sample_grads)

        loss_val = (
            masked_sq_err / norm_factor if self.reduction == "mean" else masked_sq_err
        )
        visible_mse = (
            (visible_sq_err / float(total_visible_elements))
            if total_visible_elements > 0
            else 0.0
        )
        full_mse = (masked_sq_err + visible_sq_err) / float(
            total_masked_elements + total_visible_elements
        )

        metrics = {
            "reconstruction_loss": loss_val,
            "masked_mse": (
                masked_sq_err / float(total_masked_elements)
                if total_masked_elements > 0
                else 0.0
            ),
            "visible_mse": visible_mse,
            "full_mse": full_mse,
            "total_scored_elements": float(total_masked_elements),
        }

        return loss_val, d_preds, metrics

    def compute_image_mse(
        self,
        predictions: list[list[list[list[float]]]],
        targets: list[list[list[list[float]]]],
    ) -> tuple[float, list[list[list[list[float]]]], dict[str, float]]:
        """Compute full-image spatial MSE loss for Denoising Autoencoders."""
        n_samples = len(predictions)
        if n_samples == 0:
            raise ValidationError("Empty prediction batch.")

        c = len(predictions[0])
        h = len(predictions[0][0])
        w = len(predictions[0][0][0])
        total_pixels = n_samples * c * h * w

        norm_factor = float(total_pixels) if self.reduction == "mean" else 1.0
        total_sq_err = 0.0
        d_preds: list[list[list[list[float]]]] = []

        for n in range(n_samples):
            pred_img = predictions[n]
            tgt_img = targets[n]
            img_grad: list[list[list[float]]] = []

            for ch in range(c):
                ch_grad: list[list[float]] = []
                for r in range(h):
                    row_grad: list[float] = [0.0] * w
                    for col in range(w):
                        diff = pred_img[ch][r][col] - tgt_img[ch][r][col]
                        total_sq_err += diff * diff
                        row_grad[col] = (2.0 * diff) / norm_factor
                    ch_grad.append(row_grad)
                img_grad.append(ch_grad)
            d_preds.append(img_grad)

        loss_val = (
            total_sq_err / norm_factor if self.reduction == "mean" else total_sq_err
        )
        metrics = {
            "reconstruction_loss": loss_val,
            "full_mse": total_sq_err / float(total_pixels),
            "total_scored_elements": float(total_pixels),
        }
        return loss_val, d_preds, metrics
