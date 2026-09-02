"""Normalized Temperature-scaled Cross-Entropy (NT-Xent) contrastive loss."""

from __future__ import annotations

import math


class ContrastiveNTXentLoss:
    """Normalized Temperature-scaled Cross-Entropy (NT-Xent / Info-NCE) loss.

    Computes contrastive loss and exact analytical gradients over normalized embeddings.
    """

    def __init__(self, temperature: float = 0.5) -> None:
        if temperature <= 0.0:
            raise ValueError(
                f"Temperature must be strictly positive, got {temperature}"
            )
        self.temperature = temperature

    def __call__(
        self,
        normalized_embeddings: list[list[float]],
        positive_indices: list[int],
    ) -> tuple[float, list[list[float]], dict[str, float]]:
        """Compute NT-Xent loss and analytical gradients.

        Args:
            normalized_embeddings: 2N unit-norm vectors [2N x D].
            positive_indices: list of length 2N mapping index i to positive partner j.

        Returns:
            tuple of (loss_value, grad_normalized_embeddings, metrics_dict)
        """
        total_views = len(normalized_embeddings)
        if total_views < 2:
            raise ValueError(f"NT-Xent requires at least 2 views, got {total_views}")

        dim = len(normalized_embeddings[0])
        tau = self.temperature

        # 1. Compute 2N x 2N Cosine Similarity Matrix S
        sim_matrix: list[list[float]] = [
            [0.0 for _ in range(total_views)] for _ in range(total_views)
        ]
        for i in range(total_views):
            zi = normalized_embeddings[i]
            for j in range(total_views):
                if i == j:
                    sim_matrix[i][j] = 1.0
                elif j > i:
                    zj = normalized_embeddings[j]
                    dot = sum(a * b for a, b in zip(zi, zj, strict=True))
                    sim_matrix[i][j] = dot
                    sim_matrix[j][i] = dot

        # 2. Compute Loss and Probability Weights p_ik
        # ds_matrix[i][k] stores dL / dS_ik
        ds_matrix: list[list[float]] = [
            [0.0 for _ in range(total_views)] for _ in range(total_views)
        ]
        total_loss = 0.0

        pos_sims: list[float] = []
        neg_sims: list[float] = []

        for i in range(total_views):
            pos_idx = positive_indices[i]
            pos_sim = sim_matrix[i][pos_idx]
            pos_sims.append(pos_sim)

            # Collect non-self logits: S_ik / tau for k != i
            non_self_indices = [k for k in range(total_views) if k != i]
            scaled_logits = [sim_matrix[i][k] / tau for k in non_self_indices]

            for k in non_self_indices:
                if k != pos_idx:
                    neg_sims.append(sim_matrix[i][k])

            # Max subtraction for numerical stability
            max_logit = max(scaled_logits)
            exp_terms = [math.exp(val - max_logit) for val in scaled_logits]
            sum_exp = sum(exp_terms)

            # log(sum_k!=i exp(S_ik / tau))
            log_denom = max_logit + math.log(sum_exp)
            loss_i = -(pos_sim / tau) + log_denom
            total_loss += loss_i

            # Compute softmax probabilities p_ik and gradient dL / dS_ik
            # d(loss_i) / dS_ik = (p_ik - 1(k==pos)) / tau
            scale = 1.0 / (float(total_views) * tau)

            for idx, k in enumerate(non_self_indices):
                prob_k = exp_terms[idx] / sum_exp
                indicator = 1.0 if k == pos_idx else 0.0
                ds_matrix[i][k] = scale * (prob_k - indicator)

        mean_loss = total_loss / float(total_views)

        # 3. Compute Analytical Gradient dL / d(hat_z_i)
        # S_ik = hat_z_i . hat_z_k
        # dL / d(hat_z_i) = sum_k (ds_matrix[i][k] + ds_matrix[k][i]) * hat_z_k
        d_normalized: list[list[float]] = [
            [0.0 for _ in range(dim)] for _ in range(total_views)
        ]

        for i in range(total_views):
            for k in range(total_views):
                if i == k:
                    continue
                weight = ds_matrix[i][k] + ds_matrix[k][i]
                zk = normalized_embeddings[k]
                for d in range(dim):
                    d_normalized[i][d] += weight * zk[d]

        # 4. Telemetry Metrics
        mean_pos_sim = sum(pos_sims) / len(pos_sims) if pos_sims else 1.0
        mean_neg_sim = sum(neg_sims) / len(neg_sims) if neg_sims else 0.0
        sim_gap = mean_pos_sim - mean_neg_sim

        metrics = {
            "loss": mean_loss,
            "positive_similarity": mean_pos_sim,
            "negative_similarity": mean_neg_sim,
            "similarity_gap": sim_gap,
            "temperature": tau,
        }

        return mean_loss, d_normalized, metrics
