"""Symmetric Vision-Language Contrastive Loss with Analytical Gradients."""

from __future__ import annotations

import math


class SymmetricContrastiveLoss:
    """CLIP-style symmetric contrastive loss between image and text embeddings.

    Computes:
        L_v2t = CrossEntropy(Similarity / tau, targets)
        L_t2v = CrossEntropy(Similarity^T / tau, targets)
        Total Loss = 0.5 * (L_v2t + L_t2v)
    """

    def __init__(self, temperature: float = 0.07) -> None:
        if temperature <= 0.0:
            raise ValueError(
                f"Temperature must be strictly positive, got {temperature}"
            )
        self.temperature = temperature

    def __call__(
        self,
        image_embeddings: list[list[float]],
        text_embeddings: list[list[float]],
    ) -> tuple[float, list[list[float]], list[list[float]], dict[str, float]]:
        """Compute symmetric contrastive loss and analytical gradients.

        Args:
            image_embeddings: N unit-norm vectors [N x D].
            text_embeddings: N unit-norm vectors [N x D].

        Returns:
            tuple of (total_loss, d_image, d_text, metrics_dict)
        """
        n = len(image_embeddings)
        if n == 0:
            raise ValueError("Cannot compute loss on empty batch")
        if len(text_embeddings) != n:
            raise ValueError(
                f"Image batch size {n} != text batch size {len(text_embeddings)}"
            )

        dim = len(image_embeddings[0])
        tau = self.temperature

        # 1. Compute Cosine Similarity Matrix S (N x N)
        # S[i][j] = image_i . text_j
        sim_matrix: list[list[float]] = [[0.0 for _ in range(n)] for _ in range(n)]
        pos_sims: list[float] = []
        neg_sims: list[float] = []

        for i in range(n):
            v_i = image_embeddings[i]
            for j in range(n):
                t_j = text_embeddings[j]
                dot = sum(a * b for a, b in zip(v_i, t_j, strict=True))
                sim_matrix[i][j] = dot
                if i == j:
                    pos_sims.append(dot)
                else:
                    neg_sims.append(dot)

        # 2. Compute Image -> Text Cross-Entropy Loss & Logit Gradients (row-wise)
        # d_sim_v2t[i][j] = d(L_v2t) / d(S_ij)
        d_sim_v2t: list[list[float]] = [[0.0 for _ in range(n)] for _ in range(n)]
        loss_v2t = 0.0

        for i in range(n):
            row_logits = [sim_matrix[i][j] / tau for j in range(n)]
            max_logit = max(row_logits)
            exp_row = [math.exp(val - max_logit) for val in row_logits]
            sum_exp_row = sum(exp_row)
            log_sum_exp = max_logit + math.log(sum_exp_row)

            # Target is index i
            loss_i = -(row_logits[i]) + log_sum_exp
            loss_v2t += loss_i

            # Gradient w.r.t S_ij: (1 / (N * tau)) * (prob_ij - 1(j == i))
            scale = 1.0 / (float(n) * tau)
            for j in range(n):
                prob = exp_row[j] / sum_exp_row
                indicator = 1.0 if j == i else 0.0
                d_sim_v2t[i][j] = scale * (prob - indicator)

        loss_v2t /= float(n)

        # 3. Compute Text -> Image Cross-Entropy Loss & Logit Gradients (column-wise)
        # d_sim_t2v[i][j] = d(L_t2v) / d(S_ij)
        d_sim_t2v: list[list[float]] = [[0.0 for _ in range(n)] for _ in range(n)]
        loss_t2v = 0.0

        for j in range(n):
            col_logits = [sim_matrix[i][j] / tau for i in range(n)]
            max_logit = max(col_logits)
            exp_col = [math.exp(val - max_logit) for val in col_logits]
            sum_exp_col = sum(exp_col)
            log_sum_exp = max_logit + math.log(sum_exp_col)

            # Target is index j
            loss_j = -(col_logits[j]) + log_sum_exp
            loss_t2v += loss_j

            # Gradient w.r.t S_ij: (1 / (N * tau)) * (prob_ij - 1(i == j))
            scale = 1.0 / (float(n) * tau)
            for i in range(n):
                prob = exp_col[i] / sum_exp_col
                indicator = 1.0 if i == j else 0.0
                d_sim_t2v[i][j] = scale * (prob - indicator)

        loss_t2v /= float(n)

        # 4. Total Loss & Combined Similarity Matrix Gradients
        total_loss = 0.5 * (loss_v2t + loss_t2v)

        d_sim_total: list[list[float]] = [
            [0.5 * (d_sim_v2t[i][j] + d_sim_t2v[i][j]) for j in range(n)]
            for i in range(n)
        ]

        # 5. Gradients w.r.t Normalized Image Embeddings (d_v) and Text Embeddings (d_t)
        # S_ij = v_i . t_j
        # d_v[i] = sum_j d_sim_total[i][j] * t_j
        # d_t[j] = sum_i d_sim_total[i][j] * v_i
        d_image: list[list[float]] = [[0.0 for _ in range(dim)] for _ in range(n)]
        d_text: list[list[float]] = [[0.0 for _ in range(dim)] for _ in range(n)]

        for i in range(n):
            for j in range(n):
                weight = d_sim_total[i][j]
                tj = text_embeddings[j]
                vi = image_embeddings[i]
                for d in range(dim):
                    d_image[i][d] += weight * tj[d]
                    d_text[j][d] += weight * vi[d]

        # 6. Telemetry Metrics
        mean_pos_sim = sum(pos_sims) / len(pos_sims) if pos_sims else 1.0
        mean_neg_sim = sum(neg_sims) / len(neg_sims) if neg_sims else 0.0
        sim_gap = mean_pos_sim - mean_neg_sim

        metrics = {
            "loss": total_loss,
            "image_to_text_loss": loss_v2t,
            "text_to_image_loss": loss_t2v,
            "matched_similarity": mean_pos_sim,
            "unmatched_similarity": mean_neg_sim,
            "similarity_gap": sim_gap,
            "temperature": tau,
        }

        return total_loss, d_image, d_text, metrics
