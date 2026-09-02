"""Unit tests and finite-difference gradient checks for SSL projection & loss."""

import copy

from prism.ssl.loss import ContrastiveNTXentLoss
from prism.ssl.projection import (
    SimCLRProjectionHead,
    backward_normalize_embeddings,
    normalize_embeddings,
)


def test_projection_head_finite_differences() -> None:
    """Validate analytical gradients of SimCLRProjectionHead."""
    in_dim, hidden_dim, out_dim = 4, 6, 3
    head = SimCLRProjectionHead(
        in_dim=in_dim,
        hidden_dim=hidden_dim,
        out_dim=out_dim,
        activation="relu",
        seed=42,
    )

    h_batch = [
        [0.5, -0.2, 0.8, 0.1],
        [-0.4, 0.9, -0.1, 0.3],
    ]

    # Analytical forward and backward
    head.zero_grad()
    head.forward(h_batch)
    d_z = [
        [0.1, -0.2, 0.4],
        [-0.3, 0.5, 0.2],
    ]
    d_h_analytic = head.backward(d_z)

    # Numerical gradient w.r.t input h_batch
    eps = 1e-6
    d_h_num: list[list[float]] = []

    for b in range(len(h_batch)):
        row: list[float] = []
        for i in range(in_dim):
            h_pos = copy.deepcopy(h_batch)
            h_neg = copy.deepcopy(h_batch)
            h_pos[b][i] += eps
            h_neg[b][i] -= eps

            z_pos = head.forward(h_pos)
            z_neg = head.forward(h_neg)

            # dL = sum(d_z * z)
            loss_pos = sum(
                sum(d_z[bb][k] * z_pos[bb][k] for k in range(out_dim))
                for bb in range(len(h_batch))
            )
            loss_neg = sum(
                sum(d_z[bb][k] * z_neg[bb][k] for k in range(out_dim))
                for bb in range(len(h_batch))
            )

            grad_i = (loss_pos - loss_neg) / (2.0 * eps)
            row.append(grad_i)
        d_h_num.append(row)

    for b in range(len(h_batch)):
        for i in range(in_dim):
            assert abs(d_h_analytic[b][i] - d_h_num[b][i]) < 1e-4


def test_normalize_embeddings_finite_differences() -> None:
    """Validate analytical gradients of L2 normalization against finite differences."""
    z_batch = [
        [1.2, -0.8, 0.5],
        [-0.6, 1.4, 0.9],
    ]
    _, norms = normalize_embeddings(z_batch)

    d_hat = [
        [0.3, -0.1, 0.2],
        [-0.4, 0.2, 0.5],
    ]
    d_z_analytic = backward_normalize_embeddings(d_hat, z_batch, norms)

    eps = 1e-6
    for b in range(len(z_batch)):
        for i in range(len(z_batch[0])):
            z_pos = copy.deepcopy(z_batch)
            z_neg = copy.deepcopy(z_batch)
            z_pos[b][i] += eps
            z_neg[b][i] -= eps

            hat_pos, _ = normalize_embeddings(z_pos)
            hat_neg, _ = normalize_embeddings(z_neg)

            loss_pos = sum(
                sum(d_hat[bb][k] * hat_pos[bb][k] for k in range(len(d_hat[0])))
                for bb in range(len(z_batch))
            )
            loss_neg = sum(
                sum(d_hat[bb][k] * hat_neg[bb][k] for k in range(len(d_hat[0])))
                for bb in range(len(z_batch))
            )

            grad_num = (loss_pos - loss_neg) / (2.0 * eps)
            assert abs(d_z_analytic[b][i] - grad_num) < 1e-4


def test_nt_xent_loss_and_finite_differences() -> None:
    """Validate NT-Xent loss analytical gradients against numerical derivatives."""
    loss_fn = ContrastiveNTXentLoss(temperature=0.5)

    # 4 views (2 samples: [v1_0, v2_0, v1_1, v2_1])
    raw_z = [
        [1.0, 0.2, -0.5],
        [0.9, 0.3, -0.4],
        [-0.8, 1.1, 0.2],
        [-0.7, 1.0, 0.3],
    ]
    hat_z, _ = normalize_embeddings(raw_z)
    pos_indices = [1, 0, 3, 2]

    loss_val, d_hat_analytic, metrics = loss_fn(hat_z, pos_indices)
    assert loss_val > 0.0
    assert metrics["positive_similarity"] > metrics["negative_similarity"]

    # Finite-difference check on hat_z
    eps = 1e-6
    for b in range(len(hat_z)):
        for i in range(len(hat_z[0])):
            z_pos = copy.deepcopy(hat_z)
            z_neg = copy.deepcopy(hat_z)
            z_pos[b][i] += eps
            z_neg[b][i] -= eps

            # Re-eval loss on perturbed vectors
            l_pos, _, _ = loss_fn(z_pos, pos_indices)
            l_neg, _, _ = loss_fn(z_neg, pos_indices)

            grad_num = (l_pos - l_neg) / (2.0 * eps)
            assert abs(d_hat_analytic[b][i] - grad_num) < 1e-4
