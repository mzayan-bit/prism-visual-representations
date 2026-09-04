"""Unit tests and finite-difference gradient checks for temporal aggregators."""

import copy
import math

from prism.temporal.aggregators import (
    LastFramePooling,
    LearnedTemporalPooling,
    MaxTemporalPooling,
    MeanTemporalPooling,
    SimpleRNN,
)
from prism.temporal.enums import RNNAggregationMode


def test_mean_temporal_pooling_forward_backward() -> None:
    pool = MeanTemporalPooling()
    # N=1, T=3, D=2
    features = [[[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]]
    z = pool.forward(features)
    assert len(z) == 1
    assert abs(z[0][0] - 2.0) < 1e-6
    assert abs(z[0][1] - 5.0) < 1e-6

    # Backward: d_out = [[1.0, 2.0]]
    d_out = [[1.0, 2.0]]
    d_feat = pool.backward(d_out)
    assert len(d_feat[0]) == 3
    for t in range(3):
        assert abs(d_feat[0][t][0] - 1.0 / 3.0) < 1e-6
        assert abs(d_feat[0][t][1] - 2.0 / 3.0) < 1e-6


def test_mean_temporal_pooling_with_mask() -> None:
    pool = MeanTemporalPooling()
    # N=1, T=3, D=2 with last frame padded (mask=0)
    features = [[[1.0, 2.0], [3.0, 4.0], [99.0, 99.0]]]
    mask = [[1.0, 1.0, 0.0]]
    z = pool.forward(features, mask=mask)
    assert abs(z[0][0] - 2.0) < 1e-6
    assert abs(z[0][1] - 3.0) < 1e-6

    d_out = [[1.0, 1.0]]
    d_feat = pool.backward(d_out)
    assert abs(d_feat[0][0][0] - 0.5) < 1e-6
    assert abs(d_feat[0][1][0] - 0.5) < 1e-6
    assert abs(d_feat[0][2][0] - 0.0) < 1e-6


def test_max_temporal_pooling_forward_backward() -> None:
    pool = MaxTemporalPooling()
    features = [[[1.0, 10.0], [5.0, 2.0], [3.0, 4.0]]]
    z = pool.forward(features)
    assert abs(z[0][0] - 5.0) < 1e-6
    assert abs(z[0][1] - 10.0) < 1e-6

    d_out = [[1.0, 2.0]]
    d_feat = pool.backward(d_out)
    # Feature 0 max at t=1 -> d_feat[0][1][0] = 1.0
    assert abs(d_feat[0][1][0] - 1.0) < 1e-6
    assert abs(d_feat[0][0][0] - 0.0) < 1e-6
    # Feature 1 max at t=0 -> d_feat[0][0][1] = 2.0
    assert abs(d_feat[0][0][1] - 2.0) < 1e-6
    assert abs(d_feat[0][1][1] - 0.0) < 1e-6


def test_last_frame_pooling() -> None:
    pool = LastFramePooling()
    features = [[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]]
    z = pool.forward(features)
    assert z[0] == [5.0, 6.0]

    d_out = [[1.0, 1.0]]
    d_feat = pool.backward(d_out)
    assert d_feat[0][0] == [0.0, 0.0]
    assert d_feat[0][1] == [0.0, 0.0]
    assert d_feat[0][2] == [1.0, 1.0]


def test_learned_temporal_pooling_numerical_gradients() -> None:
    # Test analytical vs finite-difference numerical gradients
    dim = 2
    pool = LearnedTemporalPooling(input_dim=dim, seed=42)
    features = [[[0.5, -0.2], [1.2, 0.8], [-0.4, 0.6]]]  # N=1, T=3, D=2
    eps = 1e-5

    # Target loss: L = 0.5 * ||z - target||^2
    target = [1.0, 0.5]

    def compute_loss(feat: list[list[list[float]]], w: list[float], b: float) -> float:
        # Forward manually
        scores = [b + sum(w[d] * feat[0][t][d] for d in range(dim)) for t in range(3)]
        max_s = max(scores)
        exp_s = [math.exp(s - max_s) for s in scores]
        sum_e = sum(exp_s)
        alphas = [e / sum_e for e in exp_s]
        z_out = [sum(alphas[t] * feat[0][t][d] for t in range(3)) for d in range(dim)]
        return 0.5 * sum((z_out[d] - target[d]) ** 2 for d in range(dim))

    # Analytical pass
    pool.zero_grad()
    z = pool.forward(features)
    d_z = [[z[0][d] - target[d] for d in range(dim)]]
    d_feat_analytical = pool.backward(d_z)
    grads_analytical = pool.get_gradients()

    # Numerical check for weights
    for d in range(dim):
        w_plus = list(pool.weights)
        w_plus[d] += eps
        loss_plus = compute_loss(features, w_plus, pool.bias)

        w_minus = list(pool.weights)
        w_minus[d] -= eps
        loss_minus = compute_loss(features, w_minus, pool.bias)

        num_grad = (loss_plus - loss_minus) / (2 * eps)
        ana_grad = grads_analytical["weights"][d]
        assert abs(num_grad - ana_grad) < 1e-4

    # Numerical check for bias
    loss_b_plus = compute_loss(features, pool.weights, pool.bias + eps)
    loss_b_minus = compute_loss(features, pool.weights, pool.bias - eps)
    num_b_grad = (loss_b_plus - loss_b_minus) / (2 * eps)
    ana_b_grad = grads_analytical["bias"]
    assert abs(num_b_grad - ana_b_grad) < 1e-4

    # Numerical check for input features
    for t in range(3):
        for d in range(dim):
            f_plus = copy.deepcopy(features)
            f_plus[0][t][d] += eps
            l_plus = compute_loss(f_plus, pool.weights, pool.bias)

            f_minus = copy.deepcopy(features)
            f_minus[0][t][d] -= eps
            l_minus = compute_loss(f_minus, pool.weights, pool.bias)

            num_f_grad = (l_plus - l_minus) / (2 * eps)
            ana_f_grad = d_feat_analytical[0][t][d]
            assert abs(num_f_grad - ana_f_grad) < 1e-4


def test_simple_rnn_bptt_numerical_gradients() -> None:
    # Test SimpleRNN BPTT analytical gradients against finite differences
    in_dim = 2
    h_dim = 2
    t_steps = 3
    rnn = SimpleRNN(
        input_dim=in_dim,
        hidden_dim=h_dim,
        mode=RNNAggregationMode.LAST_HIDDEN,
        seed=42,
    )

    features = [[[0.4, -0.3], [0.8, 0.5], [-0.2, 0.7]]]  # N=1, T=3, D_in=2
    target = [0.5, -0.5]
    eps = 1e-5

    def rnn_forward_loss(
        feat: list[list[list[float]]],
        w_x: list[list[float]],
        w_h: list[list[float]],
        b: list[float],
    ) -> float:
        h_curr = [0.0] * h_dim
        for t in range(t_steps):
            x_t = feat[0][t]
            h_next = [0.0] * h_dim
            for j in range(h_dim):
                in_term = sum(w_x[j][d] * x_t[d] for d in range(in_dim))
                hid_term = sum(w_h[j][k] * h_curr[k] for k in range(h_dim))
                act = b[j] + in_term + hid_term
                h_next[j] = math.tanh(act)
            h_curr = h_next
        return 0.5 * sum((h_curr[d] - target[d]) ** 2 for d in range(h_dim))

    # Analytical pass
    rnn.zero_grad()
    z = rnn.forward(features)
    d_z = [[z[0][d] - target[d] for d in range(h_dim)]]
    d_inputs_analytical = rnn.backward(d_z)
    grads = rnn.get_gradients()

    # 1. Check W_x gradients
    for j in range(h_dim):
        for d in range(in_dim):
            wx_plus = copy.deepcopy(rnn.W_x)
            wx_plus[j][d] += eps
            l_plus = rnn_forward_loss(features, wx_plus, rnn.W_h, rnn.bias)

            wx_minus = copy.deepcopy(rnn.W_x)
            wx_minus[j][d] -= eps
            l_minus = rnn_forward_loss(features, wx_minus, rnn.W_h, rnn.bias)

            num_g = (l_plus - l_minus) / (2 * eps)
            ana_g = grads["W_x"][j][d]
            assert abs(num_g - ana_g) < 1e-4

    # 2. Check W_h gradients
    for j in range(h_dim):
        for k in range(h_dim):
            wh_plus = copy.deepcopy(rnn.W_h)
            wh_plus[j][k] += eps
            l_plus = rnn_forward_loss(features, rnn.W_x, wh_plus, rnn.bias)

            wh_minus = copy.deepcopy(rnn.W_h)
            wh_minus[j][k] -= eps
            l_minus = rnn_forward_loss(features, rnn.W_x, wh_minus, rnn.bias)

            num_g = (l_plus - l_minus) / (2 * eps)
            ana_g = grads["W_h"][j][k]
            assert abs(num_g - ana_g) < 1e-4

    # 3. Check bias gradients
    for j in range(h_dim):
        b_plus = list(rnn.bias)
        b_plus[j] += eps
        l_plus = rnn_forward_loss(features, rnn.W_x, rnn.W_h, b_plus)

        b_minus = list(rnn.bias)
        b_minus[j] -= eps
        l_minus = rnn_forward_loss(features, rnn.W_x, rnn.W_h, b_minus)

        num_g = (l_plus - l_minus) / (2 * eps)
        ana_g = grads["bias"][j]
        assert abs(num_g - ana_g) < 1e-4

    # 4. Check input feature sequence gradients (BPTT input backprop)
    for t in range(t_steps):
        for d in range(in_dim):
            f_plus = copy.deepcopy(features)
            f_plus[0][t][d] += eps
            l_plus = rnn_forward_loss(f_plus, rnn.W_x, rnn.W_h, rnn.bias)

            f_minus = copy.deepcopy(features)
            f_minus[0][t][d] -= eps
            l_minus = rnn_forward_loss(f_minus, rnn.W_x, rnn.W_h, rnn.bias)

            num_g = (l_plus - l_minus) / (2 * eps)
            ana_g = d_inputs_analytical[0][t][d]
            assert abs(num_g - ana_g) < 1e-4
