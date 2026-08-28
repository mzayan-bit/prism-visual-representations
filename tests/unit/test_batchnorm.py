"""Unit tests for BatchNorm1D and BatchNorm2D layers and numerical gradients."""

import pytest

from prism.core.errors import ValidationError
from prism.models.normalization import BatchNorm1D, BatchNorm2D


@pytest.mark.unit
def test_batchnorm1d_forward_training_and_eval() -> None:
    """Verify BatchNorm1D normalizes batch in train mode and freezes in eval."""
    bn = BatchNorm1D(num_features=2, eps=1e-5, momentum=0.1, affine=True)
    bn.train()

    # Input batch of 3 samples, 2 features
    x = [
        [1.0, 10.0],
        [2.0, 20.0],
        [3.0, 30.0],
    ]

    out = bn.forward(x)
    assert len(out) == 3
    assert len(out[0]) == 2

    # Feature 0: mean = 2.0, var = ((1-2)^2 + (2-2)^2 + (3-2)^2)/3 = 2/3 = 0.66667
    # Normalized x_hat: [-1 / sqrt(2/3 + 1e-5), 0, 1 / sqrt(2/3 + 1e-5)]
    assert out[0][0] == pytest.approx(-1.2247, rel=1e-3)
    assert out[1][0] == pytest.approx(0.0, abs=1e-5)
    assert out[2][0] == pytest.approx(1.2247, rel=1e-3)

    # Running statistics updated:
    # running_mean = 0.9 * 0 + 0.1 * 2.0 = 0.2
    assert bn.running_mean[0] == pytest.approx(0.2)
    assert bn.num_batches_tracked == 1

    # Evaluation Mode: should use running statistics and NOT update them
    bn.eval()
    eval_out = bn.forward([[2.0, 20.0]])
    assert bn.num_batches_tracked == 1
    assert bn.running_mean[0] == pytest.approx(0.2)  # unchanged
    assert len(eval_out) == 1


@pytest.mark.unit
def test_batchnorm2d_channelwise_forward_and_running_stats() -> None:
    """Verify BatchNorm2D computes statistics channel-wise across N * H * W."""
    bn = BatchNorm2D(num_features=2, eps=1e-5, momentum=0.2, affine=True)
    bn.train()

    # 2 samples, 2 channels, 2x2 image (M = 2 * 2 * 2 = 8 elements per channel)
    # Channel 0: values 1.0 to 8.0 -> mean = 4.5
    # Channel 1: all 5.0 -> mean = 5.0, var = 0.0
    x = [
        [
            [[1.0, 2.0], [3.0, 4.0]],
            [[5.0, 5.0], [5.0, 5.0]],
        ],
        [
            [[5.0, 6.0], [7.0, 8.0]],
            [[5.0, 5.0], [5.0, 5.0]],
        ],
    ]

    out = bn.forward(x)
    assert len(out) == 2
    assert len(out[0]) == 2
    assert len(out[0][0]) == 2
    assert len(out[0][0][0]) == 2

    # Channel 0 running mean: 0.8 * 0.0 + 0.2 * 4.5 = 0.9
    assert bn.running_mean[0] == pytest.approx(0.9)
    # Channel 1 running mean: 0.8 * 0.0 + 0.2 * 5.0 = 1.0
    assert bn.running_mean[1] == pytest.approx(1.0)

    # Evaluation mode does not update stats
    bn.eval()
    _ = bn.forward(x)
    assert bn.running_mean[0] == pytest.approx(0.9)


@pytest.mark.unit
def test_batchnorm1d_numerical_gradient_check() -> None:
    """Verify finite-difference gradients for BatchNorm1D w.r.t input and params."""
    bn = BatchNorm1D(num_features=3, eps=1e-5, affine=True)
    bn.train()
    bn.gamma = [1.2, 0.8, 1.5]
    bn.beta = [-0.3, 0.5, 0.1]

    x = [
        [1.0, 2.0, 3.0],
        [4.0, 0.5, -1.0],
        [2.0, -1.5, 0.5],
        [-0.5, 3.0, 1.5],
    ]
    d_out = [
        [0.5, -0.2, 0.1],
        [-0.3, 0.4, -0.5],
        [0.2, -0.1, 0.3],
        [-0.4, 0.3, 0.1],
    ]

    # Analytic forward & backward
    _ = bn.forward(x)
    dx_ana = bn.backward(d_out)
    dgamma_ana = list(bn.grad_gamma)
    dbeta_ana = list(bn.grad_beta)

    def compute_loss(test_bn: BatchNorm1D, test_x: list[list[float]]) -> float:
        y = test_bn.forward(test_x)
        loss = 0.0
        for n in range(len(y)):
            for j in range(len(y[0])):
                loss += y[n][j] * d_out[n][j]
        return loss

    eps = 1e-5

    # 1. Check dX
    for n in range(len(x)):
        for j in range(3):
            orig = x[n][j]
            x[n][j] = orig + eps
            l_plus = compute_loss(bn, x)
            x[n][j] = orig - eps
            l_minus = compute_loss(bn, x)
            x[n][j] = orig

            dx_num = (l_plus - l_minus) / (2.0 * eps)
            assert dx_ana[n][j] == pytest.approx(dx_num, rel=1e-3, abs=1e-4)

    # 2. Check d_gamma
    for j in range(3):
        orig_g = bn.gamma[j]
        bn.gamma[j] = orig_g + eps
        l_plus = compute_loss(bn, x)
        bn.gamma[j] = orig_g - eps
        l_minus = compute_loss(bn, x)
        bn.gamma[j] = orig_g

        dg_num = (l_plus - l_minus) / (2.0 * eps)
        assert dgamma_ana[j] == pytest.approx(dg_num, rel=1e-3, abs=1e-4)

    # 3. Check d_beta
    for j in range(3):
        orig_b = bn.beta[j]
        bn.beta[j] = orig_b + eps
        l_plus = compute_loss(bn, x)
        bn.beta[j] = orig_b - eps
        l_minus = compute_loss(bn, x)
        bn.beta[j] = orig_b

        db_num = (l_plus - l_minus) / (2.0 * eps)
        assert dbeta_ana[j] == pytest.approx(db_num, rel=1e-3, abs=1e-4)


@pytest.mark.unit
def test_batchnorm2d_numerical_gradient_check() -> None:
    """Verify finite-difference gradients for BatchNorm2D w.r.t input and params."""
    bn = BatchNorm2D(num_features=2, eps=1e-5, affine=True)
    bn.train()
    bn.gamma = [1.3, 0.7]
    bn.beta = [0.2, -0.4]

    x = [
        [
            [[1.0, 2.0], [3.0, 4.0]],
            [[-1.0, 0.5], [2.0, -0.5]],
        ],
        [
            [[0.5, -0.5], [1.5, 2.5]],
            [[1.0, -1.0], [0.0, 2.0]],
        ],
    ]
    d_out = [
        [
            [[0.2, -0.1], [0.3, -0.4]],
            [[-0.3, 0.2], [0.1, -0.2]],
        ],
        [
            [[-0.2, 0.1], [0.4, -0.3]],
            [[0.2, -0.4], [0.3, 0.1]],
        ],
    ]

    # Analytic forward & backward
    _ = bn.forward(x)
    dx_ana = bn.backward(d_out)
    dgamma_ana = list(bn.grad_gamma)
    dbeta_ana = list(bn.grad_beta)

    def compute_loss(
        test_bn: BatchNorm2D, test_x: list[list[list[list[float]]]]
    ) -> float:
        y = test_bn.forward(test_x)
        loss = 0.0
        for n in range(len(y)):
            for c in range(len(y[0])):
                for h in range(len(y[0][0])):
                    for w in range(len(y[0][0][0])):
                        loss += y[n][c][h][w] * d_out[n][c][h][w]
        return loss

    eps = 1e-5

    # 1. Check dX
    for n in range(2):
        for c in range(2):
            for h in range(2):
                for w in range(2):
                    orig = x[n][c][h][w]
                    x[n][c][h][w] = orig + eps
                    l_plus = compute_loss(bn, x)
                    x[n][c][h][w] = orig - eps
                    l_minus = compute_loss(bn, x)
                    x[n][c][h][w] = orig

                    dx_num = (l_plus - l_minus) / (2.0 * eps)
                    assert dx_ana[n][c][h][w] == pytest.approx(
                        dx_num, rel=1e-3, abs=1e-4
                    )

    # 2. Check d_gamma
    for c in range(2):
        orig_g = bn.gamma[c]
        bn.gamma[c] = orig_g + eps
        l_plus = compute_loss(bn, x)
        bn.gamma[c] = orig_g - eps
        l_minus = compute_loss(bn, x)
        bn.gamma[c] = orig_g

        dg_num = (l_plus - l_minus) / (2.0 * eps)
        assert dgamma_ana[c] == pytest.approx(dg_num, rel=1e-3, abs=1e-4)

    # 3. Check d_beta
    for c in range(2):
        orig_b = bn.beta[c]
        bn.beta[c] = orig_b + eps
        l_plus = compute_loss(bn, x)
        bn.beta[c] = orig_b - eps
        l_minus = compute_loss(bn, x)
        bn.beta[c] = orig_b

        db_num = (l_plus - l_minus) / (2.0 * eps)
        assert dbeta_ana[c] == pytest.approx(db_num, rel=1e-3, abs=1e-4)


@pytest.mark.unit
def test_batchnorm_state_and_parameter_separation() -> None:
    """Verify parameters (gamma/beta) and state (running statistics) are distinct."""
    bn = BatchNorm2D(num_features=3, eps=1e-5, affine=True)
    params = bn.get_parameters()
    state = bn.get_state()

    assert "gamma" in params
    assert "beta" in params
    assert "running_mean" not in params
    assert "running_var" not in params

    assert "running_mean" in state
    assert "running_var" in state
    assert "gamma" not in state
    assert "beta" not in state

    # Test setter & state reload
    bn.set_parameters({"gamma": [2.0, 2.0, 2.0], "beta": [1.0, 1.0, 1.0]})
    bn.set_state(
        {
            "running_mean": [0.5, 0.5, 0.5],
            "running_var": [1.5, 1.5, 1.5],
            "num_batches_tracked": 10,
        }
    )

    assert bn.gamma == [2.0, 2.0, 2.0]
    assert bn.running_mean == [0.5, 0.5, 0.5]
    assert bn.num_batches_tracked == 10


@pytest.mark.unit
def test_batchnorm_invalid_configurations() -> None:
    """Verify invalid parameters raise ValidationError."""
    with pytest.raises(ValidationError, match="num_features must be positive"):
        BatchNorm1D(num_features=0)

    with pytest.raises(ValidationError, match="eps must be positive"):
        BatchNorm2D(num_features=4, eps=-1e-5)

    with pytest.raises(ValidationError, match="momentum must be in"):
        BatchNorm2D(num_features=4, momentum=1.5)
