"""Training orchestrations, optimization loops, and checkpoint management."""

from prism.training.configuration import (
    EarlyStoppingPolicy,
    GradientClipping,
    OptimizerSpecification,
    SchedulerSpecification,
    TrainingConfiguration,
)
from prism.training.loss import (
    SoftmaxCrossEntropyLoss,
    compute_accuracy,
)
from prism.training.optimizers import (
    BaseOptimizer,
    SGDOptimizer,
    create_optimizer,
)

__all__ = [
    "BaseOptimizer",
    "EarlyStoppingPolicy",
    "GradientClipping",
    "OptimizerSpecification",
    "SGDOptimizer",
    "SchedulerSpecification",
    "SoftmaxCrossEntropyLoss",
    "TrainingConfiguration",
    "compute_accuracy",
    "create_optimizer",
]
