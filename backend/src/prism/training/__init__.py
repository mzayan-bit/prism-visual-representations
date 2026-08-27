"""Training orchestrations, optimization loops, and checkpoint management."""

from prism.training.configuration import (
    EarlyStoppingPolicy,
    GradientClipping,
    OptimizerSpecification,
    SchedulerSpecification,
    TrainingConfiguration,
)
from prism.training.engine import TrainingEngine
from prism.training.loss import (
    SoftmaxCrossEntropyLoss,
    compute_accuracy,
)
from prism.training.optimizers import (
    BaseOptimizer,
    SGDOptimizer,
    create_optimizer,
)
from prism.training.results import TrainingResult

__all__ = [
    "BaseOptimizer",
    "EarlyStoppingPolicy",
    "GradientClipping",
    "OptimizerSpecification",
    "SGDOptimizer",
    "SchedulerSpecification",
    "SoftmaxCrossEntropyLoss",
    "TrainingConfiguration",
    "TrainingEngine",
    "TrainingResult",
    "compute_accuracy",
    "create_optimizer",
]
