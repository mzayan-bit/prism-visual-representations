"""Training engine, loss functions, optimizers, and learning rate schedulers."""

from prism.training.configuration import (
    GradientClipping,
    OptimizerSpecification,
    SchedulerSpecification,
    TrainingConfiguration,
)
from prism.training.engine import TrainingEngine
from prism.training.loss import SoftmaxCrossEntropyLoss, compute_accuracy
from prism.training.optimizers import (
    BaseOptimizer,
    SGDOptimizer,
    create_optimizer,
)
from prism.training.results import TrainingResult
from prism.training.schedulers import (
    BaseLRScheduler,
    ConstantLRScheduler,
    CosineAnnealingLRScheduler,
    StepLRScheduler,
    create_scheduler,
)

__all__ = [
    "BaseLRScheduler",
    "BaseOptimizer",
    "ConstantLRScheduler",
    "CosineAnnealingLRScheduler",
    "GradientClipping",
    "OptimizerSpecification",
    "SGDOptimizer",
    "SchedulerSpecification",
    "SoftmaxCrossEntropyLoss",
    "StepLRScheduler",
    "TrainingConfiguration",
    "TrainingEngine",
    "TrainingResult",
    "compute_accuracy",
    "create_optimizer",
    "create_scheduler",
]
