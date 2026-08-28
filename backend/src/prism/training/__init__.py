"""Training engine, loss functions, optimizers, and schedulers."""

from prism.training.configuration import (
    GradientClipping,
    OptimizerSpecification,
    SchedulerSpecification,
    TrainingConfiguration,
)
from prism.training.engine import TrainingEngine
from prism.training.gradient_flow import (
    ModelGradientFlowSummary,
    ParameterGradientSummary,
    compare_gradient_flow_summaries,
    compute_gradient_flow_summary,
)
from prism.training.loss import SoftmaxCrossEntropyLoss, compute_accuracy
from prism.training.optimizers import (
    BaseOptimizer,
    SGDOptimizer,
    create_optimizer,
)
from prism.training.results import TrainingResult
from prism.training.scheduler_state import SchedulerState
from prism.training.schedulers import (
    BaseLRScheduler,
    ConstantLRScheduler,
    CosineAnnealingLRScheduler,
    ExponentialLRScheduler,
    LinearWarmupScheduler,
    StepLRScheduler,
    WarmupScheduler,
    create_scheduler,
)

__all__ = [
    "BaseLRScheduler",
    "BaseOptimizer",
    "ConstantLRScheduler",
    "CosineAnnealingLRScheduler",
    "ExponentialLRScheduler",
    "GradientClipping",
    "LinearWarmupScheduler",
    "ModelGradientFlowSummary",
    "OptimizerSpecification",
    "ParameterGradientSummary",
    "SGDOptimizer",
    "SchedulerSpecification",
    "SchedulerState",
    "SoftmaxCrossEntropyLoss",
    "StepLRScheduler",
    "TrainingConfiguration",
    "TrainingEngine",
    "TrainingResult",
    "WarmupScheduler",
    "compare_gradient_flow_summaries",
    "compute_accuracy",
    "compute_gradient_flow_summary",
    "create_optimizer",
    "create_scheduler",
]
