"""Training engine, loss functions, optimizers, and gradient flow tracking."""

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
    "ModelGradientFlowSummary",
    "OptimizerSpecification",
    "ParameterGradientSummary",
    "SGDOptimizer",
    "SchedulerSpecification",
    "SoftmaxCrossEntropyLoss",
    "StepLRScheduler",
    "TrainingConfiguration",
    "TrainingEngine",
    "TrainingResult",
    "compare_gradient_flow_summaries",
    "compute_accuracy",
    "compute_gradient_flow_summary",
    "create_optimizer",
    "create_scheduler",
]
