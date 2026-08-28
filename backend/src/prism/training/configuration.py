"""Training configurations, optimizer specifications, and optimization policies."""

from typing import Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from prism.core.enums import DevicePreference, MetricDirection, PrecisionMode
from prism.core.errors import ValidationError


class OptimizerSpecification(BaseModel):
    """Declarative specification for an optimization algorithm."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    type: str = Field(description="Optimizer name (e.g. 'adam', 'adamw', 'sgd')")
    lr: float = Field(gt=0.0, description="Base learning rate (> 0.0)")
    weight_decay: float = Field(
        default=0.0,
        ge=0.0,
        description="L2 weight decay regularization coefficient",
    )
    momentum: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Momentum factor (for SGD/RMSprop)",
    )
    extra_kwargs: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional optimizer parameters (e.g. betas, eps)",
    )

    @field_validator("type")
    @classmethod
    def validate_type(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Optimizer type cannot be empty.")
        return v.strip().lower()


class SchedulerSpecification(BaseModel):
    """Declarative specification for learning rate schedules."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    type: str = Field(
        default="none",
        description="LR schedule (e.g. 'cosine', 'step', 'exponential', 'linear')",
    )
    warmup_epochs: int = Field(
        default=0, ge=0, description="Linear warmup duration in epochs"
    )
    warmup_steps: int = Field(
        default=0, ge=0, description="Linear warmup duration in discrete steps"
    )
    warmup_start_lr: float = Field(
        default=0.0, ge=0.0, description="Initial learning rate at start of warmup"
    )
    min_lr: float = Field(
        default=0.0, ge=0.0, description="Minimum learning rate floor"
    )
    step_size: int | None = Field(
        default=None,
        ge=1,
        description="Decay period in epochs/steps for step scheduler",
    )
    gamma: float | None = Field(
        default=None,
        gt=0.0,
        le=1.0,
        description="Multiplicative factor of learning rate decay",
    )
    decay_steps: int | None = Field(
        default=None,
        ge=1,
        description="Decay timescale for exponential scheduler",
    )
    step_unit: str = Field(
        default="epoch",
        description="Progress stepping unit ('epoch' or 'step')",
    )
    extra_kwargs: dict[str, Any] = Field(default_factory=dict)

    @field_validator("type")
    @classmethod
    def validate_type(cls, v: str) -> str:
        if not v or not v.strip():
            return "none"
        return v.strip().lower()

    @field_validator("step_unit")
    @classmethod
    def validate_step_unit(cls, v: str) -> str:
        unit = v.strip().lower()
        if unit not in ("epoch", "step"):
            raise ValueError(f"step_unit must be 'epoch' or 'step', got '{v}'.")
        return unit


class GradientClipping(BaseModel):
    """Declarative specification for gradient clipping."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    enabled: bool = Field(
        default=False, description="Whether gradient clipping is active"
    )
    max_norm: float | None = Field(
        default=None,
        gt=0.0,
        description="Maximum gradient norm threshold (> 0)",
    )
    norm_type: float = Field(
        default=2.0,
        gt=0.0,
        description="Type of the norm (e.g. 2.0 for L2 norm)",
    )

    @model_validator(mode="after")
    def validate_clipping(self) -> "GradientClipping":
        if self.enabled and self.max_norm is None:
            raise ValidationError(
                "Gradient clipping is enabled but max_norm is not specified."
            )
        return self


class EarlyStoppingPolicy(BaseModel):
    """Declarative early stopping criteria."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    enabled: bool = Field(default=False, description="Whether early stopping is active")
    monitor_metric: str = Field(
        default="val_loss", description="Target metric key to monitor"
    )
    patience: int = Field(
        default=5,
        ge=1,
        description="Epochs without improvement before stopping",
    )
    mode: MetricDirection = Field(
        default=MetricDirection.MINIMIZE,
        description="Optimization direction",
    )
    min_delta: float = Field(
        default=0.0,
        ge=0.0,
        description="Minimum change considered as an improvement",
    )


class TrainingConfiguration(BaseModel):
    """Training configuration governing optimization and execution budgets."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    epochs: int = Field(gt=0, description="Total training epochs budget (> 0)")
    batch_size: int = Field(gt=0, description="Per-device batch size (> 0)")
    optimizer: OptimizerSpecification = Field(description="Optimizer specification")
    scheduler: SchedulerSpecification = Field(
        default_factory=SchedulerSpecification,
        description="Learning rate scheduler policy",
    )
    gradient_clipping: GradientClipping = Field(
        default_factory=GradientClipping,
        description="Gradient clipping configuration",
    )
    precision: PrecisionMode = Field(
        default=PrecisionMode.FP32,
        description="Floating-point compute precision mode",
    )
    device: DevicePreference = Field(
        default=DevicePreference.AUTO,
        description="Target compute hardware preference",
    )
    early_stopping: EarlyStoppingPolicy = Field(
        default_factory=EarlyStoppingPolicy,
        description="Early stopping policy",
    )
    gradient_accumulation_steps: int = Field(
        default=1,
        ge=1,
        description="Number of forward/backward steps before optimizer step",
    )
    log_interval_steps: int = Field(
        default=50,
        ge=1,
        description="Step frequency for logging training telemetry",
    )
    checkpoint_interval_epochs: int = Field(
        default=1,
        ge=1,
        description="Epoch frequency for saving model checkpoints",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional training hyperparameters or flags",
    )
