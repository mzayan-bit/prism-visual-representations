"""Gradient flow tracking, depth-wise gradient summaries, and comparison analysis."""

from __future__ import annotations

import json
import math
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from prism.core.errors import SerializationError, ValidationError
from prism.models.base import BaseVisionModel
from prism.representations.summaries import _flatten_and_measure


def _infer_logical_stage_and_depth(
    param_name: str,
) -> tuple[str, int]:
    """Infer logical stage name and relative depth index from parameter identifier."""
    name = param_name.lower()

    if "stem" in name or "conv_0" in name or "layer_0" in name:
        return "stem", 0

    if "stage_" in name:
        parts = name.split("_")
        try:
            s_idx = int(parts[parts.index("stage") + 1])
            b_idx = int(parts[parts.index("block") + 1]) if "block" in parts else 0
            depth = 1 + s_idx * 10 + b_idx
            return f"stage_{s_idx}", depth
        except (ValueError, IndexError):
            return "stage", 5

    if "conv_" in name:
        parts = name.split("_")
        try:
            c_idx = int(parts[parts.index("conv") + 1])
            return f"block_{c_idx}", 1 + c_idx
        except (ValueError, IndexError):
            return "conv", 5

    if "classifier" in name or "fc_" in name or "weights_out" in name:
        return "classifier", 999

    return "body", 50


class ParameterGradientSummary(BaseModel):
    """Statistical gradient summary for a single parameter tensor across depth."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    parameter_name: str = Field(description="Unique parameter identifier")
    logical_stage: str = Field(description="Logical architectural stage name")
    depth_index: int = Field(description="Relative depth ordering index")
    norm_l2: float = Field(description="L2 Frobenius norm of the gradient tensor")
    mean: float = Field(description="Mean scalar gradient value")
    std_dev: float = Field(description="Standard deviation of scalar gradient values")
    min_value: float = Field(description="Minimum gradient value")
    max_value: float = Field(description="Maximum gradient value")
    zero_fraction: float = Field(
        description="Fraction of gradient entries equal or close to 0.0"
    )
    is_finite: bool = Field(description="True if all gradients are finite numbers")
    parameter_shape: tuple[int, ...] = Field(
        description="Dimensions of parameter tensor"
    )
    sample_count: int = Field(description="Total number of elements in parameter")


class ModelGradientFlowSummary(BaseModel):
    """Comprehensive gradient flow summary across all layers and depth of a model."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    model_id: str = Field(description="Model identifier")
    total_parameters: int = Field(description="Number of parameter tensors discovered")
    total_gradients: int = Field(
        description="Number of gradient tensors with computed values"
    )
    global_grad_norm_l2: float = Field(
        description="Global L2 norm across all model parameter gradients"
    )
    parameter_summaries: list[ParameterGradientSummary] = Field(
        description="Depth-ordered parameter gradient summaries"
    )
    depth_ordering: list[str] = Field(
        description="Ordered list of parameter names from earliest to latest layer"
    )
    is_finite: bool = Field(
        description="True if all gradients across all parameters are finite"
    )

    def get_summary(self, parameter_name: str) -> ParameterGradientSummary | None:
        """Find gradient summary for a specific parameter name."""
        for s in self.parameter_summaries:
            if s.parameter_name == parameter_name:
                return s
        return None

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")

    def to_json(self, indent: int | None = None) -> str:
        return json.dumps(
            self.to_dict(),
            indent=indent,
            sort_keys=True,
            ensure_ascii=True,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModelGradientFlowSummary:
        try:
            return cls.model_validate(data)
        except Exception as exc:
            raise SerializationError(
                f"Failed to deserialize ModelGradientFlowSummary from dict: {exc}"
            ) from exc

    @classmethod
    def from_json(cls, json_str: str) -> ModelGradientFlowSummary:
        try:
            parsed = json.loads(json_str)
            return cls.from_dict(parsed)
        except Exception as exc:
            raise SerializationError(
                f"Failed to deserialize ModelGradientFlowSummary from JSON: {exc}"
            ) from exc


def compute_gradient_flow_summary(
    model: BaseVisionModel, zero_threshold: float = 1e-12
) -> ModelGradientFlowSummary:
    """Extract and summarize gradient flow across all parameters of a model."""
    if model is None:
        raise ValidationError("Cannot compute gradient flow summary on None model.")

    params = model.get_parameters()
    grads = model.get_gradients()

    param_summaries: list[ParameterGradientSummary] = []
    global_sum_sq = 0.0
    all_finite = True

    # Identify parameter names and depth order
    ordered_keys = sorted(
        params.keys(),
        key=lambda k: _infer_logical_stage_and_depth(k)[1],
    )

    for p_name in ordered_keys:
        p_val = params[p_name]
        g_key = f"grad_{p_name}"

        _, shape = _flatten_and_measure(p_val)
        stage_name, depth_idx = _infer_logical_stage_and_depth(p_name)

        if g_key not in grads or grads[g_key] is None:
            # Missing gradient policy: represent with zero norm
            total_elements = 1
            for dim in shape:
                total_elements *= dim
            summary = ParameterGradientSummary(
                parameter_name=p_name,
                logical_stage=stage_name,
                depth_index=depth_idx,
                norm_l2=0.0,
                mean=0.0,
                std_dev=0.0,
                min_value=0.0,
                max_value=0.0,
                zero_fraction=1.0,
                is_finite=True,
                parameter_shape=shape,
                sample_count=total_elements,
            )
            param_summaries.append(summary)
            continue

        g_flat, g_shape = _flatten_and_measure(grads[g_key])
        total_count = len(g_flat)

        if total_count == 0:
            summary = ParameterGradientSummary(
                parameter_name=p_name,
                logical_stage=stage_name,
                depth_index=depth_idx,
                norm_l2=0.0,
                mean=0.0,
                std_dev=0.0,
                min_value=0.0,
                max_value=0.0,
                zero_fraction=1.0,
                is_finite=True,
                parameter_shape=g_shape,
                sample_count=0,
            )
            param_summaries.append(summary)
            continue

        # Check finiteness
        p_finite = True
        for v in g_flat:
            if math.isnan(v) or math.isinf(v):
                p_finite = False
                all_finite = False
                break

        if not p_finite:
            summary = ParameterGradientSummary(
                parameter_name=p_name,
                logical_stage=stage_name,
                depth_index=depth_idx,
                norm_l2=float("nan"),
                mean=float("nan"),
                std_dev=float("nan"),
                min_value=float("nan"),
                max_value=float("nan"),
                zero_fraction=0.0,
                is_finite=False,
                parameter_shape=g_shape,
                sample_count=total_count,
            )
            param_summaries.append(summary)
            continue

        # L2 norm, mean, variance, std_dev, min, max, zero fraction
        sum_g = 0.0
        sum_sq = 0.0
        min_g = g_flat[0]
        max_g = g_flat[0]
        zero_cnt = 0

        for v in g_flat:
            sum_g += v
            sum_sq += v * v
            if abs(v) <= zero_threshold:
                zero_cnt += 1
            if v < min_g:
                min_g = v
            if v > max_g:
                max_g = v

        norm_l2 = math.sqrt(max(0.0, sum_sq))
        mean_g = sum_g / float(total_count)
        var_g = sum((v - mean_g) ** 2 for v in g_flat) / float(total_count)
        std_g = math.sqrt(max(0.0, var_g))
        zero_frac = float(zero_cnt) / float(total_count)

        global_sum_sq += sum_sq

        summary = ParameterGradientSummary(
            parameter_name=p_name,
            logical_stage=stage_name,
            depth_index=depth_idx,
            norm_l2=norm_l2,
            mean=mean_g,
            std_dev=std_g,
            min_value=min_g,
            max_value=max_g,
            zero_fraction=zero_frac,
            is_finite=True,
            parameter_shape=g_shape,
            sample_count=total_count,
        )
        param_summaries.append(summary)

    global_norm = math.sqrt(max(0.0, global_sum_sq)) if all_finite else float("nan")

    return ModelGradientFlowSummary(
        model_id=model.model_id,
        total_parameters=len(params),
        total_gradients=len(grads),
        global_grad_norm_l2=global_norm,
        parameter_summaries=param_summaries,
        depth_ordering=ordered_keys,
        is_finite=all_finite,
    )


def compare_gradient_flow_summaries(
    summary_plain: ModelGradientFlowSummary,
    summary_residual: ModelGradientFlowSummary,
    eps: float = 1e-12,
) -> dict[str, Any]:
    """Compare gradient flow statistics between plain and residual models."""
    if not summary_plain.is_finite or not summary_residual.is_finite:
        return {
            "global_norm_plain": summary_plain.global_grad_norm_l2,
            "global_norm_residual": summary_residual.global_grad_norm_l2,
            "global_norm_ratio": float("nan"),
            "global_norm_delta": float("nan"),
            "is_finite": False,
        }

    p_norm = summary_plain.global_grad_norm_l2
    r_norm = summary_residual.global_grad_norm_l2

    ratio = r_norm / p_norm if p_norm > eps else float("nan")
    delta = r_norm - p_norm

    # Calculate early vs late gradient ratios if available
    def _early_late_ratio(summary: ModelGradientFlowSummary) -> float:
        conv_summaries = [
            s
            for s in summary.parameter_summaries
            if "conv" in s.parameter_name or "weights" in s.parameter_name
        ]
        if len(conv_summaries) < 2:
            return float("nan")
        early_norm = conv_summaries[0].norm_l2
        late_norm = conv_summaries[-1].norm_l2
        return early_norm / late_norm if late_norm > eps else float("nan")

    early_late_plain = _early_late_ratio(summary_plain)
    early_late_residual = _early_late_ratio(summary_residual)

    return {
        "global_norm_plain": p_norm,
        "global_norm_residual": r_norm,
        "global_norm_ratio": ratio,
        "global_norm_delta": delta,
        "early_to_late_ratio_plain": early_late_plain,
        "early_to_late_ratio_residual": early_late_residual,
        "is_finite": True,
    }
