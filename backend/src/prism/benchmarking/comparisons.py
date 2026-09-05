"""Factor control audits and direction-aware pairwise comparisons."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from prism.benchmarking.contracts import (
    BenchmarkResultCell,
    ComparisonControlAudit,
    MetricDefinition,
    PairwiseComparisonResult,
)
from prism.benchmarking.enums import ComparisonControlStatus, MetricDirection


def audit_comparison_control(
    factors_a: dict[str, Any],
    factors_b: dict[str, Any],
    expected_equal: Sequence[str] | None = None,
) -> ComparisonControlAudit:
    """Audit factor consistency between two benchmark candidates."""
    required = list(
        expected_equal
        if expected_equal is not None
        else ["dataset", "task", "data_budget", "seed"]
    )

    equal_factors: list[str] = []
    mismatches: dict[str, tuple[Any, Any]] = {}
    warnings: list[str] = []

    for f in required:
        val_a = factors_a.get(f)
        val_b = factors_b.get(f)
        if val_a == val_b and val_a is not None:
            equal_factors.append(f)
        else:
            mismatches[f] = (val_a, val_b)
            warnings.append(
                f"Controlled factor mismatch for '{f}': A='{val_a}' vs B='{val_b}'."
            )

    if not mismatches:
        status = ComparisonControlStatus.STRICTLY_CONTROLLED
    elif len(mismatches) == 1 and "seed" in mismatches:
        status = ComparisonControlStatus.PARTIALLY_CONTROLLED
        warnings.append(
            "Comparison evaluates different RNG seeds; cross-seed variance applies."
        )
    elif len(mismatches) <= 2:
        status = ComparisonControlStatus.DESCRIPTIVE_ONLY
        warnings.append(
            "Multiple controlled factors differ; comparison is descriptive only."
        )
    else:
        status = ComparisonControlStatus.INVALID_COMPARISON
        warnings.append("Severe factor divergence; scientific comparison is invalid.")

    return ComparisonControlAudit(
        comparison_id=f"audit_{abs(hash(str(factors_a) + str(factors_b))) % 1000000}",
        factors_expected_equal=required,
        factors_actually_equal=equal_factors,
        mismatches=mismatches,
        status=status,
        warnings=warnings,
    )


def compute_pairwise_comparison(
    cell_a: BenchmarkResultCell,
    cell_b: BenchmarkResultCell,
    metric_def: MetricDefinition | None = None,
    expected_equal_factors: Sequence[str] | None = None,
) -> PairwiseComparisonResult:
    """Compute structured descriptive comparison between two benchmark result cells."""
    val_a = cell_a.value
    val_b = cell_b.value
    metric_id = cell_a.metric_id

    direction = (
        metric_def.direction
        if metric_def is not None
        else MetricDirection.HIGHER_IS_BETTER
    )

    audit = audit_comparison_control(
        factors_a=cell_a.factors,
        factors_b=cell_b.factors,
        expected_equal=expected_equal_factors,
    )

    abs_delta: float | None = None
    pct_delta: float | None = None
    favorable: str | None = None
    interpretation: str = ""

    if val_a is not None and val_b is not None:
        abs_delta = val_b - val_a
        pct_delta = ((val_b - val_a) / abs(val_a) * 100.0) if val_a != 0.0 else None

        if direction == MetricDirection.HIGHER_IS_BETTER:
            if val_b > val_a:
                favorable = "B"
            elif val_a > val_b:
                favorable = "A"
        elif direction == MetricDirection.LOWER_IS_BETTER:
            if val_b < val_a:
                favorable = "B"
            elif val_a < val_b:
                favorable = "A"
        else:
            favorable = None

        name_a = cell_a.factors.get(
            "architecture", cell_a.factors.get("pretraining_objective", "A")
        )
        name_b = cell_b.factors.get(
            "architecture", cell_b.factors.get("pretraining_objective", "B")
        )

        interpretation = (
            f"{name_b} achieved {val_b:.4f} compared to {name_a} at {val_a:.4f} "
            f"(absolute delta {abs_delta:+.4f}) on {metric_id}."
        )
    else:
        interpretation = f"Incomplete data for comparison on {metric_id}."

    return PairwiseComparisonResult(
        comparison_id=f"pair_{cell_a.result_id}_vs_{cell_b.result_id}",
        metric_id=metric_id,
        cell_a_id=cell_a.result_id,
        cell_b_id=cell_b.result_id,
        value_a=val_a,
        value_b=val_b,
        absolute_delta=abs_delta,
        percentage_delta=pct_delta,
        direction=direction,
        favorable_candidate=favorable,
        control_audit=audit,
        descriptive_interpretation=interpretation,
    )
