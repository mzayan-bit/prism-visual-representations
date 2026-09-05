"""Repeated-seed statistical aggregation and single-seed warning attachments."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

from prism.benchmarking.contracts import (
    AggregatedBenchmarkResult,
    BenchmarkResultCell,
)
from prism.benchmarking.enums import ResultStatus
from prism.core.errors import ValidationError


def aggregate_repeated_seeds(
    cells: Sequence[BenchmarkResultCell],
) -> AggregatedBenchmarkResult:
    """Aggregate a sequence of repeated-seed benchmark result cells.

    Parameters
    ----------
    cells : Sequence[BenchmarkResultCell]
        Non-empty list of benchmark result cells for the same metric and factors.

    Returns
    -------
    AggregatedBenchmarkResult
        Aggregated statistics with sample mean, std, min, max, median, and warnings.
    """
    if not cells:
        raise ValidationError("Cannot aggregate empty sequence of result cells.")

    metric_id = cells[0].metric_id
    group_factors = {k: v for k, v in cells[0].factors.items() if k != "seed"}
    member_ids = [c.result_id for c in cells]
    warnings: list[str] = []

    # Extract valid observed numeric values
    valid_values = [
        c.value
        for c in cells
        if c.status == ResultStatus.OBSERVED and c.value is not None
    ]
    n = len(valid_values)

    if n == 0:
        return AggregatedBenchmarkResult(
            metric_id=metric_id,
            group_factors=group_factors,
            sample_count=0,
            mean=None,
            std=None,
            min_value=None,
            max_value=None,
            median_value=None,
            member_result_ids=member_ids,
            warnings=["No observed numeric values available for aggregation."],
        )

    mean_val = sum(valid_values) / float(n)
    sorted_vals = sorted(valid_values)
    min_val = sorted_vals[0]
    max_val = sorted_vals[-1]

    # Median calculation
    if n % 2 == 1:
        median_val = sorted_vals[n // 2]
    else:
        median_val = (sorted_vals[n // 2 - 1] + sorted_vals[n // 2]) / 2.0

    # Standard deviation calculation (sample standard deviation N-1)
    if n == 1:
        std_val = None
        warnings.append(
            "SINGLE_SEED_RESULT: Evaluated on a single seed (N=1); "
            "standard deviation is undefined."
        )
    else:
        var = sum((x - mean_val) ** 2 for x in valid_values) / float(n - 1)
        std_val = math.sqrt(max(0.0, var))

    # Aggregate warnings from constituent cells
    for c in cells:
        for w in c.warnings:
            if w not in warnings:
                warnings.append(w)

    return AggregatedBenchmarkResult(
        metric_id=metric_id,
        group_factors=group_factors,
        sample_count=n,
        mean=mean_val,
        std=std_val,
        min_value=min_val,
        max_value=max_val,
        median_value=median_val,
        member_result_ids=member_ids,
        warnings=warnings,
    )


def group_and_aggregate(
    cells: Sequence[BenchmarkResultCell],
    group_by_factors: Sequence[str] = ("architecture", "pretraining_objective"),
) -> list[AggregatedBenchmarkResult]:
    """Group result cells by metric and experimental factors, then aggregate."""
    groups: dict[
        tuple[str, tuple[tuple[str, Any], ...]], list[BenchmarkResultCell]
    ] = {}

    for c in cells:
        factor_items = tuple(
            sorted(
                (k, v)
                for k, v in c.factors.items()
                if k in group_by_factors and k != "seed"
            )
        )
        group_key = (c.metric_id, factor_items)
        groups.setdefault(group_key, []).append(c)

    aggregated_results: list[AggregatedBenchmarkResult] = []
    for _, cell_list in groups.items():
        agg = aggregate_repeated_seeds(cell_list)
        aggregated_results.append(agg)

    return aggregated_results
