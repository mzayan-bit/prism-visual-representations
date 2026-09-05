"""2D Benchmark matrices and scientific comparison tables."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from prism.benchmarking.contracts import (
    AggregatedBenchmarkResult,
    BenchmarkMatrix,
    BenchmarkResultCell,
    BenchmarkTable,
    MetricDefinition,
)
from prism.benchmarking.enums import (
    ComparisonControlStatus,
    MetricDirection,
    ResultStatus,
)
from prism.benchmarking.store import BenchmarkResultStore


def build_benchmark_matrix(
    store: BenchmarkResultStore,
    row_factor: str,
    column_factor: str,
    metric_id: str,
    row_values: Sequence[str] | None = None,
    column_values: Sequence[str] | None = None,
    fixed_filters: dict[str, Any] | None = None,
    title: str | None = None,
) -> BenchmarkMatrix:
    """Construct a 2D matrix of benchmark results across two factor dimensions."""
    all_cells = store.query(metric_id=metric_id, factor_filters=fixed_filters)

    # Determine unique row and column values if not explicitly provided
    r_vals = list(row_values) if row_values is not None else []
    c_vals = list(column_values) if column_values is not None else []

    if not r_vals or not c_vals:
        found_r: set[str] = set()
        found_c: set[str] = set()
        for cell in all_cells:
            r_val = cell.factors.get(row_factor)
            c_val = cell.factors.get(column_factor)
            if r_val is not None:
                found_r.add(str(r_val))
            if c_val is not None:
                found_c.add(str(c_val))
        if not r_vals:
            r_vals = sorted(found_r)
        if not c_vals:
            c_vals = sorted(found_c)

    # Initialize 2D grid
    grid: dict[
        str, dict[str, BenchmarkResultCell | AggregatedBenchmarkResult | None]
    ] = {r: dict.fromkeys(c_vals) for r in r_vals}

    # Group cells by (row_val, col_val)
    cell_groups: dict[tuple[str, str], list[BenchmarkResultCell]] = {}
    for cell in all_cells:
        r_val = str(cell.factors.get(row_factor, ""))
        c_val = str(cell.factors.get(column_factor, ""))
        if r_val in r_vals and c_val in c_vals:
            cell_groups.setdefault((r_val, c_val), []).append(cell)

    # Populate grid
    for r in r_vals:
        for c in c_vals:
            members = cell_groups.get((r, c), [])
            if len(members) == 1:
                grid[r][c] = members[0]
            elif len(members) > 1:
                # Store first or aggregated cell
                grid[r][c] = members[0]
            else:
                grid[r][c] = None

    if title is not None:
        mat_title = title
    else:
        metric_name = metric_id.replace("_", " ").title()
        mat_title = f"{metric_name} by {row_factor.title()} vs {column_factor.title()}"

    return BenchmarkMatrix(
        matrix_id=f"mat_{row_factor}_{column_factor}_{metric_id}",
        title=mat_title,
        row_factor=row_factor,
        column_factor=column_factor,
        metric_id=metric_id,
        row_values=r_vals,
        column_values=c_vals,
        cells=grid,
        warnings=[],
    )


def build_benchmark_table(
    matrix: BenchmarkMatrix,
    metric_def: MetricDefinition | None = None,
    title: str | None = None,
    research_question_id: str | None = None,
) -> BenchmarkTable:
    """Format a BenchmarkMatrix into a structured scientific BenchmarkTable."""
    rows: list[dict[str, Any]] = []
    footnotes: list[str] = []

    unit = metric_def.unit if metric_def else ""
    direction = metric_def.direction if metric_def else MetricDirection.HIGHER_IS_BETTER

    if metric_def and metric_def.methodological_notes:
        footnotes.append(metric_def.methodological_notes)

    for r_val in matrix.row_values:
        row_data: dict[str, Any] = {matrix.row_factor: r_val}
        for c_val in matrix.column_values:
            cell = matrix.cells.get(r_val, {}).get(c_val)
            if cell is None:
                row_data[c_val] = {
                    "value": None,
                    "status": ResultStatus.MISSING.value,
                    "display": "MISSING",
                    "seed_count": 0,
                }
            elif isinstance(cell, BenchmarkResultCell):
                if cell.status == ResultStatus.OBSERVED and cell.value is not None:
                    row_data[c_val] = {
                        "value": cell.value,
                        "status": cell.status.value,
                        "display": f"{cell.value:.4f}",
                        "seed_count": 1,
                        "warnings": cell.warnings,
                    }
                else:
                    row_data[c_val] = {
                        "value": None,
                        "status": cell.status.value,
                        "display": cell.status.value.upper(),
                        "seed_count": 0,
                    }
            elif isinstance(cell, AggregatedBenchmarkResult):
                if cell.mean is not None:
                    disp = f"{cell.mean:.4f}"
                    if cell.std is not None:
                        disp += f" ± {cell.std:.4f}"
                    row_data[c_val] = {
                        "value": cell.mean,
                        "std": cell.std,
                        "status": ResultStatus.AGGREGATED.value,
                        "display": disp,
                        "seed_count": cell.sample_count,
                        "warnings": cell.warnings,
                    }
                else:
                    row_data[c_val] = {
                        "value": None,
                        "status": ResultStatus.MISSING.value,
                        "display": "MISSING",
                        "seed_count": 0,
                    }

        rows.append(row_data)

    tbl_title = title if title is not None else matrix.title

    return BenchmarkTable(
        table_id=f"tbl_{matrix.matrix_id}",
        title=tbl_title,
        research_question_id=research_question_id,
        row_factor=matrix.row_factor,
        column_factor=matrix.column_factor,
        metric_id=matrix.metric_id,
        unit=unit,
        metric_direction=direction,
        rows=rows,
        footnotes=footnotes,
        control_status=ComparisonControlStatus.STRICTLY_CONTROLLED,
        warnings=matrix.warnings,
    )
