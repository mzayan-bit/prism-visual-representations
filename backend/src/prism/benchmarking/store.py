"""In-memory benchmark result store with deduplication and provenance indexing."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from prism.benchmarking.adapters import adapt_any_report
from prism.benchmarking.contracts import BenchmarkResultCell
from prism.benchmarking.enums import ResultStatus
from prism.core.errors import ValidationError


class BenchmarkResultStore:
    """In-memory repository for canonical benchmark cells with strict deduplication."""

    def __init__(self) -> None:
        self._cells: dict[str, BenchmarkResultCell] = {}
        # Composite identity index: identity_key -> result_id
        self._identity_index: dict[str, str] = {}

    def _compute_identity_key(self, cell: BenchmarkResultCell) -> str:
        """Compute strict provenance identity key for deduplication."""
        exp_fp = cell.experiment_fingerprint or cell.experiment_id
        metric = cell.metric_id
        seed_val = str(cell.seed) if cell.seed is not None else "no_seed"
        factors_str = "_".join(
            f"{k}={v}" for k, v in sorted(cell.factors.items()) if k != "seed"
        )
        return f"{exp_fp}::{metric}::{seed_val}::{factors_str}"

    def register_cell(self, cell: BenchmarkResultCell) -> None:
        """Register a benchmark result cell with deduplication and conflict checking."""
        identity_key = self._compute_identity_key(cell)

        if identity_key in self._identity_index:
            existing_id = self._identity_index[identity_key]
            existing_cell = self._cells[existing_id]

            # If values match (or both None), deduplicate silently
            if (
                existing_cell.value == cell.value
                and existing_cell.status == cell.status
            ):
                return

            # Conflicting value detected for identical provenance
            raise ValidationError(
                f"RESULT_PROVENANCE_CONFLICT: Conflicting metric value for "
                f"identity '{identity_key}'. "
                f"Existing: {existing_cell.value} ({existing_cell.status}), "
                f"Incoming: {cell.value} ({cell.status})."
            )

        self._cells[cell.result_id] = cell
        self._identity_index[identity_key] = cell.result_id

    def bulk_register(self, cells: Sequence[BenchmarkResultCell]) -> None:
        """Register multiple cells in bulk."""
        for c in cells:
            self.register_cell(c)

    def register_report(self, report: Any) -> list[BenchmarkResultCell]:
        """Adapt and register all cells from a supported PRISM report."""
        cells = adapt_any_report(report)
        self.bulk_register(cells)
        return cells

    def get(self, result_id: str) -> BenchmarkResultCell | None:
        """Retrieve a result cell by unique result ID."""
        return self._cells.get(result_id)

    def get_all(self) -> list[BenchmarkResultCell]:
        """Return all registered benchmark cells."""
        return list(self._cells.values())

    def count(self) -> int:
        """Total number of registered result cells."""
        return len(self._cells)

    def query(
        self,
        metric_id: str | None = None,
        experiment_id: str | None = None,
        architecture: str | None = None,
        pretraining_objective: str | None = None,
        task: str | None = None,
        seed: int | None = None,
        status: ResultStatus | None = None,
        factor_filters: dict[str, Any] | None = None,
    ) -> list[BenchmarkResultCell]:
        """Query benchmark result cells by metrics, factors, or lifecycle status."""
        results: list[BenchmarkResultCell] = []

        for cell in self._cells.values():
            if metric_id is not None and cell.metric_id != metric_id:
                continue
            if experiment_id is not None and cell.experiment_id != experiment_id:
                continue
            if status is not None and cell.status != status:
                continue
            if seed is not None and cell.seed != seed:
                continue

            factors = cell.factors
            if architecture is not None and factors.get("architecture") != architecture:
                continue
            if (
                pretraining_objective is not None
                and factors.get("pretraining_objective") != pretraining_objective
            ):
                continue
            if task is not None and factors.get("task") != task:
                continue

            if factor_filters:
                match = True
                for k, v in factor_filters.items():
                    if factors.get(k) != v:
                        match = False
                        break
                if not match:
                    continue

            results.append(cell)

        return results

    def list_provenance(self, result_id: str) -> dict[str, Any] | None:
        """Get complete provenance metadata for a result cell."""
        cell = self.get(result_id)
        if cell is None:
            return None
        return {
            "result_id": cell.result_id,
            "experiment_id": cell.experiment_id,
            "experiment_fingerprint": cell.experiment_fingerprint,
            "metric_id": cell.metric_id,
            "value": cell.value,
            "status": cell.status.value,
            "seed": cell.seed,
            "source_report_type": cell.source_report_type,
            "source_run_id": cell.source_run_id,
            "source_artifact": cell.source_artifact,
            "factors": cell.factors,
            "warnings": cell.warnings,
            "provenance": cell.provenance,
        }
