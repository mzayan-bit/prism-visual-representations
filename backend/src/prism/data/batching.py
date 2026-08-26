"""Deterministic batch loading and batch traceability abstractions."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
)

from prism.core.enums import OrderingStrategy
from prism.core.errors import ValidationError
from prism.data.materialized import MaterializedDataset, MaterializedSample
from prism.data.ordering import (
    compute_ordering_fingerprint,
    compute_sample_order,
)


class MaterializedBatch(BaseModel):
    """Immutable batch payload maintaining sample identity traceability."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        arbitrary_types_allowed=True,
    )

    batch_index: int = Field(
        ge=0,
        description="Zero-indexed position of batch within the epoch",
    )
    sample_ids: list[str] = Field(
        description="Traceable list of canonical sample IDs in batch"
    )
    data: Any = Field(
        description="Batch data payload (list of arrays, stacked tensors, etc.)"
    )
    targets: Any = Field(
        default=None,
        description="Batch targets / labels",
    )
    batch_size: int = Field(
        ge=1,
        description="Number of samples contained in this batch",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Auxiliary batch metadata",
    )


CollateFunction = Callable[[list[MaterializedSample], int], MaterializedBatch]


def default_collate_fn(
    samples: list[MaterializedSample],
    batch_index: int,
) -> MaterializedBatch:
    """Default collate function preserving payload list and sample IDs."""
    sample_ids = [s.sample_id for s in samples]
    data_list = [s.data for s in samples]
    target_list = [s.target for s in samples]

    return MaterializedBatch(
        batch_index=batch_index,
        sample_ids=sample_ids,
        data=data_list,
        targets=target_list,
        batch_size=len(samples),
        metadata={},
    )


class DeterministicBatchLoader:
    """Deterministic batch iterator for MaterializedDatasets."""

    def __init__(
        self,
        dataset: MaterializedDataset,
        batch_size: int = 32,
        ordering_strategy: OrderingStrategy | str = OrderingStrategy.SEQUENTIAL,
        seed: int | None = 42,
        epoch: int = 0,
        drop_last: bool = False,
        collate_fn: CollateFunction | None = None,
    ) -> None:
        if batch_size <= 0:
            raise ValidationError(f"batch_size must be positive, got {batch_size}")

        self.dataset = dataset
        self.batch_size = batch_size
        self.ordering_strategy = (
            OrderingStrategy(ordering_strategy.lower())
            if isinstance(ordering_strategy, str)
            else ordering_strategy
        )
        self.seed = seed
        self.epoch = epoch
        self.drop_last = drop_last
        self.collate_fn = collate_fn or default_collate_fn

    def set_epoch(self, epoch: int) -> None:
        """Update the active epoch for epoch-aware deterministic shuffles."""
        if epoch < 0:
            raise ValidationError(f"epoch must be non-negative, got {epoch}")
        self.epoch = epoch

    def __len__(self) -> int:
        n = len(self.dataset)
        if n == 0:
            return 0
        if self.drop_last:
            return n // self.batch_size
        return (n + self.batch_size - 1) // self.batch_size

    def get_ordering_fingerprint(self) -> str:
        """Compute the ordering fingerprint for the current configuration."""
        return compute_ordering_fingerprint(
            sample_ids=self.dataset.sample_ids,
            strategy=self.ordering_strategy,
            seed=self.seed,
            epoch=self.epoch,
        )

    def __iter__(self) -> Iterator[MaterializedBatch]:
        n = len(self.dataset)
        if n == 0:
            return

        order_indices = compute_sample_order(
            sample_ids=self.dataset.sample_ids,
            strategy=self.ordering_strategy,
            seed=self.seed,
            epoch=self.epoch,
        )

        slices = [
            order_indices[start : start + self.batch_size]
            for start in range(0, n, self.batch_size)
        ]

        for batch_idx, batch_slice in enumerate(slices):
            if len(batch_slice) < self.batch_size and self.drop_last:
                break

            samples = [self.dataset[i] for i in batch_slice]
            assert all(isinstance(s, MaterializedSample) for s in samples)
            yield self.collate_fn(samples, batch_idx)
