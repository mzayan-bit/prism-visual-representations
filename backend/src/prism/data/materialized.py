"""Executable materialized sample and dataset abstractions."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, overload

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
)

from prism.core.errors import ValidationError
from prism.core.identifiers import ensure_valid_identifier


class MaterializedSample(BaseModel):
    """Immutable runtime representation of an executable data example."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        arbitrary_types_allowed=True,
    )

    sample_id: str = Field(
        description="Globally unique sample ID (e.g. 'cifar10/train/000042')"
    )
    source_split: str = Field(
        description="Original source partition from provider (e.g. 'train')"
    )
    source_index: int = Field(
        ge=0,
        description="Zero-indexed position within provider source split",
    )
    data: Any = Field(description="Raw or preprocessed data payload")
    target: int | str | None = Field(
        default=None,
        description="Ground-truth category label or index",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Auxiliary sample metadata",
    )

    @field_validator("sample_id", "source_split")
    @classmethod
    def validate_non_empty(cls, v: str, info: Any) -> str:
        if not v or not v.strip():
            raise ValueError(f"{info.field_name} must not be empty.")
        return v.strip()


class MaterializedDataset(Sequence[MaterializedSample]):
    """Executable in-memory dataset providing indexed access to samples."""

    def __init__(
        self,
        dataset_id: str,
        samples: Sequence[MaterializedSample],
        split_name: str | None = None,
        transform: Callable[[Any], Any] | None = None,
        target_transform: Callable[[Any], Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.dataset_id = ensure_valid_identifier(
            dataset_id, field_name="dataset_id"
        )
        self.split_name = split_name.strip().lower() if split_name else None
        self._samples: list[MaterializedSample] = list(samples)
        self.transform = transform
        self.target_transform = target_transform
        self.metadata = metadata or {}

        # Index by sample_id for rapid deterministic lookups
        self._sample_id_to_index: dict[str, int] = {}
        for idx, sample in enumerate(self._samples):
            if sample.sample_id in self._sample_id_to_index:
                raise ValidationError(
                    f"Duplicate sample_id '{sample.sample_id}' in dataset."
                )
            self._sample_id_to_index[sample.sample_id] = idx

    def __len__(self) -> int:
        return len(self._samples)

    @overload
    def __getitem__(self, index: int) -> MaterializedSample: ...

    @overload
    def __getitem__(self, index: slice) -> MaterializedDataset: ...

    def __getitem__(
        self, index: int | slice
    ) -> MaterializedSample | MaterializedDataset:
        if isinstance(index, slice):
            sliced_samples = self._samples[index]
            return MaterializedDataset(
                dataset_id=self.dataset_id,
                samples=sliced_samples,
                split_name=self.split_name,
                transform=self.transform,
                target_transform=self.target_transform,
                metadata=self.metadata,
            )

        if index < 0 or index >= len(self._samples):
            raise IndexError(
                f"Index {index} out of bounds for dataset of length {len(self)}."
            )

        sample = self._samples[index]
        transformed_data = (
            self.transform(sample.data) if self.transform else sample.data
        )
        transformed_target = (
            self.target_transform(sample.target)
            if self.target_transform
            else sample.target
        )

        return MaterializedSample(
            sample_id=sample.sample_id,
            source_split=sample.source_split,
            source_index=sample.source_index,
            data=transformed_data,
            target=transformed_target,
            metadata=sample.metadata,
        )

    def get_sample(self, sample_id: str) -> MaterializedSample:
        """Retrieve a materialized sample by its unique sample ID."""
        if sample_id not in self._sample_id_to_index:
            raise KeyError(
                f"Sample '{sample_id}' not found in MaterializedDataset."
            )
        idx = self._sample_id_to_index[sample_id]
        sample = self[idx]
        assert isinstance(sample, MaterializedSample)
        return sample

    @property
    def sample_ids(self) -> list[str]:
        """Return the ordered list of all sample identifiers."""
        return [s.sample_id for s in self._samples]

    @property
    def targets(self) -> list[int | str | None]:
        """Return the ordered list of all sample targets."""
        return [s.target for s in self._samples]

    def with_transform(
        self,
        transform: Callable[[Any], Any] | None = None,
        target_transform: Callable[[Any], Any] | None = None,
    ) -> MaterializedDataset:
        """Return a new MaterializedDataset with updated transform pipelines."""
        return MaterializedDataset(
            dataset_id=self.dataset_id,
            samples=self._samples,
            split_name=self.split_name,
            transform=transform or self.transform,
            target_transform=target_transform or self.target_transform,
            metadata=self.metadata,
        )
