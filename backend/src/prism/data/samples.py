"""Canonical sample identity and universe manifest contracts."""

from collections import Counter
from typing import Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from prism.core.errors import SerializationError, ValidationError
from prism.core.identifiers import ensure_valid_identifier
from prism.experiments.hashing import compute_configuration_fingerprint


class SampleRecord(BaseModel):
    """Immutable record capturing the canonical identity of a single dataset sample."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    sample_id: str = Field(
        description="Globally stable unique sample ID (e.g. 'cifar10/train/000042')"
    )
    source_split: str = Field(
        description="Original source partition from provider (e.g. 'train')"
    )
    source_index: int = Field(
        ge=0,
        description="Zero-indexed position within provider source split",
    )
    target: int | str | None = Field(
        default=None,
        description="Ground-truth category label or target annotation",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Optional auxiliary attributes (e.g. filename)",
    )

    @field_validator("sample_id", "source_split")
    @classmethod
    def validate_non_empty(cls, v: str, info: Any) -> str:
        if not v or not v.strip():
            raise ValueError(f"{info.field_name} must not be empty.")
        return v.strip()


class CanonicalSampleManifest(BaseModel):
    """Immutable, ordered collection representing the entire sample universe."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    dataset_id: str = Field(description="Referenced DatasetManifest identifier")
    dataset_version: str = Field(
        default="1.0.0",
        description="Semantic version of the underlying dataset",
    )
    samples: list[SampleRecord] = Field(
        description="Canonical ordered sequence of all sample records"
    )
    num_samples: int = Field(
        ge=0,
        description="Total sample count in this canonical universe",
    )
    class_distribution: dict[str, int] = Field(
        default_factory=dict,
        description="Summary frequency mapping of class labels to counts",
    )
    schema_version: str = Field(
        default="1.0.0",
        description="Schema contract version",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Provenance and generator metadata",
    )

    @field_validator("dataset_id")
    @classmethod
    def validate_dataset_id(cls, v: str) -> str:
        return ensure_valid_identifier(v, field_name="dataset_id")

    @model_validator(mode="after")
    def validate_sample_integrity(self) -> "CanonicalSampleManifest":
        """Verify sample count, uniqueness, and ordering stability."""
        if len(self.samples) != self.num_samples:
            raise ValidationError(
                f"Declared num_samples ({self.num_samples}) does not match "
                f"actual sample count ({len(self.samples)})."
            )

        seen_ids: set[str] = set()
        seen_source_coords: set[tuple[str, int]] = set()

        for s in self.samples:
            if s.sample_id in seen_ids:
                raise ValidationError(
                    f"Duplicate sample_id '{s.sample_id}' in universe."
                )
            seen_ids.add(s.sample_id)

            coord = (s.source_split, s.source_index)
            if coord in seen_source_coords:
                raise ValidationError(
                    f"Duplicate source coordinate ({s.source_split}, "
                    f"{s.source_index}) for sample '{s.sample_id}'."
                )
            seen_source_coords.add(coord)

        return self

    def compute_fingerprint(self) -> str:
        """Compute deterministic SHA-256 fingerprint of the sample universe."""
        return compute_configuration_fingerprint(self.model_dump(mode="json"))

    def get_sample(self, sample_id: str) -> SampleRecord:
        """Lookup a sample record by its unique ID."""
        for s in self.samples:
            if s.sample_id == sample_id:
                return s
        raise KeyError(f"Sample '{sample_id}' not found in canonical universe.")

    def filter_by_source_split(self, source_split: str) -> list[SampleRecord]:
        """Return all samples originating from a specific source partition."""
        normalized = source_split.strip().lower()
        return [s for s in self.samples if s.source_split.lower() == normalized]

    def to_dict(self) -> dict[str, Any]:
        """Serialize the canonical sample manifest to a JSON dictionary."""
        return self.model_dump(mode="json")

    def to_json(self, indent: int | None = None) -> str:
        """Serialize the canonical sample manifest to a JSON string."""
        return self.model_dump_json(indent=indent)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CanonicalSampleManifest":
        """Deserialize a canonical sample manifest from a dictionary."""
        try:
            return cls.model_validate(data)
        except Exception as exc:
            raise SerializationError(
                f"Failed to deserialize CanonicalSampleManifest: {exc}"
            ) from exc

    @classmethod
    def from_json(cls, json_str: str) -> "CanonicalSampleManifest":
        """Deserialize a canonical sample manifest from a JSON string."""
        try:
            return cls.model_validate_json(json_str)
        except Exception as exc:
            raise SerializationError(
                f"Failed to deserialize CanonicalSampleManifest: {exc}"
            ) from exc

    @classmethod
    def create(
        cls,
        dataset_id: str,
        samples: list[SampleRecord],
        dataset_version: str = "1.0.0",
        metadata: dict[str, Any] | None = None,
    ) -> "CanonicalSampleManifest":
        """Helper constructor computing count and distribution automatically."""
        dist: Counter[str] = Counter()
        for s in samples:
            if s.target is not None:
                dist[str(s.target)] += 1

        return cls(
            dataset_id=dataset_id,
            dataset_version=dataset_version,
            samples=samples,
            num_samples=len(samples),
            class_distribution=dict(dist),
            metadata=metadata or {},
        )
