"""Deterministic dataset partitioning and fixed partition manifest contracts."""

import random
from collections import Counter, defaultdict
from collections.abc import Mapping
from typing import Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from prism.core.errors import SerializationError, ValidationError
from prism.core.identifiers import (
    ensure_valid_identifier,
    generate_partition_id,
)
from prism.data.samples import CanonicalSampleManifest, SampleRecord
from prism.experiments.hashing import compute_configuration_fingerprint


class PartitionSplit(BaseModel):
    """Immutable record capturing exact samples assigned to a named partition."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    split_name: str = Field(
        description="Partition split name (e.g. 'train', 'val', 'test')"
    )
    sample_ids: list[str] = Field(
        description="Ordered list of unique sample identifiers"
    )
    num_samples: int = Field(ge=0, description="Total sample count in this split")
    class_distribution: dict[str, int] = Field(
        default_factory=dict,
        description="Summary frequency mapping of class labels to counts",
    )

    @field_validator("split_name")
    @classmethod
    def validate_split_name(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Split name must not be empty.")
        return v.strip().lower()

    @model_validator(mode="after")
    def validate_sample_count_and_uniqueness(self) -> "PartitionSplit":
        if len(self.sample_ids) != self.num_samples:
            raise ValidationError(
                f"Declared num_samples ({self.num_samples}) does not match "
                f"sample count ({len(self.sample_ids)}) for '{self.split_name}'."
            )
        if len(self.sample_ids) != len(set(self.sample_ids)):
            raise ValidationError(
                f"Duplicate sample IDs detected within split '{self.split_name}'."
            )
        return self


class PartitionManifest(BaseModel):
    """Immutable assignment of canonical samples into mutually exclusive splits."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    partition_id: str = Field(
        description="Unique partition identifier (e.g. 'part-a1b2c3d4e5f6')"
    )
    dataset_id: str = Field(description="Referenced DatasetManifest identifier")
    canonical_manifest_fingerprint: str = Field(
        description="SHA-256 fingerprint of source CanonicalSampleManifest"
    )
    splits: dict[str, PartitionSplit] = Field(
        description="Mapping of split names to PartitionSplit records"
    )
    total_samples: int = Field(
        ge=0,
        description="Sum of samples across all partition splits",
    )
    seed: int | None = Field(
        default=None,
        description="RNG seed used to generate this partition if randomized",
    )
    strategy: str = Field(
        default="stratified",
        description="Partitioning strategy ('stratified', 'random')",
    )
    generation_metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Auxiliary generation parameters and audit notes",
    )
    schema_version: str = Field(
        default="1.0.0",
        description="Contract schema version",
    )

    @field_validator("partition_id")
    @classmethod
    def validate_partition_id(cls, v: str) -> str:
        return ensure_valid_identifier(v, field_name="partition_id")

    @field_validator("dataset_id")
    @classmethod
    def validate_dataset_id(cls, v: str) -> str:
        return ensure_valid_identifier(v, field_name="dataset_id")

    @model_validator(mode="after")
    def validate_mutual_exclusivity(self) -> "PartitionManifest":
        """Verify splits are mutually exclusive and count is consistent."""
        if not self.splits:
            raise ValidationError("PartitionManifest must contain at least one split.")

        seen_samples: dict[str, str] = {}
        computed_total = 0

        for split_name, split_obj in self.splits.items():
            computed_total += split_obj.num_samples
            for sample_id in split_obj.sample_ids:
                if sample_id in seen_samples:
                    other_split = seen_samples[sample_id]
                    raise ValidationError(
                        f"Sample '{sample_id}' appears in multiple splits: "
                        f"'{other_split}' and '{split_name}'."
                    )
                seen_samples[sample_id] = split_name

        if computed_total != self.total_samples:
            raise ValidationError(
                f"Declared total_samples ({self.total_samples}) does not match "
                f"sum of splits ({computed_total})."
            )

        return self

    def compute_fingerprint(self) -> str:
        """Compute deterministic SHA-256 fingerprint of the partition manifest."""
        return compute_configuration_fingerprint(self.model_dump(mode="json"))

    def get_split(self, split_name: str) -> PartitionSplit:
        """Lookup a PartitionSplit by split name."""
        normalized = split_name.strip().lower()
        if normalized not in self.splits:
            raise KeyError(
                f"Split '{split_name}' not found in partition manifest. "
                f"Available splits: {list(self.splits.keys())}"
            )
        return self.splits[normalized]

    def validate_against_canonical(self, canonical: CanonicalSampleManifest) -> None:
        """Validate all partition samples exist in canonical universe."""
        canonical_fingerprint = canonical.compute_fingerprint()
        if self.canonical_manifest_fingerprint != canonical_fingerprint:
            raise ValidationError(
                f"Partition fingerprint '{self.canonical_manifest_fingerprint}' "
                f"mismatches actual canonical fingerprint '{canonical_fingerprint}'."
            )

        canonical_ids = {s.sample_id for s in canonical.samples}
        for split_name, split_obj in self.splits.items():
            missing = [sid for sid in split_obj.sample_ids if sid not in canonical_ids]
            if missing:
                raise ValidationError(
                    f"Split '{split_name}' references {len(missing)} missing "
                    f"sample(s). First missing: {missing[0]}"
                )

    def to_dict(self) -> dict[str, Any]:
        """Serialize the partition manifest to a JSON dictionary."""
        return self.model_dump(mode="json")

    def to_json(self, indent: int | None = None) -> str:
        """Serialize the partition manifest to a JSON string."""
        return self.model_dump_json(indent=indent)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PartitionManifest":
        """Deserialize a partition manifest from a dictionary."""
        try:
            return cls.model_validate(data)
        except Exception as exc:
            raise SerializationError(
                f"Failed to deserialize PartitionManifest from dict: {exc}"
            ) from exc

    @classmethod
    def from_json(cls, json_str: str) -> "PartitionManifest":
        """Deserialize a partition manifest from a JSON string."""
        try:
            return cls.model_validate_json(json_str)
        except Exception as exc:
            raise SerializationError(
                f"Failed to deserialize PartitionManifest from JSON: {exc}"
            ) from exc


def generate_partition_manifest(
    canonical_manifest: CanonicalSampleManifest,
    split_ratios: Mapping[str, float],
    seed: int = 42,
    strategy: str = "stratified",
    partition_id: str | None = None,
    source_split_filter: str | None = "train",
    isolated_splits: Mapping[str, str] | None = None,
) -> PartitionManifest:
    """Generate a deterministic PartitionManifest from a canonical sample universe.

    Args:
        canonical_manifest: Complete ordered CanonicalSampleManifest.
        split_ratios: Relative partition ratios (e.g. {"train": 0.9, "val": 0.1}).
                      Must sum to 1.0 within floating point tolerance.
        seed: RNG seed for local deterministic shuffling.
        strategy: Partitioning algorithm ('stratified' or 'random').
        partition_id: Optional explicit partition ID.
        source_split_filter: If set, only partitions samples from this source split.
        isolated_splits: Mapping of {target_split: source_split} for untouched splits.

    Returns:
        Deterministic PartitionManifest instance.
    """
    if not split_ratios:
        raise ValidationError(
            "split_ratios must contain at least one split specification."
        )

    total_ratio = sum(split_ratios.values())
    if abs(total_ratio - 1.0) > 1e-5:
        raise ValidationError(
            f"Split ratios must sum to 1.0, got {total_ratio} for {split_ratios}"
        )

    # 1. Select target samples to partition
    if source_split_filter:
        samples_to_partition = canonical_manifest.filter_by_source_split(
            source_split_filter
        )
    else:
        samples_to_partition = list(canonical_manifest.samples)

    if not samples_to_partition:
        raise ValidationError(
            f"No samples found for partitioning with filter='{source_split_filter}'."
        )

    # 2. Organize samples by class for stratification
    rng = random.Random(seed)
    split_names = list(split_ratios.keys())
    split_buckets: dict[str, list[SampleRecord]] = {name: [] for name in split_names}

    if strategy == "stratified":
        class_groups: dict[str, list[SampleRecord]] = defaultdict(list)
        for s in samples_to_partition:
            key = str(s.target) if s.target is not None else "__none__"
            class_groups[key].append(s)

        # Process each class group deterministically
        for class_key in sorted(class_groups.keys()):
            group = list(class_groups[class_key])
            # Deterministic shuffle within class
            rng.shuffle(group)

            n_group = len(group)
            offset = 0
            for i, name in enumerate(split_names):
                ratio = split_ratios[name]
                if i == len(split_names) - 1:
                    # Last split receives remaining samples
                    count = n_group - offset
                else:
                    count = round(n_group * ratio)
                    count = min(count, n_group - offset)

                assigned = group[offset : offset + count]
                split_buckets[name].extend(assigned)
                offset += count
    else:
        # Pure random partition
        shuffled = list(samples_to_partition)
        rng.shuffle(shuffled)
        n_total = len(shuffled)
        offset = 0
        for i, name in enumerate(split_names):
            ratio = split_ratios[name]
            if i == len(split_names) - 1:
                count = n_total - offset
            else:
                count = round(n_total * ratio)
                count = min(count, n_total - offset)

            assigned = shuffled[offset : offset + count]
            split_buckets[name].extend(assigned)
            offset += count

    # 3. Construct PartitionSplit objects
    splits: dict[str, PartitionSplit] = {}
    for name in split_names:
        assigned_samples = split_buckets[name]
        sample_ids = [s.sample_id for s in assigned_samples]
        dist = Counter(str(s.target) for s in assigned_samples if s.target is not None)
        splits[name.lower()] = PartitionSplit(
            split_name=name.lower(),
            sample_ids=sample_ids,
            num_samples=len(sample_ids),
            class_distribution=dict(dist),
        )

    # 4. Attach isolated direct splits if requested (e.g. official test split)
    if isolated_splits:
        for target_split_name, src_split_name in isolated_splits.items():
            direct_samples = canonical_manifest.filter_by_source_split(src_split_name)
            direct_ids = [s.sample_id for s in direct_samples]
            dist = Counter(
                str(s.target) for s in direct_samples if s.target is not None
            )
            splits[target_split_name.lower()] = PartitionSplit(
                split_name=target_split_name.lower(),
                sample_ids=direct_ids,
                num_samples=len(direct_ids),
                class_distribution=dict(dist),
            )

    total_samples = sum(s.num_samples for s in splits.values())
    part_id = partition_id or generate_partition_id()

    return PartitionManifest(
        partition_id=part_id,
        dataset_id=canonical_manifest.dataset_id,
        canonical_manifest_fingerprint=canonical_manifest.compute_fingerprint(),
        splits=splits,
        total_samples=total_samples,
        seed=seed,
        strategy=strategy,
        generation_metadata={
            "split_ratios": dict(split_ratios),
            "source_split_filter": source_split_filter,
            "isolated_splits": (dict(isolated_splits) if isolated_splits else {}),
        },
    )
