"""Controlled data-budget subset manifests and nested subset generator."""

import random
from collections import Counter, defaultdict
from collections.abc import Sequence
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
    generate_subset_id,
)
from prism.data.partitions import PartitionManifest
from prism.data.samples import CanonicalSampleManifest, SampleRecord
from prism.experiments.hashing import compute_configuration_fingerprint

DEFAULT_DATA_BUDGETS: tuple[float, ...] = (0.01, 0.05, 0.10, 0.25, 0.50, 1.00)


class SubsetManifest(BaseModel):
    """Immutable manifest capturing exact samples allocated to a budget."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    subset_id: str = Field(
        description="Unique subset identifier (e.g. 'sub-cifar10-train-10pct')"
    )
    partition_id: str = Field(
        description="Referenced parent PartitionManifest identifier"
    )
    partition_fingerprint: str = Field(
        description="SHA-256 fingerprint of parent PartitionManifest"
    )
    target_split: str = Field(
        default="train",
        description="Partition split from which this subset was drawn",
    )
    budget_ratio: float = Field(
        gt=0.0,
        le=1.0,
        description="Fraction of parent split in (0.0, 1.0]",
    )
    budget_percentage: float = Field(
        gt=0.0,
        le=100.0,
        description="Percentage of the parent split (e.g. 10.0 for 0.10)",
    )
    sample_ids: list[str] = Field(
        description="Ordered list of unique sample IDs in this subset"
    )
    num_samples: int = Field(
        ge=1,
        description="Total sample count in this subset",
    )
    class_distribution: dict[str, int] = Field(
        default_factory=dict,
        description="Summary frequency mapping of class labels to counts",
    )
    selection_strategy: str = Field(
        default="nested_stratified",
        description="Selection strategy ('nested_stratified', 'nested_random')",
    )
    seed: int | None = Field(
        default=None,
        description="RNG seed used during ranking and selection",
    )
    schema_version: str = Field(
        default="1.0.0",
        description="Contract schema version",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Auxiliary generation metadata and split statistics",
    )

    @field_validator("subset_id")
    @classmethod
    def validate_subset_id(cls, v: str) -> str:
        return ensure_valid_identifier(v, field_name="subset_id")

    @field_validator("partition_id")
    @classmethod
    def validate_partition_id(cls, v: str) -> str:
        return ensure_valid_identifier(v, field_name="partition_id")

    @model_validator(mode="after")
    def validate_subset_integrity(self) -> "SubsetManifest":
        if len(self.sample_ids) != self.num_samples:
            raise ValidationError(
                f"Declared num_samples ({self.num_samples}) does not match "
                f"sample count ({len(self.sample_ids)}) in '{self.subset_id}'."
            )
        if len(self.sample_ids) != len(set(self.sample_ids)):
            raise ValidationError(
                f"Duplicate sample IDs detected in subset '{self.subset_id}'."
            )
        return self

    def compute_fingerprint(self) -> str:
        """Compute deterministic SHA-256 fingerprint of the subset manifest."""
        return compute_configuration_fingerprint(self.model_dump(mode="json"))

    def to_dict(self) -> dict[str, Any]:
        """Serialize subset manifest to a JSON dictionary."""
        return self.model_dump(mode="json")

    def to_json(self, indent: int | None = None) -> str:
        """Serialize subset manifest to a JSON string."""
        return self.model_dump_json(indent=indent)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SubsetManifest":
        """Deserialize a subset manifest from a dictionary."""
        try:
            return cls.model_validate(data)
        except Exception as exc:
            raise SerializationError(
                f"Failed to deserialize SubsetManifest from dict: {exc}"
            ) from exc

    @classmethod
    def from_json(cls, json_str: str) -> "SubsetManifest":
        """Deserialize a subset manifest from a JSON string."""
        try:
            return cls.model_validate_json(json_str)
        except Exception as exc:
            raise SerializationError(
                f"Failed to deserialize SubsetManifest from JSON: {exc}"
            ) from exc


def generate_nested_subsets(
    partition_manifest: PartitionManifest,
    canonical_manifest: CanonicalSampleManifest,
    budget_ratios: Sequence[float] = DEFAULT_DATA_BUDGETS,
    target_split: str = "train",
    seed: int = 42,
    strategy: str = "nested_stratified",
) -> dict[float, SubsetManifest]:
    """Generate strictly nested data-budget subsets for low-data studies.

    Algorithm:
    1. Filter canonical samples that belong to `target_split` in `partition_manifest`.
    2. Sort budget ratios in ascending order: b_1 < b_2 < ... < b_K <= 1.0.
    3. Generate a single master ranked sequence R = [s_1, s_2, ..., s_N]
       using deterministic per-class shuffling and round-robin class interleaving.
    4. For each budget ratio b_k, slice R[0 : round(N * b_k)].
    5. This guarantees mathematical subset nesting:
       S_{1%} ⊆ S_{5%} ⊆ S_{10%} ⊆ S_{25%} ⊆ S_{50%} ⊆ S_{100%} = ParentSplit.
    """
    if not budget_ratios:
        raise ValidationError("budget_ratios must contain at least one fraction.")

    for b in budget_ratios:
        if b <= 0.0 or b > 1.0:
            raise ValidationError(f"Budget ratios must be in (0.0, 1.0], got {b}")

    sorted_budgets = sorted(set(budget_ratios))
    split_record = partition_manifest.get_split(target_split)
    target_sample_id_set = set(split_record.sample_ids)

    # 1. Retrieve SampleRecord objects for target split
    split_samples: list[SampleRecord] = [
        s for s in canonical_manifest.samples if s.sample_id in target_sample_id_set
    ]

    if not split_samples:
        raise ValidationError(
            f"No samples found for target_split '{target_split}' in universe."
        )

    n_total = len(split_samples)
    rng = random.Random(seed)

    # 2. Construct unified ranked order R = [s_1, ..., s_N]
    if strategy == "nested_stratified":
        class_groups: dict[str, list[SampleRecord]] = defaultdict(list)
        for s in split_samples:
            key = str(s.target) if s.target is not None else "__none__"
            class_groups[key].append(s)

        # Deterministically shuffle each class group
        shuffled_groups: dict[str, list[SampleRecord]] = {}
        for class_key in sorted(class_groups.keys()):
            grp = list(class_groups[class_key])
            rng.shuffle(grp)
            shuffled_groups[class_key] = grp

        # Interleave classes round-robin to ensure balanced prefix representation
        ranked_sequence: list[SampleRecord] = []
        class_keys = sorted(shuffled_groups.keys())
        indices: dict[str, int] = dict.fromkeys(class_keys, 0)
        remaining = n_total

        while remaining > 0:
            for k in class_keys:
                if indices[k] < len(shuffled_groups[k]):
                    ranked_sequence.append(shuffled_groups[k][indices[k]])
                    indices[k] += 1
                    remaining -= 1
    else:
        # nested_random
        ranked_sequence = list(split_samples)
        rng.shuffle(ranked_sequence)

    # 3. Create nested prefixes
    partition_fp = partition_manifest.compute_fingerprint()
    results: dict[float, SubsetManifest] = {}

    for b in sorted_budgets:
        if abs(b - 1.0) < 1e-6:
            count = n_total
        else:
            count = max(1, round(n_total * b))
            count = min(count, n_total)

        selected_samples = ranked_sequence[:count]
        sample_ids = [s.sample_id for s in selected_samples]
        dist = Counter(str(s.target) for s in selected_samples if s.target is not None)
        pct = round(b * 100.0, 2)

        subset_id = generate_subset_id(
            prefix=f"sub-{partition_manifest.dataset_id}-{target_split}-{int(pct)}pct"
        )
        subset_manifest = SubsetManifest(
            subset_id=subset_id,
            partition_id=partition_manifest.partition_id,
            partition_fingerprint=partition_fp,
            target_split=target_split,
            budget_ratio=b,
            budget_percentage=pct,
            sample_ids=sample_ids,
            num_samples=len(sample_ids),
            class_distribution=dict(dist),
            selection_strategy=strategy,
            seed=seed,
            metadata={
                "parent_total_samples": n_total,
                "prefix_count": count,
            },
        )
        results[b] = subset_manifest

    return results
