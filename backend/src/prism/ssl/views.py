"""Contrastive sample pairs, view generator, and deterministic batch contracts."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from prism.data.batching import DeterministicBatchLoader
from prism.data.materialized import MaterializedDataset, MaterializedSample
from prism.ssl.context import AugmentationContext
from prism.ssl.transforms import AugmentationPolicy, AugmentationTrace


class ContrastiveSamplePair(BaseModel):
    """Immutable paired augmented views generated from a single source image."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    pair_id: str = Field(..., description="Unique semantic identifier for this pair")
    source_sample_id: str = Field(..., description="Source sample identifier")
    epoch: int = Field(ge=0, description="Epoch for which pair was generated")
    view_a: list[list[list[float]]] = Field(..., description="First augmented 3D view")
    view_b: list[list[list[float]]] = Field(..., description="Second augmented 3D view")
    traces_a: list[AugmentationTrace] = Field(
        default_factory=list, description="Augmentation trace for view A"
    )
    traces_b: list[AugmentationTrace] = Field(
        default_factory=list, description="Augmentation trace for view B"
    )
    target_metadata: int | str | None = Field(
        default=None,
        description="Ground truth label retained for post-hoc analysis only",
    )


class ContrastiveViewGenerator:
    """Generates deterministic paired augmented views from materialized samples."""

    def __init__(
        self,
        policy: AugmentationPolicy | None = None,
        global_seed: int = 42,
    ) -> None:
        self.policy = policy or AugmentationPolicy()
        self.global_seed = global_seed

    def generate_pair(
        self, sample: MaterializedSample, epoch: int = 0
    ) -> ContrastiveSamplePair:
        """Produce two independently augmented views from a single sample."""
        ctx_a = AugmentationContext(
            global_seed=self.global_seed,
            sample_id=sample.sample_id,
            epoch=epoch,
            view_index=0,
            transform_index=0,
        )
        ctx_b = AugmentationContext(
            global_seed=self.global_seed,
            sample_id=sample.sample_id,
            epoch=epoch,
            view_index=1,
            transform_index=0,
        )

        view_a, traces_a = self.policy.apply(sample.data, ctx_a)
        view_b, traces_b = self.policy.apply(sample.data, ctx_b)

        pair_id = f"{sample.sample_id}::ep_{epoch}"

        return ContrastiveSamplePair(
            pair_id=pair_id,
            source_sample_id=sample.sample_id,
            epoch=epoch,
            view_a=view_a,
            view_b=view_b,
            traces_a=traces_a,
            traces_b=traces_b,
            target_metadata=sample.target,
        )


class ContrastiveBatch(BaseModel):
    """Deterministic batch of 2N contrastive views with positive mapping."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    batch_size: int = Field(ge=1, description="Number of distinct source samples (N)")
    views: list[list[list[list[float]]]] = Field(
        ...,
        description="Flattened list of 2N 3D image views: [v1_1, v2_1, ...]",
    )
    source_sample_ids: list[str] = Field(
        ..., description="List of N source sample identifiers"
    )
    positive_indices: list[int] = Field(
        ...,
        description="For view index i in [0..2N-1], positive pair partner index j",
    )
    target_labels: list[int | str | None] = Field(
        default_factory=list,
        description="Retained ground truth labels for post-hoc validation only",
    )

    @property
    def total_views(self) -> int:
        """Total number of contrastive views in batch (2N)."""
        return len(self.views)


class ContrastiveBatchLoader:
    """Wraps MaterializedDataset to emit ContrastiveBatch items."""

    def __init__(
        self,
        dataset: MaterializedDataset,
        batch_size: int = 8,
        seed: int = 42,
        policy: AugmentationPolicy | None = None,
    ) -> None:
        self.dataset = dataset
        self.batch_size = max(1, batch_size)
        self.seed = seed
        self.generator = ContrastiveViewGenerator(policy=policy, global_seed=seed)
        self.underlying_loader = DeterministicBatchLoader(
            dataset=dataset, batch_size=batch_size, seed=seed
        )

    def get_batches(self, epoch: int = 0) -> list[ContrastiveBatch]:
        """Produce deterministic ContrastiveBatches for a given epoch."""
        self.underlying_loader.set_epoch(epoch)
        contrastive_batches: list[ContrastiveBatch] = []

        for raw_b in self.underlying_loader:
            sample_map = {s.sample_id: s for s in self.dataset.samples}
            pairs: list[ContrastiveSamplePair] = []

            for s_id in raw_b.sample_ids:
                if s_id in sample_map:
                    sample = sample_map[s_id]
                    pair = self.generator.generate_pair(sample, epoch=epoch)
                    pairs.append(pair)

            if not pairs:
                continue

            n = len(pairs)
            # Flatten into [v1_0, v2_0, v1_1, v2_1, ..., v1_{N-1}, v2_{N-1}]
            views: list[list[list[list[float]]]] = []
            pos_indices: list[int] = []
            source_ids: list[str] = []
            targets: list[int | str | None] = []

            for idx, pair in enumerate(pairs):
                # view_a is at 2*idx, view_b is at 2*idx + 1
                views.append(pair.view_a)
                views.append(pair.view_b)
                pos_indices.append(2 * idx + 1)
                pos_indices.append(2 * idx)
                source_ids.append(pair.source_sample_id)
                targets.append(pair.target_metadata)
                targets.append(pair.target_metadata)

            contrastive_batches.append(
                ContrastiveBatch(
                    batch_size=n,
                    views=views,
                    source_sample_ids=source_ids,
                    positive_indices=pos_indices,
                    target_labels=targets,
                )
            )

        return contrastive_batches
