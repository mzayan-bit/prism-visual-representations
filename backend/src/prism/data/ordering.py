"""Deterministic data ordering strategies and ordering fingerprints."""

from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Sequence
from typing import Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
)

from prism.core.enums import OrderingStrategy


class OrderingSpecification(BaseModel):
    """Declarative specification for deterministic sample ordering."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    strategy: OrderingStrategy = Field(
        default=OrderingStrategy.SEQUENTIAL,
        description="Sampling or traversal ordering strategy",
    )
    seed: int | None = Field(
        default=42,
        description="Base pseudo-random seed for shuffle strategies",
    )
    epoch: int = Field(
        default=0,
        ge=0,
        description="Epoch index for epoch-aware deterministic shuffles",
    )

    @field_validator("strategy", mode="before")
    @classmethod
    def parse_strategy(cls, v: Any) -> OrderingStrategy:
        if isinstance(v, str):
            return OrderingStrategy(v.lower())
        if isinstance(v, OrderingStrategy):
            return v
        raise ValueError(f"Invalid strategy value: {v}")


def compute_sample_order(
    sample_ids: Sequence[str],
    strategy: OrderingStrategy | str = OrderingStrategy.SEQUENTIAL,
    seed: int | None = 42,
    epoch: int = 0,
) -> list[int]:
    """Compute deterministic index permutation for a sequence of sample IDs.

    Guarantees:
    - SEQUENTIAL: returns [0, 1, ..., N-1] in identical manifest order.
    - FIXED_SHUFFLE: deterministic shuffle seeded by `seed`.
    - EPOCH_AWARE_SHUFFLE: deterministic shuffle seeded by (seed, epoch).
    - Does NOT mutate or rely on global random state.
    """
    strat = (
        OrderingStrategy(strategy.lower()) if isinstance(strategy, str) else strategy
    )
    n = len(sample_ids)
    indices = list(range(n))

    if strat == OrderingStrategy.SEQUENTIAL:
        return indices

    if strat == OrderingStrategy.FIXED_SHUFFLE:
        rng = random.Random(seed if seed is not None else 0)
        rng.shuffle(indices)
        return indices

    if strat == OrderingStrategy.EPOCH_AWARE_SHUFFLE:
        # Deterministically blend base seed and epoch
        base = seed if seed is not None else 0
        epoch_seed = ((base * 1000003) ^ (epoch * 10007)) & 0x7FFFFFFF
        rng = random.Random(epoch_seed)
        rng.shuffle(indices)
        return indices

    return indices


def compute_ordering_fingerprint(
    sample_ids: Sequence[str],
    strategy: OrderingStrategy | str = OrderingStrategy.SEQUENTIAL,
    seed: int | None = 42,
    epoch: int = 0,
) -> str:
    """Compute a deterministic SHA-256 fingerprint for a specific data ordering."""
    strat = (
        OrderingStrategy(strategy.lower()) if isinstance(strategy, str) else strategy
    )
    ordered_indices = compute_sample_order(
        sample_ids=sample_ids,
        strategy=strat,
        seed=seed,
        epoch=epoch,
    )
    ordered_ids = [sample_ids[i] for i in ordered_indices]

    fingerprint_dict = {
        "ordered_sample_ids": ordered_ids,
        "strategy": strat.value,
        "seed": seed,
        "epoch": epoch,
        "num_samples": len(ordered_ids),
    }

    canonical_json = json.dumps(
        fingerprint_dict,
        sort_keys=True,
        ensure_ascii=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()
