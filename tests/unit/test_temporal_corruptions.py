"""Unit tests for deterministic temporal corruptions and perturbations."""

from prism.core.enums import SplitName
from prism.temporal.contracts import VideoSample
from prism.temporal.corruptions import (
    apply_frame_drop,
    apply_frame_duplication,
    apply_frame_shuffle,
    apply_spatial_composite,
    apply_temporal_corruption,
    apply_temporal_subsampling,
)
from prism.temporal.enums import TemporalCorruptionType
from prism.temporal.synthetic import SyntheticVideoGenerator


def _create_sample() -> tuple[SyntheticVideoGenerator, VideoSample]:
    gen = SyntheticVideoGenerator(num_frames=4, height=12, width=12, seed=42)
    samples = gen.generate_dataset(num_samples=1, split=SplitName.TRAIN)
    return gen, samples[0]


def test_frame_drop_corruption() -> None:
    _, sample = _create_sample()
    corrupted, dropped = apply_frame_drop(sample, drop_fraction=0.5, seed=42)

    assert corrupted.frame_count < sample.frame_count
    assert len(dropped) > 0
    assert corrupted.metadata["corruption"] == "frame_drop"

    # Verify remaining frames preserve relative order
    original_ids = sample.frame_ids
    remaining_ids = corrupted.frame_ids
    orig_indices = [original_ids.index(fid) for fid in remaining_ids]
    assert orig_indices == sorted(orig_indices)


def test_frame_duplication_corruption() -> None:
    _, sample = _create_sample()
    corrupted = apply_frame_duplication(sample, dup_index=1, dup_count=2)

    id_1 = corrupted.frame_ids[1].split("_dup")[0]
    id_2 = corrupted.frame_ids[2].split("_dup")[0]
    assert id_1 == id_2


def test_frame_shuffle_corruption() -> None:
    _, sample = _create_sample()
    corrupted, perm = apply_frame_shuffle(sample, seed=42)

    assert corrupted.frame_count == sample.frame_count
    assert corrupted.metadata["corruption"] == "frame_shuffle"
    assert len(perm) == sample.frame_count
    assert sorted(perm) == list(range(sample.frame_count))


def test_temporal_subsampling_corruption() -> None:
    gen = SyntheticVideoGenerator(num_frames=8, height=12, width=12, seed=42)
    sample = gen.generate_dataset(num_samples=1, split=SplitName.TRAIN)[0]

    corrupted = apply_temporal_subsampling(sample, stride=2)
    assert corrupted.frame_count == 4
    assert corrupted.frame_ids[0] == sample.frame_ids[0]
    assert corrupted.frame_ids[1] == sample.frame_ids[2]
    assert corrupted.frame_ids[2] == sample.frame_ids[4]
    assert corrupted.frame_ids[3] == sample.frame_ids[6]


def test_spatial_composite_corruption() -> None:
    _, sample = _create_sample()
    corrupted = apply_spatial_composite(sample, noise_level=0.2, seed=42)

    assert corrupted.frame_count == sample.frame_count
    assert corrupted.frame_shape == sample.frame_shape
    assert corrupted.frame_tensors != sample.frame_tensors


def test_corruption_dispatcher() -> None:
    _, sample = _create_sample()
    c_sample, meta = apply_temporal_corruption(
        sample,
        TemporalCorruptionType.FRAME_DROP,
        drop_fraction=0.5,
    )
    assert c_sample.frame_count < sample.frame_count
    assert "dropped_frame_ids" in meta
