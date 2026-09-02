"""Unit tests for contrastive view generation, pair contracts, and batch loading."""

from prism.data.materialized import MaterializedDataset, MaterializedSample
from prism.ssl.views import ContrastiveBatchLoader, ContrastiveViewGenerator


def _make_sample(sample_id: str, label: int = 1) -> MaterializedSample:
    c, h, w = 3, 8, 8
    img = [[[float(label) * 0.2 for _ in range(w)] for _ in range(h)] for _ in range(c)]
    return MaterializedSample(
        sample_id=sample_id,
        source_split="train",
        source_index=0,
        data=img,
        target=label,
    )


def test_contrastive_view_generator() -> None:
    """Test paired view generation from a single sample."""
    sample = _make_sample("s_100", label=2)
    gen = ContrastiveViewGenerator(global_seed=42)

    pair = gen.generate_pair(sample, epoch=0)
    assert pair.source_sample_id == "s_100"
    assert pair.epoch == 0
    assert len(pair.view_a) == 3
    assert len(pair.view_b) == 3
    assert len(pair.traces_a) > 0
    assert len(pair.traces_b) > 0
    assert pair.target_metadata == 2


def test_contrastive_batch_loader() -> None:
    """Test that ContrastiveBatchLoader emits 2N views with exact positive mapping."""
    samples = [_make_sample(f"sample_{i}", label=i % 3) for i in range(6)]
    dataset = MaterializedDataset(
        dataset_id="test_ssl_ds",
        split_name="train",
        samples=samples,
    )

    loader = ContrastiveBatchLoader(dataset=dataset, batch_size=4, seed=42)
    batches = loader.get_batches(epoch=0)

    assert len(batches) > 0
    first_b = batches[0]
    assert first_b.batch_size == 4
    assert first_b.total_views == 8  # 2N = 8

    # Positive pairing verification:
    # 0 <-> 1, 2 <-> 3, 4 <-> 5, 6 <-> 7
    for i in range(8):
        pos_partner = first_b.positive_indices[i]
        assert first_b.positive_indices[pos_partner] == i
