"""Unit tests for TemporalRepresentationService and frontend benchmark export."""

import json
from pathlib import Path

from prism.api.temporal_service import TemporalRepresentationService


def test_temporal_service_payload_structure() -> None:
    service = TemporalRepresentationService(seed=42)
    payload = service.generate_benchmark_payload()

    assert "metadata" in payload
    assert "samples" in payload
    assert "objective_comparisons" in payload
    assert "layer_profiles" in payload
    assert "aggregator_comparisons" in payload
    assert "robustness_benchmarks" in payload
    assert "data_efficiency_curves" in payload
    assert "sequence_length_studies" in payload
    assert "candidate_failures" in payload

    assert len(payload["samples"]) == 16
    s0 = payload["samples"][0]
    assert "pca_trajectory" in s0
    assert "timeline_metrics" in s0
    assert "hidden_norms" in s0
    assert "attention_weights" in s0


def test_temporal_service_export_and_reload(tmp_path: Path) -> None:
    service = TemporalRepresentationService(seed=42)
    out_file = tmp_path / "test_temporal_dataset.json"
    service.export_frontend_dataset(out_file)

    assert out_file.exists()
    with open(out_file, encoding="utf-8") as f:
        data = json.load(f)

    assert data["metadata"]["phase"] == 21
    assert len(data["samples"]) == 16
