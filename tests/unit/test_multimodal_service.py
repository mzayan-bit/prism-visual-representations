"""Unit tests for multimodal API service."""

from __future__ import annotations

import json
from pathlib import Path

from prism.api.multimodal_service import MultimodalAlignmentService


def test_multimodal_alignment_service_payload(tmp_path: Path) -> None:
    """Verify service payload generation and export."""
    svc = MultimodalAlignmentService(seed=42)
    payload = svc.generate_benchmark_payload()

    assert payload["metadata"]["phase"] == 22
    assert "samples" in payload
    assert "retrieval_summary" in payload
    assert "zero_shot_summary" in payload
    assert "shared_geometry" in payload
    assert "robustness_benchmarks" in payload
    assert "objective_comparisons" in payload

    # Test file export
    out_file = tmp_path / "test_multimodal.json"
    svc.export_frontend_dataset(out_file)
    assert out_file.exists()

    with open(out_file, encoding="utf-8") as f:
        loaded = json.load(f)
    assert loaded["metadata"]["phase"] == 22
