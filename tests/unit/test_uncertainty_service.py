"""Unit tests for UncertaintyAnalysisService and payload export."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from prism.api.uncertainty_service import UncertaintyAnalysisService


def test_uncertainty_service_generate_payload() -> None:
    """Verify UncertaintyAnalysisService produces valid benchmark payload structure."""
    service = UncertaintyAnalysisService(seed=42)
    payload = service.generate_benchmark_payload()

    assert "meta" in payload
    assert payload["meta"]["phase"] == 23
    assert "report" in payload
    assert "samples" in payload
    assert len(payload["samples"]) > 0
    assert "objective_comparisons" in payload
    assert "architecture_comparisons" in payload
    assert "reference_set" in payload


def test_uncertainty_service_export_dataset() -> None:
    """Verify UncertaintyAnalysisService exports JSON dataset to filesystem."""
    service = UncertaintyAnalysisService(seed=42)
    with tempfile.TemporaryDirectory() as tmpdir:
        out_file = Path(tmpdir) / "test_dataset.json"
        service.export_frontend_dataset(out_file)

        assert out_file.exists()
        with open(out_file, encoding="utf-8") as f:
            data = json.load(f)
        assert data["meta"]["phase"] == 23
        assert len(data["samples"]) > 0
