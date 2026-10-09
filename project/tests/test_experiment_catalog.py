import csv
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/build_all_experiment_data_catalog.py"
SPEC = importlib.util.spec_from_file_location("experiment_catalog", SCRIPT)
catalog = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(catalog)


@pytest.mark.parametrize("source_path", [
    "docs/audit/review.json",
    "docs/audit/nested/review.json",
    "docs/legacy_review.json",
])
def test_catalog_retains_audit_evidence_after_directory_changes(tmp_path, monkeypatch, source_path):
    monkeypatch.setattr(catalog, "ROOT", tmp_path)
    source = tmp_path / source_path
    source.parent.mkdir(parents=True)
    payload = {"valid_metrics": {"count": 1, "mae": 0.5, "acc2_nonzero": 1.0}}
    source.write_text(json.dumps(payload), encoding="utf-8")
    output = tmp_path / "outputs/catalog"

    first = catalog.build(output, None, "2026-10-02")
    with (output / "source_inventory.csv").open(encoding="utf-8", newline="") as stream:
        sources = list(csv.DictReader(stream))
    assert len(sources) == 1
    assert sources[0]["source_path"] == source_path
    assert sources[0]["sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    with (output / "artifact_inventory.csv").open(encoding="utf-8", newline="") as stream:
        artifacts = list(csv.DictReader(stream))
    assert any(row["artifact_path"] == source_path for row in artifacts)
    assert first["counts"]["metric_records"] == 1
    assert "docs/**/*.json" in first["source_roots"]

    second = catalog.build(output, None, "2026-10-02")
    assert second["counts"] == first["counts"]
