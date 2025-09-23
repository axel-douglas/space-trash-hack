"""Tests for :mod:`app.modules.data_pipeline`."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.modules import data_pipeline


@pytest.fixture(autouse=True)
def _clean_ingestion_log() -> None:
    """Ensure the ingestion error log is removed before each test."""

    log_path = data_pipeline.INGESTION_ERROR_LOG_PATH
    if log_path.exists():
        log_path.unlink()
    if log_path.parent.exists() and not any(log_path.parent.iterdir()):
        log_path.parent.rmdir()
    yield
    if log_path.exists():
        log_path.unlink()


def test_load_inventory_logs_validation_error(tmp_path: Path) -> None:
    """Invalid inventory rows are skipped and appended to the error log."""

    csv_path = tmp_path / "inventory.csv"
    csv_path.write_text(
        """inventory_id,category,material_family,mass_kg,volume_l,flags
1,Metals,Ferrous,-1,10,
""",
        encoding="utf-8",
    )

    records = data_pipeline.load_inventory(csv_path)

    assert records == []

    log_path = data_pipeline.INGESTION_ERROR_LOG_PATH
    assert log_path.exists()

    with log_path.open("r", encoding="utf-8") as handle:
        entries = [json.loads(line) for line in handle if line.strip()]

    assert len(entries) == 1
    entry = entries[0]
    assert entry["source_file"] == str(csv_path)
    assert entry["row_index"] == 1
    assert entry["raw_entry"]["mass_kg"] == "-1"
    assert "mass kg" in entry["error"].lower()
