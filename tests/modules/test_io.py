"""Tests for the io module helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

import app.modules.io as io_module
from app.modules.io import (
    MissingDatasetError,
    format_missing_dataset_message,
    load_process_df,
    load_targets,
    load_waste_df,
)
from app.modules.problematic import problematic_mask


def test_problematic_column_matches_helper() -> None:
    df = load_waste_df()

    assert "_problematic" in df.columns, "load_waste_df should populate the _problematic column"

    helper_mask = problematic_mask(df)

    pd.testing.assert_series_equal(
        df["_problematic"],
        helper_mask,
        check_names=False,
    )


def test_load_waste_df_raises_missing_dataset(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    missing_path = tmp_path / "missing_waste.csv"
    monkeypatch.setattr(io_module, "WASTE_CSV", missing_path)
    io_module.invalidate_waste_cache()

    with pytest.raises(MissingDatasetError) as excinfo:
        load_waste_df()

    assert excinfo.value.path == missing_path
    assert str(missing_path) in format_missing_dataset_message(excinfo.value)


def test_load_process_df_raises_missing_dataset(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    missing_path = tmp_path / "missing_process.csv"
    monkeypatch.setattr(io_module, "PROC_CSV", missing_path)
    io_module.invalidate_process_cache()

    with pytest.raises(MissingDatasetError) as excinfo:
        load_process_df()

    assert excinfo.value.path == missing_path


def test_load_targets_raises_missing_dataset(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    missing_path = tmp_path / "missing_targets.json"
    monkeypatch.setattr(io_module, "TARGETS_JSON", missing_path)
    io_module.invalidate_targets_cache()

    with pytest.raises(MissingDatasetError) as excinfo:
        load_targets()

    assert excinfo.value.path == missing_path
