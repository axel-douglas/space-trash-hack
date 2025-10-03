"""Tests for the io module helpers."""

from __future__ import annotations

from pathlib import Path
from datetime import datetime

import pandas as pd
import pytest
import polars as pl

import app.modules.io as io_module
from app.modules.dataset_validation import InvalidWasteDatasetError
from app.modules.io import (
    MissingDatasetError,
    format_missing_dataset_message,
    get_last_modified,
    load_process_df,
    load_targets,
    load_waste_df,
    save_waste_df,
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


def test_get_last_modified_returns_timestamp(tmp_path: Path) -> None:
    file_path = tmp_path / "timestamp.txt"
    file_path.write_text("sample")

    timestamp = get_last_modified(file_path)

    assert isinstance(timestamp, datetime)
    assert timestamp is not None
    assert pytest.approx(file_path.stat().st_mtime, rel=1e-3) == timestamp.timestamp()


def test_get_last_modified_missing_path(tmp_path: Path) -> None:
    missing_path = tmp_path / "missing.txt"

    assert get_last_modified(missing_path) is None


def _write_waste_csv(path: Path, *, mass: float = 1.0) -> None:
    path.write_text(
        """id,category,material_family,mass_kg,volume_l,flags
W1,Plastic,Polymer,{mass},3.5,safe
""".format(mass=mass),
        encoding="utf-8",
    )


def _prepare_waste_csv(monkeypatch: pytest.MonkeyPatch, path: Path) -> None:
    monkeypatch.setattr(io_module, "WASTE_CSV", path)
    monkeypatch.setattr(io_module, "official_features_bundle", lambda: None)
    io_module.invalidate_waste_cache()


def test_load_waste_df_missing_required_column(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    waste_path = tmp_path / "waste_inventory_sample.csv"
    waste_path.write_text(
        """id,category,material_family,volume_l,flags
W1,Plastic,Polymer,3.5,safe
""",
        encoding="utf-8",
    )

    _prepare_waste_csv(monkeypatch, waste_path)

    with pytest.raises(InvalidWasteDatasetError) as excinfo:
        load_waste_df()

    assert "Faltan columnas obligatorias" in str(excinfo.value)


def test_load_waste_df_invalid_numeric_values(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    waste_path = tmp_path / "waste_inventory_sample.csv"
    _write_waste_csv(waste_path, mass=-2.0)

    _prepare_waste_csv(monkeypatch, waste_path)

    with pytest.raises(InvalidWasteDatasetError) as excinfo:
        load_waste_df()

    assert "valores inválidos" in str(excinfo.value)


def test_load_waste_df_valid_dataset(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    waste_path = tmp_path / "waste_inventory_sample.csv"
    _write_waste_csv(waste_path, mass=4.0)

    _prepare_waste_csv(monkeypatch, waste_path)

    frame = load_waste_df()

    assert not frame.empty
    assert float(frame.loc[frame.index[0], "kg"]) == pytest.approx(4.0)

    io_module.invalidate_waste_cache()


def test_save_waste_df_creates_backup(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    waste_path = tmp_path / "waste_inventory_sample.csv"
    _write_waste_csv(waste_path, mass=1.5)

    _prepare_waste_csv(monkeypatch, waste_path)

    backup_path = waste_path.with_suffix(waste_path.suffix + ".bak")

    data = pd.DataFrame(
        [
            {
                "id": "W9",
                "category": "Metal",
                "material_family": "Al",
                "mass_kg": 3.25,
                "volume_l": 0.8,
                "flags": "safe",
            }
        ]
    )

    save_waste_df(data)

    assert waste_path.read_text(encoding="utf-8").count("3.25") == 1
    assert backup_path.exists()
    assert "1.5" in backup_path.read_text(encoding="utf-8")


def test_save_waste_df_failure_preserves_original(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    waste_path = tmp_path / "waste_inventory_sample.csv"
    _write_waste_csv(waste_path, mass=2.75)

    _prepare_waste_csv(monkeypatch, waste_path)

    def _boom(*_args, **_kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(pl.DataFrame, "write_csv", _boom)

    data = pd.DataFrame(
        [
            {
                "id": "W1",
                "category": "Plastic",
                "material_family": "Polymer",
                "mass_kg": 9.0,
                "volume_l": 1.0,
                "flags": "updated",
            }
        ]
    )

    with pytest.raises(RuntimeError):
        save_waste_df(data)

    assert waste_path.read_text(encoding="utf-8").count("2.75") == 1
    assert not waste_path.with_suffix(waste_path.suffix + ".bak").exists()
