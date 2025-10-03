"""Tests covering high level IO helpers for NASA datasets."""

from __future__ import annotations

import json

from app.modules import io


def test_load_process_df_only_requires_process_dataset(tmp_path, monkeypatch):
    process_path = tmp_path / "process_catalog.csv"
    process_path.write_text("id,name\nP1,Demo\n", encoding="utf-8")

    monkeypatch.setattr(io, "PROC_CSV", process_path)
    monkeypatch.setattr(io, "WASTE_CSV", tmp_path / "waste_inventory_sample.csv")
    monkeypatch.setattr(io, "TARGETS_JSON", tmp_path / "targets_presets.json")

    io.invalidate_all_io_caches()

    frame = io.load_process_df()

    assert list(frame.columns) == ["id", "name"]
    assert frame.iloc[0]["id"] == "P1"
    assert frame.iloc[0]["name"] == "Demo"

    io.invalidate_all_io_caches()


def test_load_targets_only_requires_targets_dataset(tmp_path, monkeypatch):
    targets_path = tmp_path / "targets_presets.json"
    targets_path.write_text(json.dumps([{"id": "alpha"}]), encoding="utf-8")

    monkeypatch.setattr(io, "TARGETS_JSON", targets_path)
    monkeypatch.setattr(io, "WASTE_CSV", tmp_path / "waste_inventory_sample.csv")
    monkeypatch.setattr(io, "PROC_CSV", tmp_path / "process_catalog.csv")

    io.invalidate_all_io_caches()

    payload = io.load_targets()

    assert payload == [{"id": "alpha"}]

    io.invalidate_all_io_caches()
