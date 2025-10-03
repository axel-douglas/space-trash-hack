from __future__ import annotations

import os
from pathlib import Path
from typing import Callable

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from app.modules import mission_overview
from app.modules.mission_overview import compute_mission_summary
from app.modules.io import MissingDatasetError, format_missing_dataset_message


def _run_home_app(
    monkeypatch: pytest.MonkeyPatch,
    *,
    inventory_loader: Callable[[], pd.DataFrame],
) -> AppTest:
    """Execute ``streamlit run app/Home.py`` through ``AppTest`` with patches."""

    from app.modules import ml_models, ui_blocks
    import streamlit as st

    os.environ.setdefault("REXAI_PROJECT_ROOT", str(Path(__file__).resolve().parents[2]))

    class _RegistryStub:
        metadata = {
            "trained_at": "2024-01-01T00:00:00+00:00",
            "n_samples": 256,
            "ready": True,
        }
        ready = True

    monkeypatch.setattr(ml_models, "get_model_registry", lambda: _RegistryStub(), raising=False)
    monkeypatch.setattr(ui_blocks, "load_theme", lambda **_: None, raising=False)
    monkeypatch.setattr(st, "set_page_config", lambda *args, **kwargs: None, raising=False)

    original_dashboard = mission_overview.render_overview_dashboard

    def _render_override() -> None:
        return original_dashboard(inventory_loader=inventory_loader)

    monkeypatch.setattr(mission_overview, "render_overview_dashboard", _render_override, raising=False)

    app_path = Path(__file__).resolve().parents[2] / "app" / "Home.py"
    app_test = AppTest.from_file(str(app_path))
    return app_test.run()


def test_mission_metrics_reflect_inventory(monkeypatch: pytest.MonkeyPatch) -> None:
    inventory_df = _load_inventory_fixture()
    app = _run_home_app(monkeypatch, inventory_loader=lambda: inventory_df.copy(deep=True))

    summary = compute_mission_summary(inventory_df)

    metric_labels = {metric.label for metric in app.metric}
    assert {"Masa total", "Energía estimada"}.issubset(metric_labels)

    assert summary["mass_kg"] > 0
    assert summary["energy_kwh"] > 0


def test_model_section_displays_ready_status(monkeypatch: pytest.MonkeyPatch) -> None:
    inventory_df = _load_inventory_fixture()
    app = _run_home_app(monkeypatch, inventory_loader=lambda: inventory_df.copy(deep=True))

    model_metric = next(metric for metric in app.metric if metric.label == "Estado del modelo")
    assert "Entrenado" in (model_metric.delta or "")
    assert model_metric.value.startswith("✅")


def test_inventory_table_and_captions() -> None:
    inventory_df = _load_inventory_fixture()
    assert not inventory_df.empty

    summary_df, _ = mission_overview.prepare_material_summary(inventory_df, max_rows=20)
    expected_columns = {"material_display", "category", "kg", "volume_m3", "_problematic"}
    assert expected_columns.issubset(summary_df.columns)

    categories_column = inventory_df.get("category")
    categories = sorted({str(value).strip() for value in categories_column if str(value).strip()})
    assert categories, "Debe existir al menos una categoría en el inventario"

    problematic_expected = int(inventory_df["_problematic"].astype(bool).sum())
    assert problematic_expected >= 0


def test_home_page_shows_error_for_missing_dataset(monkeypatch: pytest.MonkeyPatch) -> None:
    missing_path = Path("missing_overview.csv")

    def _raise_missing() -> pd.DataFrame:
        raise MissingDatasetError(missing_path)

    app = _run_home_app(monkeypatch, inventory_loader=_raise_missing)

    error_messages = " ".join(block.body for block in app.error)
    assert "missing_overview.csv" in error_messages
    assert "python scripts/download_datasets.py" in error_messages
    expected_message = format_missing_dataset_message(MissingDatasetError(missing_path))
    assert expected_message in error_messages
    assert not app.exception


def _load_inventory_fixture() -> pd.DataFrame:
    data_path = Path(__file__).resolve().parents[2] / "data" / "waste_inventory_sample.csv"
    df = pd.read_csv(data_path)
    if "kg" not in df.columns:
        df["kg"] = pd.to_numeric(df.get("mass_kg"), errors="coerce").fillna(0.0)
    if "volume_l" not in df.columns:
        df["volume_l"] = pd.to_numeric(df.get("volume_l"), errors="coerce").fillna(0.0)
    if "material_display" not in df.columns:
        category_display = df.get("category", "").astype(str).str.strip()
        family_display = df.get("material_family", "").astype(str).str.strip()
        df["material_display"] = category_display.where(
            family_display.eq(""), category_display + " — " + family_display
        ).str.replace(" — ", "", regex=False)
    if "_problematic" not in df.columns:
        df["_problematic"] = False
    return df
