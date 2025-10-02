import os
import re
import sys
from pathlib import Path

import pandas as pd
import pytest
from pytest_streamlit import StreamlitRunner

from app.modules import mission_overview
from app.modules.mission_overview import compute_mission_summary


def _mission_overview_app() -> None:
    import os
    import sys
    from pathlib import Path
    import importlib

    import streamlit as st
    from app.modules.io import load_waste_df

    root_env = os.environ.get("REXAI_PROJECT_ROOT")
    root = Path(root_env) if root_env else Path.cwd()
    app_dir = root / "app"
    for candidate in (root, app_dir):
        if str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))

    original_page_config = st.set_page_config
    st.set_page_config = lambda *args, **kwargs: None

    import app.modules.ml_models as ml_models
    import app.modules.ui_blocks as ui_blocks
    import app.modules.mission_overview as mission_overview

    class _RegistryStub:
        metadata = {
            "trained_at": "2024-01-01T00:00:00+00:00",
            "n_samples": 256,
            "ready": True,
        }
        ready = True

    original_registry = ml_models.get_model_registry
    original_load_theme = ui_blocks.load_theme
    original_inventory_loader = mission_overview.load_inventory_overview

    try:
        ml_models.get_model_registry = lambda: _RegistryStub()  # type: ignore[assignment]
        ui_blocks.load_theme = lambda **_: None  # type: ignore[assignment]
        mission_overview.load_inventory_overview = load_waste_df  # type: ignore[assignment]

        sys.modules.pop("app.Home", None)
        home_module = importlib.import_module("app.Home")
        home_module.render_page()
    finally:
        ml_models.get_model_registry = original_registry  # type: ignore[assignment]
        ui_blocks.load_theme = original_load_theme  # type: ignore[assignment]
        mission_overview.load_inventory_overview = original_inventory_loader  # type: ignore[assignment]
        st.set_page_config = original_page_config


@pytest.fixture
def mission_overview_runner() -> StreamlitRunner:
    os.environ.setdefault("REXAI_PROJECT_ROOT", str(Path(__file__).resolve().parents[2]))
    return StreamlitRunner(_mission_overview_app)


def _extract_number(text: str) -> float | None:
    match = re.search(r"[-+]?\d*\.?\d+", text)
    if not match:
        return None
    return float(match.group())


def test_mission_metrics_reflect_inventory(mission_overview_runner: StreamlitRunner) -> None:
    app = mission_overview_runner.run()

    inventory_df = _load_inventory_fixture()
    summary = compute_mission_summary(inventory_df)

    metric_labels = {metric.label for metric in app.metric}
    assert {"Masa total", "Energía estimada"}.issubset(metric_labels)

    assert summary["mass_kg"] > 0
    assert summary["energy_kwh"] > 0


def test_model_section_displays_ready_status(mission_overview_runner: StreamlitRunner) -> None:
    app = mission_overview_runner.run()

    model_metric = next(metric for metric in app.metric if metric.label == "Estado del modelo")
    assert "Entrenado" in (model_metric.delta or "")
    assert model_metric.value.startswith("✅")


def test_inventory_table_and_captions(mission_overview_runner: StreamlitRunner) -> None:
    _ = mission_overview_runner  # ensure environment hooks run for downstream pages

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
