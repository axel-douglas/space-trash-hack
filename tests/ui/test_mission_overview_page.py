import importlib
import os
import re
import sys
from pathlib import Path

import pytest

pytest.importorskip("streamlit")

from pytest_streamlit import StreamlitRunner

from app.modules.io import load_waste_df
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

        sys.modules.pop("app.pages.0_Mission_Overview", None)
        importlib.import_module("app.pages.0_Mission_Overview")
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

    inventory_df = load_waste_df()
    summary = compute_mission_summary(inventory_df)

    mass_metric = next(metric for metric in app.metric if metric.label == "Masa total")
    mass_value = _extract_number(mass_metric.value)
    assert mass_value is not None
    expected_mass = summary["mass_kg"] / (1000.0 if "t" in mass_metric.value else 1.0)
    assert mass_value == pytest.approx(expected_mass, rel=0.05)

    energy_metric = next(metric for metric in app.metric if metric.label == "Energía estimada")
    energy_value = _extract_number(energy_metric.value)
    assert energy_value is not None
    expected_energy = summary["energy_kwh"] / (1000.0 if "MWh" in energy_metric.value else 1.0)
    assert energy_value == pytest.approx(expected_energy, rel=0.05)


def test_model_section_displays_ready_status(mission_overview_runner: StreamlitRunner) -> None:
    app = mission_overview_runner.run()

    model_metric = next(metric for metric in app.metric if metric.label == "Estado del modelo")
    assert "Entrenado" in (model_metric.delta or "")
    assert model_metric.value.startswith("✅")


def test_inventory_table_and_captions(mission_overview_runner: StreamlitRunner) -> None:
    app = mission_overview_runner.run()

    table_df = app.dataframe[0].value
    expected_columns = {"material_display", "category", "kg", "volume_m3", "_problematic"}
    assert expected_columns.issubset(table_df.columns)

    captions = " ".join(caption.body for caption in app.caption)
    assert "Categorías:" in captions
    inventory_df = load_waste_df()
    problematic_expected = int(inventory_df["_problematic"].astype(bool).sum())
    assert f"Problemáticos detectados: {problematic_expected}" in captions
