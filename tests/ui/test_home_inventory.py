from __future__ import annotations

import sys
from pathlib import Path
import pytest

pytest.importorskip("streamlit")

from pytest_streamlit import StreamlitRunner

from app.modules.io import load_waste_df


def _home_app() -> None:
    """Streamlit entrypoint that mirrors app/Home with lightweight stubs."""

    import sys
    from pathlib import Path
    import os
    import importlib

    root_env = os.environ.get("REXAI_PROJECT_ROOT")
    root = Path(root_env) if root_env else Path.cwd()
    app_dir = root / "app"
    for candidate in (root, app_dir):
        sys.path.insert(0, str(candidate))

    import streamlit as st

    original_page_config = st.set_page_config
    st.set_page_config = lambda *args, **kwargs: None

    import app.modules.ml_models as ml_models
    import app.modules.ui_blocks as ui_blocks

    class _RegistryStub:
        metadata = {
            "trained_at": "2024-01-01T00:00:00+00:00",
            "model_name": "stub-model",
            "n_samples": 256,
        }
        ready = True
        feature_names = ["kg", "volume_l"]

        def trained_label(self) -> str:
            return "2024-01-01 · stub-model"

    original_registry = ml_models.get_model_registry
    original_load_theme = ui_blocks.load_theme

    try:
        ml_models.get_model_registry = lambda: _RegistryStub()  # type: ignore[assignment]
        ui_blocks.load_theme = lambda: None  # type: ignore[assignment]

        sys.modules.pop("app.Home", None)
        sys.modules.pop("_bootstrap", None)
        importlib.import_module("app.Home")
    finally:
        ml_models.get_model_registry = original_registry  # type: ignore[assignment]
        ui_blocks.load_theme = original_load_theme  # type: ignore[assignment]
        st.set_page_config = original_page_config


@pytest.fixture
def home_runner() -> StreamlitRunner:
    import os
    from pathlib import Path

    os.environ.setdefault("REXAI_PROJECT_ROOT", str(Path(__file__).resolve().parents[2]))
    return StreamlitRunner(_home_app)


def test_home_inventory_uses_enriched_loader(home_runner: StreamlitRunner) -> None:
    app = home_runner.run()
    inventory_df = load_waste_df()

    expected_problematic = int(inventory_df["_problematic"].astype(bool).sum())
    inventory_metric = next(metric for metric in app.metric if metric.label == "Inventario normalizado")
    assert str(expected_problematic) in (inventory_metric.delta or ""), "Delta should show problematic count"

    volume_metric = next(metric for metric in app.metric if metric.label == "Volumen total")
    expected_volume = float(inventory_df["volume_l"].fillna(0.0).sum()) / 1000.0
    assert f"{expected_volume:.2f}" in volume_metric.body, "Volume metric should reflect enriched loader"

    table_df = app.dataframe[0].value
    for column in ["Material", "Categoría", "Masa (kg)", "Volumen (m³)", "Problemático"]:
        assert column in table_df.columns, f"Expected '{column}' column in Home inventory table"

    first_row = table_df.iloc[0]
    assert first_row["Masa (kg)"] == pytest.approx(float(inventory_df["kg"].iloc[0]))
    assert first_row["Volumen (m³)"] == pytest.approx(float(inventory_df["volume_l"].iloc[0]) / 1000.0)
    assert first_row["Problemático"] in (True, False)


def test_home_inventory_captions_include_categories(home_runner: StreamlitRunner) -> None:
    app = home_runner.run()
    inventory_df = load_waste_df()

    captions = [caption.body for caption in app.caption]
    captions_text = " ".join(captions)

    assert "Categorías:" in captions_text

    categories = sorted(
        {
            str(category).strip()
            for category in inventory_df["category"].dropna()
            if str(category).strip()
        }
    )
    for category in categories[:3]:
        assert category in captions_text, "Category summary should reference real inventory categories"

    expected_problematic = int(inventory_df["_problematic"].astype(bool).sum())
    assert f"Problemáticos detectados: {expected_problematic}" in captions
