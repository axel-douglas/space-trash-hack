from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

import pytest
from pytest_streamlit import StreamlitRunner

from app.modules.paths import DATA_ROOT


def _home_timestamp_app() -> None:
    import os
    import importlib
    import streamlit as st
    from pathlib import Path
    import sys

    root_env = os.environ.get("REXAI_PROJECT_ROOT")
    root = Path(root_env) if root_env else Path.cwd()
    app_dir = root / "app"
    for candidate in (root, app_dir):
        if str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))

    original_page_config = st.set_page_config
    st.set_page_config = lambda *args, **kwargs: None

    import app.modules.ml_models as ml_models
    import app.modules.mission_overview as mission_overview
    import app.modules.ui_blocks as ui_blocks

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
        mission_overview.load_inventory_overview = mission_overview.load_waste_df  # type: ignore[assignment]

        sys.modules.pop("app.Home", None)
        home_module = importlib.import_module("app.Home")
        home_module.render_page()
    finally:
        ml_models.get_model_registry = original_registry  # type: ignore[assignment]
        ui_blocks.load_theme = original_load_theme  # type: ignore[assignment]
        mission_overview.load_inventory_overview = original_inventory_loader  # type: ignore[assignment]
        st.set_page_config = original_page_config


@pytest.fixture
def home_timestamp_runner() -> StreamlitRunner:
    os.environ.setdefault("REXAI_PROJECT_ROOT", str(Path(__file__).resolve().parents[2]))
    return StreamlitRunner(_home_timestamp_app)


def test_home_page_shows_last_modified_caption(
    home_timestamp_runner: StreamlitRunner,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    csv_path = DATA_ROOT / "waste_inventory_sample.csv"
    assert csv_path.exists(), "El dataset de inventario debe existir bajo DATA_ROOT para la prueba"

    monkeypatch.chdir(tmp_path)

    app = home_timestamp_runner.run()

    caption_texts = [caption.body for caption in app.caption]
    assert any("Actualizado:" in text for text in caption_texts)
