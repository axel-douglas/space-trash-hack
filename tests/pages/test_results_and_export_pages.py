from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("streamlit")

from pytest_streamlit import StreamlitRunner

from app.bootstrap import ensure_project_root
from app.modules.io import format_missing_dataset_message, MissingDatasetError


def _results_page_app(*, missing_dataset: bool = False, inventory=None) -> None:
    import os
    import pandas as pd
    from types import SimpleNamespace as _SimpleNamespace
    import runpy
    import streamlit as st
    from pathlib import Path

    root_env = os.environ.get("REXAI_PROJECT_ROOT")
    start = Path(root_env) if root_env else Path(__file__).resolve()
    root = ensure_project_root(start)
    app_dir = root / "app"

    st.session_state.clear()

    base_props_values = {
        "rigidity": 0.0,
        "tightness": 0.0,
        "energy_kwh": 0.0,
        "water_l": 0.0,
        "crew_min": 0.0,
        "mass_final_kg": 0.0,
    }
    base_props = _SimpleNamespace(**base_props_values)
    st.session_state["selected"] = {
        "data": {
            "props": base_props,
            "heuristic_props": _SimpleNamespace(**base_props_values),
            "confidence_interval": {},
            "uncertainty": {},
            "model_variants": [],
            "feature_importance": [],
            "ml_prediction": {"metadata": {}},
            "latent_vector": [],
            "regolith_pct": 0.0,
            "materials": [],
            "score": 0.0,
            "process_id": "P01",
            "process_name": "Proceso demo",
            "source_ids": [],
        },
        "safety": {"level": "OK", "detail": ""},
    }
    st.session_state["target"] = {
        "rigidity": 0.0,
        "tightness": 0.0,
        "max_energy_kwh": 2.0,
        "max_water_l": 1.0,
        "max_crew_min": 60.0,
    }

    import app.modules.ui_blocks as ui_blocks
    import app.modules.navigation as navigation
    import app.modules.io as io_module

    original_page_config = st.set_page_config
    original_load_theme = ui_blocks.load_theme
    original_initialise = ui_blocks.initialise_frontend
    original_breadcrumbs = navigation.render_breadcrumbs
    original_loader = io_module.load_waste_df

    st.set_page_config = lambda *args, **kwargs: None  # type: ignore[assignment]
    ui_blocks.load_theme = lambda **_: None  # type: ignore[assignment]
    ui_blocks.initialise_frontend = lambda **_: None  # type: ignore[assignment]
    navigation.render_breadcrumbs = lambda *args, **kwargs: None  # type: ignore[assignment]

    if missing_dataset:
        missing_path = Path("missing_results.csv")

        def _raise_missing() -> pd.DataFrame:
            raise io_module.MissingDatasetError(missing_path)

        io_module.load_waste_df = _raise_missing  # type: ignore[assignment]
    else:
        io_module.load_waste_df = (
            lambda: inventory.copy() if inventory is not None else pd.DataFrame()
        )  # type: ignore[assignment]

    results_page = app_dir / "pages" / "4_Results_and_Tradeoffs.py"

    try:
        runpy.run_path(str(results_page), run_name="__main__")
    finally:
        st.set_page_config = original_page_config  # type: ignore[assignment]
        ui_blocks.load_theme = original_load_theme  # type: ignore[assignment]
        ui_blocks.initialise_frontend = original_initialise  # type: ignore[assignment]
        navigation.render_breadcrumbs = original_breadcrumbs  # type: ignore[assignment]
        io_module.load_waste_df = original_loader  # type: ignore[assignment]


def _pareto_page_app(
    *, missing_dataset: bool = False, selected_option=None, set_option: bool = False
) -> None:
    import os
    import pandas as pd
    import runpy
    from pathlib import Path
    from types import SimpleNamespace

    import streamlit as st

    root_env = os.environ.get("REXAI_PROJECT_ROOT")
    start = Path(root_env) if root_env else Path(__file__).resolve()
    root = ensure_project_root(start)
    app_dir = root / "app"

    st.session_state.clear()

    base_props = SimpleNamespace(
        energy_kwh=0.0,
        water_l=0.0,
        crew_min=0.0,
        mass_final_kg=0.0,
        rigidity=0.0,
        tightness=0.0,
    )
    st.session_state["candidates"] = [
        {
            "materials": [],
            "weights": [],
            "score": 0.0,
            "props": base_props,
            "process_id": "P01",
            "process_name": "Proceso demo",
        }
    ]
    st.session_state["target"] = {"max_energy_kwh": 2.0, "max_water_l": 1.0, "max_crew_min": 60.0}
    st.session_state["selected"] = {
        "data": {
            "materials": [],
            "weights": [],
            "props": base_props,
            "process_id": "P01",
            "process_name": "Proceso demo",
        },
        "safety": {"level": "OK", "detail": ""},
    }
    if set_option:
        st.session_state["selected_option_number"] = selected_option

    import app.modules.ui_blocks as ui_blocks
    import app.modules.navigation as navigation
    import app.modules.io as io_module

    original_page_config = st.set_page_config
    original_load_theme = ui_blocks.load_theme
    original_initialise = ui_blocks.initialise_frontend
    original_breadcrumbs = navigation.render_breadcrumbs
    original_loader = io_module.load_waste_df

    st.set_page_config = lambda *args, **kwargs: None  # type: ignore[assignment]
    ui_blocks.load_theme = lambda **_: None  # type: ignore[assignment]
    ui_blocks.initialise_frontend = lambda **_: None  # type: ignore[assignment]
    navigation.render_breadcrumbs = lambda *args, **kwargs: None  # type: ignore[assignment]

    if missing_dataset:
        missing_path = Path("missing_pareto.csv")

        def _raise_missing() -> pd.DataFrame:
            raise io_module.MissingDatasetError(missing_path)

        io_module.load_waste_df = _raise_missing  # type: ignore[assignment]
    else:
        io_module.load_waste_df = lambda: pd.DataFrame()  # type: ignore[assignment]

    pareto_page = app_dir / "pages" / "6_Pareto_and_Export.py"

    try:
        runpy.run_path(str(pareto_page), run_name="__main__")
    finally:
        st.set_page_config = original_page_config  # type: ignore[assignment]
        ui_blocks.load_theme = original_load_theme  # type: ignore[assignment]
        ui_blocks.initialise_frontend = original_initialise  # type: ignore[assignment]
        navigation.render_breadcrumbs = original_breadcrumbs  # type: ignore[assignment]
        io_module.load_waste_df = original_loader  # type: ignore[assignment]


def test_results_page_shows_error_for_missing_dataset() -> None:
    runner = StreamlitRunner(_results_page_app, kwargs={"missing_dataset": True})
    app = runner.run()

    error_messages = " ".join(block.body for block in app.error)
    assert "missing_results.csv" in error_messages
    assert "python scripts/download_datasets.py" in error_messages
    expected_message = format_missing_dataset_message(
        MissingDatasetError(Path("missing_results.csv"))
    )
    assert expected_message in error_messages
    assert not app.exception


def test_results_page_renders_without_inventory_material_columns() -> None:
    inventory = pd.DataFrame({"mass_kg": [10.0, 15.0], "notes": ["a", "b"]})
    runner = StreamlitRunner(
        _results_page_app,
        kwargs={"inventory": inventory},
    )
    app = runner.run()

    assert not app.exception


def test_pareto_page_shows_error_for_missing_dataset() -> None:
    runner = StreamlitRunner(_pareto_page_app, kwargs={"missing_dataset": True})
    app = runner.run()

    error_messages = " ".join(block.body for block in app.error)
    assert "missing_pareto.csv" in error_messages
    assert "python scripts/download_datasets.py" in error_messages
    expected_message = format_missing_dataset_message(
        MissingDatasetError(Path("missing_pareto.csv"))
    )
    assert expected_message in error_messages
    assert not app.exception


def test_pareto_page_handles_missing_selected_option_number() -> None:
    runner = StreamlitRunner(_pareto_page_app, kwargs={"set_option": False})
    app = runner.run()

    assert not app.exception


def test_pareto_page_handles_malformed_selected_option_number() -> None:
    runner = StreamlitRunner(
        _pareto_page_app,
        kwargs={"selected_option": "invalid-index", "set_option": True},
    )
    app = runner.run()

    assert not app.exception
