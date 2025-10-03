from __future__ import annotations

import pytest

pytest.importorskip("streamlit")

from pytest_streamlit import StreamlitRunner


def _feedback_page_app(*, selected_option=None, set_option: bool = False) -> None:
    import os
    import runpy
    import sys
    from pathlib import Path
    from types import SimpleNamespace

    import pandas as pd
    import streamlit as st

    root_env = os.environ.get("REXAI_PROJECT_ROOT")
    root = Path(root_env) if root_env else Path.cwd()
    app_dir = root / "app"
    for candidate in (root, app_dir):
        if str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))

    st.session_state.clear()

    base_props = SimpleNamespace(
        energy_kwh=0.0,
        water_l=0.0,
        crew_min=0.0,
        mass_final_kg=0.0,
    )
    st.session_state["selected"] = {
        "data": {
            "props": base_props,
            "materials": [],
            "weights": [],
            "process_id": "P01",
            "process_name": "Proceso demo",
            "score": 0.0,
        },
        "safety": {"level": "OK", "detail": ""},
    }
    st.session_state["target"] = {"scenario": "Alpha", "name": "Mission"}

    if set_option:
        st.session_state["selected_option_number"] = selected_option

    import app.modules.ui_blocks as ui_blocks
    import app.modules.navigation as navigation
    import app.modules.impact as impact_module

    original_page_config = st.set_page_config
    original_load_theme = ui_blocks.load_theme
    original_initialise = ui_blocks.initialise_frontend
    original_breadcrumbs = navigation.render_breadcrumbs
    original_load_impact = impact_module.load_impact_df
    original_load_feedback = impact_module.load_feedback_df
    original_append_feedback = impact_module.append_feedback
    original_append_impact = impact_module.append_impact

    st.set_page_config = lambda *args, **kwargs: None  # type: ignore[assignment]
    ui_blocks.load_theme = lambda **_: None  # type: ignore[assignment]
    ui_blocks.initialise_frontend = lambda **_: None  # type: ignore[assignment]
    navigation.render_breadcrumbs = lambda *args, **kwargs: None  # type: ignore[assignment]

    impact_df = pd.DataFrame(
        [
            {
                "ts_iso": "2024-05-01T00:00:00Z",
                "mass_final_kg": 1.0,
                "energy_kwh": 0.5,
                "water_l": 0.2,
                "crew_min": 15.0,
            }
        ]
    )
    feedback_df = pd.DataFrame(
        [
            {
                "ts_iso": "2024-05-02T00:00:00Z",
                "astronaut": "A",
                "scenario": "Alpha",
                "target_name": "Mission",
                "option_idx": 1,
                "rigidity_ok": True,
                "ease_ok": True,
                "issues": "",
                "notes": "",
                "extra": "{\"feedback_overall\": 8}",
            }
        ]
    )

    impact_module.load_impact_df = lambda: impact_df.copy()  # type: ignore[assignment]
    impact_module.load_feedback_df = lambda: feedback_df.copy()  # type: ignore[assignment]
    impact_module.append_feedback = lambda entry: "run-id"  # type: ignore[assignment]
    impact_module.append_impact = lambda entry: "run-id"  # type: ignore[assignment]

    feedback_page = app_dir / "pages" / "8_Feedback_and_Impact.py"

    try:
        runpy.run_path(str(feedback_page), run_name="__main__")
    finally:
        st.set_page_config = original_page_config  # type: ignore[assignment]
        ui_blocks.load_theme = original_load_theme  # type: ignore[assignment]
        ui_blocks.initialise_frontend = original_initialise  # type: ignore[assignment]
        navigation.render_breadcrumbs = original_breadcrumbs  # type: ignore[assignment]
        impact_module.load_impact_df = original_load_impact  # type: ignore[assignment]
        impact_module.load_feedback_df = original_load_feedback  # type: ignore[assignment]
        impact_module.append_feedback = original_append_feedback  # type: ignore[assignment]
        impact_module.append_impact = original_append_impact  # type: ignore[assignment]


def test_feedback_page_handles_missing_selected_option_number() -> None:
    runner = StreamlitRunner(_feedback_page_app, kwargs={"set_option": False})
    app = runner.run()

    assert not app.exception


def test_feedback_page_handles_malformed_selected_option_number() -> None:
    runner = StreamlitRunner(
        _feedback_page_app,
        kwargs={"selected_option": "not-a-number", "set_option": True},
    )
    app = runner.run()

    assert not app.exception
