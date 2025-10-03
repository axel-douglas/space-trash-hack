from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Callable

import pandas as pd
import pytest

pytest.importorskip("streamlit")

from pytest_streamlit import StreamlitRunner


def _compare_page_app() -> None:
    import os
    import runpy
    import sys
    import types
    from pathlib import Path
    from types import SimpleNamespace
    from typing import Callable

    import pandas as pd

    import streamlit as st

    root_env = os.environ.get("REXAI_PROJECT_ROOT")
    root = Path(root_env) if root_env else Path.cwd()
    app_dir = root / "app"
    for candidate in (root, app_dir):
        if str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))

    original_page_config = st.set_page_config
    st.set_page_config = lambda *args, **kwargs: None

    import app.modules.ui_blocks as ui_blocks
    import app.modules.io as io_module

    original_load_theme = ui_blocks.load_theme
    ui_blocks.load_theme = lambda **_: None  # type: ignore[assignment]

    original_loader: Callable[[], pd.DataFrame] = io_module.load_waste_df
    inventory_stub = pd.DataFrame(
        {
            "id": ["poly-1", "alu-1"],
            "pc_density_density_g_per_cm3": [1.15, None],
            "pc_mechanics_tensile_strength_mpa": [65.0, None],
            "pc_mechanics_modulus_gpa": [2.8, None],
            "pc_thermal_glass_transition_c": [110.0, None],
            "pc_ignition_ignition_temperature_c": [420.0, None],
            "pc_ignition_burn_time_min": [6.0, None],
            "aluminium_tensile_strength_mpa": [None, 210.0],
            "aluminium_yield_strength_mpa": [None, 180.0],
        }
    )
    io_module.load_waste_df = lambda: inventory_stub  # type: ignore[assignment]

    original_sortables = sys.modules.get("streamlit_sortables")
    sys.modules["streamlit_sortables"] = types.SimpleNamespace(
        sort_items=lambda items, **_: list(items)
    )

    st.session_state.clear()
    st.session_state["candidates"] = [
        {
            "process_id": "P02",
            "process_name": "Laminar",
            "score": 0.82,
            "materials": ["Polymer-X", "Binder-Y"],
            "weights": {"func": 0.4, "agua": 0.2, "energy": 0.2, "crew": 0.2},
            "props": SimpleNamespace(
                rigidity=0.85,
                tightness=0.66,
                energy_kwh=1.1,
                water_l=0.55,
                crew_min=38.0,
                mass_final_kg=118.0,
            ),
            "source_ids": ["poly-1"],
        },
        {
            "process_id": "P03",
            "process_name": "Sinter",
            "score": 0.76,
            "materials": ["Alloy-A"],
            "weights": {"func": 0.4, "agua": 0.2, "energy": 0.2, "crew": 0.2},
            "props": SimpleNamespace(
                rigidity=0.78,
                tightness=0.72,
                energy_kwh=1.4,
                water_l=0.62,
                crew_min=41.0,
                mass_final_kg=124.0,
            ),
            "source_ids": ["alu-1"],
        },
    ]
    st.session_state["target"] = {
        "max_energy_kwh": 2.0,
        "max_water_l": 1.5,
        "max_crew_min": 60.0,
        "crew_time_low": False,
    }

    compare_page = app_dir / "pages" / "5_Compare_and_Explain.py"
    try:
        runpy.run_path(str(compare_page), run_name="__main__")
    finally:
        st.set_page_config = original_page_config
        ui_blocks.load_theme = original_load_theme  # type: ignore[assignment]
        io_module.load_waste_df = original_loader  # type: ignore[assignment]
        if original_sortables is None:
            sys.modules.pop("streamlit_sortables", None)
        else:
            sys.modules["streamlit_sortables"] = original_sortables


@pytest.fixture
def compare_page_runner() -> StreamlitRunner:
    os.environ.setdefault("REXAI_PROJECT_ROOT", str(Path(__file__).resolve().parents[2]))
    return StreamlitRunner(_compare_page_app)


def test_compare_page_has_no_inline_styles(compare_page_runner: StreamlitRunner) -> None:
    app = compare_page_runner.run()

    markdown_bodies = [block.body for block in app.markdown]
    assert all("<style" not in body.lower() for body in markdown_bodies)


def test_compare_page_renders_table_and_storytelling(compare_page_runner: StreamlitRunner) -> None:
    app = compare_page_runner.run()

    assert app.dataframe, "Se espera al menos una tabla renderizada"
    comparison_table = app.dataframe[0].value
    expected_columns = {"Score", "Energía (kWh)", "Agua (L)", "Crew (min)"}
    assert expected_columns.issubset(set(comparison_table.columns))

    metric_labels = {metric.label for metric in app.metric}
    assert {"Opciones generadas", "Mejor Score"}.issubset(metric_labels)

    storytelling_section = " ".join(block.body for block in app.markdown)
    assert "Storytelling asistido por IA" in storytelling_section
    assert "- " in storytelling_section  # bullet list con los insights


def test_compare_page_pills_do_not_use_inline_styles(compare_page_runner: StreamlitRunner) -> None:
    app = compare_page_runner.run()

    pill_sections = [block.body for block in app.markdown if "data-mission-pill" in block.body]
    assert pill_sections, "Se espera al menos un pill renderizado"
    assert all("style=" not in section.lower() for section in pill_sections)
