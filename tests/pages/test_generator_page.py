from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import pandas as pd
import pytest

pytest.importorskip("streamlit")

from pytest_streamlit import StreamlitRunner


def _generator_page_app(inventory=None, *, force_control_expander_none: bool = False) -> None:
    import os
    import runpy
    import sys
    from pathlib import Path
    from types import SimpleNamespace

    import pandas as pd
    import streamlit as st
    from streamlit.delta_generator import DeltaGenerator

    root_env = os.environ.get("REXAI_PROJECT_ROOT")
    root = Path(root_env) if root_env else Path.cwd()
    app_dir = root / "app"
    for candidate in (root, app_dir):
        if str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))

    st.set_page_config = lambda *args, **kwargs: None  # type: ignore[assignment]

    original_expander = st.expander
    original_delta_enter = DeltaGenerator.__enter__
    original_control_expander = DeltaGenerator.expander

    def _expander_with_value(*args, **kwargs):  # pragma: no cover - UI shim
        delta = original_expander(*args, **kwargs)

        class _Wrapper:
            def __init__(self, dg):
                self._dg = dg

            def __enter__(self):
                entered = self._dg.__enter__()
                return entered if entered is not None else self._dg

            def __exit__(self, exc_type, exc, tb):
                return self._dg.__exit__(exc_type, exc, tb)

            def __getattr__(self, name):
                return getattr(self._dg, name)

        return _Wrapper(delta)

    st.expander = _expander_with_value  # type: ignore[assignment]

    class _NullExpander:
        def __enter__(self):  # pragma: no cover - UI shim
            return None

        def __exit__(self, exc_type, exc, tb):  # pragma: no cover - UI shim
            return False

    def _control_expander_none(self, *args, **kwargs):  # pragma: no cover - UI shim
        original_control_expander(self, *args, **kwargs)
        return _NullExpander()

    if force_control_expander_none:
        DeltaGenerator.expander = _control_expander_none  # type: ignore[assignment]

    def _delta_enter(self):  # pragma: no cover - UI shim
        result = original_delta_enter(self)
        return result if result is not None else self

    DeltaGenerator.__enter__ = _delta_enter  # type: ignore[assignment]

    import app.modules.ui_blocks as ui_blocks
    import app.modules.io as io_module
    import app.modules.candidate_showroom as showroom_module
    import app.modules.process_planner as process_planner
    import app.modules.safety as safety_module
    import app.modules.ml_models as ml_models
    import app.modules.visualizations as visualizations
    import app.modules.page_data as page_data
    import app.modules.generator as generator_module

    original_load_theme = ui_blocks.load_theme
    ui_blocks.load_theme = lambda **_: None  # type: ignore[assignment]

    original_waste_loader: Callable[[], pd.DataFrame] = io_module.load_waste_df
    inventory_df = inventory.copy() if isinstance(inventory, pd.DataFrame) else pd.DataFrame()
    io_module.load_waste_df = lambda: inventory_df  # type: ignore[assignment]

    process_df = pd.DataFrame(
        [
            {
                "process_id": "P02",
                "name": "Press & Heat Lamination",
                "crew_min_per_batch": 10,
                "energy_kwh_per_kg": 0.5,
                "water_l_per_kg": 0.0,
            }
        ]
    )
    original_process_loader: Callable[[], pd.DataFrame] = io_module.load_process_df
    io_module.load_process_df = lambda: process_df  # type: ignore[assignment]

    original_choose_process = process_planner.choose_process
    process_planner.choose_process = lambda *args, **kwargs: process_df

    original_showroom = showroom_module.render_candidate_showroom
    showroom_module.render_candidate_showroom = lambda cands, target: cands

    original_check_safety = safety_module.check_safety
    original_safety_badge = safety_module.safety_badge
    safety_module.check_safety = lambda *args, **kwargs: SimpleNamespace(
        pfas=False, microplastics=False, incineration=False
    )
    safety_module.safety_badge = lambda flags: {
        "level": "OK",
        "detail": "",
        "pfas": False,
        "microplastics": False,
        "incineration": False,
    }

    original_model_registry = ml_models.get_model_registry
    ml_models.get_model_registry = lambda: SimpleNamespace(
        metadata={
            "trained_at": "2024-05-01",
            "n_samples": 20,
            "random_forest": {
                "metrics": {"overall": {"mae": 0.12, "rmse": 0.21, "r2": 0.93}}
            },
        },
        feature_importance_avg=[("density", 0.42)],
        feature_names=["density"],
        label_distribution_label=lambda: "NASA/ISRU",
        label_summary={"NASA": {"count": 1, "mean_weight": 1.0}},
    )

    original_scene = visualizations.ConvergenceScene

    class DummyScene:
        def __init__(self, *args, **kwargs) -> None:
            self.args = args
            self.kwargs = kwargs

        def render(self, _st) -> None:  # pragma: no cover - UI stub
            _st.caption("dummy convergence scene")

    visualizations.ConvergenceScene = DummyScene

    original_ranking = page_data.build_ranking_table

    def fake_ranking_table(cands: list[dict[str, object]]) -> pd.DataFrame:
        if not cands:
            return pd.DataFrame()
        return pd.DataFrame(
            [
                {
                    "Rank": 1,
                    "Proceso": cands[0].get("process_name", "Proceso 1"),
                    "Score": 0.9,
                    "Rigidez": 0.8,
                    "Estanqueidad": 0.7,
                    "Energía (kWh)": 1.1,
                    "Agua (L)": 0.4,
                    "Crew (min)": 32.0,
                }
            ]
        )

    page_data.build_ranking_table = fake_ranking_table

    original_generate_candidates = generator_module.generate_candidates
    generator_module.generate_candidates = lambda *args, **kwargs: []

    st.session_state.clear()
    st.session_state.update(
        {
            "target": {
                "name": "Residence Renovations",
                "max_energy_kwh": 2.5,
                "max_water_l": 1.8,
                "max_crew_min": 60.0,
                "crew_time_low": False,
            },
            "candidates": [
                {
                    "score": 0.87,
                    "process_id": "P02",
                    "process_name": "Press & Heat Lamination",
                    "materials": ["Polymer-X", "Binder-Y"],
                    "weights": {"Polymer-X": 0.7, "Binder-Y": 0.3},
                    "props": SimpleNamespace(
                        rigidity=0.82,
                        tightness=0.74,
                        mass_final_kg=115.0,
                        energy_kwh=1.2,
                        water_l=0.6,
                        crew_min=42.0,
                    ),
                    "features": {"total_mass_kg": 18.0},
                    "auxiliary": {"passes_seal": True, "process_risk_label": "B"},
                    "source_ids": ["poly-1", "alu-1"],
                    "source_categories": ["Polymer"],
                    "source_flags": ["foam"],
                }
            ],
            "optimizer_history": pd.DataFrame(),
            "generator_button_trigger": False,
            "generator_button_state": "success",
        }
    )

    generator_page = app_dir / "pages" / "3_Generator.py"

    try:
        runpy.run_path(str(generator_page), run_name="__main__")
    finally:
        DeltaGenerator.__enter__ = original_delta_enter  # type: ignore[assignment]
        DeltaGenerator.expander = original_control_expander  # type: ignore[assignment]
        st.expander = original_expander  # type: ignore[assignment]
        ui_blocks.load_theme = original_load_theme  # type: ignore[assignment]
        io_module.load_waste_df = original_waste_loader  # type: ignore[assignment]
        io_module.load_process_df = original_process_loader  # type: ignore[assignment]
        process_planner.choose_process = original_choose_process
        showroom_module.render_candidate_showroom = original_showroom
        safety_module.check_safety = original_check_safety
        safety_module.safety_badge = original_safety_badge
        ml_models.get_model_registry = original_model_registry
        visualizations.ConvergenceScene = original_scene
        page_data.build_ranking_table = original_ranking
        generator_module.generate_candidates = original_generate_candidates


@pytest.fixture
def run_generator_page() -> Callable[..., object]:
    os.environ.setdefault("REXAI_PROJECT_ROOT", str(Path(__file__).resolve().parents[2]))

    def _run(inventory: pd.DataFrame | None, **extra: Any) -> object:
        call_kwargs = {"inventory": inventory}
        call_kwargs.update(extra)
        runner = StreamlitRunner(_generator_page_app, kwargs=call_kwargs)
        return runner.run()

    return _run


def test_generator_page_renders_histograms_with_inventory(run_generator_page: Callable[..., object]) -> None:
    inventory = pd.DataFrame(
        {
            "id": ["poly-1", "alu-1"],
            "pc_density_density_g_per_cm3": [1.15, None],
            "pc_mechanics_tensile_strength_mpa": [65.0, None],
            "pc_mechanics_modulus_gpa": [2.9, None],
            "pc_thermal_glass_transition_c": [112.0, None],
            "pc_ignition_ignition_temperature_c": [418.0, None],
            "pc_ignition_burn_time_min": [6.2, None],
            "aluminium_tensile_strength_mpa": [None, 212.0],
            "aluminium_yield_strength_mpa": [None, 182.0],
            "aluminium_elongation_pct": [None, 11.5],
            "aluminium_processing_route": [None, "Route-A"],
            "aluminium_class_id": [None, "AA2024"],
        }
    )

    app = run_generator_page(inventory)

    metric_labels = [metric.label for metric in app.metric]
    assert "ρ ref (g/cm³)" in metric_labels
    assert "σₜ ref (MPa)" in metric_labels

    info_messages = " ".join(block.body for block in app.info)
    assert "No hay densidades de polímeros" not in info_messages
    assert "No hay datos de tracción de aluminio" not in info_messages


def test_generator_page_warns_when_inventory_missing_columns(
    run_generator_page: Callable[..., object]
) -> None:
    inventory = pd.DataFrame({"id": ["poly-1"]})

    app = run_generator_page(inventory)

    info_messages = " ".join(block.body for block in app.info)
    assert "No hay densidades de polímeros en el inventario actual para comparar." in info_messages
    assert "No hay datos de tracción de aluminio en el inventario actual para comparar." in info_messages


def test_generator_page_renders_without_external_columns(
    run_generator_page: Callable[..., object]
) -> None:
    inventory = pd.DataFrame({"id": ["poly-1"], "other": [1.0]})

    app = run_generator_page(inventory)

    headers = " ".join(block.body for block in app.header)
    assert "Generador" in headers

    subheaders = " ".join(block.body for block in app.subheader)
    assert "Resultados del generador" in subheaders


def test_generator_page_falls_back_when_expander_missing(
    run_generator_page: Callable[..., object]
) -> None:
    app = run_generator_page(None, force_control_expander_none=True)

    warning_messages = " ".join(block.body for block in app.warning)
    assert "modo expandido" in warning_messages


def test_generator_page_renders_without_inventory(
    run_generator_page: Callable[..., object]
) -> None:
    app = run_generator_page(None)

    # Expect at least the generator header to render even with minimal data.
    headers = " ".join(block.body for block in app.header)
    assert "Generador asistido por IA" in headers

