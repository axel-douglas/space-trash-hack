"""Smoke tests for declarative components used on the Home page."""

import importlib
import sys
import types

import pytest

pytest.importorskip("numpy")
pytest.importorskip("plotly")
pytest.importorskip("streamlit")

if "joblib" not in sys.modules:
    sys.modules["joblib"] = types.ModuleType("joblib")

from app.modules.luxe_components import (
    ActionCard,
    ActionDeck,
    CarouselItem,
    CarouselRail,
    MissionMetrics,
)
from pytest_streamlit import StreamlitRunner


def _render_home() -> None:
    import importlib
    import sys
    from pathlib import Path

    import streamlit as st

    repo_root = Path.cwd()
    app_dir = repo_root / "app"
    for path in (repo_root, app_dir):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))

    st.set_page_config = lambda *_, **__: None  # type: ignore[assignment]
    module_name = "app.Home"
    if module_name in sys.modules:
        importlib.reload(sys.modules[module_name])
    else:
        importlib.import_module(module_name)


def test_mission_metrics_markup_variants() -> None:
    payload = [
        {
            "key": "status",
            "label": "Estado",
            "value": "✅",
            "details": ["Modelo rexai"],
            "stage_key": "inventory",
        },
        {
            "key": "results",
            "label": "Resultados",
            "value": "Listo",
            "details": ["Trade-offs listos"],
            "stage_key": "results",
        },
    ]
    metrics = MissionMetrics.from_payload(payload, title="Panel de misión")

    panel_html = metrics.markup(highlight_key="results")
    assert "luxe-mission-panel" in panel_html
    assert "is-active" in panel_html
    assert "Trade-offs listos" in panel_html

    grid_html = metrics.markup(layout="grid", detail_limit=1, show_title=False)
    assert "luxe-mission-grid" in grid_html
    assert "Panel de misión" not in grid_html


def test_carousel_rail_markup() -> None:
    rail = CarouselRail(
        items=[
            CarouselItem(title="EVA scraps", value="320 kg", description="Volumen: 450 L"),
        ],
        reveal=False,
    )

    html = rail.markup()
    assert "luxe-carousel" in html
    assert "EVA scraps" in html
    assert "320 kg" in html


def test_action_deck_markup() -> None:
    deck = ActionDeck(
        cards=[
            ActionCard(
                title="Exportar receta",
                body="Descargá Sankey + trazabilidad completa.",
                icon="📤",
            )
        ],
        reveal=False,
    )

    html = deck.markup()
    assert "luxe-action-deck" in html
    assert "📤" in html
    assert "Descargá Sankey" in html


def test_home_metrics_and_inventory_preview() -> None:
    runner = StreamlitRunner(_render_home)
    app = runner.run()

    metric_labels = [metric.label for metric in app.metric]
    assert "Estado del modelo" in metric_labels
    assert "Inventario normalizado" in metric_labels
    assert "Feedback de crew" in metric_labels

    assert app.dataframe, "La portada debe exponer al menos un DataFrame"
    inventory_df = app.dataframe[0].value
    assert not inventory_df.empty
    assert {"material", "mass_kg"}.issubset(inventory_df.columns)


def test_home_feedback_section_displays_status() -> None:
    runner = StreamlitRunner(_render_home)
    app = runner.run()

    captions = [caption.body for caption in app.caption]
    assert any("Sin registros cargados" in body for body in captions)
