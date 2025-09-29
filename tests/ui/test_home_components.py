"""Smoke tests for declarative components used on the Home page."""

import sys
import types

import pytest

pytest.importorskip("numpy")
pytest.importorskip("plotly")

if "joblib" not in sys.modules:
    sys.modules["joblib"] = types.ModuleType("joblib")

from app.modules.luxe_components import (
    ActionCard,
    ActionDeck,
    CarouselItem,
    CarouselRail,
    MissionMetrics,
)


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
