from pathlib import Path
import sys
import types

import pytest

pytest.importorskip("numpy")
pytest.importorskip("plotly")

if "joblib" not in sys.modules:
    sys.modules["joblib"] = types.ModuleType("joblib")

from app.modules.luxe_components import (  # noqa: E402
    CarouselItem,
    CarouselRail,
    MissionBoard,
    MissionMetrics,
)


def test_home_markup_prunes_legacy_classes() -> None:
    metrics = MissionMetrics.from_payload(
        [
            {
                "key": "status",
                "label": "Estado",
                "value": "✅ Modelo listo",
                "details": ["Nombre: rexai-rf-ensemble"],
                "stage_key": "inventory",
            }
        ],
        animate=False,
    )
    board = MissionBoard.from_payload(
        [
            {
                "key": "inventory",
                "title": "Inventario",
                "description": "Normalizá residuos y marcá flags EVA o multilayer.",
                "href": "./?page=1_Inventory_Builder",
                "icon": "🧱",
            }
        ],
        reveal=False,
    )
    carousel = CarouselRail(
        items=[CarouselItem(title="EVA scraps", value="320 kg")],
        reveal=False,
    )

    markup = "\n".join(
        [
            metrics.markup(with_board=True),
            board.markup(highlight_key="inventory"),
            carousel.markup(),
        ]
    )

    for legacy in ("reveal", "ghost-card", "tesla-hero", "parallax"):
        assert legacy not in markup


def test_home_css_prunes_obsolete_rules() -> None:
    css = Path("app/static/home.css").read_text(encoding="utf-8")

    assert ".home-section" in css
    for legacy in (".reveal", ".ghost-card", ".tesla-hero", "parallax"):
        assert legacy not in css
