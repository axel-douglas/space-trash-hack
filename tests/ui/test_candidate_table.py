"""Regression tests for the candidate scoring table layout."""

from __future__ import annotations

import pytest

pytest.importorskip("streamlit")

from pytest_streamlit import StreamlitRunner


def _candidate_table_app() -> None:
    import streamlit as st  # noqa: F401
    from types import SimpleNamespace as _NS

    from app.modules import candidate_showroom as showroom

    risk = {
        "process_id": "B2",
        "process_name": "Proceso Riesgoso",
        "score": 0.72,
        "props": _NS(rigidity=0.58, water_l=0.62),
        "materials": ["ptfe"],
        "auxiliary": {"passes_seal": False},
    }
    safe = {
        "process_id": "A1",
        "process_name": "Proceso Seguro",
        "score": 0.91,
        "props": _NS(rigidity=0.81, water_l=0.33),
        "materials": ["acero"],
    }
    target = {"rigidity": 0.8, "max_water_l": 1.0}

    showroom.render_candidate_showroom([risk, safe], target)


def _collect_markup(app) -> list[str]:  # noqa: ANN001
    return [
        getattr(block, "body", "")
        for block in getattr(app, "markdown", [])
        if isinstance(getattr(block, "body", ""), str)
    ]


def test_candidate_table_orders_by_score() -> None:
    runner = StreamlitRunner(_candidate_table_app)
    app = runner.run()

    markup = "".join(_collect_markup(app))
    assert "#01 · Proceso Seguro" in markup
    assert markup.index("Proceso Seguro") < markup.index("Proceso Riesgoso")


def test_candidate_table_filters_and_feedback() -> None:
    runner = StreamlitRunner(_candidate_table_app)
    app = runner.run()

    app = app.checkbox(key="showroom_only_safe").check().run()
    filtered_html = "".join(_collect_markup(app))
    assert "Proceso Riesgoso" not in filtered_html
    assert "🛡️ Filtro: seguros" in filtered_html

    from app.modules import candidate_showroom as showroom

    app.session_state[showroom._SUCCESS_KEY] = {
        "message": "Proceso Seguro confirmado. Revisá pestañas.",
        "candidate_key": "1",
    }
    app = app.run()
    success_html = "".join(_collect_markup(app))
    assert "inline-success" in success_html
    assert "Proceso Seguro confirmado. Revisá pestañas." in success_html
