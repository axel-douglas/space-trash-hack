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

    table_df = app.dataframe[0].value
    assert table_df.iloc[0]["Proceso"].startswith("Proceso Seguro")
    assert table_df["Score"].is_monotonic_decreasing


def test_candidate_table_filters_and_feedback(monkeypatch) -> None:
    runner = StreamlitRunner(_candidate_table_app)
    app = runner.run()

    app = app.checkbox(key="showroom_only_safe").check().run()
    table_df = app.dataframe[0].value
    assert not table_df["Proceso"].str.contains("Riesgoso").any()

    tags_series = table_df.get("Etiquetas")
    flattened_tags: list[str] = []
    if tags_series is not None:
        for tags in tags_series:
            if tags is None:
                continue
            if isinstance(tags, str):
                flattened_tags.append(tags)
                continue
            try:
                for tag in tags:
                    if tag:
                        flattened_tags.append(str(tag))
            except TypeError:
                flattened_tags.append(str(tags))
    assert "🛡️ Filtro: seguros" in flattened_tags
    assert "💧 Dentro de límite de agua" in flattened_tags

    from app.modules import candidate_showroom as showroom
    from app.modules import ui_blocks

    class _StatusStub:
        def __init__(self, label: str, state: str | None = None) -> None:
            self.label = label
            self.state = state

        def update(self, *, label: str | None = None, state: str | None = None, **_: object) -> None:
            if label is not None:
                self.label = label
            if state is not None:
                self.state = state

    def _status_factory(label: str, state: str | None = None, **_: object) -> _StatusStub:
        return _StatusStub(label, state)

    monkeypatch.setattr(ui_blocks.st, "status", _status_factory)

    app.session_state[showroom._SUCCESS_KEY] = {
        "message": "Proceso Seguro confirmado. Revisá pestañas.",
        "candidate_key": "1",
    }
    app = app.run()
    success_messages = [msg.body for msg in app.success]
    assert any("Proceso Seguro confirmado" in body for body in success_messages)
