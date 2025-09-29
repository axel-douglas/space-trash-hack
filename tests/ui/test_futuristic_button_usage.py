from __future__ import annotations

import re
import sys
import types
from importlib import import_module

from pytest_streamlit import StreamlitRunner

for _missing in ("joblib", "polars", "plotly"):
    sys.modules.setdefault(_missing, types.ModuleType(_missing))
sys.modules.setdefault("plotly.graph_objects", types.ModuleType("plotly.graph_objects"))


def _fx_demo_app() -> None:
    import streamlit as st

    from app.modules.ui_blocks import futuristic_button

    if "demo_fx_state" not in st.session_state:
        st.session_state["demo_fx_state"] = "idle"

    state = st.session_state["demo_fx_state"]
    if st.button("Activar loading", key="demo_fx_loading"):
        state = "loading"
    if st.button("Activar success", key="demo_fx_success"):
        state = "success"

    st.session_state["demo_fx_state"] = state

    futuristic_button(
        "Lanzar\nsecuencia orbital",
        key="demo_fx_cta",
        icon="🚀",
        state=state,
        loading_label="Sincronizando cápsula…",
        success_label="Órbita establecida",
        status_hints={
            "idle": "Listo para despegar",
            "loading": "Ajustando vector",
            "success": "Secuencia completada",
            "error": "Interrupción detectada",
        },
    )


def _extract_state_markup(html_block: str) -> str:
    match = re.search(r"data-state=\"(?P<state>[a-z]+)\"", html_block)
    assert match, f"No state attribute found in markup: {html_block!r}"
    return match.group("state")


def test_futuristic_button_transitions(monkeypatch) -> None:
    ui_blocks = import_module("app.modules.ui_blocks")

    def _capture_html(markup: str, **kwargs: object) -> dict[str, object]:
        import streamlit as st

        st.session_state["__fx_markup__"] = markup
        return {}

    monkeypatch.setattr(ui_blocks, "components_html", _capture_html)

    runner = StreamlitRunner(_fx_demo_app)
    app = runner.run()

    html_block = app.session_state["__fx_markup__"]
    assert "rexai-fx-line" in html_block
    assert _extract_state_markup(html_block) == "idle"

    app = app.button(key="demo_fx_loading").click().run()
    html_block = app.session_state["__fx_markup__"]
    assert _extract_state_markup(html_block) == "loading"

    app = app.button(key="demo_fx_success").click().run()
    html_block = app.session_state["__fx_markup__"]
    assert _extract_state_markup(html_block) == "success"
