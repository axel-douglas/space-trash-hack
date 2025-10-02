from __future__ import annotations

from importlib import import_module

import pytest

pytest.importorskip("streamlit")

from pytest_streamlit import StreamlitRunner


def _theme_app() -> None:
    import streamlit as st

    from app.modules.ui_blocks import load_theme

    load_theme()
    load_theme()


def test_interactions_script_injected_once(monkeypatch) -> None:
    ui_blocks = import_module("app.modules.ui_blocks")
    import streamlit as st

    original_markdown = ui_blocks.st.markdown

    def _tracking_markdown(body: str, **kwargs: object):
        if isinstance(body, str) and "<style" in body:
            key = "__theme_injections__"
            current = st.session_state[key] if key in st.session_state else 0
            st.session_state[key] = current + 1
        return original_markdown(body, **kwargs)

    monkeypatch.setattr(ui_blocks.st, "markdown", _tracking_markdown)

    runner = StreamlitRunner(_theme_app)
    app = runner.run()

    session_state = app.session_state
    injections = session_state["__theme_injections__"] if "__theme_injections__" in session_state else 0
    assert injections == 1
    assert "__rexai_theme_hash__" in session_state
