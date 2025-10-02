from __future__ import annotations

from importlib import import_module

import pytest

pytest.importorskip("streamlit")

from pytest_streamlit import StreamlitRunner


def _theme_app() -> None:
    import streamlit as st

    from app.modules.ui_blocks import enable_reveal_animation, load_theme

    load_theme()
    enable_reveal_animation()
    load_theme()
    enable_reveal_animation()


def test_interactions_script_injected_once(monkeypatch) -> None:
    ui_blocks = import_module("app.modules.ui_blocks")
    import streamlit as st

    original_markdown = ui_blocks.st.markdown

    def _tracking_markdown(body: str, **kwargs: object):
        if isinstance(body, str) and "data-rexai-interactions" in body:
            key = "__interactions_injections__"
            current = st.session_state[key] if key in st.session_state else 0
            st.session_state[key] = current + 1
        return original_markdown(body, **kwargs)

    monkeypatch.setattr(ui_blocks.st, "markdown", _tracking_markdown)

    runner = StreamlitRunner(_theme_app)
    app = runner.run()

    injections = (
        app.session_state["__interactions_injections__"]
        if "__interactions_injections__" in app.session_state
        else 0
    )
    assert injections == 1
    assert "__rexai_reveal_flag__" in app.session_state
