from __future__ import annotations

from importlib import import_module

import pytest

pytest.importorskip("streamlit")

from pytest_streamlit import StreamlitRunner


def _bootstrap_app() -> None:
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
        if isinstance(body, str) and "IntersectionObserver" in body:
            st.session_state["__interactions_injections__"] = (
                st.session_state.get("__interactions_injections__", 0) + 1
            )
        return original_markdown(body, **kwargs)

    monkeypatch.setattr(ui_blocks.st, "markdown", _tracking_markdown)

    runner = StreamlitRunner(_bootstrap_app)
    app = runner.run()

    assert app.session_state.get("__interactions_injections__", 0) == 1
    assert "__rexai_interactions_hash__" in app.session_state
