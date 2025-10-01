"""Regression tests for the mission HUD navigation bar."""

from __future__ import annotations

import pytest

pytest.importorskip("streamlit")

from pytest_streamlit import StreamlitRunner


def _hud_demo_app() -> None:
    from app.modules import navigation

    navigation.set_active_step("generator")
    navigation.render_mission_hud()



def test_mission_hud_limits_actions() -> None:
    """The compact HUD should never expose more than three actions."""

    runner = StreamlitRunner(_hud_demo_app)
    app = runner.run()

    hud_markup = [
        body
        for body in (
            getattr(block, "body", "") for block in getattr(app, "markdown", [])
        )
        if isinstance(body, str) and '<div class="mission-hud' in body
    ]

    assert hud_markup, "Mission HUD markup was not rendered"
    assert hud_markup[0].count("<a class='mission-hud__action") <= 3
    assert "Ajustes" in hud_markup[0]
