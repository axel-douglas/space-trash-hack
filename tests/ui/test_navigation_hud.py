"""Regression tests for the mission HUD navigation bar."""

from __future__ import annotations

import pytest

pytest.importorskip("streamlit")

from pytest_streamlit import StreamlitRunner


def _hud_demo_app() -> None:
    from app.modules import navigation

    navigation.st.session_state["target"] = {
        "scenario": "Demo",
        "max_water_l": 120,
    }
    navigation.set_active_step("generator")
    navigation.render_mission_hud()



def _fake_registry():  # noqa: D401
    class _Registry:
        ready = True
        metadata = {
            "model_name": "demo-regressor",
            "trained_label": "Simulación",
            "trained_at": "2024-01-01",
        }

        def uncertainty_label(self) -> str:  # noqa: D401
            return "reportada"

    return _Registry()


def test_mission_hud_limits_actions(monkeypatch) -> None:
    """The compact HUD should never expose more than three actions."""

    from app.modules import navigation

    monkeypatch.setattr(navigation, "get_model_registry", _fake_registry)

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
    assert "Editar target" in hud_markup[0]


def test_mission_hud_hides_settings_without_target(monkeypatch) -> None:
    """The target edit CTA should be hidden when session lacks objectives."""

    from app.modules import navigation

    monkeypatch.setattr(navigation, "get_model_registry", _fake_registry)

    def _app_without_target() -> None:
        from app.modules import navigation

        navigation.set_active_step("generator")
        navigation.render_mission_hud()

    runner = StreamlitRunner(_app_without_target)
    app = runner.run()

    hud_markup = "".join(
        body
        for body in (
            getattr(block, "body", "") for block in getattr(app, "markdown", [])
        )
        if isinstance(body, str) and '<div class="mission-hud' in body
    )

    assert hud_markup, "Mission HUD markup was not rendered"
    assert "Editar target" not in hud_markup
