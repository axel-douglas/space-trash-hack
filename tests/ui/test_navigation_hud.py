"""Regression tests for the mission HUD navigation bar."""

from __future__ import annotations

import types

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


def test_refresh_model_metadata_updates_without_restart(monkeypatch) -> None:
    """Refreshing metadata should expose latest training info without restart."""

    from app.modules import navigation

    session_proxy: dict[str, object] = {}
    mock_streamlit = types.SimpleNamespace(session_state=session_proxy)

    monkeypatch.setattr(navigation, "st", mock_streamlit)

    class _Registry:
        def __init__(self) -> None:
            self.ready = True
            self.metadata = {
                "model_name": "demo-regressor",
                "trained_label": "Simulación",
                "trained_at": "2024-01-01",
            }
            self._uncertainty = "reportada"

        def uncertainty_label(self) -> str:  # noqa: D401
            return self._uncertainty

    registry = _Registry()

    def _fake_registry() -> _Registry:
        return registry

    _fake_registry.cleared = False

    def _fake_clear() -> None:
        _fake_registry.cleared = True

    _fake_registry.clear = _fake_clear  # type: ignore[attr-defined]

    monkeypatch.setattr(navigation, "get_model_registry", _fake_registry)

    first = navigation._model_metadata()
    assert first["trained_at"] == "2024-01-01"
    assert first["uncertainty_badge"]["tone"] == "success"

    registry.metadata["trained_at"] = "2024-02-02"
    registry._uncertainty = "alta"

    cached = navigation._model_metadata()
    assert cached["trained_at"] == "2024-01-01", "Cache should remain until manual refresh"

    navigation.refresh_model_metadata()
    assert _fake_registry.cleared is True

    updated = navigation._model_metadata()
    assert updated["trained_at"] == "2024-02-02"
    assert updated["uncertainty_badge"]["tone"] == "danger"

    registry.ready = False
    registry.metadata["trained_at"] = "2024-03-03"
    registry._uncertainty = "reportada"

    auto_updated = navigation._model_metadata()
    assert auto_updated["trained_at"] == "2024-03-03"
    assert auto_updated["status_badge"]["tone"] == "danger"
    assert auto_updated["uncertainty_badge"]["tone"] == "success"
