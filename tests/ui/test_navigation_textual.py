from __future__ import annotations

import types

import pytest


@pytest.fixture
def fake_streamlit(monkeypatch: pytest.MonkeyPatch) -> types.SimpleNamespace:
    from app.modules import navigation

    proxy: dict[str, object] = {}
    st = types.SimpleNamespace(session_state=proxy, caption=lambda text: None)
    monkeypatch.setattr(navigation, "st", st)
    return st


def test_set_active_step_updates_session(fake_streamlit: types.SimpleNamespace) -> None:
    from app.modules import navigation

    step = navigation.set_active_step("generator")

    assert step.key == "generator"
    assert fake_streamlit.session_state["mission_active_step"] == "generator"


def test_render_breadcrumbs_is_plain_text(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.modules import navigation

    rendered: list[str] = []
    monkeypatch.setattr(
        navigation.st,
        "caption",
        lambda text: rendered.append(text),
        raising=False,
    )

    step = navigation.get_step("results")
    labels = navigation.render_breadcrumbs(step, extra=["Detalle"])

    assert labels == ["Home", "Brief", "Inventario", "Target", "Generador", "Resultados", "Detalle"]
    assert rendered and rendered[-1] == "Home › Brief › Inventario › Target › Generador › Resultados › Detalle"


def test_render_stepper_returns_summary(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.modules import navigation

    captured: list[str] = []
    monkeypatch.setattr(
        navigation.st,
        "caption",
        lambda text: captured.append(text),
        raising=False,
    )

    step = navigation.get_step("playbooks")
    summary = navigation.render_stepper(step)

    assert summary == "Paso 8 de 10 · Playbooks"
    assert captured[-1] == summary
