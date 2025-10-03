import pytest


@pytest.fixture(autouse=True)
def _stub_load_theme(monkeypatch):
    from app.modules import ui_blocks

    monkeypatch.setattr(ui_blocks, "load_theme", lambda show_hud=False: None)


@pytest.mark.parametrize(
    ("kind", "expected_title"),
    (
        ("info", "Referencia informativa"),
        ("accent", "Etiqueta destacada"),
    ),
)
def test_pill_serialises_extended_tones(kind, expected_title):
    from app.modules import ui_blocks

    html = ui_blocks.pill("Etiqueta", kind=kind, render=False)

    assert f"data-mission-pill='{kind}'" in html
    assert f"data-kind='{kind}'" in html
    assert f"title='{expected_title}'" in html


def test_initialise_frontend_force_resets_theme_cache(monkeypatch):
    from types import SimpleNamespace

    from app.modules import ui_blocks

    state = {ui_blocks._THEME_HASH_KEY: "cached"}
    calls: list[str] = []

    def fake_load_theme() -> None:
        assert ui_blocks._THEME_HASH_KEY not in state
        calls.append("load")

    monkeypatch.setattr(ui_blocks, "load_theme", fake_load_theme)
    monkeypatch.setattr(ui_blocks, "apply_global_visual_theme", lambda: None)
    monkeypatch.setattr(ui_blocks, "st", SimpleNamespace(session_state=state))

    ui_blocks.initialise_frontend(force=True)

    assert calls == ["load"]
    assert ui_blocks._THEME_HASH_KEY not in state
