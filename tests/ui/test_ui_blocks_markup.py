import types

import pytest


@pytest.fixture(autouse=True)
def _stub_load_theme(monkeypatch):
    from app.modules import ui_blocks

    monkeypatch.setattr(ui_blocks, "load_theme", lambda show_hud=False: None)


def test_pill_returns_markup_without_render():
    from app.modules import ui_blocks

    html = ui_blocks.pill("Seguridad", kind="ok", render=False)

    assert "data-lab-pill='ok'" in html
    assert "var(--lab-color-positive)" in html


def test_chipline_accepts_mappings(monkeypatch):
    from app.modules import ui_blocks

    fake_streamlit = types.SimpleNamespace(markdown=lambda *args, **kwargs: None)
    monkeypatch.setattr(ui_blocks, "st", fake_streamlit)

    html = ui_blocks.chipline(
        [
            {"label": "PFAS controlados", "icon": "🧪", "tone": "positive"},
            "Crew listo",
        ],
        render=False,
    )

    assert "data-lab-chip-tone='positive'" in html
    assert "🧪" in html
    assert "var(--lab-color-positive-soft)" in html
