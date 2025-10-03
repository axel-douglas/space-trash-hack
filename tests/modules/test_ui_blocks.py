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
