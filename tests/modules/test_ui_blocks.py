from pathlib import Path

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


def test_logo_markup_reuses_cached_svg(monkeypatch, tmp_path):
    from app.modules import ui_blocks

    svg_path = tmp_path / "logo.svg"
    svg_path.write_text("<svg></svg>", encoding="utf-8")

    original_static_path = ui_blocks._static_path

    def fake_static_path(filename: str | Path) -> Path:
        if Path(filename) == Path(ui_blocks._BRAND_LOGO_FILENAME):
            return svg_path
        return original_static_path(filename)

    ui_blocks._encode_svg_base64.cache_clear()
    monkeypatch.setattr(ui_blocks, "_static_path", fake_static_path)

    read_count = 0
    original_read_text = Path.read_text

    def counting_read_text(self, *args, **kwargs):
        nonlocal read_count
        if self == svg_path:
            read_count += 1
        return original_read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", counting_read_text)

    first_markup = ui_blocks._get_logo_markup()
    second_markup = ui_blocks._get_logo_markup()

    assert first_markup == second_markup
    assert read_count == 1
