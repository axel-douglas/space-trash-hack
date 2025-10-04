from pathlib import Path

from app.modules import visual_theme
from app.modules.ui_blocks import _PAGE_THEME


def test_base_css_defines_new_primitives() -> None:
    css = Path("app/static/styles/base.css").read_text(encoding="utf-8")

    for token in (
        "--mission-color-canvas",
        "--mission-color-surface",
        "--mission-space-md",
    ):
        assert token in css, f"Expected {token} token in base theme"

    for legacy in (".rex-pill", ".chipline__chip", ".rex-card", "linear-gradient"):
        assert legacy not in css


def test_page_theme_aligns_with_visual_palette() -> None:
    palette = visual_theme.get_palette()
    assert _PAGE_THEME["backgroundColor"].lower() == palette.background.lower()
