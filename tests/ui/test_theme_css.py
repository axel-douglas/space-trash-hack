from pathlib import Path


def test_base_css_defines_new_primitives() -> None:
    css = Path("app/static/styles/base.css").read_text(encoding="utf-8")

    for selector in (".rex-pill", ".chipline__chip", ".rex-card"):
        assert selector in css, f"Expected {selector} styles in base theme"

    for legacy in (".luxe-", "tesla-hero", "parallax"):
        assert legacy not in css
