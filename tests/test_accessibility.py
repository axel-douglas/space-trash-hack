from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Tuple

CSS_PATH = Path(__file__).resolve().parents[1] / "app" / "static" / "design_tokens.css"

THEME_KEYS = {
    "bg",
    "surface-card",
    "accent",
    "ink",
    "muted",
}


def _relative_luminance(rgb: Tuple[int, int, int]) -> float:
    def channel(c: int) -> float:
        c = c / 255
        return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4

    r, g, b = (channel(x) for x in rgb)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _contrast_ratio(rgb1: Tuple[int, int, int], rgb2: Tuple[int, int, int]) -> float:
    l1 = _relative_luminance(rgb1)
    l2 = _relative_luminance(rgb2)
    lighter, darker = max(l1, l2), min(l1, l2)
    return (lighter + 0.05) / (darker + 0.05)


def _parse_color(value: str) -> Tuple[int, int, int]:
    value = value.strip()
    if value.startswith("#"):
        hex_value = value[1:]
        if len(hex_value) == 3:
            hex_value = "".join(ch * 2 for ch in hex_value)
        if len(hex_value) != 6:
            raise ValueError(f"Unsupported hex color: {value}")
        return tuple(int(hex_value[i : i + 2], 16) for i in range(0, 6, 2))  # type: ignore[misc]
    if value.startswith("rgb"):
        parts = re.findall(r"[0-9]+", value)
        if len(parts) < 3:
            raise ValueError(f"Unsupported rgb color: {value}")
        return tuple(int(parts[i]) for i in range(3))  # type: ignore[misc]
    raise ValueError(f"Unsupported color format: {value}")


def _extract_blocks(pattern: str, css: str) -> Dict[str, Dict[str, str]]:
    compiled = re.compile(pattern, re.MULTILINE | re.DOTALL)
    result: Dict[str, Dict[str, str]] = {}
    for match in compiled.finditer(css):
        name = match.group("name")
        body = match.group("body")
        declarations: Dict[str, str] = {}
        for line in body.split(";"):
            if not line.strip():
                continue
            if ":" not in line:
                continue
            key, raw_value = line.split(":", 1)
            key = key.strip()
            value = raw_value.strip()
            if key.startswith("--"):
                declarations[key[2:]] = value
        result[name] = declarations
    return result


def _load_css() -> str:
    if not CSS_PATH.exists():
        raise FileNotFoundError(f"CSS tokens not found at {CSS_PATH}")
    return CSS_PATH.read_text(encoding="utf-8")


def test_theme_tokens_have_contrast() -> None:
    css = _load_css()
    themes = _extract_blocks(r'body\[data-rexai-theme="(?P<name>[^"]+)"[^\{]*\{(?P<body>[^}]*)\}', css)
    assert themes, "No theme tokens found"

    for theme_name, tokens in themes.items():
        missing = THEME_KEYS - tokens.keys()
        assert not missing, f"Theme '{theme_name}' missing tokens: {missing}"

        bg = _parse_color(tokens["bg"])
        ink = _parse_color(tokens["ink"])
        surface = _parse_color(tokens.get("surface-card", tokens["bg"]))
        accent = _parse_color(tokens["accent"])

        base_ratio = _contrast_ratio(bg, ink)
        surface_ratio = _contrast_ratio(surface, ink)
        accent_ratio = _contrast_ratio(accent, bg)

        min_base = 7.0 if "high-contrast" in theme_name else 4.5
        min_surface = 4.5 if "high-contrast" in theme_name else 3.0
        min_accent = 4.5 if "high-contrast" in theme_name else 3.0

        assert base_ratio >= min_base, f"Theme '{theme_name}' background/text contrast {base_ratio:.2f} < {min_base}"
        assert surface_ratio >= min_surface, f"Theme '{theme_name}' surface/text contrast {surface_ratio:.2f} < {min_surface}"
        assert accent_ratio >= min_accent, f"Theme '{theme_name}' accent/background contrast {accent_ratio:.2f} < {min_accent}"


def test_colorblind_accent_remains_distinct() -> None:
    css = _load_css()
    themes = _extract_blocks(r'body\[data-rexai-theme="(?P<name>[^"]+)"[^\{]*\{(?P<body>[^}]*)\}', css)
    overrides = _extract_blocks(r'body\[data-rexai-colorblind="(?P<name>[^"]+)"[^\{]*\{(?P<body>[^}]*)\}', css)

    safe = overrides.get("safe")
    assert safe, "Colorblind safe mode overrides not found"
    assert "accent" in safe, "Colorblind safe mode must override accent"

    accent_override = _parse_color(safe["accent"])
    for theme_name, tokens in themes.items():
        bg = _parse_color(tokens["bg"])
        ratio = _contrast_ratio(accent_override, bg)
        assert ratio >= 3.0, f"Colorblind accent contrast too low for theme '{theme_name}': {ratio:.2f}"
