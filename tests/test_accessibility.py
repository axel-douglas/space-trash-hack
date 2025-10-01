from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Tuple

CSS_PATH = Path(__file__).resolve().parents[1] / "app" / "static" / "styles" / "base.css"

TOKEN_KEYS = {
    "color-background",
    "color-surface",
    "color-surface-raised",
    "color-text",
    "color-accent",
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


def _extract_root_tokens(css: str) -> Dict[str, str]:
    compiled = re.compile(r":root\s*\{(?P<body>[^}]*)\}", re.MULTILINE | re.DOTALL)
    match = compiled.search(css)
    if not match:
        raise AssertionError("Root token block not found in base stylesheet")

    tokens: Dict[str, str] = {}
    body = match.group("body")
    for line in body.split(";"):
        if not line.strip():
            continue
        if ":" not in line:
            continue
        key, raw_value = line.split(":", 1)
        key = key.strip()
        value = raw_value.strip()
        if key.startswith("--"):
            tokens[key[2:]] = value
    return tokens


def _load_css() -> str:
    if not CSS_PATH.exists():
        raise FileNotFoundError(f"CSS tokens not found at {CSS_PATH}")
    return CSS_PATH.read_text(encoding="utf-8")


def test_base_tokens_present() -> None:
    css = _load_css()
    tokens = _extract_root_tokens(css)
    missing = TOKEN_KEYS - tokens.keys()
    assert not missing, f"Missing CSS variables: {missing}"


def test_base_tokens_have_contrast() -> None:
    css = _load_css()
    tokens = _extract_root_tokens(css)

    bg = _parse_color(tokens["color-background"])
    surface = _parse_color(tokens["color-surface"])
    raised = _parse_color(tokens["color-surface-raised"])
    ink = _parse_color(tokens["color-text"])
    accent = _parse_color(tokens["color-accent"])

    assert _contrast_ratio(bg, ink) >= 7.0, "Background/text contrast should be AA compliant"
    assert _contrast_ratio(surface, ink) >= 5.0, "Surface/text contrast should remain readable"
    assert _contrast_ratio(raised, ink) >= 4.5, "Raised surface contrast should remain readable"
    assert _contrast_ratio(accent, bg) >= 4.5, "Accent should pop against the canvas"
    assert _contrast_ratio(accent, surface) >= 4.0, "Accent should remain visible over surfaces"
