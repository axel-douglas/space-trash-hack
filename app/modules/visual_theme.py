"""NASA minimal visualization themes for Altair and Plotly.

The palette favours clean laboratory whites, aerospace blues, and sharply
defined contrast so that charts feel like instrumentation readouts rather than
entertainment dashboards.  Both Altair and Plotly share the same minimalist
tokens so that Streamlit pages inherit a restrained NASA mission-control
aesthetic regardless of the rendering backend.
"""
from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Dict, Literal

import altair as alt
import plotly.io as pio

ThemeMode = Literal["light", "dark"]

_THEME_ENV_VAR = "REXAI_THEME_MODE"
_DEFAULT_MODE: ThemeMode = "dark"
@dataclass(frozen=True)
class Palette:
    """Palette tokens exposed for downstream documentation or widgets."""

    background: str
    surface: str
    panel: str
    grid: str
    text: str
    muted: str
    accent: str
    accent_soft: str
    electric_gradient: tuple[str, str, str]
    categorical: tuple[str, ...]


_PALETTES: Dict[ThemeMode, Palette] = {
    "light": Palette(
        background="#F4F7FB",
        surface="#FFFFFF",
        panel="#E4EBF7",
        grid="rgba(23,63,134,0.18)",
        text="#0E2140",
        muted="#4B607D",
        accent="#234A91",
        accent_soft="#6E8FD6",
        electric_gradient=("#D7E3FF", "#7FA7FF", "#1F3E7A"),
        categorical=(
            "#234A91",
            "#2A7F9E",
            "#BF6C00",
            "#146B4E",
            "#5A3BA6",
            "#9C2F3F",
        ),
    ),
    "dark": Palette(
        background="#0B1526",
        surface="#141F32",
        panel="#1F2B41",
        grid="rgba(165,179,201,0.24)",
        text="#F4F7FB",
        muted="#9AA9C2",
        accent="#6F9BFF",
        accent_soft="#A8C1FF",
        electric_gradient=("#4469B8", "#5E87E3", "#B1C7FF"),
        categorical=(
            "#6F9BFF",
            "#49B0C9",
            "#D4973C",
            "#49B188",
            "#9082D5",
            "#CF5E73",
        ),
    ),
}

_REGISTERED = False


def _resolve_mode(mode: str | None = None) -> ThemeMode:
    env_value = os.getenv(_THEME_ENV_VAR)
    candidate = (mode or env_value or _DEFAULT_MODE).lower()
    return "dark" if candidate not in ("light", "dark") else candidate  # type: ignore[return-value]


def _altair_config(mode: ThemeMode) -> Dict[str, Dict[str, object]]:
    palette = _PALETTES[mode]
    background = palette.background
    surface = palette.surface
    return {
        "config": {
            "background": background,
            "view": {
                "stroke": palette.panel,
                "strokeOpacity": 0.35,
            },
            "padding": 16,
            "title": {
                "font": "'Source Sans 3', 'Segoe UI', sans-serif",
                "fontSize": 20,
                "fontWeight": 600,
                "color": palette.text,
            },
            "axis": {
                "labelFont": "'Source Sans 3', 'Segoe UI', sans-serif",
                "labelFontSize": 12,
                "labelColor": palette.muted,
                "titleFont": "'Source Sans 3', 'Segoe UI', sans-serif",
                "titleFontWeight": 600,
                "titleColor": palette.text,
                "grid": True,
                "gridColor": palette.grid,
                "domainColor": palette.grid,
                "tickColor": palette.grid,
                "tickSize": 4,
            },
            "legend": {
                "labelFont": "'Source Sans 3', 'Segoe UI', sans-serif",
                "labelColor": palette.muted,
                "titleColor": palette.text,
                "symbolType": "circle",
                "gradientLength": 140,
            },
            "header": {
                "labelFont": "'Source Sans 3', 'Segoe UI', sans-serif",
                "titleFont": "'Source Sans 3', 'Segoe UI', sans-serif",
                "labelColor": palette.muted,
                "titleColor": palette.text,
            },
            "mark": {
                "color": palette.accent,
                "fill": palette.accent_soft,
                "stroke": palette.accent,
                "strokeWidth": 1.2,
            },
            "range": {
                "category": list(palette.categorical),
                "diverging": list(palette.electric_gradient),
                "heatmap": list(palette.electric_gradient),
                "ramp": list(palette.electric_gradient),
            },
            "area": {
                "line": True,
                "opacity": 0.8,
            },
            "rect": {
                "stroke": surface,
                "strokeWidth": 0,
            },
            "point": {
                "size": 80,
                "filled": True,
            },
            "bar": {
                "cornerRadiusTopLeft": 2,
                "cornerRadiusTopRight": 2,
            },
        }
    }


def _plotly_template(mode: ThemeMode) -> Dict[str, object]:
    palette = _PALETTES[mode]
    gradient = [
        [0.0, palette.electric_gradient[0]],
        [0.5, palette.electric_gradient[1]],
        [1.0, palette.electric_gradient[2]],
    ]
    font_family = "'Source Sans 3', 'Segoe UI', sans-serif"
    title_family = "'Source Sans 3', 'Segoe UI', sans-serif"

    return {
        "layout": {
            "paper_bgcolor": palette.background,
            "plot_bgcolor": palette.surface,
            "font": {"family": font_family, "color": palette.text, "size": 14},
            "title": {"font": {"family": title_family, "size": 20, "color": palette.text}},
            "colorway": list(palette.categorical),
            "hoverlabel": {
                "font": {"family": font_family, "color": palette.text},
                "bgcolor": palette.surface,
                "bordercolor": palette.panel,
            },
            "xaxis": {
                "gridcolor": palette.grid,
                "zerolinecolor": palette.grid,
                "linecolor": palette.grid,
                "ticks": "outside",
                "tickcolor": palette.grid,
                "titlefont": {"family": title_family, "color": palette.muted},
                "tickfont": {"family": font_family, "color": palette.muted},
            },
            "yaxis": {
                "gridcolor": palette.grid,
                "zerolinecolor": palette.grid,
                "linecolor": palette.grid,
                "ticks": "outside",
                "tickcolor": palette.grid,
                "titlefont": {"family": title_family, "color": palette.muted},
                "tickfont": {"family": font_family, "color": palette.muted},
            },
            "legend": {
                "bgcolor": "rgba(0,0,0,0)",
                "bordercolor": "rgba(0,0,0,0)",
                "font": {"family": font_family, "color": palette.text},
            },
            "coloraxis": {
                "colorscale": gradient,
                "cmin": 0,
                "cmax": 1,
            },
        },
        "data": {
            "bar": [
                {
                    "marker": {
                        "line": {"color": palette.panel, "width": 0.6},
                        "colorscale": gradient,
                    }
                }
            ],
            "scatter": [
                {
                    "marker": {
                        "line": {"color": palette.panel, "width": 0.6},
                        "colorscale": gradient,
                        "size": 10,
                    }
                }
            ],
            "heatmap": [
                {
                    "colorscale": gradient,
                    "colorbar": {"outlinewidth": 0, "thickness": 12},
                }
            ],
        },
    }


def register_visual_themes(mode: str | None = None) -> ThemeMode:
    """Register and enable the Rex-AI visual themes for Altair and Plotly."""

    global _REGISTERED
    resolved: ThemeMode = _resolve_mode(mode)

    if not _REGISTERED:
        for key in _PALETTES:
            name = f"rexai_{key}"
            alt.themes.register(name, lambda key=key: _altair_config(key))
            pio.templates[name] = _plotly_template(key)
        _REGISTERED = True

    alt.themes.enable(f"rexai_{resolved}")
    pio.templates.default = f"rexai_{resolved}"

    return resolved


def apply_global_visual_theme(mode: str | None = None) -> ThemeMode:
    """Register and activate the visualization theme across supported libraries."""

    resolved = register_visual_themes(mode=mode)

    try:
        from app.modules.ui_blocks import load_theme
    except Exception:  # pragma: no cover - defensive import guard
        return resolved

    try:
        load_theme()
    except Exception:  # pragma: no cover - Streamlit session not ready
        pass

    return resolved


def get_palette(mode: str | None = None) -> Palette:
    """Expose the palette used by the current theme mode."""

    return _PALETTES[_resolve_mode(mode)]


__all__ = [
    "Palette",
    "ThemeMode",
    "apply_global_visual_theme",
    "get_palette",
    "register_visual_themes",
]
