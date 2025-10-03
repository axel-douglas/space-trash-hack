"""NASA minimal visualization themes for Altair and Plotly.

The module codifies a compact palette derived from the NASA "meatball"
guidelines: desaturated instrumentation neutrals, clean whites, and the
mission-operations blue used as the primary accent.  The aim is to give charts
and dashboards the feeling of a console readout—high legibility, zero
ornamentation, and consistent contrast across libraries.  The same design
tokens are exported to Altair, Plotly, and the CSS layer so that whichever
backend renders a chart it stays aligned with the mission-control aesthetic.
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
FONT_STACK = "'Source Sans 3', 'Segoe UI', sans-serif"


@dataclass(frozen=True)
class Palette:
    """Minimal high-contrast palette shared between backends."""

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
        background="#F5F7FA",
        surface="#FFFFFF",
        panel="#E1E6EF",
        grid="rgba(18,36,66,0.24)",
        text="#0B1526",
        muted="#465164",
        accent="#0B3D91",
        accent_soft="#4D6FB8",
        electric_gradient=("#C9D4E8", "#7B93C9", "#1F3D7A"),
        categorical=(
            "#0B3D91",
            "#345D9C",
            "#B65E15",
            "#1D5F48",
            "#5A4FBF",
            "#A13A4A",
        ),
    ),
    "dark": Palette(
        background="#050A14",
        surface="#0F172A",
        panel="#1C2840",
        grid="rgba(148,163,184,0.35)",
        text="#F8FAFC",
        muted="#9AA5BF",
        accent="#5A8DEE",
        accent_soft="#90AAEF",
        electric_gradient=("#1D3B6F", "#325EA3", "#8FB8FF"),
        categorical=(
            "#5A8DEE",
            "#3F9BB8",
            "#D0812A",
            "#4FA079",
            "#8579D6",
            "#C65B6C",
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
                "strokeOpacity": 0.4,
            },
            "padding": 16,
            "title": {
                "font": FONT_STACK,
                "fontSize": 20,
                "fontWeight": 600,
                "color": palette.text,
            },
            "axis": {
                "labelFont": FONT_STACK,
                "labelFontSize": 12,
                "labelColor": palette.muted,
                "titleFont": FONT_STACK,
                "titleFontWeight": 600,
                "titleColor": palette.text,
                "grid": True,
                "gridColor": palette.grid,
                "domainColor": palette.grid,
                "tickColor": palette.grid,
                "tickSize": 4,
            },
            "legend": {
                "labelFont": FONT_STACK,
                "labelColor": palette.muted,
                "titleColor": palette.text,
                "symbolType": "square",
                "gradientLength": 140,
            },
            "header": {
                "labelFont": FONT_STACK,
                "titleFont": FONT_STACK,
                "labelColor": palette.muted,
                "titleColor": palette.text,
            },
            "mark": {
                "color": palette.accent,
                "fill": palette.accent_soft,
                "stroke": palette.accent,
                "strokeWidth": 1.1,
            },
            "range": {
                "category": list(palette.categorical),
                "diverging": list(palette.electric_gradient),
                "heatmap": list(palette.electric_gradient),
                "ramp": list(palette.electric_gradient),
            },
            "area": {
                "line": True,
                "opacity": 0.85,
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
    font_family = FONT_STACK
    title_family = FONT_STACK

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
