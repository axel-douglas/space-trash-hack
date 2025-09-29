"""Shared data-visualization themes for Altair and Plotly.

The theme is inspired by premium automotive dashboards: deep, satin-finished
surfaces contrasted with electric neon highlights.  A single entry point is
provided so that pages only need to import :mod:`_bootstrap` to enable the
visual identity across Altair, Plotly and CSS styling.
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
        background="#F8FAFC",
        surface="#FFFFFF",
        panel="#E2E8F0",
        grid="rgba(30,64,175,0.14)",
        text="#0F172A",
        muted="#475569",
        accent="#1D4ED8",
        accent_soft="#60A5FA",
        electric_gradient=("#7CF4FF", "#33B9FF", "#3730FF"),
        categorical=(
            "#0F76FF",
            "#1DD3F8",
            "#F59E0B",
            "#14B8A6",
            "#7C3AED",
            "#F97316",
        ),
    ),
    "dark": Palette(
        background="#070B12",
        surface="#0E141F",
        panel="#192132",
        grid="rgba(148,163,184,0.22)",
        text="#F8FAFC",
        muted="#94A3B8",
        accent="#38BDF8",
        accent_soft="#7DD3FC",
        electric_gradient=("#7CF4FF", "#2AA8FF", "#4C4CFF"),
        categorical=(
            "#7DD3FC",
            "#34D399",
            "#FBBF24",
            "#F472B6",
            "#A855F7",
            "#F97316",
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
                "strokeOpacity": 0.0,
            },
            "padding": 16,
            "title": {
                "font": "Rajdhani, 'Segoe UI', sans-serif",
                "fontSize": 22,
                "fontWeight": 600,
                "color": palette.text,
            },
            "axis": {
                "labelFont": "Inter, 'Segoe UI', sans-serif",
                "labelFontSize": 12,
                "labelColor": palette.text,
                "titleFont": "Rajdhani, 'Segoe UI', sans-serif",
                "titleFontWeight": 500,
                "titleColor": palette.muted,
                "grid": True,
                "gridColor": palette.grid,
                "domainColor": palette.grid,
                "tickColor": palette.grid,
            },
            "legend": {
                "labelFont": "Inter, 'Segoe UI', sans-serif",
                "labelColor": palette.text,
                "titleColor": palette.muted,
                "symbolType": "circle",
                "gradientLength": 180,
            },
            "header": {
                "labelFont": "Inter, 'Segoe UI', sans-serif",
                "titleFont": "Rajdhani, 'Segoe UI', sans-serif",
                "labelColor": palette.text,
                "titleColor": palette.text,
            },
            "mark": {
                "color": palette.accent,
                "fill": palette.accent_soft,
                "stroke": palette.accent,
                "strokeWidth": 1.4,
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
                "size": 90,
                "filled": True,
            },
            "bar": {
                "cornerRadiusTopLeft": 4,
                "cornerRadiusTopRight": 4,
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
    font_family = "Inter, 'Segoe UI', sans-serif"
    title_family = "Rajdhani, 'Segoe UI', sans-serif"

    return {
        "layout": {
            "paper_bgcolor": palette.background,
            "plot_bgcolor": palette.surface,
            "font": {"family": font_family, "color": palette.text, "size": 14},
            "title": {"font": {"family": title_family, "size": 22, "color": palette.text}},
            "colorway": list(palette.categorical),
            "hoverlabel": {
                "font": {"family": font_family, "color": palette.text},
                "bgcolor": palette.panel,
                "bordercolor": palette.accent,
            },
            "xaxis": {
                "gridcolor": palette.grid,
                "zerolinecolor": palette.accent_soft,
                "linecolor": palette.grid,
                "ticks": "outside",
                "tickcolor": palette.grid,
                "titlefont": {"family": title_family, "color": palette.muted},
            },
            "yaxis": {
                "gridcolor": palette.grid,
                "zerolinecolor": palette.accent_soft,
                "linecolor": palette.grid,
                "ticks": "outside",
                "tickcolor": palette.grid,
                "titlefont": {"family": title_family, "color": palette.muted},
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
                        "line": {"color": palette.panel, "width": 0.8},
                        "colorscale": gradient,
                    }
                }
            ],
            "scatter": [
                {
                    "marker": {
                        "line": {"color": palette.panel, "width": 0.8},
                        "colorscale": gradient,
                        "size": 12,
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
    """Entry-point used by :mod:`_bootstrap` to configure the visualization theme."""

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
