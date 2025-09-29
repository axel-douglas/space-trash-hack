from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Literal

import streamlit as st

from . import luxe_components as luxe

_THEME_HASH_KEY = "__rexai_theme_hash__"
_THEME_STATE_KEY = "hud_theme"
_FONT_STATE_KEY = "hud_font"
_COLORBLIND_STATE_KEY = "hud_colorblind"

_THEME_LABELS = {
    "dark": "Oscuro",
    "dark-high-contrast": "Oscuro · Alto contraste",
    "light": "Claro",
    "light-high-contrast": "Claro · Alto contraste",
    "solarized": "Solarized",
}

_FONT_LABELS = {
    "base": "Base",
    "large": "Grande",
    "xlarge": "XL",
}

_COLORBLIND_LABELS = {
    "normal": "Sin ajustes",
    "safe": "Daltonismo (paleta segura)",
}


def _static_path(filename: str) -> Path:
    return Path(__file__).resolve().parents[1] / "static" / filename


def _theme_path() -> Path:
    return _static_path("theme.css")


def _tokens_path() -> Path:
    return _static_path("design_tokens.css")


def _ensure_defaults() -> None:
    st.session_state.setdefault(_THEME_STATE_KEY, "dark")
    st.session_state.setdefault(_FONT_STATE_KEY, "base")
    st.session_state.setdefault(_COLORBLIND_STATE_KEY, "normal")


def _read_css_bundle() -> str:
    css_parts: list[str] = []
    for path in (_tokens_path(), _theme_path()):
        try:
            css_parts.append(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            continue
    return "\n".join(css_parts)


def _inject_css_once(css: str) -> None:
    if not css:
        return

    css_hash = hashlib.sha256(css.encode("utf-8")).hexdigest()
    if st.session_state.get(_THEME_HASH_KEY) == css_hash:
        return

    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
    st.session_state[_THEME_HASH_KEY] = css_hash


def _apply_runtime_theme() -> None:
    theme = st.session_state.get(_THEME_STATE_KEY, "dark")
    font = st.session_state.get(_FONT_STATE_KEY, "base")
    colorblind = st.session_state.get(_COLORBLIND_STATE_KEY, "normal")

    script = f"""
    <script>
    const doc = window.parent?.document ?? document;
    const body = doc.body;
    if (body) {{
      body.setAttribute('data-rexai-theme', '{theme}');
      body.setAttribute('data-rexai-font', '{font}');
      body.setAttribute('data-rexai-colorblind', '{colorblind}');
    }}
    const appRoot = doc.querySelector('.stApp');
    if (appRoot) {{
      appRoot.setAttribute('data-rexai-theme', '{theme}');
      appRoot.setAttribute('data-rexai-font', '{font}');
      appRoot.setAttribute('data-rexai-colorblind', '{colorblind}');
    }}
    </script>
    """
    st.markdown(script, unsafe_allow_html=True)


def _render_hud() -> None:
    hud_container = st.container()
    with hud_container:
        st.markdown('<div class="rexai-hud">', unsafe_allow_html=True)
        st.markdown('<div class="rexai-hud__title">HUD · Accesibilidad</div>', unsafe_allow_html=True)
        cols = st.columns(3)
        with cols[0]:
            st.selectbox(
                "Tema",
                options=list(_THEME_LABELS.keys()),
                format_func=lambda value: _THEME_LABELS.get(value, value),
                key=_THEME_STATE_KEY,
                help="Ajustá contraste y estética base.",
            )
        with cols[1]:
            st.radio(
                "Tipografía",
                options=list(_FONT_LABELS.keys()),
                format_func=lambda value: _FONT_LABELS.get(value, value),
                key=_FONT_STATE_KEY,
                horizontal=True,
                help="Escala de fuente para lectura cómoda.",
            )
        with cols[2]:
            st.selectbox(
                "Modo daltónico",
                options=list(_COLORBLIND_LABELS.keys()),
                format_func=lambda value: _COLORBLIND_LABELS.get(value, value),
                key=_COLORBLIND_STATE_KEY,
                help="Paleta optimizada para protanopia/deuteranopia.",
            )
        st.markdown('</div>', unsafe_allow_html=True)


def load_theme(*, show_hud: bool = True) -> None:
    """Inject shared CSS and expose HUD toggles for the Rex-AI theme."""

    _ensure_defaults()
    css_bundle = _read_css_bundle()
    _inject_css_once(css_bundle)
    _apply_runtime_theme()

    if show_hud:
        _render_hud()


def inject_css(show_hud: bool = False) -> None:
    """Backward-compatible alias for legacy code paths."""

    load_theme(show_hud=show_hud)


def card(title: str, body: str = "") -> None:
    st.markdown(luxe.render_card(title, body), unsafe_allow_html=True)


def pill(label: str, kind: Literal["ok", "warn", "risk"] = "ok") -> None:
    st.markdown(luxe.render_pill(label, kind), unsafe_allow_html=True)


def section(title: str, subtitle: str = "") -> None:
    st.subheader(title)
    if subtitle:
        st.caption(subtitle)
