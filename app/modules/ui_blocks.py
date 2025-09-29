from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Literal, Optional

import streamlit as st

_THEME_KEY = "__rexai_theme_loaded__"


def _theme_path() -> Path:
    return Path(__file__).resolve().parents[1] / "static" / "theme.css"


def load_theme(show_hud: bool = True) -> None:
    """Inject the shared Rex-AI theme CSS once per Streamlit session and Mission HUD."""

    theme_loaded = st.session_state.get(_THEME_KEY)
    if not theme_loaded:
        theme_file = _theme_path()
        try:
            css = theme_file.read_text(encoding="utf-8")
        except FileNotFoundError:
            css = ""

        if css:
            st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
        st.session_state[_THEME_KEY] = True

    if show_hud:
        from app.modules.navigation import render_mission_hud

        render_mission_hud()


def use_token(name: str, fallback: Optional[str] = None) -> str:
    """Return a CSS var reference for the given design token."""

    sanitized = name.strip().lower().replace(" ", "-")
    sanitized = sanitized.replace("/", "-").replace("_", "-").replace(".", "-")
    css_var = f"--{sanitized}"
    if fallback is not None:
        return f"var({css_var}, {fallback})"
    return f"var({css_var})"


def _surface_markup(
    *,
    tone: str,
    padding: Optional[str],
    shadow: Optional[str],
    radius: Optional[str],
    extra_class: str = "",
) -> str:
    classes = ["rex-surface"]
    if extra_class:
        classes.append(extra_class)
    attrs = []
    if tone and tone != "base":
        attrs.append(f'data-tone="{tone}"')

    style_bits = []
    if padding:
        style_bits.append(f"padding: var(--space-{padding});")
    if shadow:
        style_bits.append(f"box-shadow: var(--shadow-{shadow});")
    if radius:
        style_bits.append(f"border-radius: {radius};")

    class_name = " ".join(classes)
    style_attr = ""
    if style_bits:
        style_value = " ".join(style_bits)
        style_attr = f' style="{style_value}"'
    attr_segment = (" " + " ".join(attrs)) if attrs else ""
    return f'<div class="{class_name}"{attr_segment}{style_attr}>'


@contextmanager
def surface(
    *,
    tone: Literal["base", "sunken", "raised"] = "base",
    padding: str | None = "lg",
    shadow: Literal["soft", "lift", "float"] | None = "soft",
    radius: str | None = None,
) -> Iterator[st.delta_generator.DeltaGenerator]:
    """Render content inside a themed surface wrapper."""

    load_theme()
    container = st.container()
    opener = _surface_markup(tone=tone, padding=padding, shadow=shadow, radius=radius)
    container.markdown(opener, unsafe_allow_html=True)
    inner = container.container()
    try:
        with inner:
            yield inner
    finally:
        container.markdown("</div>", unsafe_allow_html=True)


@contextmanager
def glass_card(
    *,
    padding: str | None = "lg",
    shadow: Literal["soft", "lift", "float"] | None = "float",
    radius: str | None = None,
) -> Iterator[st.delta_generator.DeltaGenerator]:
    """Render content inside a frosted glass style surface."""

    load_theme()
    container = st.container()
    opener = _surface_markup(
        tone="base", padding=padding, shadow=shadow, radius=radius, extra_class="rex-glass"
    )
    container.markdown(opener, unsafe_allow_html=True)
    inner = container.container()
    try:
        with inner:
            yield inner
    finally:
        container.markdown("</div>", unsafe_allow_html=True)


def inject_css():
    """Backward-compatible alias for legacy code paths."""

    load_theme()

def card(title:str, body:str=""):
    st.markdown(f"""<div class="card"><h4>{title}</h4><div class="small">{body}</div></div>""", unsafe_allow_html=True)

def pill(label:str, kind:Literal["ok","warn","risk"]="ok"):
    klass = {"ok":"badge-ok","warn":"badge-warn","risk":"badge-risk"}[kind]
    st.markdown(f"""<span class="pill {klass}">{label}</span>""", unsafe_allow_html=True)

def section(title:str, subtitle:str=""):
    st.subheader(title)
    if subtitle:
        st.caption(subtitle)
