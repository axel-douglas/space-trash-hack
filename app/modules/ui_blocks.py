from pathlib import Path
from typing import Literal

import streamlit as st

_THEME_KEY = "__rexai_theme_loaded__"


def _theme_path() -> Path:
    return Path(__file__).resolve().parents[1] / "static" / "theme.css"


def load_theme() -> None:
    """Inject the shared Rex-AI theme CSS once per Streamlit session."""

    if st.session_state.get(_THEME_KEY):
        return

    theme_file = _theme_path()
    try:
        css = theme_file.read_text(encoding="utf-8")
    except FileNotFoundError:
        return

    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
    st.session_state[_THEME_KEY] = True


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
