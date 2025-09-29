from __future__ import annotations

from typing import Iterable, Literal

import streamlit as st

_THEME_STATE_KEY = "hud_theme"
_COLORBLIND_STATE_KEY = "hud_colorblind"


def _current_theme() -> str:
    return st.session_state.get(_THEME_STATE_KEY, "dark")


def _is_high_contrast() -> bool:
    return "high-contrast" in _current_theme()


def _is_colorblind_mode() -> bool:
    return st.session_state.get(_COLORBLIND_STATE_KEY, "normal") != "normal"


def _class_names(*tokens: Iterable[str]) -> str:
    classes: list[str] = []
    for token in tokens:
        if isinstance(token, str):
            classes.append(token)
        else:
            classes.extend(t for t in token if t)
    return " ".join(cls for cls in classes if cls)


def render_card(title: str, body: str = "") -> str:
    classes = ["card"]
    if _is_high_contrast():
        classes.append("card-flat")

    body_html = f'<div class="small">{body}</div>' if body else ""
    return f'<div class="{_class_names(classes)}"><h4>{title}</h4>{body_html}</div>'


def render_pill(label: str, kind: Literal["ok", "warn", "risk"] = "ok") -> str:
    tone_map = {
        "ok": ("ok",),
        "warn": ("warn", "med"),
        "risk": ("risk", "bad"),
    }
    classes = ["pill", *tone_map.get(kind, ("ok",))]
    if _is_colorblind_mode():
        classes.append("pill-solid")
    return f'<span class="{_class_names(classes)}">{label}</span>'
