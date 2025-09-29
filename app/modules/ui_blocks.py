from __future__ import annotations

from html import escape
import json
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

import streamlit as st
from streamlit.components.v1 import html as components_html

from app import _bootstrap

_THEME_KEY = "__rexai_theme_loaded__"


def _theme_path() -> Path:
    return Path(__file__).resolve().parents[1] / "static" / "theme.css"


_MICRO_JS = _bootstrap.load_microinteractions_script()

_BUTTON_STYLES = """
.rexai-fx-wrapper{position:relative;display:flex;flex-direction:column;gap:6px;}
.rexai-fx-wrapper[data-width="full"]{width:100%;}
.rexai-fx-wrapper[data-width="auto"]{display:inline-flex;}
.rexai-fx-button{position:relative;border:none;border-radius:16px;padding:14px 18px;font-weight:600;font-size:1rem;color:#0f172a;background:linear-gradient(135deg,rgba(96,165,250,0.92),rgba(14,165,233,0.85));box-shadow:0 12px 26px rgba(14,165,233,0.28);transition:transform 0.18s ease,box-shadow 0.22s ease,filter 0.22s ease;cursor:pointer;overflow:hidden;min-height:54px;}
.rexai-fx-wrapper[data-width="full"] .rexai-fx-button{width:100%;}
.rexai-fx-button:disabled{cursor:not-allowed;opacity:0.7;filter:grayscale(0.3);}
.rexai-fx-button:hover{transform:translateY(-1px);box-shadow:0 18px 32px rgba(59,130,246,0.32);}
.rexai-fx-button:active{transform:scale(0.985);}
.rexai-fx-wrapper[data-state="loading"] .rexai-fx-button{background:linear-gradient(135deg,rgba(59,130,246,0.78),rgba(14,165,233,0.55));box-shadow:0 10px 24px rgba(14,165,233,0.25);cursor:progress;}
.rexai-fx-wrapper[data-state="success"] .rexai-fx-button{background:linear-gradient(135deg,rgba(16,185,129,0.95),rgba(59,130,246,0.65));box-shadow:0 14px 28px rgba(16,185,129,0.32);}
.rexai-fx-wrapper[data-state="error"] .rexai-fx-button{background:linear-gradient(135deg,rgba(248,113,113,0.95),rgba(239,68,68,0.75));box-shadow:0 14px 28px rgba(248,113,113,0.35);}
.rexai-fx-label{position:relative;z-index:2;display:block;text-align:center;letter-spacing:0.01em;}
.rexai-fx-status{font-size:0.78rem;letter-spacing:0.04em;text-transform:uppercase;color:rgba(148,163,184,0.92);text-align:center;transition:opacity 0.18s ease;opacity:0;height:0;}
.rexai-fx-status[data-active="true"]{opacity:1;height:auto;}
.rexai-fx-particles{position:absolute;inset:0;pointer-events:none;overflow:visible;}
.rexai-particle{position:absolute;top:50%;left:50%;width:var(--size);height:var(--size);border-radius:999px;transform:translate(-50%,-50%) scale(0.3);transition:transform 0.45s ease,opacity 0.45s ease;}
.rexai-fx-wrapper[data-state="loading"] .rexai-fx-button::after{content:"";position:absolute;inset:12px;width:28px;height:28px;margin:auto;border-radius:999px;border:3px solid rgba(255,255,255,0.38);border-top-color:rgba(15,23,42,0.82);animation:rexai-spin 0.8s linear infinite;z-index:1;}
.rexai-fx-wrapper[data-state="loading"] .rexai-fx-label{opacity:0.35;}
.rexai-fx-wrapper[data-state="error"] .rexai-fx-button{color:#fff;}
.rexai-fx-help{font-size:0.82rem;color:rgba(148,163,184,0.92);margin:0 4px;}
@keyframes rexai-spin{from{transform:rotate(0deg);}to{transform:rotate(360deg);}}
"""


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


def futuristic_button(
    label: str,
    key: str,
    *,
    state: Literal["idle", "loading", "success", "error"] = "idle",
    width: Literal["full", "auto"] = "full",
    help_text: str | None = None,
    loading_label: str | None = None,
    success_label: str | None = None,
    error_label: str | None = None,
    sound: bool = True,
    enable_vibration: bool = False,
    disabled: bool = False,
    status_hints: dict[str, str] | None = None,
) -> bool:
    """Render the futuristic CTA microinteraction button and return ``True`` on click."""

    if state not in {"idle", "loading", "success", "error"}:
        raise ValueError(f"Estado no soportado: {state}")

    state_messages = {
        "idle": label,
        "loading": loading_label or "Procesando…",
        "success": success_label or "Listo",
        "error": error_label or "Reintentar",
    }
    status_hints = status_hints or {
        "idle": "",
        "loading": "Optimizando parámetros",
        "success": "Listo para revisar",
        "error": "Revisá parámetros o intenta de nuevo",
    }

    container_width = "full" if width not in {"full", "auto"} else width
    status_text = status_hints.get(state, "")
    help_html = (
        f'<div class="rexai-fx-help">{escape(help_text)}</div>' if help_text else ""
    )

    label_current = state_messages.get(state, label)
    button_id = f"rexai-fx-{uuid4().hex}"

    config: dict[str, Any] = {
        "state": state,
        "stateMessages": state_messages,
        "statusHints": status_hints,
        "sound": sound,
        "vibration": bool(enable_vibration),
        "vibrationPattern": [8, 14, 4, 18] if enable_vibration else [],
        "particleColors": [
            "rgba(125,211,252,0.85)",
            "rgba(59,130,246,0.95)",
            "rgba(129,140,248,0.85)",
            "rgba(244,114,182,0.8)",
        ],
        "disabled": bool(disabled),
    }

    script_parts = []
    if _MICRO_JS:
        script_parts.append(_MICRO_JS)
    script_parts.append(
        "(function(){"
        "const styleId='rexai-fx-style';"
        f"const styleCSS={json.dumps(_BUTTON_STYLES)};"
        "if(!document.getElementById(styleId)){const style=document.createElement('style');style.id=styleId;style.textContent=styleCSS;document.head.appendChild(style);}"  # noqa: E501
        f"const cfg={json.dumps(config)};"
        f"const wrapperId='{button_id}';"
        "const Streamlit=window.parent && window.parent.Streamlit;"
        "if(!Streamlit){return;}"
        "const wrapper=document.getElementById(wrapperId);"
        "if(!wrapper){return;}"
        "Streamlit.setComponentReady();"
        "const buttonEl=wrapper.querySelector('button');"
        "if(buttonEl){buttonEl.disabled=cfg.disabled;}"
        "const statusEl=wrapper.querySelector('.rexai-fx-status');"
        "if(statusEl){const hint=(cfg.statusHints && cfg.statusHints[cfg.state])||'';statusEl.textContent=hint;statusEl.setAttribute('data-active',hint?'true':'false');}"
        "if(window.RexAIMicro){const controller=window.RexAIMicro.mount(wrapper,cfg);if(controller){controller.applyState(cfg.state);}}else{wrapper.setAttribute('data-state',cfg.state);}"
        "const send=(payload)=>Streamlit.setComponentValue(payload);"
        "if(buttonEl && !cfg.disabled){buttonEl.addEventListener('click',()=>send({event:'click',ts:Date.now()}));}"
        "const sync=()=>Streamlit.setFrameHeight(document.body.scrollHeight);"
        "sync();window.addEventListener('resize',sync);"
        "})();"
    )
    script = "".join(script_parts)

    html_markup = f"""
    <div id="{button_id}" class="rexai-fx-wrapper" data-state="{state}" data-width="{container_width}">
      <button type="button" class="rexai-fx-button" {'disabled="disabled"' if disabled else ''}>
        <span class="rexai-fx-particles"></span>
        <span class="rexai-fx-label">{escape(label_current)}</span>
      </button>
      <span class="rexai-fx-status" data-active="{'true' if status_text else 'false'}">{escape(status_text)}</span>
      {help_html}
    </div>
    <script>{script}</script>
    """

    result = components_html(html_markup, height=120 if help_text else 100, key=key)
    session_key = f"__rexai_fx_ts::{key}"
    if isinstance(result, dict) and result.get("event") == "click":
        ts = result.get("ts")
        if ts and st.session_state.get(session_key) != ts:
            st.session_state[session_key] = ts
            return True
    return False
