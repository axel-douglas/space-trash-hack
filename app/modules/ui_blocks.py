from __future__ import annotations

import hashlib
import json
from contextlib import contextmanager
from html import escape
from pathlib import Path
from typing import Any, Generator, Iterable, Iterator, Literal, Optional
from uuid import uuid4

import streamlit as st
from streamlit.components.v1 import html as components_html
from streamlit.delta_generator import DeltaGenerator
from streamlit.runtime.scriptrunner.script_runner import get_script_run_ctx
from streamlit.runtime.state.safe_session_state import SafeSessionState

from app import _bootstrap

from . import luxe_components as luxe


_THEME_HASH_KEY = "__rexai_theme_hash__"
_INTERACTIONS_HASH_KEY = "__rexai_interactions_hash__"
_THEME_STATE_KEY = "hud_theme"
_FONT_STATE_KEY = "hud_font"
_COLORBLIND_STATE_KEY = "hud_colorblind"
_REVEAL_FLAG_KEY = "__rexai_reveal_flag__"
_HUD_PLACEHOLDER: Optional[DeltaGenerator] = None

_THEME_LABELS = {
    "mars-minimal": "Marte minimal",
    "dark": "Oscuro (fallback)",
    "dark-high-contrast": "Oscuro · Alto contraste",
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


if not hasattr(SafeSessionState, "get"):

    def _safe_state_get(self: SafeSessionState, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default

    SafeSessionState.get = _safe_state_get  # type: ignore[attr-defined]


def _static_path(filename: str | Path) -> Path:
    return Path(__file__).resolve().parents[1] / "static" / Path(filename)


def _base_css_path() -> Path:
    return _static_path(Path("styles") / "base.css")


def _ensure_defaults() -> None:
    st.session_state.setdefault(_THEME_STATE_KEY, "mars-minimal")
    st.session_state.setdefault(_FONT_STATE_KEY, "base")
    st.session_state.setdefault(_COLORBLIND_STATE_KEY, "normal")

    current = st.session_state.get(_THEME_STATE_KEY)
    if current not in _THEME_LABELS:
        name = str(current)
        fallback = "dark-high-contrast" if "high-contrast" in name else "dark"
        st.session_state[_THEME_STATE_KEY] = fallback


def _read_css_bundle() -> str:
    try:
        return _base_css_path().read_text(encoding="utf-8")
    except FileNotFoundError:
        return ""


def _inject_css_once() -> None:
    css = _read_css_bundle()
    if not css:
        return

    css_hash = hashlib.sha256(css.encode("utf-8")).hexdigest()
    if st.session_state.get(_THEME_HASH_KEY) == css_hash:
        return

    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
    st.session_state[_THEME_HASH_KEY] = css_hash


def _inject_interactions_script_once() -> None:
    if not _INTERACTIONS_JS:
        return

    script_hash = hashlib.sha256(_INTERACTIONS_JS.encode("utf-8")).hexdigest()
    if st.session_state.get(_INTERACTIONS_HASH_KEY) == script_hash:
        return

    st.markdown(f"<script>{_INTERACTIONS_JS}</script>", unsafe_allow_html=True)
    st.session_state[_INTERACTIONS_HASH_KEY] = script_hash


_MICRO_JS = _bootstrap.load_microinteractions_script()
_INTERACTIONS_JS = _bootstrap.load_interactions_script()

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
.rexai-fx-label{position:relative;z-index:2;display:flex;align-items:center;justify-content:center;gap:10px;text-align:center;letter-spacing:0.01em;flex-wrap:wrap;}
.rexai-fx-label[data-layout="stack"]{flex-direction:column;gap:6px;}
.rexai-fx-icon{font-size:1.25rem;line-height:1;filter:drop-shadow(0 0 6px rgba(14,165,233,0.18));}
.rexai-fx-text{display:flex;flex-direction:column;gap:2px;line-height:1.2;align-items:center;text-align:center;}
.rexai-fx-line{display:block;}
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

_MINIMAL_BUTTON_STYLES = """
.rexai-minimal-wrapper{display:flex;flex-direction:column;gap:6px;}
.rexai-minimal-wrapper[data-width="full"]{width:100%;}
.rexai-minimal-wrapper[data-width="auto"]{display:inline-flex;}
.rexai-minimal-button{border:1px solid rgba(148,163,184,0.32);border-radius:14px;padding:14px 18px;font-weight:600;font-size:1rem;color:var(--rexai-button-fg,rgba(15,23,42,0.94));background:var(--rexai-button-bg,rgba(148,163,184,0.08));box-shadow:0 4px 12px rgba(15,23,42,0.04);transition:transform 0.16s ease,box-shadow 0.18s ease,background 0.2s ease,color 0.2s ease;cursor:pointer;min-height:52px;text-align:center;display:flex;justify-content:center;align-items:center;gap:10px;}
.rexai-minimal-button:disabled{cursor:not-allowed;opacity:0.6;}
.rexai-minimal-wrapper[data-width="full"] .rexai-minimal-button{width:100%;}
.rexai-minimal-wrapper[data-state="loading"] .rexai-minimal-button{background:rgba(148,163,184,0.14);box-shadow:0 4px 14px rgba(59,130,246,0.18);cursor:progress;}
.rexai-minimal-wrapper[data-state="success"] .rexai-minimal-button{border-color:rgba(16,185,129,0.42);background:rgba(16,185,129,0.12);color:rgba(15,118,110,0.92);}
.rexai-minimal-wrapper[data-state="error"] .rexai-minimal-button{border-color:rgba(248,113,113,0.48);background:rgba(248,113,113,0.1);color:rgba(153,27,27,0.92);}
.rexai-minimal-icon{font-size:1.2rem;line-height:1;}
.rexai-minimal-text{display:flex;flex-direction:column;gap:4px;line-height:1.2;align-items:center;justify-content:center;}
.rexai-minimal-text[data-layout="inline"]{flex-direction:row;gap:10px;}
.rexai-minimal-line{display:block;}
.rexai-minimal-status{font-size:0.78rem;color:rgba(71,85,105,0.82);text-transform:uppercase;letter-spacing:0.04em;transition:opacity 0.18s ease;height:0;opacity:0;}
.rexai-minimal-status[data-active="true"]{height:auto;opacity:1;}
.rexai-minimal-help{font-size:0.82rem;color:rgba(100,116,139,0.9);margin:0 4px;}
@media (prefers-color-scheme:dark){
  .rexai-minimal-button{border-color:rgba(148,163,184,0.24);background:rgba(100,116,139,0.12);color:rgba(226,232,240,0.94);box-shadow:0 6px 16px rgba(15,23,42,0.3);}
  .rexai-minimal-wrapper[data-state="loading"] .rexai-minimal-button{background:rgba(148,163,184,0.18);}
  .rexai-minimal-wrapper[data-state="success"] .rexai-minimal-button{background:rgba(34,197,94,0.14);color:rgba(190,242,100,0.95);}
  .rexai-minimal-wrapper[data-state="error"] .rexai-minimal-button{background:rgba(248,113,113,0.18);color:rgba(254,226,226,0.92);}
  .rexai-minimal-status{color:rgba(148,163,184,0.86);}
  .rexai-minimal-help{color:rgba(148,163,184,0.88);}
}
"""


def load_theme(*, show_hud: bool = True) -> None:
    """Inject the lightweight NASA-inspired base stylesheet."""

    del show_hud  # compatibility no-op
    _inject_css_once()
    _inject_interactions_script_once()


def enable_reveal_animation() -> None:
    """Signal the front-end to activate scroll-based reveal animations."""

    if st.session_state.get(_THEME_HASH_KEY):
        load_theme(show_hud=False)
    else:
        load_theme()
    if st.session_state.get(_REVEAL_FLAG_KEY):
        return

    st.markdown(
        '<span data-rexai-interactions="reveal" style="display:none"></span>',
        unsafe_allow_html=True,
    )
    st.session_state[_REVEAL_FLAG_KEY] = True


def inject_css(show_hud: bool = False) -> None:
    """Backward-compatible alias for legacy code paths."""

    load_theme(show_hud=show_hud)


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
) -> Iterator[DeltaGenerator]:
    """Render content inside a themed surface wrapper."""

    load_theme(show_hud=False)
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
) -> Iterator[DeltaGenerator]:
    """Render content inside a frosted glass style surface."""

    load_theme(show_hud=False)
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
    ctx = get_script_run_ctx()
    if ctx and {
        _THEME_STATE_KEY,
        _FONT_STATE_KEY,
        _COLORBLIND_STATE_KEY,
    }.intersection(ctx.widget_user_keys_this_run):
        return

    global _HUD_PLACEHOLDER
    if _HUD_PLACEHOLDER is None:
        _HUD_PLACEHOLDER = st.empty()
    hud_container = _HUD_PLACEHOLDER.container()
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


def card(title: str, body: str = "") -> None:
    st.markdown(luxe.render_card(title, body), unsafe_allow_html=True)


def pill(label: str, kind: Literal["ok", "warn", "risk"] = "ok") -> None:
    st.markdown(luxe.render_pill(label, kind), unsafe_allow_html=True)


def section(title: str, subtitle: str = "") -> None:
    st.subheader(title)
    if subtitle:
        st.caption(subtitle)


_BUTTON_STATES: set[str] = {"idle", "loading", "success", "error"}


def _split_lines(text: str) -> list[str]:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        stripped = text.strip()
        return [stripped or text]
    return lines


def _normalize_button_options(
    label: str,
    *,
    state: Literal["idle", "loading", "success", "error"],
    loading_label: str | None,
    success_label: str | None,
    error_label: str | None,
    status_hints: dict[str, str] | None,
    help_text: str | None,
    icon: str | None,
) -> dict[str, Any]:
    if state not in _BUTTON_STATES:
        raise ValueError(f"Estado no soportado: {state}")

    state_messages = {
        "idle": label,
        "loading": loading_label or "Procesando…",
        "success": success_label or "Listo",
        "error": error_label or "Reintentar",
    }
    hints = status_hints or {
        "idle": "",
        "loading": "Optimizando parámetros",
        "success": "Listo para revisar",
        "error": "Revisá parámetros o intenta de nuevo",
    }

    current_label = state_messages.get(state, label)
    label_lines = _split_lines(current_label)
    line_count = max(1, len(label_lines))
    layout_mode = "stack" if (len(label_lines) > 1 and not icon) else "inline"
    status_text = hints.get(state, "")
    return {
        "state": state,
        "state_messages": state_messages,
        "status_hints": hints,
        "label_lines": label_lines,
        "line_count": line_count,
        "layout_mode": layout_mode,
        "status_text": status_text,
        "help_text": help_text,
    }


def _render_button_component(
    markup: str,
    *,
    key: str,
    line_count: int,
    has_help: bool,
) -> bool:
    base_height = 100 + max(0, line_count - 1) * 8
    component_kwargs = {"height": base_height + (20 if has_help else 0), "key": key}
    try:
        result = components_html(markup, **component_kwargs)
    except TypeError:
        component_kwargs.pop("key", None)
        result = components_html(markup, **component_kwargs)

    session_key = f"__rexai_fx_ts::{key}"
    if isinstance(result, dict) and result.get("event") == "click":
        ts = result.get("ts")
        if ts and st.session_state.get(session_key) != ts:
            st.session_state[session_key] = ts
            return True
    return False


def minimal_button(
    label: str,
    key: str,
    *,
    state: Literal["idle", "loading", "success", "error"] = "idle",
    width: Literal["full", "auto"] = "full",
    help_text: str | None = None,
    loading_label: str | None = None,
    success_label: str | None = None,
    error_label: str | None = None,
    disabled: bool = False,
    status_hints: dict[str, str] | None = None,
    icon: str | None = None,
) -> bool:
    """Render a minimal CTA button with the shared Rex-AI API."""

    options = _normalize_button_options(
        label,
        state=state,
        loading_label=loading_label,
        success_label=success_label,
        error_label=error_label,
        status_hints=status_hints,
        help_text=help_text,
        icon=icon,
    )

    container_width = "full" if width not in {"full", "auto"} else width
    button_id = f"rexai-minimal-{uuid4().hex}"
    label_lines = options["label_lines"]
    layout_mode = options["layout_mode"]
    line_count = options["line_count"]
    status_text = options["status_text"]
    icon_html = (
        f'<span class="rexai-minimal-icon" aria-hidden="true">{escape(icon)}</span>'
        if icon
        else ""
    )
    text_block = "".join(
        f'<span class="rexai-minimal-line">{escape(line)}</span>' for line in label_lines
    )
    text_wrapper = (
        f'<span class="rexai-minimal-text" data-layout="{layout_mode}" '
        f'data-lines="{line_count}">{text_block}</span>'
    )

    help_html = (
        f'<div class="rexai-minimal-help">{escape(help_text)}</div>' if help_text else ""
    )

    script = "".join(
        [
            "(function(){",
            "const styleId='rexai-minimal-style';",
            f"const styleCSS={json.dumps(_MINIMAL_BUTTON_STYLES)};",
            "if(!document.getElementById(styleId)){const style=document.createElement('style');style.id=styleId;style.textContent=styleCSS;document.head.appendChild(style);}",
            f"const cfg={{state:{json.dumps(state)},statusText:{json.dumps(status_text)},disabled:{json.dumps(bool(disabled))},width:{json.dumps(container_width)}}};",
            "const wrapperId='" + button_id + "';",
            "const Streamlit=window.parent && window.parent.Streamlit;",
            "if(!Streamlit){return;}",
            "const wrapper=document.getElementById(wrapperId);",
            "if(!wrapper){return;}",
            "Streamlit.setComponentReady();",
            "wrapper.setAttribute('data-fx','minimal');",
            "wrapper.setAttribute('data-state',cfg.state);",
            "wrapper.setAttribute('data-width',cfg.width);",
            "const statusEl=wrapper.querySelector('.rexai-minimal-status');",
            "if(statusEl){statusEl.textContent=cfg.statusText||'';statusEl.setAttribute('data-active',cfg.statusText?'true':'false');}",
            "const buttonEl=wrapper.querySelector('button');",
            "if(buttonEl){buttonEl.disabled=cfg.disabled || cfg.state==='loading';",
            "if(cfg.state==='loading'){buttonEl.setAttribute('aria-busy','true');}else{buttonEl.removeAttribute('aria-busy');}",
            "if(!cfg.disabled){buttonEl.addEventListener('click',()=>Streamlit.setComponentValue({event:'click',ts:Date.now()}));}}",
            "const sync=()=>Streamlit.setFrameHeight(document.body.scrollHeight);",
            "sync();",
            "window.addEventListener('resize',sync);",
            "})();",
        ]
    )

    html_markup = f"""
    <div id="{button_id}" class="rexai-minimal-wrapper" data-state="{state}" data-width="{container_width}" data-fx="minimal">
      <button type="button" class="rexai-minimal-button" {'disabled="disabled"' if disabled else ''}>
        {icon_html}{text_wrapper}
      </button>
      <span class="rexai-minimal-status" data-active="{'true' if status_text else 'false'}">{escape(status_text)}</span>
      {help_html}
    </div>
    <script>{script}</script>
    """

    return _render_button_component(
        html_markup,
        key=key,
        line_count=line_count,
        has_help=bool(help_text),
    )


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
    icon: str | None = None,
    mode: Literal["minimal", "cinematic"] = "minimal",
) -> bool:
    """Render the CTA button with optional cinematic microinteractions."""

    if mode == "minimal":
        return minimal_button(
            label,
            key,
            state=state,
            width=width,
            help_text=help_text,
            loading_label=loading_label,
            success_label=success_label,
            error_label=error_label,
            disabled=disabled,
            status_hints=status_hints,
            icon=icon,
        )

    if mode != "cinematic":
        raise ValueError(f"Modo no soportado para futuristic_button: {mode}")

    options = _normalize_button_options(
        label,
        state=state,
        loading_label=loading_label,
        success_label=success_label,
        error_label=error_label,
        status_hints=status_hints,
        help_text=help_text,
        icon=icon,
    )

    container_width = "full" if width not in {"full", "auto"} else width
    status_text = options["status_text"]
    help_html = (
        f'<div class="rexai-fx-help">{escape(help_text)}</div>' if help_text else ""
    )
    label_lines = options["label_lines"]
    line_count = options["line_count"]
    layout_mode = options["layout_mode"]
    button_id = f"rexai-fx-{uuid4().hex}"
    icon_html = (
        f'<span class="rexai-fx-icon" aria-hidden="true">{escape(icon)}</span>'
        if icon
        else ""
    )
    text_block = "".join(
        f'<span class="rexai-fx-line">{escape(line)}</span>' for line in label_lines
    )
    label_html = (
        f'<span class="rexai-fx-label" data-layout="{layout_mode}" '
        f'data-lines="{line_count}">{icon_html}'
        f'<span class="rexai-fx-text">{text_block}</span></span>'
    )

    config: dict[str, Any] = {
        "state": state,
        "stateMessages": options["state_messages"],
        "statusHints": options["status_hints"],
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
    script_parts.extend(
        [
            "(function(){",
            "const styleId='rexai-fx-style';",
            f"const styleCSS={json.dumps(_BUTTON_STYLES)};",
            "if(!document.getElementById(styleId)){const style=document.createElement('style');style.id=styleId;style.textContent=styleCSS;document.head.appendChild(style);}",
            f"const cfg={json.dumps(config)};",
            f"const wrapperId='{button_id}';",
            "const Streamlit=window.parent && window.parent.Streamlit;",
            "if(!Streamlit){return;}",
            "const wrapper=document.getElementById(wrapperId);",
            "if(!wrapper){return;}",
            "Streamlit.setComponentReady();",
            "wrapper.setAttribute('data-fx','cinematic');",
            "wrapper.setAttribute('data-state',cfg.state);",
            "wrapper.setAttribute('data-width'," + json.dumps(container_width) + ");",
            "const buttonEl=wrapper.querySelector('button');",
            "if(buttonEl){buttonEl.disabled=cfg.disabled;}",
            "const statusEl=wrapper.querySelector('.rexai-fx-status');",
            "if(statusEl){const hint=(cfg.statusHints && cfg.statusHints[cfg.state])||'';statusEl.textContent=hint;statusEl.setAttribute('data-active',hint?'true':'false');}",
            "if(window.RexAIMicro){const controller=window.RexAIMicro.mount(wrapper,cfg);if(controller){controller.applyState(cfg.state);}}",
            "const send=(payload)=>Streamlit.setComponentValue(payload);",
            "if(buttonEl && !cfg.disabled){buttonEl.addEventListener('click',()=>send({event:'click',ts:Date.now()}));}",
            "const sync=()=>Streamlit.setFrameHeight(document.body.scrollHeight);",
            "sync();window.addEventListener('resize',sync);",
            "})();",
        ]
    )
    script = "".join(script_parts)

    html_markup = f"""
    <div id="{button_id}" class="rexai-fx-wrapper" data-state="{state}" data-width="{container_width}" data-fx="cinematic">
      <button type="button" class="rexai-fx-button" {'disabled="disabled"' if disabled else ''}>
        <span class="rexai-fx-particles"></span>
        {label_html}
      </button>
      <span class="rexai-fx-status" data-active="{'true' if status_text else 'false'}">{escape(status_text)}</span>
      {help_html}
    </div>
    <script>{script}</script>
    """

    return _render_button_component(
        html_markup,
        key=key,
        line_count=line_count,
        has_help=bool(help_text),
    )


@contextmanager
def layout_block(
    classes: str,
    *,
    parent: DeltaGenerator | None = None,
) -> Generator[DeltaGenerator, None, None]:
    """Yield a Streamlit container wrapped in custom layout classes."""

    target = parent if parent is not None else st.container()
    target.markdown(f"<div class=\"{classes}\">", unsafe_allow_html=True)
    inner = target.container()
    try:
        yield inner
    finally:
        target.markdown("</div>", unsafe_allow_html=True)


@contextmanager
def layout_stack(*, parent: DeltaGenerator | None = None) -> Generator[DeltaGenerator, None, None]:
    """Convenience wrapper for a vertical flex stack."""

    with layout_block("layout-stack", parent=parent) as block:
        yield block


@contextmanager
def pane_block(*, parent: DeltaGenerator | None = None) -> Generator[DeltaGenerator, None, None]:
    """Render content inside a frosted pane surface."""

    with layout_block("pane", parent=parent) as block:
        yield block


def chipline(labels: Iterable[str], *, parent: DeltaGenerator | None = None) -> None:
    """Render a list of chips using the shared chipline styles."""

    if not labels:
        return

    pills = "".join(f'<span>{escape(label)}</span>' for label in labels)
    target = parent if parent is not None else st
    target.markdown(f"<div class=\"chipline\">{pills}</div>", unsafe_allow_html=True)


def badge_group(labels: Iterable[str], *, parent: DeltaGenerator | None = None) -> None:
    """Render pill badges inside the shared badge group wrapper."""

    items = [f'<span class="badge">{escape(label)}</span>' for label in labels]
    if not items:
        return

    html = f"<div class=\"badge-group\">{''.join(items)}</div>"
    target = parent if parent is not None else st
    target.markdown(html, unsafe_allow_html=True)


def micro_divider(*, parent: DeltaGenerator | None = None) -> None:
    """Insert a subtle divider matching the Rex-AI style guide."""

    target = parent if parent is not None else st
    target.markdown('<div class="hr-micro"></div>', unsafe_allow_html=True)
