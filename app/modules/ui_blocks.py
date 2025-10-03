from __future__ import annotations

import hashlib
from contextlib import contextmanager
from html import escape
from pathlib import Path
from typing import Any, Generator, Iterable, Literal, Mapping, Optional

import streamlit as st
from streamlit.delta_generator import DeltaGenerator

from app.modules.visual_theme import apply_global_visual_theme

_LAYOUT_STYLE_MAP: dict[str, str] = {
    "layout-stack": "display:flex; flex-direction:column; gap: var(--mission-space-md);",
    "layout-grid": (
        "display:grid; gap: var(--mission-space-md); align-items:start; "
        "width:min(100%, var(--mission-layout-max-width)); margin-inline:auto;"
    ),
    "layout-grid--dual": "grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));",
    "layout-grid--flow": "grid-auto-flow: row;",
    "depth-stack": "display:flex; flex-direction:column; gap: var(--mission-space-sm);",
    "side-panel": (
        "display:flex; flex-direction:column; gap: var(--mission-space-sm); "
        "background-color: var(--mission-color-surface); "
        "border: var(--mission-line-weight) solid var(--mission-color-border); "
        "border-radius: var(--mission-radius-md); padding: var(--mission-space-lg); "
        "color: var(--mission-color-text);"
    ),
    "pane": (
        "display:flex; flex-direction:column; gap: var(--mission-space-sm); "
        "background-color: var(--mission-color-surface); "
        "border: var(--mission-line-weight) solid var(--mission-color-border); "
        "border-radius: var(--mission-radius-md); padding: var(--mission-space-lg); "
        "color: var(--mission-color-text);"
    ),
    "layer-shadow": (
        "background-color: var(--mission-color-panel); "
        "border-color: var(--mission-color-border-strong);"
    ),
    "fade-in": "",
    "fade-in-delayed": "",
}
_THEME_HASH_KEY = "__rexai_theme_hash__"

def _static_path(filename: str | Path) -> Path:
    return Path(__file__).resolve().parents[1] / "static" / Path(filename)


def _base_css_path() -> Path:
    return _static_path(Path("styles") / "base.css")


def _read_css_bundle() -> str:
    try:
        return _base_css_path().read_text(encoding="utf-8")
    except FileNotFoundError:
        return ""


def load_theme(*, show_hud: bool = True) -> None:
    """Inject the lightweight NASA-inspired base stylesheet."""

    del show_hud  # compatibility no-op
    css = _read_css_bundle()
    if not css:
        return

    css_hash = hashlib.sha256(css.encode("utf-8")).hexdigest()
    if st.session_state.get(_THEME_HASH_KEY) == css_hash:
        return

    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
    st.session_state[_THEME_HASH_KEY] = css_hash


def initialise_frontend() -> None:
    """Prepare the visual styling for Streamlit pages."""

    load_theme()
    apply_global_visual_theme()


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


def card(title: str, body: str = "", *, render: bool = True) -> str:
    """Render a minimalist mission panel."""

    load_theme(show_hud=False)
    base_style = (
        "display:flex; flex-direction:column; gap: var(--mission-space-xs); "
        "background-color: var(--mission-color-surface); "
        "border: var(--mission-line-weight) solid var(--mission-color-border); "
        "border-radius: var(--mission-radius-md); "
        "padding: var(--mission-space-lg);"
    )
    title_style = (
        "margin:0; font-size:1.1rem; color: var(--mission-color-text); "
        "letter-spacing:0.01em;"
    )
    body_style = "margin:0; color: var(--mission-color-muted); font-size:0.95rem;"
    title_html = f"<h3 style=\"{title_style}\">{escape(title)}</h3>" if title else ""
    body_html = f"<p style=\"{body_style}\">{escape(body)}</p>" if body else ""
    markup = (
        f"<article data-mission-card data-lab-card style=\"{base_style}\">"
        f"{title_html}{body_html}</article>"
    )
    if render:
        st.markdown(markup, unsafe_allow_html=True)
    return markup



_PILL_KINDS = {
    "ok": "Rango nominal",
    "warn": "Monitoreo",
    "risk": "Riesgo",
    "info": "Referencia informativa",
    "accent": "Etiqueta destacada",
}

_CHIP_TONES: dict[str, tuple[str, str]] = {
    "positive": ("var(--mission-color-positive-soft)", "var(--mission-color-positive)"),
    "ok": ("var(--mission-color-positive-soft)", "var(--mission-color-positive)"),
    "warning": ("var(--mission-color-warning-soft)", "var(--mission-color-warning)"),
    "warn": ("var(--mission-color-warning-soft)", "var(--mission-color-warning)"),
    "danger": ("var(--mission-color-critical-soft)", "var(--mission-color-critical)"),
    "risk": ("var(--mission-color-critical-soft)", "var(--mission-color-critical)"),
    "info": ("var(--mission-color-panel)", "var(--mission-color-accent)"),
    "accent": ("var(--mission-color-accent-soft)", "var(--mission-color-accent)"),
}


def pill(
    label: str,
    kind: Literal["ok", "warn", "risk", "info", "accent"] = "ok",
    *,
    render: bool = True,
) -> str:
    """Render a mission-status pill using the base palette."""

    load_theme(show_hud=False)
    tone = kind if kind in _PILL_KINDS else "ok"
    title_attr = escape(_PILL_KINDS[tone])
    tone_attr = escape(tone)
    label_html = escape(label)
    markup = (
        f"<span class='mission-pill mission-pill--{tone_attr}' "
        f"data-mission-pill='{tone_attr}' data-lab-pill='{tone_attr}' "
        f"data-kind='{tone_attr}' title='{title_attr}'>"
        f"{label_html}</span>"
    )
    if render:
        st.markdown(markup, unsafe_allow_html=True)
    return markup


def section(title: str, subtitle: str = "") -> None:
    st.subheader(title)
    if subtitle:
        st.caption(subtitle)


_BUTTON_STATES: set[str] = {"idle", "loading", "success", "error"}


def _compose_button_label(label: str, icon: str | None) -> str:
    text = label or ""
    lines = [ln for ln in text.splitlines() if ln]
    if not lines:
        return icon or text
    if not icon:
        return "\n".join(lines)
    first, *rest = lines
    prefixed = f"{icon} {first}".strip()
    return "\n".join([prefixed, *rest]) if rest else prefixed


def _state_labels(
    default: str,
    *,
    overrides: Mapping[str, str] | None = None,
) -> dict[str, str]:
    labels = {state: default for state in _BUTTON_STATES}
    labels["idle"] = default
    if overrides:
        for state, text in overrides.items():
            if state in _BUTTON_STATES:
                labels[state] = str(text)
    return labels


def _state_messages(
    *,
    overrides: Mapping[str, str] | None = None,
) -> dict[str, str]:
    messages = {
        "idle": "",
        "loading": "Procesando…",
        "success": "Listo",
        "error": "Reintentar",
    }
    if overrides:
        for state, text in overrides.items():
            if state in _BUTTON_STATES:
                messages[state] = str(text)
    return messages


def action_button(
    label: str,
    key: str,
    *,
    state: Literal["idle", "loading", "success", "error"] = "idle",
    width: Literal["full", "auto"] = "full",
    help_text: str | None = None,
    tooltip: str | None = None,
    icon: str | None = None,
    disabled: bool = False,
    download_data: Any | None = None,
    download_file_name: str | None = None,
    download_mime: str | None = None,
    on_click: Any | None = None,
    on_click_args: tuple[Any, ...] | None = None,
    on_click_kwargs: Mapping[str, Any] | None = None,
    button_type: Literal["primary", "secondary"] = "secondary",
    state_labels: Mapping[str, str] | None = None,
    state_messages: Mapping[str, str] | None = None,
) -> bool:
    """Render a Streamlit button with Rex-AI convenience features."""

    load_theme(show_hud=False)
    if state not in _BUTTON_STATES:
        raise ValueError(f"Estado no soportado: {state}")

    labels = _state_labels(label, overrides=state_labels)
    messages = _state_messages(overrides=state_messages)
    button_label = _compose_button_label(labels.get(state, label), icon)
    use_container_width = width == "full"
    disabled_flag = bool(disabled) or state == "loading"
    args = on_click_args or ()
    kwargs = dict(on_click_kwargs or {})

    def _render_button() -> bool:
        if download_data is not None:
            return st.download_button(
                button_label,
                data=download_data,
                file_name=download_file_name,
                mime=download_mime,
                key=key,
                help=tooltip,
                on_click=on_click,
                args=args,
                kwargs=kwargs,
                disabled=disabled_flag,
                use_container_width=use_container_width,
            )
        return st.button(
            button_label,
            key=key,
            help=tooltip,
            on_click=on_click,
            args=args,
            kwargs=kwargs,
            type=button_type,
            disabled=disabled_flag,
            use_container_width=use_container_width,
        )

    status_text = messages.get(state, "")
    if state == "loading" and status_text:
        with st.spinner(status_text):
            clicked = _render_button()
    else:
        clicked = _render_button()

    if state == "success" and status_text:
        st.status(status_text, state="complete")
    elif state == "error" and status_text:
        st.status(status_text, state="error")

    if help_text:
        st.caption(help_text)

    return clicked


@contextmanager
def layout_block(
    classes: str,
    *,
    parent: DeltaGenerator | None = None,
) -> Generator[DeltaGenerator, None, None]:
    """Yield a Streamlit container wrapped in custom layout classes."""

    tokens = [token for token in classes.split() if token.strip()]
    style_bits: list[str] = []
    for token in tokens:
        mapped = _LAYOUT_STYLE_MAP.get(token)
        if mapped:
            style_bits.append(mapped)

    if not style_bits:
        style_bits.append("display:flex; flex-direction:column; gap: var(--mission-space-sm);")

    style_attr = " ".join(style_bits)
    data_attr = ""
    if tokens:
        token_attr = escape(" ".join(tokens))
        data_attr = (
            f' data-mission-classes="{token_attr}" data-lab-classes="{token_attr}"'
        )

    target = parent if parent is not None else st.container()
    target.markdown(
        f"<div style=\"{style_attr}\"{data_attr}>",
        unsafe_allow_html=True,
    )
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


def chipline(
    labels: Iterable[str | Mapping[str, object]],
    *,
    parent: DeltaGenerator | None = None,
    render: bool = True,
) -> str:
    """Render a list of chips using the shared chipline styles."""

    items = list(labels)
    if not items:
        return ""

    load_theme(show_hud=False)

    row_style = (
        "display:flex; flex-wrap:wrap; align-items:center; gap: var(--mission-space-xs); "
        "margin-block: var(--mission-space-xs);"
    )
    chip_base = (
        "display:inline-flex; align-items:center; gap: var(--mission-space-2xs); "
        "padding: 0.3rem 0.75rem; border-radius: 999px; font-size: 0.9rem; "
        "border: var(--mission-line-weight) solid var(--mission-color-border); "
        "background-color: var(--mission-color-panel); color: var(--mission-color-text);"
    )
    icon_style = "font-size:1rem; line-height:1;"
    label_style = "display:inline-flex; align-items:center; gap: var(--mission-space-2xs);"

    html: list[str] = [f"<div style=\"{row_style}\">"]
    for item in items:
        if isinstance(item, Mapping):
            label_text = str(item.get("label", ""))
            icon_text = item.get("icon")
            tone = str(item.get("tone", "") or "").strip()
        else:
            label_text = str(item)
            icon_text = None
            tone = ""

        tone_key = tone.lower()
        bg_color, fg_color = _CHIP_TONES.get(
            tone_key, ("var(--mission-color-panel)", "var(--mission-color-text)")
        )
        chip_style = (
            f"{chip_base} background-color: {bg_color}; color: {fg_color}; "
            f"border-color: {fg_color};"
            if tone_key in _CHIP_TONES
            else chip_base
        )
        icon_html = (
            f"<span style=\"{icon_style}\">{escape(str(icon_text))}</span>"
            if icon_text
            else ""
        )
        tone_attr = (
            f" data-mission-chip-tone='{escape(tone_key)}'"
            f" data-lab-chip-tone='{escape(tone_key)}'"
        ) if tone_key else ""
        html.append(
            f"<span{tone_attr} style=\"{chip_style}\">"
            f"{icon_html}<span style=\"{label_style}\">{escape(label_text)}</span>"
            "</span>"
        )
    html.append("</div>")
    markup = "".join(html)

    if render:
        target = parent if parent is not None else st
        target.markdown(markup, unsafe_allow_html=True)
    return markup


def badge_group(labels: Iterable[str], *, parent: DeltaGenerator | None = None) -> None:
    """Render pill badges inside the shared badge group wrapper."""

    palette_cycle = [
        ("var(--mission-color-positive-soft)", "var(--mission-color-positive)"),
        ("var(--mission-color-warning-soft)", "var(--mission-color-warning)"),
        ("var(--mission-color-critical-soft)", "var(--mission-color-critical)"),
    ]
    row_style = (
        "display:flex; flex-wrap:wrap; gap: var(--mission-space-xs); margin-block: var(--mission-space-xs);"
    )
    badge_style = (
        "display:inline-flex; align-items:center; justify-content:center; "
        "padding: 0.35rem 0.75rem; border-radius: 999px; font-weight:600; "
        "letter-spacing:0.03em; text-transform:uppercase; font-size:0.75rem;"
    )

    items = list(labels)
    if not items:
        return

    badge_markup: list[str] = [f"<div style=\"{row_style}\">"]
    for index, label in enumerate(items):
        bg_color, fg_color = palette_cycle[index % len(palette_cycle)]
        style = (
            f"{badge_style} background-color: {bg_color}; color: {fg_color}; "
            f"border: var(--mission-line-weight) solid {fg_color};"
        )
        badge_markup.append(
            f"<span data-mission-badge-index='{index}' data-lab-badge-index='{index}' "
            f"style=\"{style}\">{escape(label)}</span>"
        )
    badge_markup.append("</div>")

    html = "".join(badge_markup)
    target = parent if parent is not None else st
    target.markdown(html, unsafe_allow_html=True)


def micro_divider(*, parent: DeltaGenerator | None = None) -> None:
    """Insert a subtle divider matching the mission style guide."""

    target = parent if parent is not None else st
    divider_style = (
        "height:1px; width:100%; background-color: var(--mission-color-divider); "
        "margin-block: var(--mission-space-md);"
    )
    target.markdown(f"<div style=\"{divider_style}\"></div>", unsafe_allow_html=True)
