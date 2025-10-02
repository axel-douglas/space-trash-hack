from __future__ import annotations

import hashlib
from contextlib import contextmanager
from html import escape
from pathlib import Path
from typing import Any, Generator, Iterable, Iterator, Literal, Mapping, Optional

import streamlit as st
from streamlit.delta_generator import DeltaGenerator

_THEME_HASH_KEY = "__rexai_theme_hash__"
_REVEAL_FLAG_KEY = "__rexai_reveal_flag__"

def _static_path(filename: str | Path) -> Path:
    return Path(__file__).resolve().parents[1] / "static" / Path(filename)


def _base_css_path() -> Path:
    return _static_path(Path("styles") / "base.css")


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







def load_theme(*, show_hud: bool = True) -> None:
    """Inject the lightweight NASA-inspired base stylesheet."""

    del show_hud  # compatibility no-op
    _inject_css_once()


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


def card(title: str, body: str = "", *, render: bool = True) -> str:
    """Render a simple Rex-AI card block."""

    load_theme(show_hud=False)
    title_html = f"<h3 class='rex-card__title'>{escape(title)}</h3>" if title else ""
    body_html = f"<p class='rex-card__body'>{escape(body)}</p>" if body else ""
    markup = (
        "<article class='rex-card'>"
        f"{title_html}{body_html}"
        "</article>"
    )
    if render:
        st.markdown(markup, unsafe_allow_html=True)
    return markup


_PILL_KINDS = {
    "ok": "Rango nominal",
    "warn": "Monitoreo",
    "risk": "Riesgo",
}


def pill(
    label: str,
    kind: Literal["ok", "warn", "risk"] = "ok",
    *,
    render: bool = True,
) -> str:
    """Render a lab-status pill using the base mission palette."""

    load_theme(show_hud=False)
    tone = kind if kind in _PILL_KINDS else "ok"
    markup = (
        "<span class='rex-pill' "
        f"data-kind='{tone}' "
        f"title='{escape(_PILL_KINDS[tone])}'>"
        f"{escape(label)}"
        "</span>"
    )
    if render:
        st.markdown(markup, unsafe_allow_html=True)
    return markup


def section(title: str, subtitle: str = "") -> None:
    st.subheader(title)
    if subtitle:
        st.caption(subtitle)


_BUTTON_STATES: set[str] = {"idle", "loading", "success", "error"}

_STATUS_STATE_MAP: dict[str, str] = {
    "loading": "running",
    "success": "complete",
    "error": "error",
}


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


def _compose_button_label(lines: list[str], icon: str | None) -> str:
    if not lines:
        return icon or ""
    if icon:
        first, *rest = lines
        prefixed = f"{icon} {first}".strip()
        return "\n".join([prefixed, *rest]) if rest else prefixed
    return "\n".join(lines)


def _status_label(options: Mapping[str, Any], state: str) -> str:
    hint = str(options.get("status_text", "")).strip()
    if hint:
        return hint
    state_messages = options.get("state_messages", {})
    return str(state_messages.get(state, "")).strip()


def action_button(
    label: str,
    key: str,
    *,
    state: Literal["idle", "loading", "success", "error"] = "idle",
    width: Literal["full", "auto"] = "full",
    help_text: str | None = None,
    tooltip: str | None = None,
    loading_label: str | None = None,
    success_label: str | None = None,
    error_label: str | None = None,
    status_hints: dict[str, str] | None = None,
    icon: str | None = None,
    disabled: bool = False,
    download_data: Any | None = None,
    download_file_name: str | None = None,
    download_mime: str | None = None,
    on_click: Any | None = None,
    on_click_args: tuple[Any, ...] | None = None,
    on_click_kwargs: Mapping[str, Any] | None = None,
    button_type: Literal["primary", "secondary"] = "secondary",
) -> bool:
    """Render a Streamlit button with Rex-AI convenience features."""

    load_theme(show_hud=False)
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

    use_container_width = width == "full"
    disabled_flag = bool(disabled) or state == "loading"
    button_label = _compose_button_label(options["label_lines"], icon)
    args = on_click_args or ()
    kwargs = dict(on_click_kwargs or {})

    if download_data is not None:
        clicked = st.download_button(
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
    else:
        clicked = st.button(
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

    status_state = _STATUS_STATE_MAP.get(state)
    if status_state:
        status_label = _status_label(options, state)
        if status_label:
            status = st.status(status_label, state=status_state)
            status.update(label=status_label, state=status_state, expanded=False)

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

    html: list[str] = ["<div class=\"chipline\">"]
    for item in items:
        if isinstance(item, Mapping):
            label_text = str(item.get("label", ""))
            icon_text = item.get("icon")
            tone = str(item.get("tone", "") or "").strip()
        else:
            label_text = str(item)
            icon_text = None
            tone = ""

        tone_attr = f" data-tone='{escape(tone)}'" if tone else ""
        icon_html = (
            f"<span class='chipline__icon'>{escape(str(icon_text))}</span>"
            if icon_text
            else ""
        )
        html.append(
            "<span class='chipline__chip'"
            f"{tone_attr}>"
            f"{icon_html}<span class='chipline__label'>{escape(label_text)}</span>"
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
