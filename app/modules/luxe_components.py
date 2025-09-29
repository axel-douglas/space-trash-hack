"""High-fidelity UI primitives for the Rex-AI Streamlit experience."""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Literal, Mapping, Optional, Sequence, Tuple

import numpy as np
import plotly.graph_objects as go
import streamlit as st


_CSS_KEY = "__rexai_luxe_css__"


# ---------------------------------------------------------------------------
# CSS snippets
# ---------------------------------------------------------------------------
_BRIEFING_AND_TARGET_CSS = """
<style>
.briefing-grid {
    display: grid;
    grid-template-columns: minmax(280px, 1fr) minmax(320px, 1fr);
    gap: 32px;
    align-items: stretch;
    margin-top: 18px;
}
.briefing-video {
    position: relative;
    overflow: hidden;
    border-radius: 28px;
    border: 1px solid rgba(148, 163, 184, 0.18);
    box-shadow: 0 32px 80px -40px rgba(15, 23, 42, 0.8);
    min-height: 280px;
    background: radial-gradient(circle at top left, rgba(56, 189, 248, 0.4), rgba(15, 23, 42, 0.85));
}
.briefing-video video {
    position: absolute;
    inset: 0;
    width: 100%;
    height: 100%;
    object-fit: cover;
    filter: saturate(1.05) contrast(1.05);
}
.briefing-fallback {
    position: absolute;
    inset: 0;
    display: flex;
    align-items: center;
    justify-content: center;
    color: rgba(226, 232, 240, 0.88);
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    background: linear-gradient(120deg, rgba(56, 189, 248, 0.25), rgba(59, 130, 246, 0.05));
    animation: aurora 16s linear infinite;
}
.briefing-cards {
    display: grid;
    gap: 18px;
}
.briefing-card {
    position: relative;
    padding: 22px 24px;
    border-radius: 24px;
    border: 1px solid rgba(96, 165, 250, 0.18);
    background: rgba(15, 23, 42, 0.78);
    color: var(--ink);
    box-shadow: 0 24px 48px -32px rgba(30, 64, 175, 0.55);
    overflow: hidden;
}
.briefing-card::after {
    content: "";
    position: absolute;
    inset: -60% 40% 20% -40%;
    background: var(--card-accent);
    opacity: 0.16;
    filter: blur(60px);
    transform: rotate(8deg);
    transition: transform 600ms ease, opacity 600ms ease;
}
.briefing-card h3 {
    margin: 0 0 6px;
    font-size: 1.1rem;
    letter-spacing: 0.02em;
}
.briefing-card p {
    margin: 0;
    color: rgba(226, 232, 240, 0.86);
    font-size: 0.95rem;
}
.briefing-card:hover::after {
    opacity: 0.32;
    transform: rotate(-6deg) scale(1.08);
}
.briefing-stepper {
    display: flex;
    flex-direction: column;
    gap: 8px;
    margin-top: 24px;
}
.briefing-step {
    display: grid;
    grid-template-columns: 44px 1fr;
    gap: 12px;
    align-items: start;
    padding: 12px 14px;
    border-radius: 18px;
    background: rgba(15, 23, 42, 0.6);
    border: 1px solid rgba(148, 163, 184, 0.16);
    animation: rise-in 460ms ease backwards;
}
.briefing-step span {
    width: 44px;
    height: 44px;
    border-radius: 999px;
    display: grid;
    place-items: center;
    font-weight: 700;
    background: rgba(56, 189, 248, 0.18);
    color: var(--ink);
}
.briefing-step strong {
    display: block;
    font-size: 0.98rem;
    margin-bottom: 4px;
}
.briefing-step small {
    color: rgba(226, 232, 240, 0.78);
    font-size: 0.85rem;
}
.orbital-timeline {
    perspective: 1200px;
    margin: 32px 0 12px;
}
.orbital-track {
    position: relative;
    transform-style: preserve-3d;
    transform: rotateX(18deg);
    display: flex;
    gap: 32px;
    padding: 32px 18px;
    border-radius: 28px;
    background: linear-gradient(135deg, rgba(59, 130, 246, 0.12), rgba(15, 23, 42, 0.85));
    border: 1px solid rgba(96, 165, 250, 0.22);
    overflow-x: auto;
}
.orbital-track::-webkit-scrollbar {height: 6px;}
.orbital-track::-webkit-scrollbar-thumb {
    background: rgba(59, 130, 246, 0.3);
    border-radius: 999px;
}
.orbital-node {
    min-width: 220px;
    padding: 18px 20px;
    border-radius: 20px;
    background: rgba(15, 23, 42, 0.9);
    border: 1px solid rgba(148, 163, 184, 0.16);
    box-shadow: 0 18px 32px -20px rgba(30, 64, 175, 0.7);
    position: relative;
    transform: translateZ(var(--depth, 0px));
    transition: transform 400ms ease, box-shadow 400ms ease;
}
.orbital-node::after {
    content: "";
    position: absolute;
    top: 50%;
    right: -16px;
    width: 32px;
    height: 2px;
    background: linear-gradient(90deg, rgba(56, 189, 248, 0.0), rgba(56, 189, 248, 0.65));
}
.orbital-node:last-child::after {display: none;}
.orbital-node:hover {
    transform: translateZ(calc(var(--depth, 0px) + 28px));
    box-shadow: 0 30px 60px -28px rgba(37, 99, 235, 0.85);
}
.orbital-node span {
    font-size: 1.5rem;
}
.orbital-node h4 {
    margin: 12px 0 6px;
    font-size: 1.05rem;
}
.orbital-node p {
    margin: 0;
    color: rgba(226, 232, 240, 0.8);
    font-size: 0.9rem;
}
.guided-overlay {
    position: fixed;
    inset: 0;
    pointer-events: none;
    display: grid;
    place-items: center;
    z-index: 900;
}
.guided-overlay.hidden {display: none;}
.guided-panel {
    pointer-events: auto;
    max-width: 420px;
    padding: 26px 28px;
    border-radius: 24px;
    background: rgba(15, 23, 42, 0.94);
    border: 1px solid rgba(96, 165, 250, 0.24);
    box-shadow: 0 40px 120px -60px rgba(15, 23, 42, 0.9);
    backdrop-filter: blur(12px);
    text-align: center;
    animation: pulse 820ms ease-in-out infinite alternate;
}
.guided-panel h3 {
    margin: 0 0 10px;
}
.guided-panel p {
    margin: 0;
    color: rgba(226, 232, 240, 0.85);
}
.guided-panel footer {
    margin-top: 18px;
    font-size: 0.85rem;
    color: rgba(148, 163, 184, 0.85);
}
@keyframes aurora {
    0% {background-position: 0% 50%;}
    50% {background-position: 100% 50%;}
    100% {background-position: 0% 50%;}
}
@keyframes rise-in {
    from {opacity: 0; transform: translateY(18px);}
    to {opacity: 1; transform: translateY(0);}
}
@keyframes pulse {
    from {transform: scale(1);}
    to {transform: scale(1.02);}
}
.luxe-card-grid {display:flex; gap:1rem; flex-wrap:wrap;}
.luxe-card {
    border-radius: 18px;
    padding: 1rem;
    width: 220px;
    background: linear-gradient(145deg, rgba(15,25,36,0.9), rgba(48,74,102,0.85));
    color: #f5f9ff;
    position: relative;
    box-shadow: 0 25px 45px rgba(2,12,27,0.45);
    transition: transform 0.3s ease, box-shadow 0.3s ease;
    border: 1px solid rgba(255,255,255,0.08);
}
.luxe-card.is-active {
    transform: translateY(-8px) scale(1.02);
    box-shadow: 0 35px 60px rgba(0, 150, 255, 0.45);
    border-color: rgba(94,174,255,0.8);
}
.luxe-card h4 {margin: 0; font-size: 1.1rem;}
.luxe-card .tagline {opacity: 0.85; font-size: 0.85rem; margin-top: 0.4rem;}
.luxe-card .image {
    width: 100%;
    height: 120px;
    border-radius: 14px;
    margin-bottom: 0.8rem;
    background: radial-gradient(circle at top left, rgba(120,195,255,0.8), rgba(8,15,27,0.6));
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 3rem;
    text-shadow: 0 12px 30px rgba(0,0,0,0.35);
}
.target-3d-card {
    perspective: 1200px;
}
.target-3d-card .inner {
    background: linear-gradient(160deg, rgba(8,21,35,0.9), rgba(45,90,120,0.88));
    border-radius: 24px;
    padding: 1.5rem;
    min-height: 260px;
    color: #eaf4ff;
    box-shadow: 0 35px 55px rgba(3, 12, 32, 0.55);
    border: 1px solid rgba(255,255,255,0.08);
    transform: rotateY(-12deg) rotateX(6deg);
    transform-style: preserve-3d;
    position: relative;
}
.target-3d-card .inner::after {
    content: "";
    position: absolute;
    inset: 10px;
    border-radius: 20px;
    border: 1px solid rgba(255,255,255,0.05);
    pointer-events: none;
}
.circular-indicator {
    width: 60px;
    height: 60px;
    border-radius: 50%;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    font-weight: 600;
    background: conic-gradient(var(--accent-color, #59b1ff) calc(var(--value, 0) * 1%), rgba(255,255,255,0.08) 0);
    color: #0e2137;
    margin-right: 0.6rem;
}
.circular-indicator span {
    font-size: 0.85rem;
}
.slider-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 0.75rem;
}
.slider-row .stSlider {flex: 1;}
.feedback-pill {
    display: inline-flex;
    gap: 0.3rem;
    align-items: center;
    padding: 0.35rem 0.6rem;
    border-radius: 999px;
    background: rgba(88, 184, 255, 0.12);
    color: #cfe9ff;
    font-size: 0.78rem;
}
</style>
"""

_LUXE_COMPONENT_CSS = """
<style>
:root {
  --luxe-surface: rgba(12, 17, 27, 0.78);
  --luxe-border: rgba(148, 163, 184, 0.32);
  --luxe-border-strong: rgba(148, 163, 184, 0.48);
  --luxe-ink: #e9f0ff;
  --luxe-muted: rgba(226, 232, 240, 0.76);
  --luxe-accent: #60a5fa;
  --luxe-positive: #34d399;
  --luxe-warning: #f59e0b;
  --luxe-danger: #f87171;
}

@keyframes heroGlow {
  0% { box-shadow: 0 25px 70px rgba(96, 165, 250, 0.24); }
  50% { box-shadow: 0 35px 110px rgba(56, 189, 248, 0.45); }
  100% { box-shadow: 0 25px 70px rgba(96, 165, 250, 0.24); }
}

@keyframes parallaxDrift {
  0% { transform: translate3d(0, 0, 0) scale(1); opacity: 0.45; }
  50% { transform: translate3d(12px, -10px, 0) scale(1.05); opacity: 0.75; }
  100% { transform: translate3d(0, 0, 0) scale(1); opacity: 0.45; }
}

@keyframes sparkleShift {
  0% { background-position: 0% 50%; }
  50% { background-position: 100% 50%; }
  100% { background-position: 0% 50%; }
}

.luxe-hero {
  position: relative;
  overflow: hidden;
  border-radius: 32px;
  border: 1px solid var(--luxe-border);
  background: var(--hero-gradient, linear-gradient(135deg, rgba(59, 130, 246, 0.16), rgba(14, 165, 233, 0.06)));
  padding: var(--hero-padding, 2.6rem 3.1rem);
  color: var(--luxe-ink);
  isolation: isolate;
  backdrop-filter: blur(18px);
  box-shadow: 0 24px 60px rgba(8, 15, 35, 0.45);
  animation: heroGlow 14s ease-in-out infinite;
}

.luxe-hero::after {
  content: "";
  position: absolute;
  inset: -30%;
  background: radial-gradient(circle at 20% 20%, var(--hero-glow, rgba(96, 165, 250, 0.4)), transparent 62%);
  filter: blur(40px);
  opacity: 0.9;
  pointer-events: none;
  animation: sparkleShift 18s ease-in-out infinite;
}

.luxe-hero__content {
  position: relative;
  z-index: 1;
  max-width: 520px;
}

.luxe-hero__content h1 {
  font-size: clamp(2.5rem, 4vw, 3.2rem);
  margin-bottom: 0.7rem;
}

.luxe-hero__content p {
  font-size: 1.05rem;
  color: rgba(226, 232, 240, 0.82);
  margin: 0 0 1rem 0;
}

.luxe-hero__icon {
  font-size: 2rem;
  margin-bottom: 1rem;
  opacity: 0.85;
}

.luxe-hero__layer {
  position: absolute;
  top: 20%;
  left: 60%;
  font-size: var(--layer-size, 3.5rem);
  opacity: 0.5;
  filter: drop-shadow(0 18px 25px rgba(15, 23, 42, 0.45));
  animation: parallaxDrift var(--layer-speed, 18s) ease-in-out infinite;
}

.luxe-chip-row {
  display: inline-flex;
  flex-wrap: wrap;
  gap: var(--chip-gap, 0.6rem);
  margin-top: 1.2rem;
}

.luxe-chip {
  display: inline-flex;
  align-items: center;
  gap: 0.35rem;
  padding: var(--chip-padding, 0.4rem 0.9rem);
  font-size: var(--chip-size, 0.82rem);
  border-radius: 999px;
  border: 1px solid rgba(148, 163, 184, 0.25);
  background: rgba(15, 23, 42, 0.6);
  color: var(--luxe-ink);
  letter-spacing: 0.04em;
}

.luxe-chip[data-tone="accent"] { background: rgba(96, 165, 250, 0.16); border-color: rgba(125, 211, 252, 0.45); }
.luxe-chip[data-tone="info"] { background: rgba(14, 165, 233, 0.16); border-color: rgba(56, 189, 248, 0.5); }
.luxe-chip[data-tone="positive"] { background: rgba(52, 211, 153, 0.14); border-color: rgba(52, 211, 153, 0.45); }
.luxe-chip[data-tone="warning"] { background: rgba(245, 158, 11, 0.12); border-color: rgba(245, 158, 11, 0.45); }
.luxe-chip[data-tone="danger"] { background: rgba(248, 113, 113, 0.16); border-color: rgba(248, 113, 113, 0.45); }

.luxe-metric-galaxy {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(var(--metric-min, 13rem), 1fr));
  gap: var(--metric-gap, 1rem);
  margin-top: 1.8rem;
}

.luxe-metric {
  position: relative;
  border-radius: 20px;
  border: 1px solid var(--luxe-border);
  background: rgba(13, 17, 23, 0.76);
  padding: var(--metric-padding, 1.2rem 1.4rem);
  box-shadow: inset 0 0 0 1px rgba(148, 163, 184, 0.1), 0 18px 40px rgba(8, 15, 35, 0.38);
  display: flex;
  flex-direction: column;
  gap: 0.35rem;
  position: relative;
  overflow: hidden;
}

.luxe-metric[data-glow="true"]::after {
  content: "";
  position: absolute;
  inset: 0;
  background: radial-gradient(circle at 15% 20%, rgba(96, 165, 250, 0.24), transparent 65%);
  opacity: 0.85;
  z-index: 0;
  pointer-events: none;
}

.luxe-metric__icon {
  font-size: 1.15rem;
  opacity: 0.7;
}

.luxe-metric__label {
  font-size: 0.78rem;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  opacity: 0.72;
}

.luxe-metric__value {
  font-size: 1.48rem;
  font-weight: 700;
}

.luxe-metric__delta {
  font-size: 0.82rem;
  opacity: 0.75;
}

.luxe-metric__caption {
  font-size: 0.82rem;
  color: var(--luxe-muted);
}

.luxe-stack {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(var(--stack-min, 16rem), 1fr));
  gap: var(--stack-gap, 1.1rem);
  margin: var(--stack-margin, 1.6rem 0 0);
}

.luxe-card {
  position: relative;
  border-radius: 22px;
  border: 1px solid var(--luxe-border);
  background: rgba(12, 17, 27, 0.72);
  padding: var(--card-padding, 1.3rem 1.4rem);
  box-shadow: 0 18px 40px rgba(8, 15, 35, 0.35);
  overflow: hidden;
  backdrop-filter: blur(18px);
  color: var(--luxe-ink);
}

.luxe-card::before {
  content: "";
  position: absolute;
  inset: 0;
  background: linear-gradient(135deg, rgba(96, 165, 250, 0.14), transparent 55%);
  opacity: 0.75;
  pointer-events: none;
}

.luxe-card__icon {
  font-size: 1.4rem;
  margin-bottom: 0.4rem;
}

.luxe-card__title {
  font-size: 1.05rem;
  margin: 0 0 0.45rem 0;
}

.luxe-card__body {
  font-size: 0.92rem;
  color: var(--luxe-muted);
}

.luxe-card__footer {
  margin-top: 0.7rem;
  font-size: 0.8rem;
  color: rgba(226, 232, 240, 0.64);
}
</style>
"""

_UTILITY_CSS = """
<style>
.card {
  border-radius: 18px;
  padding: 1rem 1.25rem;
  background: rgba(15, 23, 42, 0.72);
  border: 1px solid rgba(148, 163, 184, 0.22);
  color: rgba(226, 232, 240, 0.92);
  box-shadow: 0 22px 45px -24px rgba(15, 23, 42, 0.75);
}
.card-flat {
  box-shadow: none;
  border-color: rgba(148, 163, 184, 0.35);
}
.pill {
  display: inline-flex;
  align-items: center;
  gap: 0.35rem;
  padding: 0.25rem 0.75rem;
  border-radius: 999px;
  font-size: 0.82rem;
  font-weight: 600;
  letter-spacing: 0.04em;
  text-transform: uppercase;
  border: 1px solid rgba(94, 234, 212, 0.45);
  background: rgba(45, 212, 191, 0.16);
  color: rgba(15, 23, 42, 0.88);
}
.pill.ok {
  border-color: rgba(34, 197, 94, 0.45);
  background: rgba(34, 197, 94, 0.18);
  color: rgba(15, 23, 42, 0.9);
}
.pill.warn,
.pill.med {
  border-color: rgba(245, 158, 11, 0.55);
  background: rgba(245, 158, 11, 0.18);
}
.pill.risk,
.pill.bad {
  border-color: rgba(239, 68, 68, 0.6);
  background: rgba(239, 68, 68, 0.2);
  color: rgba(15, 23, 42, 0.92);
}
.pill-solid {
  color: rgba(226, 232, 240, 0.95);
  background: rgba(15, 23, 42, 0.75);
  border-color: rgba(226, 232, 240, 0.45);
}
.small {
  font-size: 0.85rem;
  opacity: 0.82;
}
</style>
"""


def _load_css() -> None:
    """Inject shared CSS snippets once per session."""

    if st.session_state.get(_CSS_KEY):
        return

    for css in (_BRIEFING_AND_TARGET_CSS, _LUXE_COMPONENT_CSS, _UTILITY_CSS):
        st.markdown(css, unsafe_allow_html=True)

    st.session_state[_CSS_KEY] = True


# ---------------------------------------------------------------------------
# Basic helpers
# ---------------------------------------------------------------------------

def _current_theme() -> str:
    return st.session_state.get("hud_theme", "dark")


def _is_high_contrast() -> bool:
    return "high-contrast" in _current_theme()


def _is_colorblind_mode() -> bool:
    return st.session_state.get("hud_colorblind", "normal") != "normal"


def _class_names(*tokens: Iterable[str]) -> str:
    classes: list[str] = []
    for token in tokens:
        if isinstance(token, str):
            classes.append(token)
        else:
            classes.extend(t for t in token if t)
    return " ".join(cls for cls in classes if cls)


def render_card(title: str, body: str = "") -> str:
    """Return a luxe card block, injecting CSS if necessary."""

    _load_css()

    classes = ["card"]
    if _is_high_contrast():
        classes.append("card-flat")

    body_html = f'<div class="small">{body}</div>' if body else ""
    return f'<div class="{_class_names(classes)}"><h4>{title}</h4>{body_html}</div>'


def render_pill(label: str, kind: Literal["ok", "warn", "risk"] = "ok") -> str:
    """Return a status pill supporting tone variants."""

    _load_css()

    tone_map = {
        "ok": ("ok",),
        "warn": ("warn", "med"),
        "risk": ("risk", "bad"),
    }
    classes = ["pill", *tone_map.get(kind, ("ok",))]
    if _is_colorblind_mode():
        classes.append("pill-solid")

    return f'<span class="{_class_names(classes)}">{label}</span>'


# ---------------------------------------------------------------------------
# Data descriptors
# ---------------------------------------------------------------------------


@dataclass
class BriefingCard:
    """Descriptor for each animated briefing card."""

    title: str
    body: str
    accent: str = "#38bdf8"


@dataclass
class TimelineMilestone:
    """Descriptor for an orbital timeline milestone."""

    label: str
    description: str
    icon: str = "🛰️"


@dataclass
class TargetPresetMeta:
    """Metadata used to render the Tesla-style preset cards."""

    icon: str
    tagline: str


_PRESET_META: Dict[str, TargetPresetMeta] = {
    "Container": TargetPresetMeta(
        icon="📦",
        tagline="Para almacenaje hermético y modular sin sacrificar estilo.",
    ),
    "Utensil": TargetPresetMeta(
        icon="🍴",
        tagline="Diseñado para tareas delicadas con acabado pulido lunar.",
    ),
    "Interior": TargetPresetMeta(
        icon="🛋️",
        tagline="Habitáculos confortables que optimizan espacio y calor.",
    ),
    "Tool": TargetPresetMeta(
        icon="🛠️",
        tagline="Robustez industrial lista para cualquier misión orbital.",
    ),
}


# ---------------------------------------------------------------------------
# Mission briefing and guided flows
# ---------------------------------------------------------------------------

def _video_as_base64(video_path: Path) -> Optional[str]:
    try:
        data = video_path.read_bytes()
    except FileNotFoundError:
        return None
    if not data:
        return None
    encoded = base64.b64encode(data).decode("utf-8")
    return f"data:video/mp4;base64,{encoded}"


def mission_briefing(
    *,
    title: str,
    tagline: str,
    video_path: Optional[Path | str] = None,
    cards: Sequence[BriefingCard] = (),
    steps: Sequence[tuple[str, str]] = (),
) -> None:
    """Render the mission briefing hero with media loop and animated cards."""

    _load_css()

    st.markdown(f"## {title}")
    st.caption(tagline)

    media_src: Optional[str] = None
    if video_path:
        media_src = _video_as_base64(Path(video_path))

    st.markdown("<div class='briefing-grid'>", unsafe_allow_html=True)

    media_html = (
        f"<video autoplay loop muted playsinline src='{media_src}'></video>"
        if media_src
        else "<div class='briefing-fallback'>Simulación orbital</div>"
    )

    st.markdown(
        f"<div class='briefing-video'>{media_html}</div>",
        unsafe_allow_html=True,
    )

    cards_html = "".join(
        f"""
        <div class='briefing-card' style="--card-accent: {card.accent};">
            <h3>{card.title}</h3>
            <p>{card.body}</p>
        </div>
        """
        for card in cards
    )

    steps_html = "".join(
        f"""
        <div class='briefing-step' style="animation-delay: {idx * 120}ms;">
            <span>{idx + 1}</span>
            <div>
                <strong>{step_title}</strong>
                <small>{copy}</small>
            </div>
        </div>
        """
        for idx, (step_title, copy) in enumerate(steps)
    )

    st.markdown(
        f"""
        <div class='briefing-cards'>
            {cards_html}
            <div class='briefing-stepper'>
                {steps_html}
            </div>
        </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def orbital_timeline(milestones: Iterable[TimelineMilestone]) -> None:
    """Render the animated orbital timeline."""

    _load_css()

    st.markdown("<div class='orbital-timeline'>", unsafe_allow_html=True)
    nodes = []
    for depth, milestone in enumerate(milestones):
        nodes.append(
            f"""
            <div class='orbital-node' style="--depth: {depth * 18}px;">
                <span>{milestone.icon}</span>
                <h4>{milestone.label}</h4>
                <p>{milestone.description}</p>
            </div>
            """
        )
    st.markdown(
        f"<div class='orbital-track'>{''.join(nodes)}</div></div>",
        unsafe_allow_html=True,
    )


def guided_demo(
    *,
    steps: Sequence[TimelineMilestone],
    session_key: str = "guided_demo_step",
    step_duration: float = 8.0,
) -> Optional[TimelineMilestone]:
    """Render an overlay with rotating guidance steps."""

    _load_css()

    if not steps:
        return None

    state = st.session_state.setdefault(session_key, {"index": 0, "last_tick": 0.0})
    current_index = state["index"]

    with st.sidebar:
        st.write("### Demo asistida")
        option_labels = [step.label for step in steps]
        chosen_label = st.radio(
            "Seleccioná el foco actual",
            option_labels,
            index=current_index,
            key=f"{session_key}_selector",
        )
        current_index = option_labels.index(chosen_label)
        state["index"] = current_index

    placeholder = st.empty()
    active_step = steps[current_index]
    payload = json.dumps(
        {
            "label": active_step.label,
            "description": active_step.description,
            "icon": active_step.icon,
            "duration": step_duration,
        }
    )
    placeholder.markdown(
        f"""
        <div class='guided-overlay'>
            <div class='guided-panel' data-payload='{payload}'>
                <h3>{active_step.label}</h3>
                <p>{active_step.description}</p>
                <footer>Actualiza cada {step_duration:.1f} segundos</footer>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    return active_step


# ---------------------------------------------------------------------------
# Target configurator
# ---------------------------------------------------------------------------

def _render_preset_cards(presets: Iterable[Dict]) -> str:
    st.markdown("<div class='luxe-card-grid'>", unsafe_allow_html=True)
    columns = st.columns(len(presets))
    selected_name = st.session_state.get("_target_selected", "")

    for column, preset in zip(columns, presets):
        meta = _PRESET_META.get(
            preset.get("category", ""),
            TargetPresetMeta(icon="🛰️", tagline="Optimizado para desafíos orbitales."),
        )
        is_selected = selected_name == preset["name"]
        with column:
            if st.button(
                f"{preset['name']}",
                key=f"preset_{preset['name']}",
                help=meta.tagline,
            ):
                selected_name = preset["name"]
            st.markdown(
                f"""
                <div class='luxe-card {'is-active' if is_selected else ''}'>
                    <div class='image'>{meta.icon}</div>
                    <h4>{preset['name']}</h4>
                    <div class='tagline'>{meta.tagline}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.session_state["_target_selected"] = selected_name or presets[0]["name"]
    st.markdown("</div>", unsafe_allow_html=True)
    return st.session_state["_target_selected"]


def _render_indicator(value: float, max_value: float, accent: str, label: str) -> None:
    percentage = 100 * value / max_value if max_value else 0
    st.markdown(
        f"<div class='circular-indicator' style='--value:{percentage}; --accent-color:{accent};'>"
        f"<span>{percentage:.0f}%</span></div><div class='feedback-pill'>{label}: {value:.2f}</div>",
        unsafe_allow_html=True,
    )


def _gauge(title: str, value: float, max_value: float, unit: str, color: str = "#59b1ff") -> None:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=value,
            number={"suffix": unit},
            gauge={
                "axis": {"range": [0, max_value]},
                "bar": {"color": color},
                "bgcolor": "rgba(6,20,33,0.65)",
                "borderwidth": 1,
                "bordercolor": "rgba(255,255,255,0.1)",
            },
            title={"text": title},
        )
    )
    fig.update_layout(
        margin=dict(l=10, r=10, t=30, b=10),
        height=200,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#eaf4ff"},
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def _build_audio_clip() -> bytes:
    """Create a short sine beep encoded as WAV bytes."""

    sample_rate = 44100
    duration = 0.25
    frequency = 880
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    tone = 0.5 * np.sin(2 * np.pi * frequency * t)
    audio = np.int16(tone * 32767)

    from io import BytesIO
    import wave

    buffer = BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio.tobytes())
    return buffer.getvalue()


def target_configurator(
    presets: List[Dict],
    scenario_options: Tuple[str, ...] | List[str] | None = None,
) -> Dict:
    """Render the luxe target configurator and return the resulting target spec."""

    if not presets:
        st.warning("No hay presets disponibles para configurar el objetivo.")
        return {}

    _load_css()
    scenario_options = tuple(scenario_options or ())

    st.subheader("Target Configurator ✨")
    st.caption("Seleccioná un preset y refiná el objetivo con feedback inmediato.")

    selected_name = _render_preset_cards(presets)
    selected_preset = next(p for p in presets if p["name"] == selected_name)

    current_target = st.session_state.get("target")
    default_scenario = scenario_options[0] if scenario_options else ""
    if current_target is None:
        st.session_state["target"] = {
            **selected_preset,
            "scenario": default_scenario,
            "crew_time_low": False,
        }
    elif current_target.get("name") != selected_name:
        st.session_state["target"] = {
            **selected_preset,
            "scenario": current_target.get("scenario", default_scenario),
            "crew_time_low": current_target.get("crew_time_low", False),
        }

    current_target = st.session_state["target"]

    previous_values = st.session_state.get("_target_prev_values", {})

    base = {
        key: float(selected_preset[key]) if "max_" not in key else selected_preset[key]
        for key in ("rigidity", "tightness", "max_water_l", "max_energy_kwh", "max_crew_min")
    }

    main_col, summary_col = st.columns([3, 1])

    with main_col:
        preview_col, controls_col = st.columns([1.2, 1.8])
        with preview_col:
            st.markdown(
                f"""
                <div class="target-3d-card">
                    <div class="inner">
                        <h3>{selected_name}</h3>
                        <p>Render conceptual en vivo del objeto seleccionado.</p>
                        <div style="margin-top:2rem; font-size:0.85rem; opacity:0.85;">
                            Rigidez base: {base['rigidity']:.2f}<br/>
                            Estanqueidad base: {base['tightness']:.2f}
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            scenario = ""
            if scenario_options:
                default_scenario = current_target.get("scenario", scenario_options[0])
                if default_scenario not in scenario_options:
                    default_idx = 0
                else:
                    default_idx = scenario_options.index(default_scenario)
                scenario = st.selectbox(
                    "Escenario del reto",
                    scenario_options,
                    index=default_idx,
                )
            crew_low = st.toggle(
                "Crew-time Low",
                value=current_target.get("crew_time_low", False),
                help="Prioriza procesos con poco tiempo de tripulación.",
            )
        with controls_col:
            st.markdown("#### Ajustes dinámicos")
            rigidity = st.slider(
                "Rigidez deseada",
                0.0,
                1.0,
                float(current_target.get("rigidity", selected_preset["rigidity"])),
                0.05,
            )
            _render_indicator(rigidity, 1.0, "#4ecdc4", "Rigidez")

            tightness = st.slider(
                "Estanqueidad deseada",
                0.0,
                1.0,
                float(current_target.get("tightness", selected_preset["tightness"])),
                0.05,
            )
            _render_indicator(tightness, 1.0, "#ff8c69", "Estanqueidad")

            max_water = st.slider(
                "Agua máxima (L)",
                0.0,
                3.0,
                float(current_target.get("max_water_l", selected_preset["max_water_l"])),
                0.1,
            )
            _render_indicator(max_water, 3.0, "#59b1ff", "Agua")

            max_energy = st.slider(
                "Energía máxima (kWh)",
                0.0,
                3.0,
                float(current_target.get("max_energy_kwh", selected_preset["max_energy_kwh"])),
                0.1,
            )
            _render_indicator(max_energy, 3.0, "#f8d66d", "Energía")

            max_crew = st.slider(
                "Tiempo máximo de tripulación (min)",
                5,
                60,
                int(current_target.get("max_crew_min", selected_preset["max_crew_min"])),
                1,
            )
            _render_indicator(max_crew, 60.0, "#d277ff", "Crew")

            audio_enabled = st.checkbox(
                "Audio feedback", value=st.session_state.get("_target_audio", False)
            )
            haptic_enabled = st.checkbox(
                "Vibración háptica", value=st.session_state.get("_target_haptic", False)
            )
            st.session_state["_target_audio"] = audio_enabled
            st.session_state["_target_haptic"] = haptic_enabled

            if audio_enabled:
                st.audio(_build_audio_clip(), format="audio/wav", sample_rate=44100)

    with main_col:
        st.markdown("#### Simulación visual")
        gauges = st.columns(3)
        with gauges[0]:
            _gauge("Agua", max_water, 3.0, " L")
        with gauges[1]:
            _gauge("Energía", max_energy, 3.0, " kWh", color="#f8d66d")
        with gauges[2]:
            _gauge("Crew", max_crew, 60.0, " min", color="#d277ff")

        feedback_area = st.empty()

    current_values = {
        "rigidity": rigidity,
        "tightness": tightness,
        "max_water_l": max_water,
        "max_energy_kwh": max_energy,
        "max_crew_min": max_crew,
    }

    if previous_values and current_values != previous_values:
        messages = []
        if st.session_state.get("_target_audio"):
            messages.append("🔊 Audio feedback (simulado)")
        if st.session_state.get("_target_haptic"):
            messages.append("🤲 Haptic pulse (simulado)")
        if messages:
            feedback_area.success(" ".join(messages))
    else:
        feedback_area.empty()

    st.session_state["_target_prev_values"] = current_values

    with summary_col:
        st.markdown("### Resumen")
        st.caption("Comparación contra el preset seleccionado.")

        def metric(label: str, value: float, base_value: float, unit: str = "") -> None:
            delta = value - base_value
            st.metric(label, f"{value:.2f}{unit}", f"{delta:+.2f}{unit}")

        metric("Rigidez", rigidity, base["rigidity"], "")
        metric("Estanqueidad", tightness, base["tightness"], "")
        metric("Agua máx.", max_water, base["max_water_l"], " L")
        metric("Energía máx.", max_energy, base["max_energy_kwh"], " kWh")
        metric("Crew máx.", max_crew, base["max_crew_min"], " min")

    target = {
        "name": selected_name,
        "rigidity": rigidity,
        "tightness": tightness,
        "max_water_l": max_water,
        "max_energy_kwh": max_energy,
        "max_crew_min": max_crew,
        "scenario": scenario if scenario_options else current_target.get("scenario", ""),
        "crew_time_low": crew_low,
    }

    st.session_state["target"] = target
    return target


# ---------------------------------------------------------------------------
# Luxe hero components
# ---------------------------------------------------------------------------

def _merge_styles(base: Mapping[str, str], extra: Mapping[str, str] | None = None) -> str:
    merged: dict[str, str] = dict(base)
    if extra:
        merged.update({k: v for k, v in extra.items() if v is not None})
    return "; ".join(f"{k}: {v}" for k, v in merged.items())


def ChipRow(
    chips: Sequence[str | Mapping[str, str]],
    *,
    tone: str | None = None,
    size: str = "md",
    render: bool = True,
    gap: str | None = None,
) -> str:
    """Render a compact list of chips with luxe styling."""

    _load_css()

    size_map: Mapping[str, tuple[str, str]] = {
        "sm": ("0.32rem 0.8rem", "0.74rem"),
        "md": ("0.38rem 0.9rem", "0.82rem"),
        "lg": ("0.45rem 1.05rem", "0.92rem"),
    }
    padding, font_size = size_map.get(size, size_map["md"])
    row_style = {
        "--chip-padding": padding,
        "--chip-size": font_size,
    }
    if gap:
        row_style["--chip-gap"] = gap

    html = [f"<div class='luxe-chip-row' style='{_merge_styles(row_style, {})}'>"]
    for item in chips:
        if isinstance(item, Mapping):
            label = item.get("label", "")
            icon = item.get("icon")
            chip_tone = item.get("tone", tone)
        else:
            label = str(item)
            icon = None
            chip_tone = tone
        icon_fragment = f"<span>{icon}</span>" if icon else ""
        html.append(
            f"<span class='luxe-chip' data-tone='{chip_tone or ''}'>"
            f"{icon_fragment}<span>{label}</span>"
            "</span>"
        )
    html.append("</div>")
    html_markup = "".join(html)
    if render:
        st.markdown(html_markup, unsafe_allow_html=True)
    return html_markup


@dataclass
class TeslaHero:
    title: str
    subtitle: str
    chips: Sequence[str | Mapping[str, str]] = field(default_factory=list)
    icon: str | None = None
    gradient: str | None = None
    glow: str | None = None
    density: str = "cozy"
    parallax_icons: Sequence[Mapping[str, str]] = field(default_factory=list)

    def render(self) -> None:
        _load_css()
        padding_map = {
            "compact": "1.9rem 2.2rem",
            "cozy": "2.5rem 2.9rem",
            "roomy": "3.1rem 3.4rem",
        }
        padding = padding_map.get(self.density, padding_map["cozy"])
        hero_style = {
            "--hero-padding": padding,
        }
        if self.gradient:
            hero_style["--hero-gradient"] = self.gradient
        if self.glow:
            hero_style["--hero-glow"] = self.glow

        layers = []
        for idx, layer in enumerate(self.parallax_icons):
            icon = layer.get("icon", "✦")
            top = layer.get("top", f"{10 + idx * 12}%")
            left = layer.get("left", f"{55 + idx * 8}%")
            size = layer.get("size", "4rem")
            speed = layer.get("speed", f"{16 + idx * 4}s")
            layers.append(
                f"<span class='luxe-hero__layer' style='top:{top};left:{left};--layer-size:{size};--layer-speed:{speed};'>"
                f"{icon}</span>"
            )

        chips_html = ChipRow(self.chips, render=False) if self.chips else ""
        icon_html = f"<div class='luxe-hero__icon'>{self.icon}</div>" if self.icon else ""

        html = f"""
        <div class='luxe-hero' style='{_merge_styles(hero_style, {})}'>
          {''.join(layers)}
          <div class='luxe-hero__content'>
            {icon_html}
            <h1>{self.title}</h1>
            <p>{self.subtitle}</p>
            {chips_html}
          </div>
        </div>
        """
        st.markdown(html, unsafe_allow_html=True)


@dataclass
class MetricItem:
    label: str
    value: str
    caption: str | None = None
    delta: str | None = None
    icon: str | None = None
    tone: str | None = None


@dataclass
class MetricGalaxy:
    metrics: Sequence[MetricItem]
    glow: bool = True
    density: str = "cozy"
    min_width: str = "13rem"

    def render(self) -> None:
        _load_css()
        padding_map = {
            "compact": "1rem 1.15rem",
            "cozy": "1.2rem 1.4rem",
            "roomy": "1.45rem 1.65rem",
        }
        gap_map = {
            "compact": "0.8rem",
            "cozy": "1rem",
            "roomy": "1.3rem",
        }
        style = {
            "--metric-padding": padding_map.get(self.density, padding_map["cozy"]),
            "--metric-gap": gap_map.get(self.density, gap_map["cozy"]),
            "--metric-min": self.min_width,
        }
        html = [f"<div class='luxe-metric-galaxy' style='{_merge_styles(style, {})}'>"]
        for metric in self.metrics:
            tone = metric.tone or ("positive" if (metric.delta and metric.delta.startswith("+")) else "")
            icon_html = f"<div class='luxe-metric__icon'>{metric.icon}</div>" if metric.icon else ""
            delta_html = f"<div class='luxe-metric__delta'>{metric.delta}</div>" if metric.delta else ""
            caption_html = f"<div class='luxe-metric__caption'>{metric.caption}</div>" if metric.caption else ""
            html.append(
                f"<div class='luxe-metric' data-glow='{str(self.glow).lower()}' data-tone='{tone}'>"
                f"{icon_html}"
                f"<div class='luxe-metric__label'>{metric.label}</div>"
                f"<div class='luxe-metric__value'>{metric.value}</div>"
                f"{delta_html}"
                f"{caption_html}"
                "</div>"
            )
        html.append("</div>")
        st.markdown("".join(html), unsafe_allow_html=True)


@dataclass
class GlassCard:
    title: str
    body: str
    icon: str | None = None
    footer: str | None = None


@dataclass
class GlassStack:
    cards: Sequence[GlassCard]
    columns_min: str = "16rem"
    density: str = "cozy"

    def render(self) -> None:
        _load_css()
        padding_map = {
            "compact": "1.05rem 1.15rem",
            "cozy": "1.3rem 1.4rem",
            "roomy": "1.6rem 1.75rem",
        }
        gap_map = {
            "compact": "0.9rem",
            "cozy": "1.1rem",
            "roomy": "1.45rem",
        }
        style = {
            "--card-padding": padding_map.get(self.density, padding_map["cozy"]),
            "--stack-gap": gap_map.get(self.density, gap_map["cozy"]),
            "--stack-min": self.columns_min,
        }
        html = [f"<div class='luxe-stack' style='{_merge_styles(style, {})}'>"]
        for card in self.cards:
            icon_html = f"<div class='luxe-card__icon'>{card.icon}</div>" if card.icon else ""
            footer_html = f"<div class='luxe-card__footer'>{card.footer}</div>" if card.footer else ""
            html.append(
                f"<div class='luxe-card'>"
                f"{icon_html}"
                f"<h3 class='luxe-card__title'>{card.title}</h3>"
                f"<div class='luxe-card__body'>{card.body}</div>"
                f"{footer_html}"
                "</div>"
            )
        html.append("</div>")
        st.markdown("".join(html), unsafe_allow_html=True)


__all__ = [
    "BriefingCard",
    "TimelineMilestone",
    "TargetPresetMeta",
    "mission_briefing",
    "orbital_timeline",
    "guided_demo",
    "target_configurator",
    "render_card",
    "render_pill",
    "ChipRow",
    "TeslaHero",
    "MetricGalaxy",
    "MetricItem",
    "GlassStack",
    "GlassCard",
]
