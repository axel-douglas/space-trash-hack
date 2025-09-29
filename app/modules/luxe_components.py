"""High-fidelity UI primitives for the Rex-AI Streamlit experience."""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

import streamlit as st


_CSS_KEY = "__rexai_luxe_css__"


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


def _load_css() -> None:
    """Inject shared CSS snippets once per session."""

    if st.session_state.get(_CSS_KEY):
        return

    st.markdown(
        """
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
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.session_state[_CSS_KEY] = True


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
    """Render a faux-3D orbital timeline."""

    _load_css()

    nodes = []
    for depth, milestone in enumerate(milestones):
        nodes.append(
            f"""
            <div class='orbital-node' style="--depth: {depth * 16}px;">
                <span>{milestone.icon}</span>
                <h4>{milestone.label}</h4>
                <p>{milestone.description}</p>
            </div>
            """
        )

    st.markdown(
        f"""
        <div class='orbital-timeline'>
            <div class='orbital-track'>
                {''.join(nodes)}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def guided_demo(
    *,
    steps: Sequence[TimelineMilestone],
    param_key: str = "demo",
    query_value: str = "mission",
    step_duration: float = 7.0,
    loop: bool = True,
) -> Optional[TimelineMilestone]:
    """Drive a guided overlay demo using query parameters.

    Returns the current step so callers can optionally adapt their layout.
    """

    _load_css()

    params = st.experimental_get_query_params()
    is_active = params.get(param_key, [None])[0] == query_value

    col1, col2 = st.columns(2)
    with col1:
        if st.button("▶️ Activar demo guiada", disabled=is_active or not steps):
            new_params = dict(params)
            new_params[param_key] = [query_value]
            new_params["step"] = ["0"]
            st.experimental_set_query_params(**new_params)
            st.experimental_rerun()

    with col2:
        if st.button("⏹️ Detener demo", disabled=not is_active):
            new_params = {k: v for k, v in params.items() if k not in {param_key, "step"}}
            st.experimental_set_query_params(**new_params)
            st.experimental_rerun()

    if not is_active or not steps:
        return None

    step_raw = params.get("step", ["0"])[0]
    try:
        step_index = int(step_raw)
    except ValueError:
        step_index = 0

    step_index = max(0, min(step_index, len(steps) - 1))
    active_step = steps[step_index]

    # Schedule automatic progression via a lightweight JS snippet.
    if step_duration > 0 and (loop or step_index < len(steps) - 1):
        next_index = (step_index + 1) % len(steps) if loop else min(step_index + 1, len(steps) - 1)
        payload = json.dumps(
            {
                "param_key": param_key,
                "param_value": query_value,
                "next_step": str(next_index),
                "duration_ms": int(step_duration * 1000),
            }
        )
        st.markdown(
            f"""
            <script>
            const cfg = {payload};
            setTimeout(() => {{
                const url = new URL(window.location.href);
                url.searchParams.set(cfg.param_key, cfg.param_value);
                url.searchParams.set('step', cfg.next_step);
                window.location.href = url.toString();
            }}, cfg.duration_ms);
            </script>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        f"""
        <div class='guided-overlay'>
            <div class='guided-panel'>
                <h3>{active_step.icon} {active_step.label}</h3>
                <p>{active_step.description}</p>
                <footer>Paso {step_index + 1} de {len(steps)}</footer>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    return active_step

