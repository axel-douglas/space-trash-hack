"""Mission HUD utilities shared across Rex-AI pages."""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from typing import Sequence

import streamlit as st
from streamlit.components.v1 import html

from app.modules.ml_models import get_model_registry


@dataclass(frozen=True)
class MissionStep:
    """Definition of a mission step in the guided flow."""

    key: str
    label: str
    icon: str
    page: str
    description: str


MISSION_STEPS: tuple[MissionStep, ...] = (
    MissionStep("brief", "Brief", "🛰️", "Home", "Resumen y preparación"),
    MissionStep("inventory", "Inventario", "🧱", "1_Inventory_Builder", "Normalizar residuos"),
    MissionStep("target", "Target", "🎯", "2_Target_Designer", "Definir objetivos"),
    MissionStep("generator", "Generador", "🤖", "3_Generator", "Recetas asistidas"),
    MissionStep("results", "Resultados", "📊", "4_Results_and_Tradeoffs", "Trade-offs y métricas"),
    MissionStep("compare", "Comparar", "🧪", "5_Compare_and_Explain", "Explicabilidad"),
    MissionStep("export", "Export", "📦", "6_Pareto_and_Export", "Pareto y export"),
    MissionStep("playbooks", "Playbooks", "📚", "7_Scenario_Playbooks", "Escenarios"),
    MissionStep("feedback", "Feedback", "📝", "8_Feedback_and_Impact", "Impacto y retraining"),
    MissionStep("capacity", "Capacidad", "⚙️", "9_Capacity_Simulator", "Simulación"),
)

_HUD_STATE_KEY = "__mission_hud_injected__"


def set_active_step(step_key: str) -> None:
    """Persist the active step so the HUD can highlight it."""

    st.session_state["mission_active_step"] = step_key


def _step_from_key(step_key: str | None) -> MissionStep | None:
    if not step_key:
        return None
    for step in MISSION_STEPS:
        if step.key == step_key:
            return step
    return None


@lru_cache(maxsize=1)
def _model_metadata() -> dict[str, str]:
    """Read lightweight metadata from the model registry once per run."""

    registry = get_model_registry()
    ready = "✅ Listo" if registry.ready else "⚠️ Requiere entrenamiento"
    trained_label = registry.metadata.get("trained_label") or registry.metadata.get("trained_on") or "—"
    trained_at = registry.metadata.get("trained_at") or "sin metadata"
    return {
        "status": ready,
        "model_name": registry.metadata.get("model_name", "rexai-rf-ensemble"),
        "trained_label": str(trained_label),
        "trained_at": str(trained_at),
        "uncertainty": registry.uncertainty_label(),
    }


def _page_url(page: str) -> str:
    return f"./?page={page}"


def _hud_css() -> str:
    return """
    <style>
      .mission-hud {position: sticky; top: 0; z-index: 999; margin-bottom: 1.4rem;}
      .mission-hud__wrap {backdrop-filter: blur(12px); background: rgba(13,17,23,0.82); border: 1px solid rgba(96,165,250,0.18);
        border-radius: 20px; padding: 14px 20px; display: grid; grid-template-columns: auto 1fr auto; gap: 18px; align-items: center;}
      .mission-hud__logo {display: flex; align-items: center; gap: 10px; font-weight: 700; letter-spacing: .02em; color: var(--ink, #e2e8f0); font-size: 1.05rem;}
      .mission-hud__steps {display: flex; gap: 10px; overflow-x: auto; padding-bottom: 6px;}
      .mission-hud__step {display: inline-flex; align-items: center; gap: 8px; padding: 6px 12px; border-radius: 999px; border: 1px solid rgba(148,163,184,0.22); color: rgba(226,232,240,0.78); font-size: 0.82rem; transition: all .3s ease; text-decoration: none; white-space: nowrap;}
      .mission-hud__step:hover {border-color: rgba(96,165,250,0.45); color: #f8fafc; box-shadow: 0 6px 18px rgba(15,118,110,0.18);}
      .mission-hud__step.is-active {background: linear-gradient(135deg, rgba(59,130,246,0.35), rgba(14,165,233,0.18)); border-color: rgba(96,165,250,0.75); color: #f8fafc;}
      .mission-hud__status {display: grid; gap: 4px; text-align: right;}
      .mission-hud__status small {opacity: 0.68; font-size: 0.7rem;}
      .mission-hud__progress {height: 4px; margin-top: 8px; background: rgba(96,165,250,0.18); border-radius: 99px; overflow: hidden;}
      .mission-hud__progress-bar {height: 100%; background: linear-gradient(90deg, rgba(59,130,246,1), rgba(14,165,233,1)); transition: width .4s ease;}
      .mission-hud__ctas {display: flex; gap: 8px; margin-top: 6px; justify-content: flex-end;}
      .mission-hud__cta {padding: 6px 12px; border-radius: 10px; border: 1px solid rgba(96,165,250,0.4); background: rgba(15,23,42,0.75); color: #e2e8f0; font-size: 0.78rem; text-decoration: none; transition: all .3s ease;}
      .mission-hud__cta:hover {background: rgba(59,130,246,0.18);}
      .mission-breadcrumbs {margin-bottom: 1rem; font-size: 0.78rem; display: flex; gap: 6px; align-items: center; color: rgba(226,232,240,0.76);}
      .mission-breadcrumbs a {color: rgba(148,197,255,0.85); text-decoration: none; font-weight: 600;}
      .mission-breadcrumbs span {opacity: 0.65;}
    </style>
    """


def _render_shortcuts_script(step_urls: dict[str, str]) -> None:
    payload = json.dumps(step_urls)
    escaped = payload.replace("\\", "\\\\").replace("'", "\\'")
    html(
        """
        <div id="mission-hud-portal"></div>
        <script>
        (function() {
          const initKey = '__missionHudHotkeys';
          if (window[initKey]) return;
          const urls = JSON.parse('%s');
          function goTo(url) {
            if (!url) return;
            const base = window.parent?.location ?? window.location;
            base.href = url;
          }
          function onKey(event) {
            if (event.altKey || event.metaKey || event.ctrlKey) return;
            const tag = (event.target?.tagName || '').toLowerCase();
            if (['input','textarea','select'].includes(tag) || event.target?.isContentEditable) return;
            const url = urls[event.code];
            if (url) {
              event.preventDefault();
              goTo(url);
            }
          }
          function animateSteps() {
            const attempt = () => {
              const host = window.parent?.document ?? document;
              const chips = host.querySelectorAll('.mission-hud__step');
              if (!chips.length) return;
              if (!window.framerMotion || !window.framerMotion.animate) return;
              chips.forEach((chip, idx) => {
                window.framerMotion.animate(
                  chip,
                  { opacity: [0, 1], transform: ['translateY(-6px)', 'translateY(0px)'] },
                  { duration: 0.6, delay: idx * 0.05, easing: 'easeOut' }
                );
              });
            };
            if (window.framerMotion) {
              attempt();
            } else {
              const scriptId = 'framer-motion-umd';
              if (!document.getElementById(scriptId)) {
                const script = document.createElement('script');
                script.id = scriptId;
                script.src = 'https://unpkg.com/framer-motion@10.16.5/dist/framer-motion.umd.js';
                script.onload = attempt;
                document.head.appendChild(script);
              }
            }
          }
          window.addEventListener('keydown', onKey, true);
          const observer = new MutationObserver(animateSteps);
          observer.observe(document.body, { childList: true, subtree: true });
          animateSteps();
          window[initKey] = true;
        })();
        </script>
        """
        % escaped,
        height=0,
    )


def render_mission_hud() -> None:
    """Render the Mission HUD (logo, steps, model state and CTAs)."""

    st.markdown(_hud_css(), unsafe_allow_html=True)

    if st.session_state.get(_HUD_STATE_KEY):
        st.session_state[_HUD_STATE_KEY] += 1
    else:
        st.session_state[_HUD_STATE_KEY] = 1

    metadata = _model_metadata()
    active_step = _step_from_key(st.session_state.get("mission_active_step"))

    progress_index = 0
    steps_markup = []
    for idx, step in enumerate(MISSION_STEPS, start=1):
        url = _page_url(step.page)
        is_active = active_step.key == step.key if active_step else False
        if is_active:
            progress_index = idx
        class_attr = "is-active" if is_active else ""
        steps_markup.append(
            (
                f"<a class='mission-hud__step {class_attr}' href='{url}' title='{step.description}'>"
                f"<span>{step.icon}</span><strong>{idx} · {step.label}</strong>"
                "</a>"
            )
        )

    progress = (progress_index / len(MISSION_STEPS)) if progress_index else 0.0

    next_step = None
    prev_step = None
    if active_step:
        current_idx = MISSION_STEPS.index(active_step)
        if current_idx + 1 < len(MISSION_STEPS):
            next_step = MISSION_STEPS[current_idx + 1]
        if current_idx - 1 >= 0:
            prev_step = MISSION_STEPS[current_idx - 1]

    ctas: list[str] = []
    if prev_step:
        ctas.append(
            f'<a class="mission-hud__cta" href="{_page_url(prev_step.page)}">⬅️ {prev_step.label}</a>'
        )
    if next_step:
        ctas.append(
            f'<a class="mission-hud__cta" href="{_page_url(next_step.page)}">{next_step.label} ➡️</a>'
        )

    st.markdown(
        f"""
        <div class="mission-hud">
          <div class="mission-hud__wrap">
            <div class="mission-hud__logo">🛰️ Mission HUD <span style="opacity:0.6;font-weight:500;">Rex-AI</span></div>
            <div>
              <div class="mission-hud__steps">{''.join(steps_markup)}</div>
              <div class="mission-hud__progress"><div class="mission-hud__progress-bar" style="width:{progress*100:.1f}%;"></div></div>
            </div>
            <div class="mission-hud__status">
              <div><strong>{metadata['status']}</strong> · {metadata['model_name']}</div>
              <small>Entrenado: {metadata['trained_label']} · {metadata['trained_at']} · {metadata['uncertainty']}</small>
              <div class="mission-hud__ctas">{''.join(ctas)}</div>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    ordered_urls = {
        ("Digit0" if idx == 10 else f"Digit{idx}"): _page_url(step.page)
        for idx, step in enumerate(MISSION_STEPS, start=1)
        if idx <= 10
    }
    _render_shortcuts_script(ordered_urls)


def render_breadcrumbs(current_step_key: str, extra: Sequence[tuple[str, str]] | None = None) -> None:
    """Render breadcrumbs using mission steps and optional extra nodes."""

    trail: list[tuple[str, str | None]] = [("Home", _page_url("Home"))]
    for step in MISSION_STEPS:
        trail.append((f"{step.icon} {step.label}", _page_url(step.page)))
        if step.key == current_step_key:
            break

    if extra:
        trail.extend((label, url) for label, url in extra)

    if not trail:
        return

    fragments: list[str] = []
    for idx, (label, url) in enumerate(trail):
        if idx:
            fragments.append('<span>›</span>')
        if url and idx < len(trail) - 1:
            fragments.append(f'<a href="{url}">{label}</a>')
        else:
            fragments.append(f'<span>{label}</span>')

    st.markdown(
        f"<nav class='mission-breadcrumbs'>{''.join(fragments)}</nav>",
        unsafe_allow_html=True,
    )

