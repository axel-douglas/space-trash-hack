"""Mission HUD utilities shared across Rex-AI pages."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Sequence

import streamlit as st
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
def _model_metadata() -> dict[str, str | dict[str, str]]:
    """Read lightweight metadata from the model registry once per run."""

    registry = get_model_registry()
    ready = "✅ Listo" if registry.ready else "⚠️ Requiere entrenamiento"
    trained_label = registry.metadata.get("trained_label") or registry.metadata.get("trained_on") or "—"
    trained_at = registry.metadata.get("trained_at") or "sin metadata"

    status_badge = {
        "label": "Modelo listo" if registry.ready else "Entrenamiento pendiente",
        "tone": "success" if registry.ready else "danger",
    }

    uncertainty_raw = registry.uncertainty_label()
    uncertainty_tone: str
    lowered = uncertainty_raw.lower()
    if lowered in {"reportada", "reported"}:
        uncertainty_tone = "success"
    elif lowered in {"alta", "high"}:
        uncertainty_tone = "danger"
    else:
        uncertainty_tone = "warning"

    uncertainty_badge = {
        "label": f"Incertidumbre {uncertainty_raw}",
        "tone": uncertainty_tone,
    }

    return {
        "status": ready,
        "model_name": registry.metadata.get("model_name", "rexai-rf-ensemble"),
        "trained_label": str(trained_label),
        "trained_at": str(trained_at),
        "uncertainty": uncertainty_raw,
        "status_badge": status_badge,
        "uncertainty_badge": uncertainty_badge,
    }


def _page_url(page: str) -> str:
    return f"./?page={page}"


def _hud_css() -> str:
    return """
    <style>
      .mission-hud {position: sticky; top: 0; z-index: 999; margin-bottom: var(--mission-hud-stack, 1.1rem);}
      .mission-hud__wrap {backdrop-filter: blur(14px); background: var(--mission-hud-bg, rgba(13,17,23,0.85)); border: 1px solid var(--mission-hud-border, rgba(96,165,250,0.28));
        border-radius: var(--mission-hud-radius, 18px); padding: 12px 18px; display: grid; grid-template-columns: auto 1fr auto; gap: var(--mission-hud-gap, 16px); align-items: center;
        box-shadow: var(--mission-hud-shadow, 0 18px 38px rgba(8,18,36,0.32));}
      .mission-hud__logo {display: flex; align-items: center; gap: 10px; font-weight: 700; letter-spacing: .02em; color: var(--ink, #e2e8f0); font-size: 1.05rem;}
      .mission-hud__actions {display: flex; gap: 10px; align-items: center; flex-wrap: wrap;}
      .mission-hud__action {display: inline-flex; align-items: center; gap: 8px; padding: 8px 14px; border-radius: 14px; border: 1px solid rgba(148,163,184,0.22);
        background: rgba(15,23,42,0.35); color: rgba(226,232,240,0.78); font-size: 0.82rem; text-decoration: none; transition: all .3s ease; white-space: nowrap;}
      .mission-hud__action:hover {border-color: rgba(96,165,250,0.45); color: #f8fafc;}
      .mission-hud__action.is-current {border-color: rgba(96,165,250,0.78); background: linear-gradient(135deg, rgba(59,130,246,0.38), rgba(14,165,233,0.22)); color: #f8fafc;}
      .mission-hud__action--back {border-style: dashed; border-color: rgba(148,163,184,0.35); background: rgba(15,23,42,0.2); font-size: 0.78rem; opacity: 0.85;}
      .mission-hud__action--back:hover {opacity: 1; border-color: rgba(96,165,250,0.5);}
      .mission-hud__meta {display: flex; align-items: center; gap: 12px; flex-wrap: wrap;}
      .mission-hud__status {display: inline-flex; gap: 8px; flex-wrap: wrap; align-items: center;}
      .mission-hud__badge {display: inline-flex; align-items: center; gap: 6px; padding: 5px 12px; border-radius: 999px; font-size: 0.74rem; font-weight: 600; letter-spacing: 0.01em;
        text-transform: uppercase; border: 1px solid transparent; background: rgba(148,163,184,0.18); color: rgba(226,232,240,0.88);}
      .mission-hud__badge--success {background: rgba(34,197,94,0.18); border-color: rgba(34,197,94,0.35); color: rgba(187,247,208,0.95);}
      .mission-hud__badge--warning {background: rgba(250,204,21,0.16); border-color: rgba(250,204,21,0.32); color: rgba(254,240,138,0.95);}
      .mission-hud__badge--danger {background: rgba(248,113,113,0.16); border-color: rgba(248,113,113,0.36); color: rgba(254,202,202,0.95);}
      .mission-hud__target {display: grid; gap: 4px; font-size: 0.78rem; color: rgba(226,232,240,0.82);}
      .mission-hud__target-label {font-weight: 600; letter-spacing: 0.01em;}
      .mission-hud__target-limits {display: inline-flex; gap: 6px; flex-wrap: wrap;}
      .mission-hud__target-pill {display: inline-flex; align-items: center; padding: 4px 10px; border-radius: 999px; border: 1px solid rgba(148,163,184,0.28); background: rgba(15,23,42,0.42);
        font-size: 0.72rem; color: rgba(226,232,240,0.8);}
      .mission-hud__target-pill.is-empty {opacity: 0.6; font-style: italic;}
      .mission-hud__settings {padding: 7px 12px; border-radius: 12px; border: 1px solid rgba(96,165,250,0.32); background: rgba(15,23,42,0.55);
        color: #e2e8f0; font-size: 0.78rem; text-decoration: none; transition: all .3s ease; display: inline-flex; align-items:center; gap: 6px;}
      .mission-hud__settings:hover {background: rgba(59,130,246,0.18);}
      .mission-hud__details {margin: 0; position: relative;}
      .mission-hud__details summary {list-style: none; cursor: pointer; display: inline-flex; align-items: center; gap: 6px; padding: 6px 10px; border-radius: 12px;
        border: 1px solid rgba(96,165,250,0.22); background: rgba(15,23,42,0.42); color: rgba(226,232,240,0.82); font-size: 0.78rem; transition: all .3s ease;}
      .mission-hud__details[open] summary {background: rgba(59,130,246,0.18); color: #f8fafc;}
      .mission-hud__details summary::-webkit-details-marker {display: none;}
      .mission-hud__details-content {position: absolute; right: 0; top: calc(100% + 10px); width: 260px; padding: 14px; border-radius: 14px; background: rgba(11,17,27,0.96);
        border: 1px solid rgba(59,130,246,0.32); box-shadow: 0 18px 48px rgba(8,18,36,0.35); display: grid; gap: 8px;}
      .mission-hud__details dl {margin: 0; display: grid; gap: 6px;}
      .mission-hud__details dt {font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.04em; color: rgba(148,163,184,0.78);}
      .mission-hud__details dd {margin: 0; font-size: 0.82rem; color: rgba(226,232,240,0.9);}
      @media (max-width: 900px) {
        .mission-hud__wrap {grid-template-columns: 1fr; gap: 14px;}
        .mission-hud__logo {justify-content: space-between;}
        .mission-hud__actions {flex-wrap: wrap;}
        .mission-hud__meta {justify-content: space-between;}
        .mission-hud__details-content {position: static; width: 100%;}
      }
      .mission-breadcrumbs {margin-bottom: 1rem; font-size: 0.78rem; display: flex; gap: 6px; align-items: center; color: rgba(226,232,240,0.76);}
      .mission-breadcrumbs a {color: rgba(148,197,255,0.85); text-decoration: none; font-weight: 600;}
      .mission-breadcrumbs span {opacity: 0.65;}
    </style>
    """



def render_mission_hud() -> None:
    """Render the Mission HUD (logo, steps, model state and CTAs)."""

    st.markdown(_hud_css(), unsafe_allow_html=True)

    if st.session_state.get(_HUD_STATE_KEY):
        st.session_state[_HUD_STATE_KEY] += 1
    else:
        st.session_state[_HUD_STATE_KEY] = 1

    if not MISSION_STEPS:
        return

    metadata = _model_metadata()
    active_step = _step_from_key(st.session_state.get("mission_active_step"))

    if active_step:
        current_index = MISSION_STEPS.index(active_step)
    else:
        current_index = 0
        active_step = MISSION_STEPS[0]

    core_keys = ("inventory", "target", "generator", "results")
    core_steps = [step for step in MISSION_STEPS if step.key in core_keys]

    active_core_key: str | None
    if active_step.key in core_keys:
        active_core_key = active_step.key
    else:
        inventory_index = next((i for i, step in enumerate(MISSION_STEPS) if step.key == "inventory"), 0)
        results_index = next((i for i, step in enumerate(MISSION_STEPS) if step.key == "results"), len(MISSION_STEPS) - 1)
        if current_index <= inventory_index:
            active_core_key = "inventory"
        elif current_index >= results_index:
            active_core_key = "results"
        else:
            active_core_key = None

    visible_steps: list[tuple[MissionStep, bool]] = [
        (step, step.key == active_core_key)
        for step in core_steps
    ]

    if active_core_key is None and core_steps:
        closest = min(
            core_steps,
            key=lambda step: abs(MISSION_STEPS.index(step) - current_index),
        )
        visible_steps = [
            (step, step is closest)
            for step in core_steps
        ]

    quick_back_steps: list[MissionStep] = []
    if current_index > 0:
        quick_back_steps.append(MISSION_STEPS[current_index - 1])
    if current_index > 1:
        quick_back_steps.append(MISSION_STEPS[current_index - 2])

    actions_markup = []
    for step in reversed(quick_back_steps):
        actions_markup.append(
            """
            <a class="mission-hud__action mission-hud__action--back" href="{url}" title="Volver a {title}">
              ‹ {label}
            </a>
            """.format(
                url=_page_url(step.page),
                title=step.description,
                label=step.label,
            ).strip()
        )

    for step, is_current in visible_steps:
        step_idx = MISSION_STEPS.index(step) + 1
        label = f"<span>{step.icon}</span><strong>{step_idx} · {step.label}</strong>"
        class_attr = "is-current" if is_current else ""
        actions_markup.append(
            f"<a class=\"mission-hud__action {class_attr}\" href=\"{_page_url(step.page)}\" title=\"{step.description}\">"
            f"{label}"
            "</a>"
        )

    target_state = st.session_state.get("target", {})
    has_target = bool(target_state)

    def _format_limit(value: object) -> str:
        if isinstance(value, (int, float)):
            if isinstance(value, float) and value.is_integer():
                value = int(value)
            return f"{value:,}".replace(",", "·")
        return str(value)

    scenario_label = target_state.get("scenario") or "Sin escenario asignado"
    limit_fields = (
        ("max_water_l", "Agua", "L"),
        ("max_energy_kwh", "Energía", "kWh"),
        ("max_crew_min", "Crew", "min"),
    )
    limit_pills = []
    for field, label, unit in limit_fields:
        value = target_state.get(field)
        if value is None:
            continue
        limit_pills.append(
            f"<span class='mission-hud__target-pill'>{label} ≤ {_format_limit(value)} {unit}</span>"
        )
    if target_state.get("crew_time_low"):
        limit_pills.append("<span class='mission-hud__target-pill'>Crew-time low</span>")
    if not limit_pills:
        limit_pills.append("<span class='mission-hud__target-pill is-empty'>Sin límites definidos</span>")

    def _badge_markup(data: dict[str, str]) -> str:
        tone = data.get("tone", "warning")
        label = data.get("label", "Estado")
        return f"<span class='mission-hud__badge mission-hud__badge--{tone}'>{label}</span>"

    details_rows = """
        <dl>
          <div><dt>Estado</dt><dd>{status}</dd></div>
          <div><dt>Modelo</dt><dd>{model}</dd></div>
          <div><dt>Entrenado en</dt><dd>{trained_label}</dd></div>
          <div><dt>Actualizado</dt><dd>{trained_at}</dd></div>
          <div><dt>Incertidumbre</dt><dd>{uncertainty}</dd></div>
        </dl>
    """.format(
        status=metadata["status"],
        model=metadata["model_name"],
        trained_label=metadata["trained_label"],
        trained_at=metadata["trained_at"],
        uncertainty=metadata["uncertainty"],
    )

    settings_cta = ""
    if has_target:
        settings_cta = (
            f"<a class=\"mission-hud__settings\" href=\"{_page_url('2_Target_Designer')}\" "
            "title=\"Editar target\">🎯 Editar target</a>"
        )

    st.markdown(
        f"""
        <div class="mission-hud mission-hud--compact">
            <div class="mission-hud__wrap">
            <div class="mission-hud__logo">🛰️ Mission HUD <span style="opacity:0.6;font-weight:500;">Rex-AI</span></div>
            <div class="mission-hud__actions">{''.join(actions_markup)}</div>
            <div class="mission-hud__meta">
              <div class="mission-hud__status">
                {_badge_markup(metadata["status_badge"])}
                {_badge_markup(metadata["uncertainty_badge"])}
              </div>
              <div class="mission-hud__target">
                <span class="mission-hud__target-label">Escenario · {scenario_label}</span>
                <div class="mission-hud__target-limits">{''.join(limit_pills)}</div>
              </div>
              <details class="mission-hud__details">
                <summary>Detalles del modelo</summary>
                <div class="mission-hud__details-content">
                  {details_rows}
                </div>
              </details>
              {settings_cta}
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

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

