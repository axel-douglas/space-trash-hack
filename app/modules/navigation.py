"""Utility helpers for textual navigation components."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import streamlit as st


@dataclass(frozen=True)
class MissionStep:
    """Definition of a mission step used across the Rex-AI flow."""

    key: str
    label: str
    page: str
    description: str


MISSION_STEPS: tuple[MissionStep, ...] = (
    MissionStep("brief", "Brief", "Home", "Resumen y preparación"),
    MissionStep("inventory", "Inventario", "1_Inventory_Builder", "Normalizar residuos"),
    MissionStep("target", "Target", "2_Target_Designer", "Definir objetivos"),
    MissionStep("generator", "Generador", "3_Generator", "Recetas asistidas"),
    MissionStep("results", "Resultados", "4_Results_and_Tradeoffs", "Trade-offs y métricas"),
    MissionStep("compare", "Comparar", "5_Compare_and_Explain", "Explicabilidad"),
    MissionStep("export", "Export", "6_Pareto_and_Export", "Pareto y export"),
    MissionStep("playbooks", "Playbooks", "7_Scenario_Playbooks", "Escenarios"),
    MissionStep("feedback", "Feedback", "8_Feedback_and_Impact", "Impacto y retraining"),
    MissionStep("capacity", "Capacidad", "9_Capacity_Simulator", "Simulación"),
)

_STEP_LOOKUP = {step.key: step for step in MISSION_STEPS}


def get_step(step_key: str) -> MissionStep:
    """Return the mission step associated with ``step_key``.

    Raises:
        KeyError: If the provided key is not part of the mission flow.
    """

    try:
        return _STEP_LOOKUP[step_key]
    except KeyError as exc:  # pragma: no cover - tiny guardrail
        raise KeyError(f"Step '{step_key}' is not defined") from exc


def set_active_step(step_key: str) -> MissionStep:
    """Persist the active step in session state and return it."""

    step = get_step(step_key)
    st.session_state["mission_active_step"] = step.key
    return step


def breadcrumb_labels(step: MissionStep, extra: Sequence[str] | None = None) -> list[str]:
    """Compute the textual breadcrumb trail for ``step``.

    The resulting list always starts with ``Home`` and ends with the active step.
    Additional labels can be appended through ``extra``.
    """

    labels: list[str] = ["Home"]
    for candidate in MISSION_STEPS:
        labels.append(candidate.label)
        if candidate.key == step.key:
            break
    if extra:
        labels.extend(extra)
    return labels


def render_breadcrumbs(step: MissionStep, extra: Sequence[str] | None = None) -> list[str]:
    """Render a textual breadcrumb trail using ``st.caption``.

    The rendered value is also returned to simplify testing.
    """

    labels = breadcrumb_labels(step, extra)
    st.caption(" › ".join(labels))
    return labels


def format_stepper(step: MissionStep) -> str:
    """Return a short textual representation of the active step."""

    index = MISSION_STEPS.index(step) + 1
    total = len(MISSION_STEPS)
    return f"Paso {index} de {total} · {step.label}"


def render_stepper(step: MissionStep) -> str:
    """Render the textual stepper using ``st.caption`` and return the string."""

    summary = format_stepper(step)
    st.caption(summary)
    return summary


def iter_steps() -> Iterable[MissionStep]:
    """Yield the mission steps in their canonical order."""

    return iter(MISSION_STEPS)
