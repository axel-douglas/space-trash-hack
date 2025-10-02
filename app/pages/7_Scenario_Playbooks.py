"""Simplified scenario playbooks with actionable summaries."""

import _bootstrap  # noqa: F401

from typing import Iterable

import pandas as pd
import streamlit as st

from app.modules.navigation import render_breadcrumbs, set_active_step
from app.modules.scenarios import PLAYBOOKS
from app.modules.ui_blocks import layout_stack, load_theme


st.set_page_config(page_title="Scenario Playbooks", page_icon="📚", layout="wide")

current_step = set_active_step("playbooks")

load_theme()

render_breadcrumbs(current_step)

FEATURED_PLAYBOOKS: tuple[str, ...] = ("Residence Renovations", "Daring Discoveries")
GENERATOR_FILTER_PRESETS: dict[str, dict[str, bool]] = {
    "Residence Renovations": {
        "showroom_only_safe": True,
        "showroom_limit_energy": True,
        "showroom_limit_water": True,
        "showroom_limit_crew": True,
    },
    "Daring Discoveries": {
        "showroom_only_safe": False,
        "showroom_limit_energy": True,
        "showroom_limit_water": False,
        "showroom_limit_crew": True,
    },
}


def _ordered_scenarios(options: Iterable[str]) -> list[str]:
    favourites = [name for name in FEATURED_PLAYBOOKS if name in options]
    remainder = [name for name in options if name not in favourites]
    return favourites + remainder


target = st.session_state.get("target")
state_sel = st.session_state.get("selected")
selected_candidate = state_sel.get("data") if isinstance(state_sel, dict) else None
props = selected_candidate.get("props") if isinstance(selected_candidate, dict) else None

if not target:
    st.info("Definí un objetivo en **2 · Target Designer** para habilitar los playbooks.")
    st.stop()

with layout_stack():
    st.title("📚 Scenario Playbooks")
    st.caption("Procedimientos listos para ejecutar según el escenario operativo seleccionado.")

scenarios = list(PLAYBOOKS.keys())
if not scenarios:
    st.warning("No hay playbooks configurados en `app/modules/scenarios.py`.")
    st.stop()

scenario_default = str(target.get("scenario") or scenarios[0])
ordered_scenarios = _ordered_scenarios(scenarios)
if scenario_default not in ordered_scenarios:
    scenario_default = ordered_scenarios[0]

scenario = st.selectbox(
    "Escenario de misión",
    ordered_scenarios,
    index=ordered_scenarios.index(scenario_default),
)

playbook = PLAYBOOKS.get(scenario)
if not playbook:
    st.warning("No se encontró el playbook seleccionado.")
    st.stop()

summary_columns = st.columns(2)
with summary_columns[0]:
    st.subheader(playbook.name)
    st.write(playbook.summary)

with summary_columns[1]:
    st.subheader("Estado de la misión")
    st.markdown(
        """
        - Target activo: **{target_name}**
        - Límite de energía: **{energy} kWh**
        - Límite de agua: **{water} L**
        - Límite de crew: **{crew} min**
        """.format(
            target_name=target.get("name", "—"),
            energy=target.get("max_energy_kwh", "—"),
            water=target.get("max_water_l", "—"),
            crew=target.get("max_crew_min", "—"),
        )
    )

st.subheader("Filtros recomendados")
filters_payload = GENERATOR_FILTER_PRESETS.get(playbook.name, {})
if filters_payload:
    st.json(filters_payload, expanded=False)
else:
    st.caption("Este playbook no define filtros automáticos para el generador.")

if st.button("Abrir generador con estos filtros"):
    st.session_state["_playbook_generator_filters"] = {"scenario": playbook.name, "filters": filters_payload}
    st.session_state["mission_active_step"] = "generator"
    st.switch_page("pages/3_Generator.py")

st.subheader("Recursos del candidato activo")
metric_columns = st.columns(4)
metrics = [
    ("Masa final (kg)", getattr(props, "mass_final_kg", None)),
    ("Energía por corrida (kWh)", getattr(props, "energy_kwh", None)),
    ("Agua por corrida (L)", getattr(props, "water_l", None)),
    ("Crew por corrida (min)", getattr(props, "crew_min", None)),
]
for column, (label, value) in zip(metric_columns, metrics):
    text = "—"
    if value is not None:
        text = f"{float(value):.2f}" if "Crew" not in label else f"{float(value):.0f}"
    column.metric(label, text)
if not props:
    st.caption("Seleccioná un candidato en **3 · Generador** para ver métricas reales.")

st.subheader("Pasos del playbook")
steps_df = pd.DataFrame(
    {
        "Paso": list(range(1, len(playbook.steps) + 1)),
        "Actividad": [step.title for step in playbook.steps],
        "Detalle": [step.detail for step in playbook.steps],
    }
)
st.dataframe(steps_df, use_container_width=True, hide_index=True)

st.subheader("Checklist editable")
default_lines = [
    "- Verificar disponibilidad de materiales",
    "- Preparar equipo de proceso",
    "- Registrar lote, hora y operador",
    "- Cargar feedback en 8) Feedback & Impact",
]
checklist_value = st.text_area("Checklist", "\n".join(default_lines), height=160)
st.download_button(
    "Descargar checklist (.txt)",
    data=checklist_value.encode("utf-8"),
    file_name="scenario_checklist.txt",
    mime="text/plain",
)
