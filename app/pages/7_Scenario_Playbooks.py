import sys
from pathlib import Path

if not __package__:
    repo_root = Path(__file__).resolve().parents[2]
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)

from app.bootstrap import ensure_streamlit_entrypoint

ensure_streamlit_entrypoint(__file__)

__doc__ = """Simplified scenario playbooks with actionable summaries."""

from dataclasses import asdict
from typing import Iterable

import pandas as pd
import streamlit as st

from app.modules.navigation import render_breadcrumbs, set_active_step
from app.modules.scenarios import PLAYBOOKS, ExampleRecipe
from app.modules.ui_blocks import (
    configure_page,
    initialise_frontend,
    layout_stack,
    render_brand_header,
    render_nasa_badge,
)

configure_page(page_title="Scenario Playbooks", page_icon="📚")
initialise_frontend()

current_step = set_active_step("playbooks")

render_brand_header()

render_breadcrumbs(current_step)

render_nasa_badge()

FEATURED_PLAYBOOKS: tuple[str, ...] = ("Residence Renovations", "Daring Discoveries")


def _ordered_scenarios(options: Iterable[str]) -> list[str]:
    favourites = [name for name in FEATURED_PLAYBOOKS if name in options]
    remainder = [name for name in options if name not in favourites]
    return favourites + remainder


target = st.session_state.get("target")
state_sel = st.session_state.get("selected")
selected_candidate = state_sel.get("data") if isinstance(state_sel, dict) else None
props = selected_candidate.get("props") if isinstance(selected_candidate, dict) else None

if not target:
    st.info(
        "Necesitás un objetivo activo para recomendar procedimientos. Configuralo"
        " en **2 · Target Designer** y volvé a esta pantalla."
    )
    st.stop()

with layout_stack():
    st.title("📚 Scenario Playbooks")
    st.caption(
        "Accedé a guías paso a paso calibradas para cada escenario. Cada playbook"
        " incluye filtros sugeridos, métricas y checklist editable."
    )

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

recipe: ExampleRecipe = playbook.example_recipe

with st.container():
    st.markdown("### 🧪 Receta destacada")
    recipe_columns = st.columns((3, 2))
    with recipe_columns[0]:
        st.markdown(f"**Producto objetivo:** {playbook.product_label}")
        st.caption(playbook.product_end_use)
        st.markdown(f"**Proceso:** `{recipe.process_id}`")
        st.markdown(recipe.batch_notes)
        st.markdown("**Mezcla sugerida**")
        for component in recipe.mix:
            if component.role:
                st.markdown(
                    f"- **{component.material}** — {component.quantity} _(Rol: {component.role})_"
                )
            else:
                st.markdown(f"- **{component.material}** — {component.quantity}")

    with recipe_columns[1]:
        st.markdown("**Metadatos operativos**")
        for label, value in playbook.metadata.items():
            st.markdown(f"- {label}: **{value}**")

        filters_payload = playbook.generator_filters or recipe.generator_filters or {}
        if filters_payload:
            st.markdown("**Filtros sugeridos**")
            st.json(filters_payload, expanded=False)
        else:
            st.caption("Este playbook no define filtros automáticos para el generador.")

        if st.button("Aplicar receta en generador"):
            st.session_state["_playbook_generator_filters"] = {
                "scenario": playbook.name,
                "filters": filters_payload,
                "recipe": {
                    "process_id": recipe.process_id,
                    "mix": [asdict(component) for component in recipe.mix],
                    "product_label": playbook.product_label,
                },
            }
            st.session_state["mission_active_step"] = "generator"
            st.switch_page("pages/3_Generator.py")

st.subheader("Recursos del candidato activo")
st.caption("Mirá el consumo estimado de la opción elegida antes de lanzar el playbook.")
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
st.caption("Cada paso combina tareas operativas y notas técnicas para guiar a la tripulación.")
steps_df = pd.DataFrame(
    {
        "Paso": list(range(1, len(playbook.steps) + 1)),
        "Actividad": [step.title for step in playbook.steps],
        "Detalle": [step.detail for step in playbook.steps],
    }
)
st.dataframe(steps_df, use_container_width=True, hide_index=True)

st.subheader("Checklist editable")
st.caption("Personalizá la lista de control y compartila con el equipo en segundos.")
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
