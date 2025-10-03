from pathlib import Path
import sys

if __package__ in {None, ""}:
    project_root = Path(__file__).resolve().parents[1]
    project_root_str = str(project_root)
    if project_root_str not in sys.path:
        sys.path.insert(0, project_root_str)

from app.bootstrap import ensure_streamlit_entrypoint

_PROJECT_ROOT = ensure_streamlit_entrypoint(__file__)

import streamlit as st

from app.modules.io import load_targets
from app.modules.navigation import render_breadcrumbs, set_active_step
from app.modules.target_limits import compute_target_limits
from app.modules.ui_blocks import initialise_frontend, load_theme, render_brand_header

# ⚠️ Debe ser la PRIMERA llamada de Streamlit en la página
st.set_page_config(page_title="Objetivo", page_icon="🎯", layout="wide")
initialise_frontend()

current_step = set_active_step("target")

load_theme()

render_brand_header()

render_breadcrumbs(current_step)

st.title("2) Definir objetivo (TargetSpec)")

presets = load_targets()
if not presets:
    st.error("No se encontraron presets de objetivos. Verifica `data/targets_presets.json`.")
    st.stop()

scenario_options = (
    "Residence Renovations",
    "Cosmic Celebrations",
    "Daring Discoveries",
)

preset_names = [preset["name"] for preset in presets if "name" in preset]
if not preset_names:
    st.error("Los presets cargados no contienen nombres válidos.")
    st.stop()

stored_target = st.session_state.get("target", {})
default_name = stored_target.get("name", preset_names[0])
default_index = preset_names.index(default_name) if default_name in preset_names else 0

selected_name = st.selectbox("Preset objetivo", preset_names, index=default_index)
selected_preset = next(p for p in presets if p.get("name") == selected_name)

slider_limits = compute_target_limits(presets)

default_scenario = scenario_options[0] if scenario_options else ""
scenario = stored_target.get("scenario", default_scenario)
if scenario not in scenario_options:
    scenario = default_scenario

scenario = st.selectbox("Escenario del reto", scenario_options, index=scenario_options.index(scenario))

crew_time_low = st.checkbox(
    "Priorizar procesos de bajo tiempo de tripulación",
    value=stored_target.get("crew_time_low", False),
)

rigidity_bounds = slider_limits["rigidity"]
rigidity_default = float(stored_target.get("rigidity", selected_preset.get("rigidity", 0.0)))
rigidity = st.slider(
    "Rigidez deseada",
    rigidity_bounds["min"],
    rigidity_bounds["max"],
    min(max(rigidity_default, rigidity_bounds["min"]), rigidity_bounds["max"]),
    rigidity_bounds["step"],
    help=rigidity_bounds["help"],
)

tightness_bounds = slider_limits["tightness"]
tightness_default = float(stored_target.get("tightness", selected_preset.get("tightness", 0.0)))
tightness = st.slider(
    "Estanqueidad deseada",
    tightness_bounds["min"],
    tightness_bounds["max"],
    min(max(tightness_default, tightness_bounds["min"]), tightness_bounds["max"]),
    tightness_bounds["step"],
    help=tightness_bounds["help"],
)

water_bounds = slider_limits["max_water_l"]
water_default = float(stored_target.get("max_water_l", selected_preset.get("max_water_l", 0.0)))
max_water = st.slider(
    "Agua máxima (L)",
    water_bounds["min"],
    water_bounds["max"],
    min(max(water_default, water_bounds["min"]), water_bounds["max"]),
    water_bounds["step"],
    help=water_bounds["help"],
)

energy_bounds = slider_limits["max_energy_kwh"]
energy_default = float(
    stored_target.get("max_energy_kwh", selected_preset.get("max_energy_kwh", 0.0))
)
max_energy = st.slider(
    "Energía máxima (kWh)",
    energy_bounds["min"],
    energy_bounds["max"],
    min(max(energy_default, energy_bounds["min"]), energy_bounds["max"]),
    energy_bounds["step"],
    help=energy_bounds["help"],
)

crew_bounds = slider_limits["max_crew_min"]
crew_default = float(stored_target.get("max_crew_min", selected_preset.get("max_crew_min", crew_bounds["min"])))
max_crew = st.number_input(
    "Tiempo máximo de tripulación (min)",
    min_value=float(crew_bounds["min"]),
    max_value=float(crew_bounds["max"]),
    value=float(min(max(crew_default, crew_bounds["min"]), crew_bounds["max"])),
    step=float(crew_bounds["step"]),
    help=crew_bounds["help"],
)

target = {
    **selected_preset,
    "name": selected_name,
    "scenario": scenario,
    "crew_time_low": crew_time_low,
    "rigidity": rigidity,
    "tightness": tightness,
    "max_water_l": float(max_water),
    "max_energy_kwh": float(max_energy),
    "max_crew_min": float(max_crew),
}

st.session_state["target"] = target

summary_cols = st.columns(3)
summary_cols[0].metric("Rigidez", f"{rigidity:.2f}")
summary_cols[1].metric("Estanqueidad", f"{tightness:.2f}")
summary_cols[2].metric("Crew (min)", f"{max_crew:.0f}")

st.caption("Límites de recursos establecidos")
limits_table = {
    "Recurso": ["Agua (L)", "Energía (kWh)"],
    "Máximo": [f"{max_water:.2f}", f"{max_energy:.2f}"],
}
st.table(limits_table)

st.success("Objetivo listo. Abrí la página **3) Generador** para obtener recetas.")
