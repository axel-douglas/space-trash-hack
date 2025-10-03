from app.bootstrap import ensure_streamlit_entrypoint

ensure_streamlit_entrypoint(__file__)

import streamlit as st

from app.modules.io import load_targets
from app.modules.navigation import render_breadcrumbs, set_active_step
from app.modules.target_limits import compute_target_limits
from app.modules.ui_blocks import initialise_frontend, render_brand_header

# ⚠️ Debe ser la PRIMERA llamada de Streamlit en la página
st.set_page_config(page_title="Objetivo", page_icon="🎯", layout="wide")
initialise_frontend()

current_step = set_active_step("target")

render_brand_header()

render_breadcrumbs(current_step)

st.title("2) Definir objetivo (TargetSpec)")

st.info(
    "En esta misión podés elegir entre tres escenarios: **Residence Renovations** "
    "(rehabilitar hábitats existentes), **Cosmic Celebrations** (montajes rápidos para"
    " eventos especiales) y **Daring Discoveries** (expediciones de laboratorio)."
    " Configurá la rigidez y la estanqueidad para alinear la ingeniería del producto "
    "con cada reto, y fijá límites de agua, energía y minutos de tripulación para "
    "asegurarte de que la receta resultante respete los recursos disponibles en la "
    "estación."
)

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

scenario_descriptions = {
    "Residence Renovations": "rehabilitar y ampliar espacios existentes a largo plazo",
    "Cosmic Celebrations": "crear montajes efímeros y visuales para eventos orbitales",
    "Daring Discoveries": "apoyar misiones científicas con prototipos ajustados",
}

scenario = st.selectbox(
    "Escenario del reto",
    scenario_options,
    index=scenario_options.index(scenario),
    help="Elegí el contexto operativo principal para adaptar los límites a su propósito específico.",
)

crew_time_low = st.checkbox(
    "Priorizar procesos de bajo tiempo de tripulación",
    value=stored_target.get("crew_time_low", False),
)

rigidity_bounds = slider_limits["rigidity"]
rigidity_default = float(stored_target.get("rigidity", selected_preset.get("rigidity", 0.0)))
rigidity_help = (
    f"{rigidity_bounds['help']} Ejemplo: cascos modulares para `Residence Renovations`"
    " requieren más rigidez que decoraciones temporales."
)
rigidity = st.slider(
    "Rigidez deseada (resistencia estructural)",
    rigidity_bounds["min"],
    rigidity_bounds["max"],
    min(max(rigidity_default, rigidity_bounds["min"]), rigidity_bounds["max"]),
    rigidity_bounds["step"],
    help=rigidity_help,
)

tightness_bounds = slider_limits["tightness"]
tightness_default = float(stored_target.get("tightness", selected_preset.get("tightness", 0.0)))
tightness_help = (
    f"{tightness_bounds['help']} Ejemplo: cápsulas para `Daring Discoveries`"
    " necesitan sellos herméticos, mientras que instalaciones festivas pueden"
    " tolerar más fugas."
)
tightness = st.slider(
    "Estanqueidad deseada (control de fugas)",
    tightness_bounds["min"],
    tightness_bounds["max"],
    min(max(tightness_default, tightness_bounds["min"]), tightness_bounds["max"]),
    tightness_bounds["step"],
    help=tightness_help,
)

water_bounds = slider_limits["max_water_l"]
water_default = float(stored_target.get("max_water_l", selected_preset.get("max_water_l", 0.0)))
water_help = (
    f"{water_bounds['help']} Ejemplo: limpiar módulos habitables exige más agua que"
    " ajustes decorativos en `Cosmic Celebrations`."
)
max_water = st.slider(
    "Agua máxima permitida (L)",
    water_bounds["min"],
    water_bounds["max"],
    min(max(water_default, water_bounds["min"]), water_bounds["max"]),
    water_bounds["step"],
    help=water_help,
)

energy_bounds = slider_limits["max_energy_kwh"]
energy_default = float(
    stored_target.get("max_energy_kwh", selected_preset.get("max_energy_kwh", 0.0))
)
energy_help = (
    f"{energy_bounds['help']} Ejemplo: experimentos de `Daring Discoveries`"
    " consumen más energía que iluminación ambiente en `Cosmic Celebrations`."
)
max_energy = st.slider(
    "Energía máxima asignada (kWh)",
    energy_bounds["min"],
    energy_bounds["max"],
    min(max(energy_default, energy_bounds["min"]), energy_bounds["max"]),
    energy_bounds["step"],
    help=energy_help,
)

crew_bounds = slider_limits["max_crew_min"]
crew_default = float(stored_target.get("max_crew_min", selected_preset.get("max_crew_min", crew_bounds["min"])))
crew_help = (
    f"{crew_bounds['help']} Ejemplo: `Crew` significa minutos humanos disponibles;"
    " tareas automatizadas en `Residence Renovations` reducen este valor."
)
max_crew = st.number_input(
    "Tiempo máximo de tripulación humana (min)",
    min_value=float(crew_bounds["min"]),
    max_value=float(crew_bounds["max"]),
    value=float(min(max(crew_default, crew_bounds["min"]), crew_bounds["max"])),
    step=float(crew_bounds["step"]),
    help=crew_help,
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
summary_cols[2].metric("Tiempo de tripulación (min)", f"{max_crew:.0f}")

st.caption("Límites de recursos establecidos")
limits_table = {
    "Recurso": ["Agua (L)", "Energía (kWh)"],
    "Máximo": [f"{max_water:.2f}", f"{max_energy:.2f}"],
}
st.table(limits_table)

crew_focus_text = "priorizar procesos rápidos" if crew_time_low else "permitir procesos con más supervisión"
scenario_text = scenario_descriptions.get(scenario, scenario.lower())
st.markdown(
    "**Resumen narrativo:** Configuraste el objetivo para "
    f"**{scenario}**, enfocado en {scenario_text}. La receta deberá respetar una "
    f"rigidez de {rigidity:.2f} (protege la estructura), una estanqueidad de {tightness:.2f} "
    "(evita fugas críticas) y usar hasta "
    f"{max_water:.2f} L de agua y {max_energy:.2f} kWh de energía. También indicaste "
    f"{crew_focus_text}, con un máximo de {max_crew:.0f} minutos humanos disponibles."
)

st.success("Objetivo listo. Abrí la página **3) Generador** para obtener recetas.")
