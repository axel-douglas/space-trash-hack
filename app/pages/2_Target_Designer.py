import sys
from pathlib import Path

if not __package__:
    repo_root = Path(__file__).resolve().parents[2]
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)

from app.bootstrap import ensure_streamlit_entrypoint

ensure_streamlit_entrypoint(__file__)

import streamlit as st

from app.modules.io import load_targets
from app.modules.navigation import render_breadcrumbs, set_active_step
from app.modules.target_limits import compute_target_limits
from app.modules.ui_blocks import (
    configure_page,
    initialise_frontend,
    render_brand_header,
)

# ⚠️ Debe ser la PRIMERA llamada de Streamlit en la página
configure_page(page_title="Objetivo", page_icon="🎯")
initialise_frontend()

current_step = set_active_step("target")

render_brand_header()

render_breadcrumbs(current_step)

st.title("2) Definir objetivo (TargetSpec)")

st.info(
    "Seleccioná el escenario que describe tu misión y usá los deslizadores para "
    "balancear desempeño y logística antes de pasar al generador:\n"
    "- **Residence Renovations** prioriza refuerzos estructurales y sellado estable "
    "  para módulos habitables.\n"
    "- **Cosmic Celebrations** busca montajes rápidos con bajo consumo de crew y "
    "  recursos.\n"
    "- **Daring Discoveries** habilita prototipos de laboratorio donde la precisión "
    "  del sellado y la energía disponible son críticas."
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
    "Residence Renovations": (
        "rehabilitar y ampliar espacios habitables con énfasis en estabilidad"
        " estructural"
    ),
    "Cosmic Celebrations": (
        "montar instalaciones efímeras donde manda la agilidad y el consumo"
        " acotado de recursos"
    ),
    "Daring Discoveries": (
        "respaldar expediciones científicas con prototipos precisos y soporte"
        " instrumental"
    ),
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
    "\n".join(
        [
            "**Resumen de misión**",
            f"- Escenario elegido: **{scenario}** ({scenario_text}).",
            f"- Rigidez objetivo: `{rigidity:.2f}` (controla la resistencia del producto).",
            f"- Estanqueidad objetivo: `{tightness:.2f}` (evita fugas y pérdida de presión).",
            f"- Recursos máximos: `{max_water:.2f}` L de agua y `{max_energy:.2f}` kWh disponibles.",
            f"- Tiempo humano asignado: `{max_crew:.0f}` minutos para operadores ({crew_focus_text}).",
        ]
    )
)

st.success(
    "Objetivo listo. Pasá al paso **3 · Generador asistido** para crear recetas"
    " compatibles con estas restricciones."
)
