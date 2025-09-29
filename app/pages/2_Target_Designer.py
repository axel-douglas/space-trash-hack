import _bootstrap  # noqa: F401

import streamlit as st

# ⚠️ Debe ser la PRIMERA llamada de Streamlit en la página
st.set_page_config(page_title="Objetivo", page_icon="🎯", layout="wide")

from app.modules.io import load_targets
from app.modules.ui_blocks import section

st.title("2) Definir objetivo (TargetSpec)")

presets = load_targets()
names = [p["name"] for p in presets] if presets else []
if not names:
    st.error("No se encontraron presets de objetivos. Verifica `data/targets_presets.json`.")
    st.stop()

colL, colR = st.columns([1,2])

with colL:
    choice = st.radio("¿Qué necesitás fabricar?", names, index=0)

    scenario = st.selectbox(
        "Escenario del reto",
        ["Residence Renovations","Cosmic Celebrations","Daring Discoveries"],
        index=0
    )

    crew_low = st.toggle(
        "Modo Crew-time Low (priorizar poco tiempo de tripulación)",
        value=False,
        help="Aumenta el peso del tiempo de tripulación en el score y filtra procesos más cortos."
    )

with colR:
    preset = next(p for p in presets if p["name"]==choice)
    section("Prioridades y límites", "Afiná requisitos del target y recursos máximos.")
    rigidity = st.slider("Rigidez deseada", 0.0, 1.0, float(preset["rigidity"]), 0.05)
    tight    = st.slider("Estanqueidad deseada", 0.0, 1.0, float(preset["tightness"]), 0.05)
    max_w    = st.slider("Agua máx. (L)", 0.0, 3.0, float(preset["max_water_l"]), 0.1)
    max_e    = st.slider("Energía máx. (kWh)", 0.0, 3.0, float(preset["max_energy_kwh"]), 0.1)
    max_c    = st.slider("Tiempo tripulación máx. (min)", 5, 60, int(preset["max_crew_min"]), 1)

st.session_state["target"] = {
    "name": choice,
    "rigidity": rigidity,
    "tightness": tight,
    "max_water_l": max_w,
    "max_energy_kwh": max_e,
    "max_crew_min": max_c,
    "scenario": scenario,
    "crew_time_low": crew_low
}

st.success("Objetivo listo. Abrí la página **3) Generador** para obtener recetas.")
