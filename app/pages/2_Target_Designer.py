import streamlit as st
from app.modules.io import load_targets

st.set_page_config(page_title="Objetivo", page_icon="🎯", layout="wide")
st.title("2) Definir objetivo (TargetSpec)")

presets = load_targets()
names = [p["name"] for p in presets]
colL, colR = st.columns([1,2])

with colL:
    choice = st.radio("¿Qué necesitás fabricar?", names, index=0)
    preset = next(p for p in presets if p["name"]==choice)

with colR:
    st.subheader("Prioridades y límites")
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
    "max_crew_min": max_c
}

st.success("Objetivo listo. Abrí la página **3) Generador** para obtener recetas.")
