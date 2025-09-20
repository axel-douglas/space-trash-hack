# --- path guard universal (funciona en Home.py y en pages/*) ---
import sys, pathlib
_here = pathlib.Path(__file__).resolve()
p = _here.parent
while p.name != "app" and p.parent != p:
    p = p.parent
repo_root = p.parent if p.name == "app" else _here.parent  # fallback
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))
# ----------------------------------------------------------------

import streamlit as st

# ⚠️ Primero
st.set_page_config(page_title="Capacity Simulator", page_icon="🧮", layout="wide")

from app.modules.capacity import LineConfig, simulate

st.title("9) Simulador de capacidad productiva (por turnos)")
st.caption("Modela producción acumulada con una línea/equipo del hábitat, considerando lotes por turno, kg por lote y recursos.")

colA, colB = st.columns(2)
with colA:
    batches_per_shift = st.number_input("Lotes por turno", 1, 100, 3, 1)
    kg_per_batch = st.number_input("Kg por lote", 0.1, 50.0, 0.95, 0.05)
    energy_kwh_per_batch = st.number_input("kWh por lote", 0.01, 10.0, 1.2, 0.01)
with colB:
    water_l_per_batch = st.number_input("Agua (L) por lote", 0.0, 10.0, 0.1, 0.1)
    crew_min_per_batch = st.number_input("Crew (min) por lote", 1.0, 180.0, 25.0, 1.0)

st.markdown("---")
st.subheader("Horizonte")
shifts_per_sol = st.slider("Turnos por sol marciano", 1, 6, 2)
num_sols = st.slider("Soles simulados", 1, 120, 30)

if st.button("▶️ Simular"):
    cfg = LineConfig(
        batches_per_shift=int(batches_per_shift),
        kg_per_batch=float(kg_per_batch),
        energy_kwh_per_batch=float(energy_kwh_per_batch),
        water_l_per_batch=float(water_l_per_batch),
        crew_min_per_batch=float(crew_min_per_batch)
    )
    res = simulate(cfg, shifts_per_sol=int(shifts_per_sol), num_sols=int(num_sols))
    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("Lotes", res["batches"])
    c2.metric("Kg", f"{res['kg']:.2f}")
    c3.metric("kWh", f"{res['kwh']:.2f}")
    c4.metric("Agua (L)", f"{res['water_l']:.2f}")
    c5.metric("Crew (min)", f"{res['crew_min']:.0f}")

    st.info("Tip: combina este simulador con parámetros reales de tu proceso ganador (kg por lote y recursos por lote).")
