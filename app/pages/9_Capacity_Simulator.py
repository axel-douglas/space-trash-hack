from app.bootstrap import ensure_streamlit_entrypoint

ensure_streamlit_entrypoint(__file__)

__doc__ = """Lightweight capacity simulator driven by shared helpers."""

import pandas as pd
import streamlit as st

from app.modules.capacity import LineConfig, simulate
from app.modules.navigation import render_breadcrumbs, set_active_step
from app.modules.ui_blocks import (
    initialise_frontend,
    layout_stack,
    render_brand_header,
)

st.set_page_config(page_title="Capacity Simulator", page_icon="🧮", layout="wide")
initialise_frontend()

current_step = set_active_step("capacity")

render_brand_header()

render_breadcrumbs(current_step)

target = st.session_state.get("target")
selected_state = st.session_state.get("selected")
selected_candidate = selected_state.get("data") if isinstance(selected_state, dict) else None
props = selected_candidate.get("props") if isinstance(selected_candidate, dict) else None

with layout_stack():
    st.title("🧮 Capacity Simulator")
    st.caption("Evalúa la producción diaria frente a límites de recursos y downtime estimado.")

col_left, col_right = st.columns(2)
with col_left:
    shifts_per_sol = st.slider("Turnos por sol", 1, 6, 2)
    num_sols = st.slider("Horizonte (soles)", 1, 120, 30)
    batches_per_shift = st.number_input("Lotes por turno", min_value=1, value=3)
    efficiency = st.slider("Eficiencia de la línea", 0.5, 1.2, 0.95, step=0.01)
with col_right:
    downtime_pct = st.slider("Downtime estimado (%)", 0, 40, 5)
    energy_limit = st.number_input("Límite kWh por sol", min_value=0.0, value=float(target.get("max_energy_kwh", 250.0)) if isinstance(target, dict) else 250.0)
    water_limit = st.number_input("Límite de agua (L) por sol", min_value=0.0, value=float(target.get("max_water_l", 30.0)) if isinstance(target, dict) else 30.0)
    crew_limit = st.number_input("Límite de crew (min) por sol", min_value=0.0, value=float(target.get("max_crew_min", 600.0)) if isinstance(target, dict) else 600.0)

default_mass = float(getattr(props, "mass_final_kg", 1.0) or 1.0)
default_energy = float(getattr(props, "energy_kwh", 1.2) or 1.2)
default_water = float(getattr(props, "water_l", 0.1) or 0.1)
default_crew = float(getattr(props, "crew_min", 25.0) or 25.0)

resource_cols = st.columns(4)
kg_per_batch = resource_cols[0].number_input("Kg por lote", min_value=0.01, value=default_mass, step=0.05)
energy_per_batch = resource_cols[1].number_input("kWh por lote", min_value=0.0, value=default_energy, step=0.01)
water_per_batch = resource_cols[2].number_input("Agua (L) por lote", min_value=0.0, value=default_water, step=0.01)
crew_per_batch = resource_cols[3].number_input("Crew (min) por lote", min_value=0.0, value=default_crew, step=1.0)

simulate_clicked = st.button("Simular", type="primary")

if simulate_clicked:
    config = LineConfig(
        batches_per_shift=int(batches_per_shift),
        kg_per_batch=float(kg_per_batch) * efficiency,
        energy_kwh_per_batch=float(energy_per_batch),
        water_l_per_batch=float(water_per_batch),
        crew_min_per_batch=float(crew_per_batch),
    )
    aggregate = simulate(config, shifts_per_sol=int(shifts_per_sol), num_sols=int(num_sols))

    batches_per_day = config.batches_per_shift * int(shifts_per_sol)
    base_kg = config.kg_per_batch * batches_per_day
    base_kwh = config.energy_kwh_per_batch * batches_per_day
    base_water = config.water_l_per_batch * batches_per_day
    base_crew = config.crew_min_per_batch * batches_per_day

    downtime_factor = 1.0 - (float(downtime_pct) / 100.0)
    per_day_rows = []
    cumulative_kg = 0.0
    for day in range(1, int(num_sols) + 1):
        kg_day = base_kg * downtime_factor
        kwh_day = base_kwh * downtime_factor
        water_day = base_water * downtime_factor
        crew_day = base_crew * downtime_factor
        ratios = []
        if energy_limit:
            ratios.append(energy_limit / max(kwh_day, 1e-6))
        if water_limit:
            ratios.append(water_limit / max(water_day, 1e-6))
        if crew_limit:
            ratios.append(crew_limit / max(crew_day, 1e-6))
        limit_factor = min(1.0, *ratios) if ratios else 1.0
        kg_day *= limit_factor
        kwh_day *= limit_factor
        water_day *= limit_factor
        crew_day *= limit_factor
        cumulative_kg += kg_day
        per_day_rows.append(
            {
                "Sol": day,
                "Kg": kg_day,
                "kWh": kwh_day,
                "Agua (L)": water_day,
                "Crew (min)": crew_day,
                "Utilización vs límites": limit_factor,
                "Downtime aplicado": downtime_pct,
            }
        )

    per_day_df = pd.DataFrame(per_day_rows)
    total_row = {
        "Sol": "Total",
        "Kg": per_day_df["Kg"].sum(),
        "kWh": per_day_df["kWh"].sum(),
        "Agua (L)": per_day_df["Agua (L)"].sum(),
        "Crew (min)": per_day_df["Crew (min)"].sum(),
        "Utilización vs límites": per_day_df["Utilización vs límites"].mean(),
        "Downtime aplicado": downtime_pct,
    }

    metric_summary = st.columns(4)
    metric_summary[0].metric("Lotes totales", aggregate.get("batches", 0))
    metric_summary[1].metric("Kg totales", f"{total_row['Kg']:.2f}")
    metric_summary[2].metric("kWh totales", f"{total_row['kWh']:.2f}")
    metric_summary[3].metric("Crew total (min)", f"{total_row['Crew (min)']:.0f}")

    st.subheader("Producción por sol")
    result_table = pd.concat([per_day_df, pd.DataFrame([total_row])], ignore_index=True)
    st.dataframe(result_table, use_container_width=True, hide_index=True)
else:
    st.caption("Configurá los parámetros y presioná **Simular** para ver la proyección por sol.")
