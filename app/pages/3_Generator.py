import streamlit as st
from app.modules.io import load_waste_df, load_process_df
from app.modules.process_planner import choose_process
from app.modules.generator import generate_candidates
from app.modules.safety import check_safety, safety_badge
from app.modules.ui_blocks import pill

st.set_page_config(page_title="Generador", page_icon="⚙️", layout="wide")
st.title("3) Generar recetas y procesos")

target = st.session_state.get("target", None)
if not target:
    st.warning("Definí primero el objetivo en **2) Target Designer**.")
    st.stop()

waste_df = load_waste_df()
proc_df  = load_process_df()
filtered_proc = choose_process(
    target["name"], proc_df,
    scenario=target.get("scenario"),
    crew_time_low=target.get("crew_time_low", False)
)

n = st.slider("Número de candidatos", 3, 12, 6)
if st.button("🚀 Generar opciones", type="primary"):
    cands = generate_candidates(
        waste_df, filtered_proc, target, n=n,
        crew_time_low=target.get("crew_time_low", False)
    )
    st.session_state["candidates"] = cands

cands = st.session_state.get("candidates", [])
for i,c in enumerate(cands):
    flags = check_safety(c["materials"], c["process_name"], c["process_id"])
    badge = safety_badge(flags)
    header = f"Opción {i+1} — Score {c['score']} — Proceso {c['process_id']} {c['process_name']}"

    with st.expander(header, expanded=(i==0)):
        st.write("**Materiales:**", ", ".join(c["materials"]))
        st.write("**Pesos:**", c["weights"])
        p = c["props"]
        st.write(f"**Predicción** → Rigidez: {p.rigidity:.2f} | Estanqueidad: {p.tightness:.2f} | Masa final: {p.mass_final_kg:.2f} kg")
        st.write(f"**Recursos** → Energía: {p.energy_kwh:.2f} kWh | Agua: {p.water_l:.2f} L | Crew: {p.crew_min:.0f} min")

        st.markdown("**Seguridad**")
        if badge["level"]=="Riesgo":
            pill("Riesgo", "risk"); st.warning(badge["detail"])
        else:
            pill("OK", "ok"); st.success(badge["detail"])

        if st.button(f"✅ Seleccionar Opción {i+1}", key=f"pick_{i}"):
            st.session_state["selected"] = {"data": c, "safety": badge}
            st.success("Opción seleccionada. Pasá a **4) Resultados**, **5) Comparar & Explicar** o **6) Pareto & Export**.")
