import streamlit as st
from app.modules.io import load_waste_df, load_process_df
from app.modules.process_planner import choose_process
from app.modules.generator import generate_candidates

st.set_page_config(page_title="Generador", page_icon="⚙️", layout="wide")
st.title("3) Generar recetas y procesos")

target = st.session_state.get("target", None)
if not target:
    st.warning("Definí primero el objetivo en **2) Target Designer**.")
    st.stop()

waste_df = load_waste_df()
proc_df  = load_process_df()
filtered_proc = choose_process(target["name"], proc_df)

n = st.slider("Número de candidatos", 3, 12, 6)
if st.button("🚀 Generar opciones", type="primary"):
    cands = generate_candidates(waste_df, filtered_proc, target, n=n)
    st.session_state["candidates"] = cands

cands = st.session_state.get("candidates", [])
for i,c in enumerate(cands):
    with st.expander(f"Opción {i+1} — Score {c['score']} — Proceso {c['process_id']} {c['process_name']}", expanded=(i==0)):
        st.write("**Materiales:**", ", ".join(c["materials"]))
        st.write("**Pesos:**", c["weights"])
        p = c["props"]
        st.write(f"**Predicción** → Rigidez: {p.rigidity:.2f} | Estanqueidad: {p.tightness:.2f} | Masa final: {p.mass_final_kg:.2f} kg")
        st.write(f"**Recursos** → Energía: {p.energy_kwh:.2f} kWh | Agua: {p.water_l:.2f} L | Crew: {p.crew_min:.0f} min")
        if st.button(f"✅ Seleccionar Opción {i+1}", key=f"pick_{i}"):
            st.session_state["selected"] = c
            st.success("Opción seleccionada. Pasá a **4) Resultados**.")
