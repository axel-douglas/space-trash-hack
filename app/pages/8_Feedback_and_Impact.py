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
st.set_page_config(page_title="Feedback & Impact", page_icon="📝", layout="wide")

from datetime import datetime
from app.modules.impact import (
    ImpactEntry, FeedbackEntry, append_impact, append_feedback,
    load_impact_df, load_feedback_df, summarize_impact
)

st.title("8) Feedback del astronauta & Impacto acumulado")

# --- Bloque A: Registrar impacto del candidato seleccionado ---
state_sel = st.session_state.get("selected", None)
target = st.session_state.get("target", None)

st.subheader("A) Registrar impacto de la corrida")
if not state_sel or not target:
    st.info("Seleccioná y registra una opción desde **3) Generador** / **4) Resultados** antes de guardar impacto.")
else:
    sel = state_sel["data"]; p = sel["props"]
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"**Escenario:** {target.get('scenario','-')}")
        st.write(f"**Target:** {target.get('name','-')}")
        st.write(f"**Proceso:** {sel['process_id']} {sel['process_name']}")
        st.write(f"**Materiales:** {', '.join(sel['materials'])}")

    with col2:
        st.write(f"**Masa final (kg):** {p.mass_final_kg:.2f}")
        st.write(f"**Energía (kWh):** {p.energy_kwh:.2f}")
        st.write(f"**Agua (L):** {p.water_l:.2f}")
        st.write(f"**Crew (min):** {p.crew_min:.0f}")
        st.write(f"**Score:** {sel['score']:.2f}")

    if st.button("💾 Guardar impacto de esta corrida", type="primary"):
        entry = ImpactEntry(
            ts_iso=datetime.utcnow().isoformat(),
            scenario=target.get("scenario","-"),
            target_name=target.get("name","-"),
            materials="|".join(sel["materials"]),
            weights="|".join(map(str, sel["weights"])),
            process_id=sel["process_id"],
            process_name=sel["process_name"],
            mass_final_kg=p.mass_final_kg,
            energy_kwh=p.energy_kwh, water_l=p.water_l, crew_min=p.crew_min,
            score=sel["score"]
        )
        append_impact(entry)
        st.success("Impacto registrado.")

st.markdown("---")

# --- Bloque B: Feedback humano en el loop ---
st.subheader("B) Feedback del astronauta")
with st.form("feedback_form"):
    astronaut = st.text_input("Nombre (opcional para registro)", "")
    option_idx = st.number_input("Opción elegida #", min_value=1, step=1, value=1)
    rigidity_ok = st.toggle("La rigidez fue suficiente", value=True)
    ease_ok = st.toggle("El proceso fue fácil de ejecutar", value=True)
    issues = st.text_input("Problemas observados (bordes, olor, slip, etc.)", "")
    notes = st.text_area("Notas libres", "")
    submitted = st.form_submit_button("Enviar feedback")
    if submitted:
        entry = FeedbackEntry(
            ts_iso=datetime.utcnow().isoformat(),
            astronaut=astronaut or "anon",
            scenario=target.get("scenario","-") if target else "-",
            target_name=target.get("name","-") if target else "-",
            option_idx=int(option_idx),
            rigidity_ok=bool(rigidity_ok),
            ease_ok=bool(ease_ok),
            issues=issues,
            notes=notes
        )
        append_feedback(entry)
        st.success("Feedback guardado. Los próximos cálculos podrán ajustar pesos/penalizaciones con estas señales.")

st.markdown("---")

# --- Bloque C: Panel de impacto acumulado ---
st.subheader("C) Panel de impacto acumulado")
idf = load_impact_df()
sumy = summarize_impact(idf)
c1,c2,c3,c4,c5 = st.columns(5)
c1.metric("Corridas", sumy["runs"])
c2.metric("Kg valorizados", f"{sumy['kg']:.2f} kg")
c3.metric("Energía total", f"{sumy['kwh']:.2f} kWh")
c4.metric("Agua total", f"{sumy['water_l']:.2f} L")
c5.metric("Crew total", f"{sumy['crew_min']:.0f} min")

st.caption("Debajo: detalle por corrida")
st.dataframe(idf, use_container_width=True)
