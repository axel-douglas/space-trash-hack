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
st.set_page_config(page_title="Comparar & Explicar", page_icon="🧪", layout="wide")

from app.modules.explain import compare_table, score_breakdown

st.title("5) Comparar candidatos y explicar decisiones")

cands = st.session_state.get("candidates", [])
target = st.session_state.get("target", None)
if not cands or not target:
    st.warning("Generá opciones en **3) Generador** primero.")
    st.stop()

df = compare_table(cands, target, crew_time_low=target.get("crew_time_low", False))
st.dataframe(df, use_container_width=True, hide_index=True)

pick = st.number_input("Ver desglose de la Opción #", min_value=1, max_value=len(cands), value=1, step=1)
c = cands[pick-1]
parts = score_breakdown(c["props"], target, crew_time_low=target.get("crew_time_low", False))

colA, colB = st.columns(2)
with colA:
    st.subheader(f"Desglose Score — Opción {pick}")
    st.bar_chart(parts.set_index("component")["contribution"])
with colB:
    st.subheader("Resumen del candidato")
    st.write(f"**Proceso:** {c['process_id']} {c['process_name']}")
    st.write(f"**Materiales:** {', '.join(c['materials'])}")
    st.write(f"**Pesos:** {c['weights']}")
    st.write(f"**Score:** {c['score']}")
    p=c["props"]
    st.write(f"**Predicción** → Rigidez {p.rigidity:.2f} / Estanqueidad {p.tightness:.2f}")
    st.write(f"**Recursos** → Energía {p.energy_kwh:.2f} kWh | Agua {p.water_l:.2f} L | Crew {p.crew_min:.0f} min")
