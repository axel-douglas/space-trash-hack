# --- path guard para Streamlit Cloud ---
import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]  # carpeta raíz del repo
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# ---------------------------------------

import streamlit as st
import plotly.graph_objects as go
from app.modules.explain import score_breakdown
from app.modules.ui_blocks import pill

st.set_page_config(page_title="Resultados", page_icon="📊", layout="wide")
st.title("4) Resultados y trade-offs")

state_sel = st.session_state.get("selected", None)
if not state_sel:
    st.warning("Seleccioná una receta en **3) Generador**.")
    st.stop()

sel = state_sel["data"]
badge = state_sel["safety"]
target = st.session_state.get("target")
p = sel["props"]

col1, col2, col3 = st.columns(3)
col1.metric("Rigidez", f"{p.rigidity:.2f}")
col2.metric("Estanqueidad", f"{p.tightness:.2f}")
col3.metric("Score", f"{sel['score']:.2f}")

st.markdown("**Seguridad**")
if badge["level"]=="Riesgo":
    pill("Riesgo","risk"); st.warning(badge["detail"])
else:
    pill("OK","ok"); st.success(badge["detail"])

st.subheader("Desglose del Score (explicabilidad)")
parts = score_breakdown(p, target, crew_time_low=target.get("crew_time_low", False))
st.bar_chart(parts.set_index("component")["contribution"])

st.subheader("Sankey (residuos → proceso → producto)")
labels = sel["materials"] + [sel["process_name"], "Producto"]
src = list(range(len(sel["materials"])))
tgt = [len(sel["materials"])]*len(sel["materials"])
val = [round(w*100,1) for w in sel["weights"]]
src += [len(sel["materials"])]
tgt += [len(sel["materials"])+1]
val += [100.0]

fig = go.Figure(data=[go.Sankey(
    node=dict(label=labels, pad=20, thickness=18),
    link=dict(source=src, target=tgt, value=val)
)])
st.plotly_chart(fig, use_container_width=True)

st.subheader("Checklist de fabricación")
st.markdown(f"""
1. Triturar/Preparar materiales seleccionados (**{', '.join(sel['materials'])}**).  
2. Ejecutar **{sel['process_name']}** según parámetros estándar del hábitat.  
3. Enfriar y verificar bordes/ajuste.  
4. Registrar feedback rápido (rigidez percibida, facilidad de uso, problemas).
""")

st.info(f"Recursos estimados — Energía: {p.energy_kwh:.2f} kWh • Agua: {p.water_l:.2f} L • Tiempo tripulación: {p.crew_min:.0f} min")
