# --- path guard para Streamlit Cloud ---
import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]  # carpeta raíz del repo
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# ---------------------------------------

import streamlit as st
import plotly.express as px
from app.modules.explain import compare_table
from app.modules.analytics import pareto_front
from app.modules.exporters import candidate_to_json, candidate_to_csv

st.set_page_config(page_title="Pareto & Export", page_icon="📤", layout="wide")
st.title("6) Pareto & Export")

cands = st.session_state.get("candidates", [])
target = st.session_state.get("target", None)
state_sel = st.session_state.get("selected", None)

if not cands or not target:
    st.warning("Generá opciones en **3) Generador** primero.")
    st.stop()

df = compare_table(cands, target, crew_time_low=target.get("crew_time_low", False))
front_idx = pareto_front(df)
df["Pareto"] = df.index.isin(front_idx)

st.subheader("Frontera de Pareto (Energía vs Agua vs Crew)")
fig = px.scatter_3d(
    df, x="Energía (kWh)", y="Agua (L)", z="Crew (min)",
    color="Pareto", size="Score", hover_data=["Opción","Proceso","Materiales"]
)
st.plotly_chart(fig, use_container_width=True)

st.subheader("Tabla consolidada")
st.dataframe(df, use_container_width=True, hide_index=True)

st.markdown("---")
st.subheader("Exportar candidato seleccionado")
if not state_sel:
    st.info("Seleccioná una opción en **3) Generador** para habilitar export.")
else:
    selected = state_sel["data"]
    safety = state_sel["safety"]
    json_bytes = candidate_to_json(selected, target, safety)
    csv_bytes  = candidate_to_csv(selected)
    st.download_button("⬇️ Descargar JSON (plan completo)", data=json_bytes, file_name="candidate_plan.json", mime="application/json")
    st.download_button("⬇️ Descargar CSV (resumen)", data=csv_bytes, file_name="candidate_summary.csv", mime="text/csv")
