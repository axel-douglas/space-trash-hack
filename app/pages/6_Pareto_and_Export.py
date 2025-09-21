# --- path guard para Streamlit Cloud ---
import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]  # carpeta raíz del repo
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# ---------------------------------------

import math
import random
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from app.modules.explain import compare_table
from app.modules.analytics import pareto_front
from app.modules.exporters import candidate_to_json, candidate_to_csv

# ⚠️ Debe ser la PRIMERA llamada de Streamlit en la página
st.set_page_config(page_title="Pareto & Export", page_icon="📤", layout="wide")
st.title("6) Pareto & Export")

# --------------------------------------------------------------------
# Data in-session
# --------------------------------------------------------------------
cands = st.session_state.get("candidates", [])
target = st.session_state.get("target", None)
state_sel = st.session_state.get("selected", None)

if not cands or not target:
    st.warning("Generá opciones en **3) Generador** primero.")
    st.stop()

# Tabla de comparación consolidada (Score + recursos + etiquetas)
df = compare_table(cands, target, crew_time_low=target.get("crew_time_low", False))

# Índices de frontera de Pareto (minimizar energía/agua/crew y maximizar score)
front_idx = pareto_front(df)
df["Pareto"] = df.index.isin(front_idx)

# --------------------------------------------------------------------
# Bloque A — Frontera de Pareto
# --------------------------------------------------------------------
st.subheader("Frontera de Pareto (Energía vs Agua vs Crew)")

fig = px.scatter_3d(
    df, x="Energía (kWh)", y="Agua (L)", z="Crew (min)",
    color="Pareto", size="Score",
    hover_data=["Opción","Proceso","Materiales"]
)
st.plotly_chart(fig, use_container_width=True)

with st.expander("¿Qué estoy viendo aquí? (explicación simple)"):
    st.markdown("""
- **Cada punto** es una opción candidata (receta + proceso).  
- La **frontera de Pareto** son las opciones "eficientes": no existe otra que sea **mejor en todo a la vez**.  
- Elegimos dentro de esa frontera según prioridades de misión (por ejemplo, priorizar **menos tiempo de tripulación**).
    """)

# --------------------------------------------------------------------
# Bloque B — Predicciones de ensayo (con incertidumbre)
# --------------------------------------------------------------------
st.markdown("---")
st.subheader("Predicciones de ensayo (con incertidumbre)")

def _demo_seed_from_candidate(c):
    """
    Creamos una semilla estable a partir de atributos del candidato
    para que los intervalos sean reproducibles entre corridas de la demo.
    """
    base = hash(c["process_id"] + c["process_name"] + "|".join(c["materials"])) % (10**6)
    return base

def estimate_ci_for_candidate(c):
    """
    Estima media (score) e intervalo de confianza 95% para la demo.
    Lógica:
      - partimos del score calculado por el sistema,
      - el ancho del intervalo crece si la receta usa muchos materiales o
        si el proceso es más complejo (heurística simple).
    """
    mean = float(c["score"])
    k_materials = max(1, len(c.get("materials", [])))
    # Heurística de "complejidad" del proceso por longitud del nombre/id
    complexity = 1.0 + 0.02 * (len(c.get("process_name","")) + len(c.get("process_id","")))
    # Desvío estándar relativo (2% a 10% del score), acotado
    rel_std = min(0.10, max(0.02, 0.03 * math.log1p(k_materials) * complexity))
    # Semilla reproducible por candidato
    random.seed(_demo_seed_from_candidate(c))
    jitter = random.uniform(0.9, 1.1)
    std = rel_std * mean * jitter
    # IC 95% aproximado: mean ± 1.96*std
    low = max(0.0, mean - 1.96 * std)
    high = mean + 1.96 * std
    return mean, low, high

# Construimos un DataFrame con medias e IC para los candidatos vigentes
pred_rows = []
for i, c in enumerate(cands, start=1):
    m, lo, hi = estimate_ci_for_candidate(c)
    pred_rows.append({
        "Opción": f"Op{i}",
        "Proceso": f"{c['process_id']} {c['process_name']}",
        "Score (media)": round(m, 2),
        "CI 95% - inf": round(lo, 2),
        "CI 95% - sup": round(hi, 2),
    })
pred_df = pd.DataFrame(pred_rows)

# Gráfico tipo "error bar" (similar al de la referencia visual)
fig_pred = go.Figure()

fig_pred.add_trace(go.Scatter(
    x=list(range(1, len(pred_df)+1)),
    y=pred_df["Score (media)"],
    mode="markers",
    marker=dict(size=10),
    name="Predicción media",
    error_y=dict(
        type="data",
        symmetric=False,
        array=(pred_df["CI 95% - sup"] - pred_df["Score (media)"]),
        arrayminus=(pred_df["Score (media)"] - pred_df["CI 95% - inf"]),
        thickness=1.5,
        width=0
    ),
    text=pred_df["Proceso"],
    hovertemplate="Opción %{x}<br>Score %{y}<br>%{text}<extra></extra>"
))

fig_pred.update_layout(
    xaxis=dict(
        title="Candidato",
        tickmode="array",
        tickvals=list(range(1, len(pred_df)+1)),
        ticktext=pred_df["Opción"].tolist()
    ),
    yaxis=dict(title="Score (con IC 95%)"),
    margin=dict(l=10, r=10, t=10, b=10),
    height=420
)

st.plotly_chart(fig_pred, use_container_width=True)

colA, colB = st.columns([2,1])
with colA:
    st.caption("Tabla de predicciones con intervalos de confianza")
    st.dataframe(pred_df, use_container_width=True, hide_index=True)
with colB:
    st.markdown("#### ¿Para qué sirve?")
    st.markdown("""
- **Incertidumbre**: toda predicción tiene margen de error; lo mostramos explícitamente.  
- **Comparación honesta**: si dos opciones tienen medias similares pero una tiene **IC más estrecho**, es **más confiable**.  
- **Decisión operativa**: combiná esto con Pareto (recursos) para elegir la receta que **maximiza utilidad** y **minimiza riesgo**.
    """)

with st.expander("Notas técnicas (demo)"):
    st.markdown("""
- Estos intervalos de confianza están **estimados** para la demo con una heurística reproducible
  (varía con la cantidad de materiales y la “complejidad” del proceso).  
- En producción, se reemplaza por IC derivados del modelo real (ensembles/MC dropout/bootstrapping) o de la variabilidad de banco de pruebas.
    """)

# --------------------------------------------------------------------
# Bloque C — Tabla consolidada
# --------------------------------------------------------------------
st.markdown("---")
st.subheader("Tabla consolidada")
st.dataframe(df, use_container_width=True, hide_index=True)

# --------------------------------------------------------------------
# Bloque D — Export
# --------------------------------------------------------------------
st.markdown("---")
st.subheader("Exportar candidato seleccionado")

if not state_sel:
    st.info("Seleccioná una opción en **3) Generador** para habilitar export.")
else:
    selected = state_sel["data"]
    safety = state_sel["safety"]
    json_bytes = candidate_to_json(selected, target, safety)
    csv_bytes  = candidate_to_csv(selected)
    st.download_button("⬇️ Descargar JSON (plan completo)", data=json_bytes,
                       file_name="candidate_plan.json", mime="application/json")
    st.download_button("⬇️ Descargar CSV (resumen)", data=csv_bytes,
                       file_name="candidate_summary.csv", mime="text/csv")
