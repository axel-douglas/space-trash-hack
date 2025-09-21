# --- path guard para Streamlit Cloud ---
import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]  # carpeta raíz del repo
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# ---------------------------------------

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from app.modules.explain import compare_table
from app.modules.analytics import pareto_front
from app.modules.exporters import candidate_to_json, candidate_to_csv

# ⚠️ Debe ser la PRIMERA llamada
st.set_page_config(page_title="Pareto & Export", page_icon="📤", layout="wide")

# ======== estado requerido ========
cands     = st.session_state.get("candidates", [])
target    = st.session_state.get("target", None)
state_sel = st.session_state.get("selected", None)

if not cands or not target:
    st.warning("Generá opciones en **3) Generador** primero.")
    st.stop()

# ======== estilo visual ========
st.markdown("""
<style>
.kpi {border:1px solid rgba(128,128,128,0.25); border-radius:14px; padding:14px; margin-bottom:12px;}
.kpi h3 {margin:0 0 6px 0; font-size:0.95rem; opacity:0.8;}
.kpi .v {font-size:1.6rem; font-weight:700; letter-spacing:0.2px;}
.kpi .hint {font-size:0.85rem; opacity:0.75;}
.section-note {font-size:0.92rem; opacity:0.85;}
.pill {display:inline-block; padding:3px 10px; border-radius:999px; font-weight:600; font-size:0.80rem; border:1px solid rgba(128,128,128,0.25);}
.pill.ok {background:#e8f7ee; color:#136c3a; border-color:#b3e2c4;}
.pill.info {background:#e7f1ff; color:#174ea6; border-color:#c6dcff;}
.pill.warn {background:#fff3cd; color:#8a6d1d; border-color:#ffe69b;}
</style>
""", unsafe_allow_html=True)

st.markdown("# 📤 Pareto & Export")
st.caption("Elegí con cabeza de misión: explorá la **frontera de Pareto** (agua/energía/crew) y exportá el plan elegido para ejecución y trazabilidad.")

# ======== tabla base y saneo robusto ========
df_raw = compare_table(cands, target, crew_time_low=target.get("crew_time_low", False)).copy()

# Normalizamos nombres esperados si hiciera falta
rename_map = {}
for col in df_raw.columns:
    low = col.lower().strip()
    if low in ["energia (kwh)", "energía (kwh)", "energia kwh"]:
        rename_map[col] = "Energía (kWh)"
    if low in ["agua (l)", "agua l", "agua"]:
        rename_map[col] = "Agua (L)"
    if low in ["crew (min)", "crew min", "crew"]:
        rename_map[col] = "Crew (min)"
    if low in ["masa (kg)", "masa kg", "kg"]:
        rename_map[col] = "Masa (kg)"
    if low == "materiales":
        rename_map[col] = "Materiales"
    if low == "proceso":
        rename_map[col] = "Proceso"
    if low == "opción" or low == "opcion":
        rename_map[col] = "Opción"
    if low == "score":
        rename_map[col] = "Score"
df_raw.rename(columns=rename_map, inplace=True)

# Asegurar tipos y columnas mínimas
for k in ["Opción","Score","Proceso","Materiales","Energía (kWh)","Agua (L)","Crew (min)"]:
    if k not in df_raw.columns:
        df_raw[k] = np.nan

# “Materiales” como string (por si viniera como lista/objeto)
df_raw["Materiales"] = df_raw["Materiales"].apply(lambda v: ", ".join(v) if isinstance(v, (list, tuple)) else (str(v) if pd.notna(v) else ""))

# Coerción numérica segura
for k in ["Score","Energía (kWh)","Agua (L)","Crew (min)","Masa (kg)"]:
    if k in df_raw.columns:
        df_raw[k] = pd.to_numeric(df_raw[k], errors="coerce")

# Drop de filas totalmente inválidas para el 3D (evita ValueError de Plotly)
df_plot = df_raw.dropna(subset=["Energía (kWh)","Agua (L)","Crew (min)","Score"]).copy()

# ======== KPIs ========
colA, colB, colC, colD = st.columns(4)
with colA:
    st.markdown(f'<div class="kpi"><h3>Opciones válidas</h3><div class="v">{len(df_plot)}</div><div class="hint">Con métricas completas</div></div>', unsafe_allow_html=True)
with colB:
    st.markdown(f'<div class="kpi"><h3>Score máximo</h3><div class="v">{df_plot["Score"].max():.2f}</div><div class="hint">Tope actual</div></div>', unsafe_allow_html=True)
with colC:
    st.markdown(f'<div class="kpi"><h3>Mín. Agua</h3><div class="v">{df_plot["Agua (L)"].min():.2f} L</div><div class="hint">Menor consumo</div></div>', unsafe_allow_html=True)
with colD:
    st.markdown(f'<div class="kpi"><h3>Mín. Energía</h3><div class="v">{df_plot["Energía (kWh)"].min():.2f} kWh</div><div class="hint">Menor consumo</div></div>', unsafe_allow_html=True)

# ======== What-If de restricciones (no cambia sesión; solo filtro visual) ========
st.markdown("### 🎛️ What-If (filtro visual de límites)")
f1, f2, f3 = st.columns(3)
with f1:
    lim_e = st.number_input("Límite de Energía (kWh)", min_value=0.0, value=float(target["max_energy_kwh"]), step=0.1)
with f2:
    lim_w = st.number_input("Límite de Agua (L)", min_value=0.0, value=float(target["max_water_l"]), step=0.1)
with f3:
    lim_c = st.number_input("Límite de Crew (min)", min_value=0.0, value=float(target["max_crew_min"]), step=1.0)

mask_ok = (df_plot["Energía (kWh)"] <= lim_e) & (df_plot["Agua (L)"] <= lim_w) & (df_plot["Crew (min)"] <= lim_c)
df_view = df_plot.copy()
df_view["Dentro_límites"] = np.where(mask_ok, "Dentro de límites", "Excede límites")

# ======== Frontera de Pareto (con fallback seguro) ========
try:
    front_idx = pareto_front(df_plot)  # debe devolver índices del df_plot
    front_mask = df_plot.index.isin(front_idx)
except Exception:
    # fallback simple: los 5 mejores scores
    front_mask = df_plot["Score"].rank(ascending=False, method="first") <= 5

df_view["Pareto"] = np.where(front_mask, "Pareto", "No Pareto")

# ======== Scatter 3D Pareto ========
st.markdown("### 🌌 Explorador 3D (Energía vs Agua vs Crew)")
# Tamaño positivo seguro
df_view["ScorePos"] = np.clip(df_view["Score"].fillna(0.0), 0.01, None)

# Evita ValueError por valores no válidos
usable = df_view.dropna(subset=["Energía (kWh)","Agua (L)","Crew (min)","ScorePos"]).copy()
if usable.empty:
    st.info("No hay suficientes datos válidos para dibujar el 3D. Revisá que existan candidatos con métricas completas.")
else:
    fig3d = px.scatter_3d(
        usable,
        x="Energía (kWh)", y="Agua (L)", z="Crew (min)",
        color="Pareto", size="ScorePos",
        hover_data=["Opción","Proceso","Materiales","Dentro_límites","Score"]
    )
    fig3d.update_layout(margin=dict(l=10,r=10,t=10,b=10), height=520)
    st.plotly_chart(fig3d, use_container_width=True)

    # Slice de sólo Pareto (tabla)
    st.markdown("#### Tabla — Frontera de Pareto")
    table_pareto = usable[usable["Pareto"] == "Pareto"].sort_values("Score", ascending=False)
    st.dataframe(table_pareto[["Opción","Score","Proceso","Materiales","Energía (kWh)","Agua (L)","Crew (min)"]], use_container_width=True, hide_index=True)

st.markdown('<span class="pill info">Tip</span> En el 3D buscá **abajo/izquierda** (menos agua/energía) y **adelante** (menos crew). Si estás en modo *Crew-time Low*, priorizá el eje **Crew**.', unsafe_allow_html=True)

st.markdown("---")

# ======== Centro de Exportación ========
st.markdown("## 📦 Export Center")
st.markdown('<p class="section-note">Llevate lo necesario para documentar la decisión y ejecutar el proceso en el hábitat.</p>', unsafe_allow_html=True)

colL, colR = st.columns([1.1, 1.0])

with colL:
    st.markdown("### Exportar seleccionados/tabla")
    # Export CSV del universo visible y de la frontera Pareto
    csv_all = df_view.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ CSV — Tabla completa (vista actual)", data=csv_all, file_name="pareto_view.csv", mime="text/csv")

    if not usable.empty:
        csv_front = table_pareto.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ CSV — Frontera Pareto", data=csv_front, file_name="pareto_frontier.csv", mime="text/csv")

    st.markdown('<span class="pill ok">Sugerencia</span> Esta tabla es ideal para **revisión técnica** y para adjuntar en el **log de misión**.', unsafe_allow_html=True)

with colR:
    st.markdown("### Exportar candidato elegido")
    if not state_sel:
        st.info("Seleccioná una opción en **3) Generador** (o en **4/5**) para habilitar export de plan.")
    else:
        selected = state_sel["data"]
        safety   = state_sel["safety"]
        try:
            json_bytes = candidate_to_json(selected, target, safety)
            st.download_button("⬇️ JSON — Plan completo", data=json_bytes, file_name="candidate_plan.json", mime="application/json")
        except Exception as e:
            st.warning(f"No se pudo construir JSON: {e}")
        try:
            csv_bytes  = candidate_to_csv(selected)
            st.download_button("⬇️ CSV — Resumen candidato", data=csv_bytes, file_name="candidate_summary.csv", mime="text/csv")
        except Exception as e:
            st.warning(f"No se pudo construir CSV: {e}")

# ======== Ayuda didáctica ========
st.markdown("---")
h1, h2 = st.columns(2)
with h1:
    pop = st.popover("¿Qué es la frontera de Pareto?")
    with pop:
        st.markdown("""
- Es el **conjunto de opciones que nadie puede mejorar en un eje sin empeorar otro**.
- Acá los ejes son **energía**, **agua** y **minutos de tripulación**.
- Elegir en la frontera = decisión **eficiente** dada tu realidad operativa en Marte.
""")
with h2:
    pop2 = st.popover("¿Cómo exporto bien para ejecución?")
    with pop2:
        st.markdown("""
- **JSON** del candidato: incluye objetivo, proceso, materiales y predicciones (para reproducibilidad).
- **CSV**: rápido para compartir y para pegar en hojas de cálculo del turno.
- **Pareto CSV**: adjuntalo a la justificación de ingeniería (por qué esta opción y no otra).
""")
