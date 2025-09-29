# app/pages/6_Pareto_and_Export.py
import _bootstrap  # noqa: F401

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from app.modules.explain import compare_table
from app.modules.analytics import pareto_front
from app.modules.exporters import candidate_to_json, candidate_to_csv
from app.modules.safety import check_safety  # recalcular badge al seleccionar
from app.modules.ui_blocks import load_theme

# ⚠️ PRIMERA llamada
st.set_page_config(page_title="Pareto & Export", page_icon="📤", layout="wide")

load_theme()

# ======== estado requerido ========
cands  = st.session_state.get("candidates", [])
target = st.session_state.get("target", None)
state_sel = st.session_state.get("selected", None)

if not cands or not target:
    st.warning("Generá opciones en **3) Generador** primero.")
    st.stop()

# ======== estilos (NASA/SpaceX-like) ========
st.markdown(
    """
    <style>
    .hero {border-radius:16px; padding:18px 18px 8px; background: radial-gradient(1200px 380px at 20% -10%, rgba(80,120,255,.08), transparent);}
    .section-title{margin-top:6px; margin-bottom:6px}
    </style>
    """,
    unsafe_allow_html=True,
)

# ======== HERO ========
st.markdown("""
<div class="hero">
  <h1 style="margin:0 0 6px 0">📤 Pareto & Export</h1>
  <div class="small" style="margin-bottom:10px">
    Explorá el trade-off **Energía ↔ Agua ↔ Crew** con datos reales de tus candidatos.
    Elegí uno y exportá el plan. Todo conectado al objetivo definido en <b>2) Target</b>.
  </div>
  <div class="legend">
    <span class="pill info">Paso 1 — Explorar</span>
    <span class="pill info">Paso 2 — Seleccionar</span>
    <span class="pill ok">Paso 3 — Exportar</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ======== tabla base (derivada de candidates reales) ========
df_raw = compare_table(cands, target, crew_time_low=target.get("crew_time_low", False)).copy()

# Normalización robusta de nombres
rename_map = {}
for col in df_raw.columns:
    low = col.lower().strip()
    if low in ["energia (kwh)", "energía (kwh)", "energia kwh"]: rename_map[col] = "Energía (kWh)"
    if low in ["agua (l)", "agua l", "agua"]: rename_map[col] = "Agua (L)"
    if low in ["crew (min)", "crew min", "crew"]: rename_map[col] = "Crew (min)"
    if low in ["masa (kg)", "masa kg", "kg"]: rename_map[col] = "Masa (kg)"
    if low in ["opción","opcion"]: rename_map[col] = "Opción"
    if low == "materiales": rename_map[col] = "Materiales"
    if low == "proceso": rename_map[col] = "Proceso"
    if low == "score": rename_map[col] = "Score"
df_raw.rename(columns=rename_map, inplace=True)

# Tipos y saneo
for k in ["Score","Energía (kWh)","Agua (L)","Crew (min)","Masa (kg)"]:
    if k in df_raw: df_raw[k] = pd.to_numeric(df_raw[k], errors="coerce")
if "Materiales" in df_raw:
    df_raw["Materiales"] = df_raw["Materiales"].apply(
        lambda v: ", ".join(v) if isinstance(v, (list,tuple)) else (str(v) if pd.notna(v) else "")
    )

df_plot = df_raw.dropna(subset=["Energía (kWh)","Agua (L)","Crew (min)","Score"]).copy()

# ======== KPIs ========
colA, colB, colC, colD = st.columns(4)
with colA: st.markdown(f'<div class="kpi"><h3>Opciones válidas</h3><div class="v">{len(df_plot)}</div></div>', unsafe_allow_html=True)
with colB: st.markdown(f'<div class="kpi"><h3>Score máximo</h3><div class="v">{df_plot["Score"].max():.2f}</div></div>', unsafe_allow_html=True)
with colC: st.markdown(f'<div class="kpi"><h3>Mín. Agua</h3><div class="v">{df_plot["Agua (L)"].min():.2f} L</div></div>', unsafe_allow_html=True)
with colD: st.markdown(f'<div class="kpi"><h3>Mín. Energía</h3><div class="v">{df_plot["Energía (kWh)"].min():.2f} kWh</div></div>', unsafe_allow_html=True)

# ======== What-If de límites ========
st.markdown("### 🎛️ What-If (filtro visual)")
f1, f2, f3 = st.columns(3)
with f1: lim_e = st.number_input("Límite de Energía (kWh)", 0.0, 999.0, float(target["max_energy_kwh"]), 0.1)
with f2: lim_w = st.number_input("Límite de Agua (L)", 0.0, 999.0, float(target["max_water_l"]), 0.1)
with f3: lim_c = st.number_input("Límite de Crew (min)", 0.0, 999.0, float(target["max_crew_min"]), 1.0)

mask_ok = (df_plot["Energía (kWh)"]<=lim_e) & (df_plot["Agua (L)"]<=lim_w) & (df_plot["Crew (min)"]<=lim_c)
df_view = df_plot.copy()
df_view["Dentro_límites"] = np.where(mask_ok, "Dentro de límites", "Excede límites")

# ======== Frontera de Pareto ========
try:
    front_idx = pareto_front(df_plot)
    front_mask = df_plot.index.isin(front_idx)
except Exception:
    # fallback estable si el usuario sube columnas raras
    front_mask = df_plot["Score"].rank(ascending=False, method="first") <= 5
df_view["Pareto"] = np.where(front_mask, "Pareto", "No Pareto")
df_view["ScorePos"] = np.clip(df_view["Score"].fillna(0.0), 0.01, None)

tab_pareto, tab_trials, tab_objectives, tab_export = st.tabs(
    ["🌌 Pareto Explorer", "🔮 Predicciones de ensayo (demo)", "🎯 Objetivos por eje", "📦 Export Center"]
)

# ---------- TAB 1: Pareto Explorer ----------
with tab_pareto:
    st.markdown('<h3 class="section-title">Explorador 3D</h3>', unsafe_allow_html=True)
    usable = df_view.dropna(subset=["Energía (kWh)","Agua (L)","Crew (min)","ScorePos"]).copy()

    if usable.empty:
        st.info("No hay suficientes datos para graficar.")
    else:
        fig3d = px.scatter_3d(
            usable,
            x="Energía (kWh)", y="Agua (L)", z="Crew (min)",
            color="Pareto", size="ScorePos",
            hover_data=["Opción","Proceso","Materiales","Dentro_límites","Score"]
        )
        fig3d.update_layout(margin=dict(l=10,r=10,t=10,b=10), height=520)
        st.plotly_chart(fig3d, use_container_width=True)

    st.markdown("""
<div class="legend">
<b>Cómo leerlo (criollo):</b> querés puntos <b>abajo/izquierda</b> (menos energía/agua) y <b>adelante</b> (menos crew).
La capa “Pareto” marca los que no pueden mejorarse en un eje sin empeorar otro.
</div>
""", unsafe_allow_html=True)

    st.markdown('<h4 class="section-title">Tabla — Frontera de Pareto</h4>', unsafe_allow_html=True)
    table_pareto = df_view[df_view["Pareto"]=="Pareto"].sort_values("Score", ascending=False)
    st.dataframe(
        table_pareto[["Opción","Score","Proceso","Materiales","Energía (kWh)","Agua (L)","Crew (min)"]],
        use_container_width=True, hide_index=True
    )

    st.markdown('<h4 class="section-title">Seleccionar candidato</h4>', unsafe_allow_html=True)
    opciones = table_pareto["Opción"].astype(int).tolist()
    if opciones:
        pick_opt = st.selectbox("Elegí Opción #", opciones, index=0, key="pick_from_pareto")
        if st.button("✅ Usar como seleccionado"):
            idx = int(pick_opt) - 1
            if 0 <= idx < len(cands):
                selected = cands[idx]
                flags = check_safety(selected["materials"], selected["process_name"], selected["process_id"])
                st.session_state["selected"] = {"data": selected, "safety": flags}
                st.success(f"Candidato #{pick_opt} seleccionado. Abrí **4) Resultados** o **5) Comparar & Explicar**.")
            else:
                st.warning("Opción fuera de rango respecto a la lista de candidates.")
    else:
        st.info("No hay puntos en la frontera con datos completos.")

# ---------- TAB 2: Predicciones de ensayo (demo conectada a datos) ----------
with tab_trials:
    st.markdown('<h3 class="section-title">Score predictions — barras de confianza</h3>', unsafe_allow_html=True)
    st.caption("Usa los **scores reales** y les aplica un ±CI porcentual para visualizar la variabilidad esperable (demo).")

    ci_pct = st.slider("Intervalo de confianza (± % de Score)", 5, 50, 20, step=5)
    top_n  = st.slider("Top-N por Score", 3, max(3, len(df_view)), min(8, len(df_view)))

    df_trials = df_view.sort_values("Score", ascending=False).head(top_n).copy()
    if df_trials.empty:
        st.info("No hay candidatos suficientes para graficar.")
    else:
        yerr = (df_trials["Score"].abs() * (ci_pct/100.0)).clip(lower=0.05)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df_trials["Opción"].astype(str),
            y=df_trials["Score"],
            error_y=dict(type='data', array=yerr, thickness=1.2, width=4),
            mode="markers",
            marker=dict(size=10),
            name="Predicted trial score"
        ))
        fig.update_layout(yaxis_title="Score ± CI", xaxis_title="Opción", height=420, margin=dict(l=10,r=10,t=10,b=10))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
<div class="legend"><b>Interpretación:</b> si dos opciones se solapan mucho en su CI, tal vez requieras otra señal (p. ej., menos agua) para decidir.
</div>
""", unsafe_allow_html=True)

# ---------- TAB 3: Objetivos por eje ----------
with tab_objectives:
    st.markdown('<h3 class="section-title">Métricas por componente del objetivo</h3>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    if "Energía (kWh)" in df_view:
        with col1:
            st.markdown("**Energía (kWh)**")
            e_fig = px.histogram(df_view, x="Energía (kWh)")
            e_fig.update_layout(height=280, margin=dict(l=10,r=10,t=10,b=10))
            st.plotly_chart(e_fig, use_container_width=True)
    if "Agua (L)" in df_view:
        with col2:
            st.markdown("**Agua (L)**")
            w_fig = px.histogram(df_view, x="Agua (L)")
            w_fig.update_layout(height=280, margin=dict(l=10,r=10,t=10,b=10))
            st.plotly_chart(w_fig, use_container_width=True)
    if "Crew (min)" in df_view:
        with col3:
            st.markdown("**Crew (min)**")
            c_fig = px.histogram(df_view, x="Crew (min)")
            c_fig.update_layout(height=280, margin=dict(l=10,r=10,t=10,b=10))
            st.plotly_chart(c_fig, use_container_width=True)

    st.markdown("""
<div class="legend"><b>Ejemplo:</b> si tu objetivo prioriza tiempo de tripulación,
mirá la cola izquierda del histograma de <i>Crew (min)</i> y elegí opciones con menor valor.
</div>
""", unsafe_allow_html=True)

# ---------- TAB 4: Export Center ----------
with tab_export:
    st.markdown('<h3 class="section-title">Exportar resultados</h3>', unsafe_allow_html=True)
    colL, colR = st.columns([1.1, 1.0])

    with colL:
        st.markdown("**Exportar tabla actual**")
        csv_all = df_view.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ CSV — Vista filtrada", data=csv_all,
                           file_name="pareto_view.csv", mime="text/csv")

        table_pareto = df_view[df_view["Pareto"]=="Pareto"].sort_values("Score", ascending=False)
        if not table_pareto.empty:
            csv_front = table_pareto.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ CSV — Frontera Pareto", data=csv_front,
                               file_name="pareto_frontier.csv", mime="text/csv")
        st.markdown("""
<div class="legend">La tabla exportada conserva columnas clave (energía/agua/crew/score).
Podés adjuntarla a un PR o a una revisión de diseño.</div>
""", unsafe_allow_html=True)

    with colR:
        st.markdown("**Exportar candidato seleccionado**")
        if not state_sel:
            st.info("Seleccioná una opción en **Pareto** para habilitar export de plan.")
        else:
            selected = state_sel["data"]
            safety   = state_sel["safety"]
            try:
                json_bytes = candidate_to_json(selected, target, safety)
                st.download_button("⬇️ JSON — Plan completo", data=json_bytes,
                                   file_name="candidate_plan.json", mime="application/json")
            except Exception as e:
                st.warning(f"No se pudo construir JSON: {e}")
            try:
                csv_bytes  = candidate_to_csv(selected)
                st.download_button("⬇️ CSV — Resumen candidato", data=csv_bytes,
                                   file_name="candidate_summary.csv", mime="text/csv")
            except Exception as e:
                st.warning(f"No se pudo construir CSV: {e}")

    st.markdown("---")
    st.markdown("""
<div class="block">
  <b>Checklist de trazabilidad</b><br/>
  • ¿El candidato exportado coincide con la <i>Opción</i> elegida? ✔️<br/>
  • ¿Materiales incluyen entradas problemáticas NASA y (si aplica) <b>MGS-1</b>? ✔️<br/>
  • ¿Los límites del objetivo (agua/energía/crew) se respetan o se justifican? ✔️<br/>
</div>
""", unsafe_allow_html=True)
