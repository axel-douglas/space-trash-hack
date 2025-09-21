# --- path guard para Streamlit Cloud ---
import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]
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
from app.modules.safety import check_safety  # para recalcular badge al seleccionar

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
.card {border:1px solid rgba(128,128,128,0.25); border-radius:14px; padding:14px;}
h1, h2, h3 { letter-spacing:.2px }
</style>
""", unsafe_allow_html=True)

st.markdown("# 📤 Pareto & Export")
st.caption("Todo lo que ves aquí usa **los candidatos reales generados**. Explora la **frontera de Pareto** (agua/energía/crew), mira **predicciones** y **exportá** el plan elegido.")

# ======== tabla base y saneo robusto (derivado de candidates reales) ========
df_raw = compare_table(cands, target, crew_time_low=target.get("crew_time_low", False)).copy()

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
    if low in ["opción","opcion"]:
        rename_map[col] = "Opción"
    if low == "score":
        rename_map[col] = "Score"
df_raw.rename(columns=rename_map, inplace=True)

for k in ["Opción","Score","Proceso","Materiales","Energía (kWh)","Agua (L)","Crew (min)"]:
    if k not in df_raw.columns:
        df_raw[k] = np.nan

df_raw["Materiales"] = df_raw["Materiales"].apply(
    lambda v: ", ".join(v) if isinstance(v, (list, tuple)) else (str(v) if pd.notna(v) else "")
)
for k in ["Score","Energía (kWh)","Agua (L)","Crew (min)","Masa (kg)"]:
    if k in df_raw.columns:
        df_raw[k] = pd.to_numeric(df_raw[k], errors="coerce")

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

# ======== What-If de límites (todo deriva de df_plot) ========
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

# ======== Frontera de Pareto (sobre datos reales) ========
try:
    front_idx = pareto_front(df_plot)  # índices de df_plot
    front_mask = df_plot.index.isin(front_idx)
except Exception:
    front_mask = df_plot["Score"].rank(ascending=False, method="first") <= 5
df_view["Pareto"] = np.where(front_mask, "Pareto", "No Pareto")

# ======== Tabs ========
tab_pareto, tab_trials, tab_objectives, tab_export = st.tabs(
    ["🌌 Pareto Explorer", "🔮 Predicciones de ensayo (demo)", "🎯 Objetivos por eje", "📦 Export Center"]
)

# ---------- TAB 1: PARETO ----------
with tab_pareto:
    st.markdown("#### Explorador 3D (Energía vs Agua vs Crew)")
    df_view["ScorePos"] = np.clip(df_view["Score"].fillna(0.0), 0.01, None)
    usable = df_view.dropna(subset=["Energía (kWh)","Agua (L)","Crew (min)","ScorePos"]).copy()

    if usable.empty:
        st.info("No hay suficientes datos válidos para dibujar el 3D.")
    else:
        fig3d = px.scatter_3d(
            usable,
            x="Energía (kWh)", y="Agua (L)", z="Crew (min)",
            color="Pareto", size="ScorePos",
            hover_data=["Opción","Proceso","Materiales","Dentro_límites","Score"]
        )
        fig3d.update_layout(margin=dict(l=10,r=10,t=10,b=10), height=520)
        st.plotly_chart(fig3d, use_container_width=True)

        st.markdown("##### Tabla — Frontera de Pareto")
        table_pareto = usable[usable["Pareto"] == "Pareto"].sort_values("Score", ascending=False)
        st.dataframe(table_pareto[["Opción","Score","Proceso","Materiales","Energía (kWh)","Agua (L)","Crew (min)"]],
                     use_container_width=True, hide_index=True)

        # ---- Selección del candidato (basada en datos reales) ----
        st.markdown("##### Seleccionar candidato desde la frontera")
        opciones = table_pareto["Opción"].astype(int).tolist()
        if opciones:
            pick_opt = st.selectbox("Elegí Opción #", opciones, index=0, key="pick_from_pareto")
            if st.button("✅ Usar como seleccionado"):
                idx = int(pick_opt) - 1
                if 0 <= idx < len(cands):
                    selected = cands[idx]
                    # Recalcular banderas de seguridad reales
                    flags = check_safety(selected["materials"], selected["process_name"], selected["process_id"])
                    st.session_state["selected"] = {"data": selected, "safety": flags}
                    st.success(f"Candidato #{pick_opt} seleccionado. Ya podés ir a **4) Resultados** o **5) Comparar & Explicar**.")
                else:
                    st.warning("Opción fuera de rango respecto a la lista de candidates.")

    st.markdown('<span class="pill info">Tip</span> En el 3D buscá **abajo/izquierda** (menos agua/energía) y **adelante** (menos crew). Si estás en modo *Crew-time Low*, priorizá el eje **Crew**.',
                unsafe_allow_html=True)

# ---------- TAB 2: PREDICCIONES DE ENSAYO (DEMO, sobre df_view real) ----------
with tab_trials:
    st.markdown("#### Score predictions — barras de confianza")
    st.caption("Usa los **scores reales** de tus candidatos y aplica un intervalo de confianza proporcional (demo visual).")

    ci_pct = st.slider("Intervalo de confianza (± % de Score)", 5, 50, 20, step=5)
    top_n  = st.slider("Mostrar Top-N por Score", 3, max(3, len(df_view)), min(8, len(df_view)))

    df_trials = df_view.sort_values("Score", ascending=False).head(top_n).copy()
    df_trials["y"] = df_trials["Score"]
    df_trials["y_err"] = (df_trials["Score"].abs() * (ci_pct/100.0)).clip(lower=0.05)

    if df_trials.empty:
        st.info("No hay candidatos suficientes para graficar.")
    else:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df_trials["Opción"].astype(str),
            y=df_trials["y"],
            error_y=dict(type='data', array=df_trials["y_err"], thickness=1.2, width=4),
            mode="markers",
            marker=dict(size=10),
            name="Predicted trial score"
        ))
        fig.update_layout(yaxis_title="Score ± CI", xaxis_title="Opción", height=460, margin=dict(l=10,r=10,t=10,b=10))
        st.plotly_chart(fig, use_container_width=True)

# ---------- TAB 3: OBJETIVOS POR EJE (sobre df_view real) ----------
with tab_objectives:
    st.markdown("#### Métricas por componente del objetivo")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**Energía (kWh)**")
        if "Energía (kWh)" in df_view:
            e_fig = px.histogram(df_view, x="Energía (kWh)")
            e_fig.update_layout(height=280, margin=dict(l=10,r=10,t=10,b=10))
            st.plotly_chart(e_fig, use_container_width=True)
    with col2:
        st.markdown("**Agua (L)**")
        if "Agua (L)" in df_view:
            w_fig = px.histogram(df_view, x="Agua (L)")
            w_fig.update_layout(height=280, margin=dict(l=10,r=10,t=10,b=10))
            st.plotly_chart(w_fig, use_container_width=True)
    with col3:
        st.markdown("**Crew (min)**")
        if "Crew (min)" in df_view:
            c_fig = px.histogram(df_view, x="Crew (min)")
            c_fig.update_layout(height=280, margin=dict(l=10,r=10,t=10,b=10))
            st.plotly_chart(c_fig, use_container_width=True)

# ---------- TAB 4: EXPORT (todo desde estado real) ----------
with tab_export:
    st.markdown("#### Exportar resultados")
    colL, colR = st.columns([1.1, 1.0])
    with colL:
        st.markdown("**Exportar tabla**")
        csv_all = df_view.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ CSV — Vista actual", data=csv_all, file_name="pareto_view.csv", mime="text/csv")

        usable = df_view.dropna(subset=["Energía (kWh)","Agua (L)","Crew (min)","Score"]).copy()
        table_pareto = usable[usable["Pareto"] == "Pareto"].sort_values("Score", ascending=False)
        if not table_pareto.empty:
            csv_front = table_pareto.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ CSV — Frontera Pareto", data=csv_front, file_name="pareto_frontier.csv", mime="text/csv")

    with colR:
        st.markdown("**Exportar candidato elegido**")
        if not state_sel:
            st.info("Seleccioná una opción (arriba) para habilitar export de plan.")
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
