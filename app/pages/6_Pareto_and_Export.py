# app/pages/6_Pareto_and_Export.py
import _bootstrap  # noqa: F401

import io
import json
from datetime import datetime

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.colors import sample_colorscale

from app.modules.explain import compare_table
from app.modules.analytics import pareto_front
from app.modules.exporters import candidate_to_json, candidate_to_csv
from app.modules.safety import check_safety, safety_badge  # recalcular badge al seleccionar
from app.modules.ui_blocks import load_theme

# ⚠️ PRIMERA llamada
st.set_page_config(page_title="Pareto & Export", page_icon="📤", layout="wide")

load_theme()

# ======== estado requerido ========
cands  = st.session_state.get("candidates", [])
target = st.session_state.get("target", None)
state_sel = st.session_state.get("selected", None)

st.session_state.setdefault("export_history", [])
st.session_state.setdefault("export_wizard_step", 1)
st.session_state.setdefault("selected_export_format", "Plan JSON")
st.session_state.setdefault("last_export_payload", None)
st.session_state.setdefault("selected_option_number", None)
st.session_state.setdefault("export_payload_cache", {})

selected_candidate = state_sel["data"] if state_sel else None
safety_flags = state_sel["safety"] if state_sel else None

if selected_candidate and not st.session_state.get("selected_option_number"):
    try:
        matched_idx = next(idx for idx, cand in enumerate(cands, start=1) if cand is selected_candidate)
        st.session_state["selected_option_number"] = matched_idx
    except StopIteration:
        pass

safety_summary = safety_badge(safety_flags) if safety_flags else {"level": "Sin datos", "detail": "Seleccioná un candidato."}

if not cands or not target:
    st.warning("Generá opciones en **3) Generador** primero.")
    st.stop()

# ======== estilos (NASA/SpaceX-like) ========
st.markdown(
    """
    <style>
    .hero {border-radius:16px; padding:18px 18px 8px; background: radial-gradient(1200px 380px at 20% -10%, rgba(80,120,255,.08), transparent);}
    .section-title{margin-top:6px; margin-bottom:6px}
    .legend{margin-top:12px; font-size:0.9rem; color:var(--muted);}
    .pill{display:inline-flex; align-items:center; padding:6px 12px; border-radius:999px; font-size:0.78rem; margin-right:8px; border:1px solid rgba(148,163,184,0.32); background:rgba(15,23,42,0.6); box-shadow:0 0 18px rgba(59,130,246,0.18);}
    .pill.ok{border-color:rgba(74,222,128,0.45); box-shadow:0 0 18px rgba(74,222,128,0.25);}
    .kpi{border-radius:18px; padding:16px 18px; background:rgba(15,23,42,0.68); border:1px solid rgba(148,163,184,0.28); box-shadow:0 12px 28px rgba(15,23,42,0.35); text-align:center;}
    .kpi .v{font-size:1.8rem; font-weight:600; margin-top:4px; color:#e0f2fe;}
    .legend b{color:#f8fafc;}
    .nebula-panel{background:rgba(15,23,42,0.65); border-radius:20px; padding:16px 18px; border:1px solid rgba(148,163,184,0.22); box-shadow:0 0 40px rgba(59,130,246,0.12);}
    .sidebar-flight h2{margin-top:0; color:#e0f2fe;}
    .flight-card{border-radius:16px; padding:14px 16px; margin-bottom:10px; background:linear-gradient(135deg, rgba(30,64,175,0.25), rgba(15,23,42,0.75)); border:1px solid rgba(96,165,250,0.35); box-shadow:0 8px 22px rgba(15,23,42,0.45); transition:transform 0.25s ease, box-shadow 0.25s ease;}
    .flight-card:hover{transform:translateY(-2px); box-shadow:0 14px 30px rgba(30,64,175,0.4);}
    .flight-card.active{border-color:rgba(74,222,128,0.65); box-shadow:0 0 32px rgba(74,222,128,0.35);}
    .flight-card strong{display:block; font-size:1rem; color:#f8fafc;}
    .flight-card span{display:block; font-size:0.78rem; color:rgba(226,232,240,0.75);}
    .flight-alert{border-radius:18px; padding:12px 16px; margin-bottom:12px; border:1px solid rgba(74,222,128,0.4); background:rgba(15,118,110,0.35); color:#f0fdfa; box-shadow:0 0 18px rgba(45,212,191,0.35);}
    .flight-alert.animate{animation:flightGlow 1.6s ease-in-out 2;}
    @keyframes flightGlow{0%{box-shadow:0 0 0 rgba(45,212,191,0.0);}50%{box-shadow:0 0 28px rgba(45,212,191,0.65);}100%{box-shadow:0 0 0 rgba(45,212,191,0.0);}}
    .nebula-preview{border-radius:16px; padding:12px 16px; border:1px solid rgba(148,163,184,0.28); background:rgba(15,23,42,0.6); color:#e2e8f0; font-size:0.85rem;}
    .wizard-container{margin-top:12px;}
    .wizard-panel{border-radius:20px; padding:18px 20px; background:linear-gradient(145deg, rgba(30,64,175,0.28), rgba(15,23,42,0.85)); border:1px solid rgba(96,165,250,0.35); box-shadow:0 12px 42px rgba(15,23,42,0.55);}
    .translucent-panel{border-radius:22px; padding:20px 22px; background:linear-gradient(160deg, rgba(148,197,255,0.18), rgba(15,23,42,0.78)); border:1px solid rgba(148,197,255,0.45); backdrop-filter:blur(10px); box-shadow:0 18px 40px rgba(30,64,175,0.35);}
    .safety-badges{display:flex; gap:10px; flex-wrap:wrap; margin-bottom:12px;}
    .safety-badge{padding:6px 12px; border-radius:999px; font-size:0.78rem; letter-spacing:0.02em; border:1px solid rgba(148,163,184,0.35); background:rgba(15,23,42,0.75); box-shadow:0 0 22px rgba(59,130,246,0.18); color:#e0f2fe;}
    .safety-badge.ok{border-color:rgba(74,222,128,0.55); box-shadow:0 0 22px rgba(74,222,128,0.35);}
    .safety-badge.alert{border-color:rgba(248,113,113,0.55); box-shadow:0 0 22px rgba(248,113,113,0.35);}
    .stRadio > div[role='radiogroup']{display:flex; flex-wrap:wrap; gap:10px;}
    .stRadio > div[role='radiogroup'] > label{border-radius:999px; padding:10px 18px; border:1px solid rgba(148,163,184,0.35); background:rgba(15,23,42,0.55); box-shadow:0 0 18px rgba(148,197,255,0.18); cursor:pointer; transition:all 0.25s ease;}
    .stRadio > div[role='radiogroup'] > label:hover{box-shadow:0 0 26px rgba(148,197,255,0.38); transform:translateY(-1px);}
    .stRadio > div[role='radiogroup'] > label input:checked + div{background:linear-gradient(135deg, rgba(59,130,246,0.85), rgba(56,189,248,0.75)); color:#0f172a; border-radius:999px; font-weight:600; box-shadow:0 0 32px rgba(56,189,248,0.55); padding:6px 12px;}
    .stButton>button{background:linear-gradient(135deg, rgba(56,189,248,0.95), rgba(59,130,246,0.95)); color:#0f172a; border:1px solid rgba(125,211,252,0.9); font-weight:700; text-transform:uppercase; letter-spacing:0.04em; border-radius:999px; padding:0.6rem 1.8rem; box-shadow:0 18px 38px rgba(56,189,248,0.35); transition:all 0.25s ease;}
    .stButton>button:hover{box-shadow:0 24px 46px rgba(59,130,246,0.45); transform:translateY(-2px);}
    .stButton>button:focus{outline:none; box-shadow:0 0 0 3px rgba(125,211,252,0.4);}
    .history-table{margin-top:18px;}
    .history-table table{width:100%; border-collapse:collapse; font-size:0.82rem;}
    .history-table th,.history-table td{padding:8px 10px; border-bottom:1px solid rgba(148,163,184,0.2); text-align:left; color:#e2e8f0;}
    .history-table tr:last-child td{border-bottom:none;}
    .history-table th{color:#bae6fd; font-weight:600;}
    </style>
    """,
    unsafe_allow_html=True,
)

def render_safety_badges_html(flags) -> str:
    if not flags:
        return "<span class='safety-badge'>Sin evaluación</span>"
    badges = []
    checklist = [
        (not getattr(flags, "pfas", False), "PFAS sweep"),
        (not getattr(flags, "microplastics", False), "Microplásticos"),
        (not getattr(flags, "incineration", False), "Incineración"),
    ]
    for ok, label in checklist:
        state = "OK" if ok else "Revisar"
        css = "safety-badge ok" if ok else "safety-badge alert"
        badges.append(f"<span class='{css}'>{label}: {state}</span>")
    return "".join(badges)

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
table_pareto = df_view[df_view["Pareto"] == "Pareto"].sort_values("Score", ascending=False)
pareto_options = table_pareto["Opción"].astype(int).tolist() if "Opción" in table_pareto else []

selected_option_number = st.session_state.get("selected_option_number")

# ======== Sidebar – Flight Plans & Previews ========
sidebar = st.sidebar
sidebar.markdown("<div class='sidebar-flight'><h2>🛫 Flight Plans</h2></div>", unsafe_allow_html=True)

flash_event = st.session_state.pop("flight_flash", None)
if flash_event:
    sidebar.markdown(
        f"<div class='flight-alert animate'>Flight plan #{flash_event.get('option', '—')} listo para export. Revisá la Nebula preview.</div>",
        unsafe_allow_html=True,
    )

if table_pareto.empty:
    sidebar.info("Generá candidatos para visualizar planes de vuelo.")
else:
    for _, row in table_pareto.head(4).iterrows():
        option_number = int(row.get("Opción", 0))
        energy = row.get("Energía (kWh)", 0.0)
        water = row.get("Agua (L)", 0.0)
        crew = row.get("Crew (min)", 0.0)
        score_val = row.get("Score", 0.0)
        active_cls = " active" if selected_option_number and option_number == int(selected_option_number) else ""
        sidebar.markdown(
            f"<div class='flight-card{active_cls}'>"
            f"<strong>Plan #{option_number}</strong>"
            f"<span>Score: {score_val:.2f}</span>"
            f"<span>Energía: {energy:.2f} kWh · Agua: {water:.2f} L · Crew: {crew:.1f} min</span>"
            "</div>",
            unsafe_allow_html=True,
        )

if selected_candidate:
    props = selected_candidate.get("props")
    materials = ", ".join(selected_candidate.get("materials", [])[:3])
    if selected_candidate.get("materials") and len(selected_candidate["materials"]) > 3:
        materials += "…"
    sidebar.markdown(
        "<h3 style='margin-top:14px;'>Nebula preview</h3>",
        unsafe_allow_html=True,
    )
    sidebar.markdown(
        "<div class='safety-badges'>" + render_safety_badges_html(safety_flags) + "</div>",
        unsafe_allow_html=True,
    )
    if props:
        sidebar.markdown(
            """
            <div class='nebula-preview'>
              <strong>{label}</strong><br/>
              Proceso: {proc}<br/>
              Materiales: {mats}<br/>
              Score: {score:.2f} · Energía: {energy:.2f} kWh · Agua: {water:.2f} L · Crew: {crew:.1f} min
            </div>
            """.format(
                label=f"Plan #{selected_option_number or '—'}",
                proc=f"{selected_candidate.get('process_id', '')} {selected_candidate.get('process_name', '')}".strip(),
                mats=materials or "—",
                score=selected_candidate.get("score", 0.0),
                energy=getattr(props, "energy_kwh", 0.0),
                water=getattr(props, "water_l", 0.0),
                crew=getattr(props, "crew_min", 0.0),
            ),
            unsafe_allow_html=True,
        )

    fmt_choice = st.session_state.get("selected_export_format", "Plan JSON")
    sidebar.caption(f"Formato seleccionado: {fmt_choice}")
    try:
        if fmt_choice == "Plan JSON" and safety_flags:
            preview = candidate_to_json(selected_candidate, target, safety_flags).decode("utf-8")
            preview_lines = preview.splitlines()
            sidebar.code("\n".join(preview_lines[:10]) + ("\n…" if len(preview_lines) > 10 else ""), language="json")
        elif fmt_choice == "Resumen CSV":
            csv_text = candidate_to_csv(selected_candidate).decode("utf-8")
            preview_lines = csv_text.splitlines()
            sidebar.code("\n".join(preview_lines[:8]) + ("\n…" if len(preview_lines) > 8 else ""), language="csv")
        elif fmt_choice == "Pareto CSV":
            sidebar.dataframe(table_pareto.head(6), use_container_width=True, hide_index=True)
    except Exception as preview_error:
        sidebar.warning(f"No se pudo generar preview: {preview_error}")


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
        if "Opción" in usable:
            usable["Opción"] = pd.to_numeric(usable["Opción"], errors="coerce")

        pareto_points = usable[usable["Pareto"] == "Pareto"].copy()
        other_points = usable[usable["Pareto"] != "Pareto"].copy()

        def _color_scale(values: pd.Series, scale: str) -> list[str]:
            if values.empty:
                return []
            vals = values.fillna(values.mean() if not values.empty else 0.0)
            vmin, vmax = float(vals.min()), float(vals.max())
            if abs(vmax - vmin) < 1e-9:
                norm = [0.5] * len(vals)
            else:
                norm = ((vals - vmin) / (vmax - vmin)).clip(0.0, 1.0).tolist()
            return sample_colorscale(scale, norm)

        def _safe_series(df: pd.DataFrame, key: str) -> pd.Series:
            if key in df:
                return df[key]
            return pd.Series(np.nan, index=df.index)

        fig3d = go.Figure()

        if not other_points.empty:
            fig3d.add_trace(
                go.Scatter3d(
                    x=other_points["Energía (kWh)"],
                    y=other_points["Agua (L)"],
                    z=other_points["Crew (min)"],
                    mode="markers",
                    name="Candidatos",
                    marker=dict(
                        size=6,
                        color=_color_scale(other_points["Score"], "Viridis"),
                        opacity=0.45,
                        line=dict(width=1.2, color="rgba(148,163,184,0.4)"),
                        symbol="circle",
                    ),
                    hovertemplate="<b>Opción %{customdata[0]}</b><br>Score %{customdata[1]:.2f}<extra></extra>",
                    customdata=np.stack([
                        _safe_series(other_points, "Opción").fillna(0.0),
                        other_points["Score"].fillna(0.0),
                    ], axis=-1),
                )
            )

        if not pareto_points.empty:
            pareto_colors = _color_scale(pareto_points["Score"], "IceFire")
            fig3d.add_trace(
                go.Scatter3d(
                    x=pareto_points["Energía (kWh)"],
                    y=pareto_points["Agua (L)"],
                    z=pareto_points["Crew (min)"],
                    mode="markers",
                    name="Pareto Prime",
                    marker=dict(
                        size=11,
                        color=pareto_colors,
                        opacity=0.98,
                        symbol="diamond",
                        line=dict(width=3, color="rgba(240,249,255,0.9)"),
                        lighting=dict(ambient=0.62, diffuse=0.9, specular=0.88, roughness=0.2, fresnel=0.25),
                        lightposition=dict(x=200, y=120, z=140),
                    ),
                    hovertemplate=(
                        "<b>Plan %{customdata[0]}</b><br>Score %{customdata[1]:.2f}<br>"
                        "Energía %{x:.2f} kWh<br>Agua %{y:.2f} L<br>Crew %{z:.1f} min<extra></extra>"
                    ),
                    customdata=np.stack([
                        _safe_series(pareto_points, "Opción").fillna(0.0),
                        pareto_points["Score"].fillna(0.0),
                    ], axis=-1),
                )
            )

            score_min = float(pareto_points["ScorePos"].min()) if not pareto_points.empty else 0.0
            score_max = float(pareto_points["ScorePos"].max()) if not pareto_points.empty else 1.0
            size_span = max(score_max - score_min, 0.01)
            halo_sizes = 18 + 20 * (pareto_points["ScorePos"] - score_min) / size_span
            fig3d.add_trace(
                go.Scatter3d(
                    x=pareto_points["Energía (kWh)"],
                    y=pareto_points["Agua (L)"],
                    z=pareto_points["Crew (min)"],
                    mode="markers",
                    name="Nebula halo",
                    marker=dict(
                        size=halo_sizes,
                        color="rgba(125,211,252,0.18)",
                        opacity=0.22,
                    ),
                    hoverinfo="skip",
                    showlegend=False,
                )
            )

            # Nebula cloud around Pareto points
            x_vals = pareto_points["Energía (kWh)"].to_numpy()
            y_vals = pareto_points["Agua (L)"].to_numpy()
            z_vals = pareto_points["Crew (min)"].to_numpy()
            x_span = max(float(usable["Energía (kWh)"].max() - usable["Energía (kWh)"].min()), 1e-3)
            y_span = max(float(usable["Agua (L)"].max() - usable["Agua (L)"].min()), 1e-3)
            z_span = max(float(usable["Crew (min)"].max() - usable["Crew (min)"].min()), 1e-3)
            nebula_points = []
            for option, xv, yv, zv in zip(pareto_points.get("Opción", []), x_vals, y_vals, z_vals):
                rng = np.random.default_rng(int(option * 997) if not pd.isna(option) else 42)
                spread = np.array([x_span, y_span, z_span]) * 0.04
                cloud = rng.normal(loc=[xv, yv, zv], scale=np.maximum(spread, 1e-3), size=(24, 3))
                nebula_points.append(cloud)
            if nebula_points:
                nebula = np.vstack(nebula_points)
                fig3d.add_trace(
                    go.Scatter3d(
                        x=nebula[:, 0],
                        y=nebula[:, 1],
                        z=nebula[:, 2],
                        mode="markers",
                        marker=dict(size=3, opacity=0.18, color="rgba(148,197,255,0.18)"),
                        hoverinfo="skip",
                        showlegend=False,
                    )
                )

        # Highlight selected candidate in scene
        if selected_option_number:
            selected_trace = usable[usable.get("Opción") == int(selected_option_number)] if "Opción" in usable else pd.DataFrame()
            if not selected_trace.empty:
                fig3d.add_trace(
                    go.Scatter3d(
                        x=selected_trace["Energía (kWh)"],
                        y=selected_trace["Agua (L)"],
                        z=selected_trace["Crew (min)"],
                        mode="markers",
                        name="Seleccionado",
                        marker=dict(
                            size=14,
                            color="rgba(74,222,128,0.95)",
                            line=dict(width=4, color="rgba(255,255,255,0.95)"),
                            opacity=1.0,
                            symbol="circle",
                        ),
                        hovertemplate="Plan seleccionado %{customdata[0]}<extra></extra>",
                        customdata=np.stack([
                            _safe_series(selected_trace, "Opción").fillna(0.0)
                        ], axis=-1),
                    )
                )

        # Illuminated axes
        x_min, x_max = float(usable["Energía (kWh)"].min()), float(usable["Energía (kWh)"].max())
        y_min, y_max = float(usable["Agua (L)"].min()), float(usable["Agua (L)"].max())
        z_min, z_max = float(usable["Crew (min)"].min()), float(usable["Crew (min)"].max())
        axis_lines = [
            ([x_min, x_max], [y_min, y_min], [z_min, z_min]),
            ([x_min, x_min], [y_min, y_max], [z_min, z_min]),
            ([x_min, x_min], [y_min, y_min], [z_min, z_max]),
        ]
        for idx, (xs, ys, zs) in enumerate(axis_lines):
            fig3d.add_trace(
                go.Scatter3d(
                    x=xs,
                    y=ys,
                    z=zs,
                    mode="lines",
                    line=dict(color="rgba(148,197,255,0.85)", width=6),
                    hoverinfo="skip",
                    showlegend=False,
                    name=f"axis-{idx}",
                )
            )

        fig3d.update_layout(
            height=540,
            margin=dict(l=0, r=0, b=0, t=24),
            paper_bgcolor="rgba(4,7,20,1)",
            scene=dict(
                xaxis=dict(
                    title="Energía (kWh)",
                    backgroundcolor="rgba(8,12,35,0.92)",
                    gridcolor="rgba(96,165,250,0.12)",
                    zerolinecolor="rgba(148,197,255,0.6)",
                    showbackground=True,
                    showspikes=True,
                    spikecolor="rgba(125,211,252,0.8)",
                    spikethickness=2,
                    tickfont=dict(color="#cbd5f5"),
                    titlefont=dict(color="#bae6fd"),
                ),
                yaxis=dict(
                    title="Agua (L)",
                    backgroundcolor="rgba(6,11,32,0.9)",
                    gridcolor="rgba(96,165,250,0.12)",
                    zerolinecolor="rgba(148,197,255,0.6)",
                    showbackground=True,
                    showspikes=True,
                    spikecolor="rgba(56,189,248,0.75)",
                    spikethickness=2,
                    tickfont=dict(color="#cbd5f5"),
                    titlefont=dict(color="#bae6fd"),
                ),
                zaxis=dict(
                    title="Crew (min)",
                    backgroundcolor="rgba(5,9,28,0.9)",
                    gridcolor="rgba(96,165,250,0.12)",
                    zerolinecolor="rgba(148,197,255,0.6)",
                    showbackground=True,
                    showspikes=True,
                    spikecolor="rgba(14,165,233,0.8)",
                    spikethickness=2,
                    tickfont=dict(color="#cbd5f5"),
                    titlefont=dict(color="#bae6fd"),
                ),
                camera=dict(eye=dict(x=1.65, y=1.72, z=1.45)),
                dragmode="orbit",
                aspectmode="cube",
            ),
            legend=dict(
                bgcolor="rgba(8,12,35,0.82)",
                font=dict(color="#e0f2fe"),
                orientation="h",
                yanchor="bottom",
                y=0.01,
                x=0.02,
            ),
        )

        st.plotly_chart(
            fig3d,
            use_container_width=True,
            config={"displaylogo": False, "modeBarButtonsToRemove": ["resetCameraDefault3d"], "scrollZoom": True},
        )

    st.markdown("""
<div class="legend">
<b>Cómo leerlo (criollo):</b> querés puntos <b>abajo/izquierda</b> (menos energía/agua) y <b>adelante</b> (menos crew).
La capa “Pareto” marca los que no pueden mejorarse en un eje sin empeorar otro.
</div>
""", unsafe_allow_html=True)

    st.markdown('<h4 class="section-title">Tabla — Frontera de Pareto</h4>', unsafe_allow_html=True)
    if not table_pareto.empty:
        st.dataframe(
            table_pareto[["Opción","Score","Proceso","Materiales","Energía (kWh)","Agua (L)","Crew (min)"]],
            use_container_width=True, hide_index=True
        )
    else:
        st.info("No hay puntos en la frontera con datos completos.")

    st.markdown('<h4 class="section-title">Seleccionar candidato</h4>', unsafe_allow_html=True)
    if pareto_options:
        default_index = 0
        if selected_option_number and int(selected_option_number) in pareto_options:
            default_index = pareto_options.index(int(selected_option_number))
        pick_opt = st.selectbox("Elegí Opción #", pareto_options, index=default_index, key="pick_from_pareto")
        if st.button("✅ Usar como seleccionado"):
            idx = int(pick_opt) - 1
            if 0 <= idx < len(cands):
                selected = cands[idx]
                flags = check_safety(selected["materials"], selected["process_name"], selected["process_id"])
                st.session_state["selected"] = {"data": selected, "safety": flags}
                st.session_state["selected_option_number"] = pick_opt
                st.session_state["flight_flash"] = {"option": pick_opt}
                st.session_state["export_wizard_step"] = 1
                st.session_state["last_export_payload"] = None
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
    st.markdown('<h3 class="section-title">🧭 Mission Export Assistant</h3>', unsafe_allow_html=True)

    if not selected_candidate:
        st.info("Seleccioná primero un plan en la pestaña **Pareto Explorer** para habilitar el asistente.")
    else:
        st.markdown(
            "<div class='safety-badges'>" + render_safety_badges_html(safety_flags) + "</div>",
            unsafe_allow_html=True,
        )
        st.caption(f"Estado de seguridad: {safety_summary['level']} — {safety_summary['detail']}")

        step_labels = ["1️⃣ Formato", "2️⃣ Previsualizar", "3️⃣ Confirmar"]
        current_step = st.session_state.get("export_wizard_step", 1)
        step_choice = st.radio(
            "Asistente de exportación",
            step_labels,
            index=max(0, min(len(step_labels) - 1, current_step - 1)),
            horizontal=True,
            key="export_step_radio",
        )
        current_step = step_labels.index(step_choice) + 1
        st.session_state["export_wizard_step"] = current_step

        format_options = ["Plan JSON", "Resumen CSV", "Pareto CSV"]
        if st.session_state["selected_export_format"] not in format_options:
            st.session_state["selected_export_format"] = "Plan JSON"

        def generate_payload(fmt: str):
            if fmt == "Plan JSON":
                if not safety_flags:
                    raise ValueError("Se requiere evaluación de seguridad para exportar JSON.")
                data = candidate_to_json(selected_candidate, target, safety_flags)
                filename = f"flight_plan_{int(selected_option_number or 0):02d}.json"
                mime = "application/json"
            elif fmt == "Resumen CSV":
                data = candidate_to_csv(selected_candidate)
                filename = f"candidate_{int(selected_option_number or 0):02d}_summary.csv"
                mime = "text/csv"
            elif fmt == "Pareto CSV":
                dataset = table_pareto if not table_pareto.empty else df_view
                data = dataset.to_csv(index=False).encode("utf-8")
                filename = "pareto_frontier.csv"
                mime = "text/csv"
            else:
                raise ValueError(f"Formato no soportado: {fmt}")
            return data, mime, filename

        with st.container():
            st.markdown("<div class='wizard-container'>", unsafe_allow_html=True)
            if current_step == 1:
                st.markdown(
                    """
                    <div class='wizard-panel'>
                      <h4>Paso 1 — Elegí tu carga útil</h4>
                      <p>Seleccioná el formato con el que vas a compartir el plan. Podés moverte de paso cuando quieras.</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                fmt_index = format_options.index(st.session_state.get("selected_export_format", "Plan JSON"))
                fmt_choice = st.radio(
                    "Formato de export",
                    format_options,
                    index=fmt_index,
                    key="export_format_selector",
                )
                st.session_state["selected_export_format"] = fmt_choice
                if st.button("Siguiente ➡️", key="wizard_next_1", use_container_width=True):
                    st.session_state["export_wizard_step"] = 2
                    current_step = 2
            elif current_step == 2:
                fmt_choice = st.session_state.get("selected_export_format", "Plan JSON")
                st.markdown(
                    """
                    <div class='wizard-panel'>
                      <h4>Paso 2 — Nebula preview</h4>
                      <p>Verificá los datos renderizados con el formato seleccionado antes de autorizar la exportación.</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                try:
                    payload_bytes, _, _ = generate_payload(fmt_choice)
                    if fmt_choice == "Plan JSON":
                        st.json(json.loads(payload_bytes.decode("utf-8")))
                    elif fmt_choice == "Resumen CSV":
                        preview_df = pd.read_csv(io.StringIO(payload_bytes.decode("utf-8")))
                        st.dataframe(preview_df, use_container_width=True, hide_index=True)
                    elif fmt_choice == "Pareto CSV":
                        st.dataframe(table_pareto if not table_pareto.empty else df_view, use_container_width=True, hide_index=True)
                except Exception as preview_error:
                    st.warning(f"No se pudo generar la previsualización: {preview_error}")

                col_back, col_next = st.columns([1, 1])
                with col_back:
                    if st.button("⬅️ Volver", key="wizard_back_2", use_container_width=True):
                        st.session_state["export_wizard_step"] = 1
                        current_step = 1
                with col_next:
                    if st.button("Continuar ➡️", key="wizard_next_2", use_container_width=True):
                        st.session_state["export_wizard_step"] = 3
                        current_step = 3
            else:
                fmt_choice = st.session_state.get("selected_export_format", "Plan JSON")
                st.markdown(
                    """
                    <div class='wizard-panel'>
                      <h4>Paso 3 — Checklist & confirmación</h4>
                      <p>Confirmá la exportación desde la consola translúcida. Podés volver atrás para ajustar.</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.markdown("<div class='translucent-panel'>", unsafe_allow_html=True)
                st.markdown(
                    f"""
                    <h4 style='margin-top:0;'>Checklist operativo</h4>
                    <ul>
                      <li>Formato elegido: <b>{fmt_choice}</b></li>
                      <li>Plan vinculado: <b>#{selected_option_number or '—'}</b> — {selected_candidate.get('process_name', 'sin proceso')}</li>
                      <li>Seguridad: <b>{safety_summary['level']}</b> · {safety_summary['detail']}</li>
                    </ul>
                    """,
                    unsafe_allow_html=True,
                )
                st.markdown(
                    "<div class='safety-badges'>" + render_safety_badges_html(safety_flags) + "</div>",
                    unsafe_allow_html=True,
                )
                col_confirm, col_back = st.columns([1.2, 1])
                confirm_clicked = False
                with col_confirm:
                    confirm_clicked = st.button("🚀 Generar paquete", key="wizard_confirm", use_container_width=True)
                with col_back:
                    if st.button("⬅️ Ajustar", key="wizard_back_3", use_container_width=True):
                        st.session_state["export_wizard_step"] = 2
                        current_step = 2

                if confirm_clicked:
                    try:
                        payload_bytes, mime, filename = generate_payload(fmt_choice)
                        timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
                        st.session_state["last_export_payload"] = {
                            "data": payload_bytes,
                            "mime": mime,
                            "filename": filename,
                            "format": fmt_choice,
                            "timestamp": timestamp,
                        }
                        history = st.session_state.get("export_history", [])
                        history.insert(
                            0,
                            {
                                "timestamp": timestamp,
                                "plan": f"#{selected_option_number or '—'}",
                                "format": fmt_choice,
                                "safety": safety_summary["level"],
                            },
                        )
                        st.session_state["export_history"] = history[:12]
                        st.success(f"Paquete {fmt_choice} listo para descargar.")
                    except Exception as export_error:
                        st.warning(f"No se pudo generar el paquete: {export_error}")

                payload = st.session_state.get("last_export_payload")
                if payload and payload.get("format") == fmt_choice:
                    st.download_button(
                        "⬇️ Descargar misión",
                        data=payload["data"],
                        file_name=payload["filename"],
                        mime=payload["mime"],
                        key="export_download_button",
                    )

                st.markdown("</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("### 📜 Historial de exportaciones")
        history = st.session_state.get("export_history", [])
        if history:
            hist_df = pd.DataFrame(history)
            ordered_cols = [c for c in ["timestamp", "plan", "format", "safety"] if c in hist_df.columns]
            if ordered_cols:
                hist_df = hist_df[ordered_cols]
            st.markdown(
                "<div class='history-table'>" + hist_df.to_html(index=False, escape=False) + "</div>",
                unsafe_allow_html=True,
            )
        else:
            st.caption("Aún no generaste exportaciones en esta sesión.")
