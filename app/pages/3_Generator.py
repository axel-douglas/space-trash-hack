import app  # noqa: F401

from datetime import datetime

import altair as alt
import pandas as pd
import streamlit as st

from app.modules.generator import generate_candidates
from app.modules.io import load_waste_df, load_process_df  # si tu IO usa load_process_catalog, cámbialo aquí
from app.modules.ml_models import MODEL_REGISTRY
from app.modules.process_planner import choose_process
from app.modules.safety import check_safety, safety_badge
from app.modules.ui_blocks import load_theme

st.set_page_config(page_title="Rex-AI • Generador", page_icon="🤖", layout="wide")

load_theme()

# ----------------------------- CSS local -----------------------------
st.markdown(
    """
    <style>
    .layout {display:flex; flex-direction:column; gap:1.6rem;}
    .pane {background: rgba(15,18,26,0.75); border:1px solid rgba(148,163,184,0.18); padding:22px 24px; border-radius:20px;}
    .pane h3 {margin-bottom:0.6rem;}
    .hero-gen {padding:28px 30px; border-radius:26px; background: linear-gradient(135deg, rgba(59,130,246,0.18), rgba(14,165,233,0.08)); border:1px solid rgba(59,130,246,0.32);}
    .hero-gen h1 {margin-bottom:0.4rem;}
    .hero-gen p {margin:0; opacity:0.82; max-width:760px;}
    .chipline {display:flex; gap:10px; margin-top:14px; flex-wrap:wrap;}
    .chipline span {padding:5px 12px; border-radius:999px; border:1px solid rgba(148,163,184,0.26); font-size:0.8rem; opacity:0.85;}
    .candidate {border-radius:20px; border:1px solid rgba(148,163,184,0.2); padding:20px 22px; margin-bottom:16px; background: rgba(13,17,23,0.7);}
    .candidate h4 {margin-bottom:0.4rem;}
    .candidate-grid {display:grid; grid-template-columns: repeat(auto-fit,minmax(180px,1fr)); gap:12px; margin:12px 0;}
    .candidate-grid div {background:rgba(148,163,184,0.12); border-radius:14px; padding:12px;}
    .candidate-grid strong {display:block; font-size:1.2rem;}
    .confidence {font-size:0.86rem; opacity:0.8; margin-top:4px;}
    .badge-ai {display:inline-flex; align-items:center; gap:6px; padding:4px 10px; border-radius:999px; border:1px solid rgba(148,163,184,0.25); font-size:0.78rem;}
    .delta {font-size:0.82rem; opacity:0.8;}
    .hr-micro {height:1px; background:rgba(148,163,184,0.25); margin:14px 0;}
    .badge {padding:4px 10px; border-radius:999px; font-size:0.78rem; background:rgba(96,165,250,0.16); color:#e6eefc; margin-right:6px; border:1px solid rgba(148,163,184,0.25);}
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------- Helpers -----------------------------
TARGET_DISPLAY = {
    "rigidez": "Rigidez",
    "estanqueidad": "Estanqueidad",
    "energy_kwh": "Energía (kWh)",
    "water_l": "Agua (L)",
    "crew_min": "Crew (min)",
}


def _format_label_summary(summary: dict[str, dict[str, float]] | None) -> str:
    if not summary:
        return ""

    parts: list[str] = []
    for source, stats in summary.items():
        if not isinstance(stats, dict):
            parts.append(str(source))
            continue

        label = str(source)
        count = stats.get("count")
        mean_weight = stats.get("mean_weight")

        fragment = label
        try:
            if count is not None:
                fragment = f"{label}×{int(count)}"
        except (TypeError, ValueError):
            fragment = label

        try:
            if mean_weight is not None:
                fragment = f"{fragment} (w≈{float(mean_weight):.2f})"
        except (TypeError, ValueError):
            pass

        parts.append(fragment)

    return " · ".join(parts)

# ----------------------------- Hero -----------------------------
st.markdown(
    """
    <div class="hero-gen">
      <h1>🤖 Generador asistido por IA</h1>
      <p>Rex-AI explora combinaciones de residuos NASA, optimiza parámetros y explica cada predicción con bandas de confianza e importancias de features.</p>
      <div class="chipline">
        <span>RandomForest + XGBoost (alternativo)</span>
        <span>Confianza 95%</span>
        <span>Comparación heurística vs IA</span>
        <span>Trazabilidad NASA + MGS-1</span>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ----------------------------- Pre-condición: target -----------------------------
target = st.session_state.get("target")
if not target:
    st.warning("Configura primero el objetivo en **2 · Target Designer** para habilitar el generador.")
    st.stop()

# ----------------------------- Datos base -----------------------------
waste_df = load_waste_df()
proc_df = load_process_df()
proc_filtered = choose_process(
    target["name"], proc_df,
    scenario=target.get("scenario"),
    crew_time_low=target.get("crew_time_low", False)
)
if proc_filtered is None or proc_filtered.empty:
    proc_filtered = proc_df.copy()

# ----------------------------- Panel de control + IA -----------------------------
col_control, col_ai = st.columns([1.3, 0.9])
with col_control:
    st.markdown("### 🎛️ Configuración")
    stored_mode = st.session_state.get("prediction_mode", "Modo Rex-AI (ML)")
    mode = st.radio(
        "Motor de predicción",
        ("Modo Rex-AI (ML)", "Modo heurístico"),
        index=0 if stored_mode == "Modo Rex-AI (ML)" else 1,
        help="Usá Rex-AI para predicciones ML o quedate con la estimación heurística reproducible.",
    )
    st.session_state["prediction_mode"] = mode
    use_ml = mode == "Modo Rex-AI (ML)"
    n_candidates = st.slider("Recetas a explorar", 3, 12, 6)
    opt_evals = st.slider(
        "Iteraciones de optimización (Ax/BoTorch)",
        0, 60, 18,
        help="Loop bayesiano para maximizar score sin violar límites de recursos."
    )
    crew_low = target.get("crew_time_low", False)
    st.caption("Los resultados privilegian %s" % ("tiempo de tripulación" if crew_low else "un balance general"))
    run = st.button("Generar recomendaciones", type="primary", use_container_width=True)
    if not use_ml:
        st.info("Modo heurístico activo: las métricas se basan en reglas físicas y no en ML.")

with col_ai:
    st.markdown("### 🧠 Modelo Rex-AI")
    trained_at = MODEL_REGISTRY.metadata.get("trained_at", "—")
    n_samples = MODEL_REGISTRY.metadata.get("n_samples", "—")
    top_features = MODEL_REGISTRY.feature_importance_avg[:5]
    if top_features:
        df_feat = pd.DataFrame(top_features, columns=["feature", "weight"])
        chart = alt.Chart(df_feat).mark_bar(color="#60a5fa").encode(
            x=alt.X("weight", title="Importancia promedio"),
            y=alt.Y("feature", sort="-x", title="Feature"),
            tooltip=["feature", alt.Tooltip("weight", format=".3f")],
        ).properties(height=180)
        st.altair_chart(chart, use_container_width=True)
    st.caption(f"Entrenado: {trained_at} · Muestras: {n_samples} · Features: {len(MODEL_REGISTRY.feature_names)}")
    if MODEL_REGISTRY.metadata.get("random_forest", {}).get("metrics", {}).get("overall"):
        overall = MODEL_REGISTRY.metadata["random_forest"]["metrics"]["overall"]
        try:
            st.caption(f"MAE promedio: {overall.get('mae', float('nan')):.3f} · RMSE: {overall.get('rmse', float('nan')):.3f} · R²: {overall.get('r2', float('nan')):.3f}")
        except Exception:
            pass
    label_summary_text = MODEL_REGISTRY.label_distribution_label()
    if label_summary_text and label_summary_text != "—":
        st.caption(f"Fuentes de labels: {label_summary_text}")

# ----------------------------- Generación -----------------------------
if run:
    result = generate_candidates(
        waste_df,
        proc_filtered,
        target,
        n=n_candidates,
        crew_time_low=target.get("crew_time_low", False),
        optimizer_evals=opt_evals,
        use_ml=use_ml,
    )
    if isinstance(result, tuple):
        cands, history = result
    else:
        cands, history = result, pd.DataFrame()
    # normalizar a lista de dicts
    if isinstance(cands, pd.DataFrame):
        cands = cands.to_dict("records")
    elif isinstance(cands, dict):
        cands = [cands]
    else:
        cands = [dict(c) for c in cands]
    history_df = history if isinstance(history, pd.DataFrame) else pd.DataFrame()
    st.session_state["candidates"] = cands
    st.session_state["optimizer_history"] = history_df

# ----------------------------- Si no hay candidatos aún -----------------------------
st.markdown('<div class="hr-micro"></div>', unsafe_allow_html=True)
cands = st.session_state.get("candidates", [])
history_df = st.session_state.get("optimizer_history", pd.DataFrame())

if not cands:
    st.info(
        "Todavía no hay candidatos. Configurá los controles y presioná **Generar recomendaciones**. "
        "Asegurate de que el inventario tenga pouches, espumas, EVA/CTB, textiles o nitrilo; "
        "y que el catálogo incluya P02/P03/P04."
    )
    with st.expander("¿Qué hace el generador (en criollo)?", expanded=False):
        st.markdown(
            "- **Mira tus residuos** (enfocado en los problemáticos de NASA).\n"
            "- **Elige un proceso** coherente (laminar, sinter con regolito, reconfigurar CTB, etc.).\n"
            "- **Predice** propiedades y recursos de la receta.\n"
            "- **Puntúa** balanceando objetivos y costos.\n"
            "- **Muestra trazabilidad** para ver qué basura se valorizó."
        )
    st.stop()

# ----------------------------- Helpers -----------------------------
def _res_bar(current: float, limit: float) -> float:
    if limit is None or float(limit) <= 0:
        return 0.0
    return max(0.0, min(1.0, current/float(limit)))

# ----------------------------- Historial del optimizador -----------------------------
if isinstance(history_df, pd.DataFrame) and not history_df.empty:
    st.subheader("Convergencia del optimizador")
    st.caption("Seguimiento rápido de hipervolumen y porcentaje de soluciones dominadas.")
    valid_hist = history_df.dropna(subset=["hypervolume"])
    if not valid_hist.empty:
        last = valid_hist.iloc[-1]
        m1, m2, m3 = st.columns([1, 1, 1])
        m1.metric("Hipervolumen", f"{last['hypervolume']:.3f}")
        m2.metric("Dominancia", f"{last['dominance_ratio']*100:.1f}%")
        m3.metric("Tamaño Pareto", f"{int(last['pareto_size'])}")
        chart_data = valid_hist.set_index("iteration")[["hypervolume", "dominance_ratio"]]
        st.line_chart(chart_data)

# ----------------------------- Resumen de ranking -----------------------------
summary_rows: list[dict[str, object]] = []
for idx, cand in enumerate(cands, start=1):
    props = cand.get("props")
    if props is None:
        continue
    aux = cand.get("auxiliary") or {}
    summary_rows.append(
        {
            "Rank": idx,
            "Score": cand.get("score"),
            "Proceso": f"{cand.get('process_id', '')} · {cand.get('process_name', '')}",
            "Rigidez": getattr(props, "rigidity", float("nan")),
            "Estanqueidad": getattr(props, "tightness", float("nan")),
            "Energía (kWh)": getattr(props, "energy_kwh", float("nan")),
            "Agua (L)": getattr(props, "water_l", float("nan")),
            "Crew (min)": getattr(props, "crew_min", float("nan")),
            "Seal": "✅" if aux.get("passes_seal", True) else "⚠️",
            "Riesgo": aux.get("process_risk_label", "—"),
        }
    )

if summary_rows:
    st.subheader("Ranking multiobjetivo")
    st.caption("Ordenado por score total, con sellado y riesgo resumidos.")
    summary_df = pd.DataFrame(summary_rows)
    st.dataframe(summary_df, hide_index=True, use_container_width=True)

# ----------------------------- Render de candidatos -----------------------------
st.subheader("Resultados del generador")
st.caption(
    "Cada ‘Opción’ es una combinación concreta de residuos + proceso, con predicción de propiedades y consumo de recursos. "
    "Usá los expanders para ver detalles y trazabilidad NASA."
)

for i, c in enumerate(cands):
    p = c["props"]
    header = f"Opción {i+1} — Score {c['score']} — Proceso {c['process_id']} {c['process_name']}"
    with st.expander(header, expanded=(i == 0)):
        # Badges
        badges = []
        if c.get("regolith_pct", 0) > 0:
            badges.append("⛰️ ISRU: +MGS-1")
        src_cats = " ".join(map(str, c.get("source_categories", []))).lower()
        src_flags = " ".join(map(str, c.get("source_flags", []))).lower()
        problem_present = any([
            "pouches" in src_cats, "multilayer" in src_flags,
            "foam" in src_cats, "ctb" in src_flags, "eva" in src_cats,
            "nitrile" in src_cats, "wipe" in src_flags
        ])
        if problem_present:
            badges.append("♻️ Valorización de problemáticos")
        aux = c.get("auxiliary") or {}
        if aux:
            if aux.get("passes_seal") is False:
                badges.append("⚠️ Revisar estanqueidad")
            elif aux.get("passes_seal"):
                badges.append("✅ Sellado OK")
            risk_label = aux.get("process_risk_label")
            if risk_label:
                badges.append(f"🏷️ Riesgo {risk_label}")
        if badges:
            st.markdown(" ".join([f'<span class="badge">{b}</span>' for b in badges]), unsafe_allow_html=True)

        pred_error = c.get("prediction_error")
        if pred_error:
            st.error(f"Predicción ML no disponible: {pred_error}")

        # Resumen técnico
        colA, colB = st.columns([1.1, 1])
        with colA:
            st.markdown("**🧪 Materiales**")
            st.write(", ".join(c["materials"]))
            st.markdown("**⚖️ Pesos en mezcla**")
            st.write(c["weights"])

            st.markdown("**🔬 Predicción**" if not pred_error else "**🔬 Estimación heurística**")
            colA1, colA2, colA3 = st.columns(3)
            if pred_error:
                colA1.write(f"Rigidez: {p.rigidity:.2f}")
                colA2.write(f"Estanqueidad: {p.tightness:.2f}")
                colA3.write(f"Masa final: {p.mass_final_kg:.2f} kg")
            else:
                colA1.metric("Rigidez", f"{p.rigidity:.2f}")
                colA2.metric("Estanqueidad", f"{p.tightness:.2f}")
                colA3.metric("Masa final", f"{p.mass_final_kg:.2f} kg")
            src = c.get("prediction_source", "heuristic")
            meta_payload = {}
            if isinstance(c.get("ml_prediction"), dict):
                meta_payload = c["ml_prediction"].get("metadata", {}) or {}
            if pred_error:
                st.caption("Fallback heurístico mostrado por indisponibilidad del modelo.")
            elif str(src).startswith("rexai"):
                t_at = meta_payload.get("trained_at", "?")
                latent = c.get("latent_vector", [])
                latent_note = "" if not latent else f" · Vector latente {len(latent)}D"
                st.caption(f"Predicción por modelo ML (**{src}**, entrenado {t_at}){latent_note}.")
                summary_text = _format_label_summary(
                    meta_payload.get("label_summary") or MODEL_REGISTRY.label_summary
                )
                if summary_text:
                    st.caption(f"Dataset Rex-AI: {summary_text}")
            else:
                st.caption("Predicción heurística basada en reglas.")

            ci = c.get("confidence_interval") or {}
            unc = c.get("uncertainty") or {}
            if ci:
                rows: list[dict[str, float | str]] = []
                for key, bounds in ci.items():
                    label = TARGET_DISPLAY.get(key, key)
                    try:
                        lo_val, hi_val = float(bounds[0]), float(bounds[1])
                    except (TypeError, ValueError, IndexError):
                        lo_val, hi_val = float("nan"), float("nan")
                    row: dict[str, float | str] = {
                        "Variable": label,
                        "Lo": lo_val,
                        "Hi": hi_val,
                    }
                    sigma_val = unc.get(key)
                    if sigma_val is not None:
                        try:
                            row["σ (std)"] = float(sigma_val)
                        except (TypeError, ValueError):
                            row["σ (std)"] = float("nan")
                    rows.append(row)
                if rows and not pred_error:
                    st.markdown("**📉 Intervalos de confianza (95%)**")
                    ci_df = pd.DataFrame(rows)
                    st.dataframe(ci_df, hide_index=True, use_container_width=True)

            feature_imp = c.get("feature_importance") or []
            if feature_imp and not pred_error:
                st.markdown("**🪄 Features que más influyen**")
                fi_df = pd.DataFrame(feature_imp, columns=["feature", "impact"])
                chart = alt.Chart(fi_df).mark_bar(color="#60a5fa").encode(
                    x=alt.X("impact", title="Impacto relativo"),
                    y=alt.Y("feature", sort="-x", title="Feature"),
                    tooltip=["feature", alt.Tooltip("impact", format=".3f")],
                ).properties(height=180)
                st.altair_chart(chart, use_container_width=True)

        with colB:
            st.markdown("**🔧 Proceso**")
            st.write(f"{c['process_id']} — {c['process_name']}")
            st.markdown("**📉 Recursos estimados**")
            colB1, colB2, colB3 = st.columns([1,1,1])
            colB1.write("Energía (kWh)")
            colB1.progress(_res_bar(p.energy_kwh, target["max_energy_kwh"]))
            colB1.caption(f"{p.energy_kwh:.2f} / {target['max_energy_kwh']}")

            colB2.write("Agua (L)")
            colB2.progress(_res_bar(p.water_l, target["max_water_l"]))
            colB2.caption(f"{p.water_l:.2f} / {target['max_water_l']}")

            colB3.write("Crew (min)")
            colB3.progress(_res_bar(p.crew_min, target["max_crew_min"]))
            colB3.caption(f"{p.crew_min:.0f} / {target['max_crew_min']}")

        st.markdown('<div class="hr-micro"></div>', unsafe_allow_html=True)

        # Trazabilidad NASA
        st.markdown("**🛰️ Trazabilidad NASA**")
        st.write("IDs usados:", ", ".join(c.get("source_ids", [])) or "—")
        st.write("Categorías:", ", ".join(map(str, c.get("source_categories", []))) or "—")
        st.write("Flags:", ", ".join(map(str, c.get("source_flags", []))) or "—")
        if c.get("regolith_pct", 0) > 0:
            st.write(f"**MGS-1 agregado:** {c['regolith_pct']*100:.0f}%")

        # Features resumen (si los hay)
        feat = c.get("features", {})
        if feat:
            feat_summary = {
                "Masa total (kg)": feat.get("total_mass_kg"),
                "Densidad (kg/m³)": feat.get("density_kg_m3"),
                "Humedad": feat.get("moisture_frac"),
                "Dificultad": feat.get("difficulty_index"),
                "Recupero gas": feat.get("gas_recovery_index"),
                "Reuso logístico": feat.get("logistics_reuse_index"),
            }
            feat_df = pd.DataFrame([feat_summary])
            st.markdown("**Features NASA/ML (alimentan la IA)**")
            st.dataframe(feat_df, hide_index=True, use_container_width=True)
            if feat.get("latent_vector"):
                st.caption("Latente Rex-AI incluido para análisis generativo.")

        breakdown = c.get("score_breakdown") or {}
        contribs = breakdown.get("contributions") or {}
        penalties = breakdown.get("penalties") or {}
        if contribs or penalties:
            st.markdown("**⚖️ Desglose del score**")
            col_sc1, col_sc2 = st.columns(2)
            if contribs:
                contrib_df = pd.DataFrame([
                    {"Factor": k, "+": float(v)} for k, v in contribs.items()
                ]).sort_values("+", ascending=False)
                col_sc1.dataframe(contrib_df, hide_index=True, use_container_width=True)
            if penalties:
                pen_df = pd.DataFrame([
                    {"Penalización": k, "-": float(v)} for k, v in penalties.items()
                ]).sort_values("-", ascending=False)
                col_sc2.dataframe(pen_df, hide_index=True, use_container_width=True)

        # Seguridad
        flags = check_safety(c["materials"], c["process_name"], c["process_id"])
        badge = safety_badge(flags)
        st.info(f"Seguridad: {badge['level']} · {badge['detail']}")

        # Botón de selección
        if st.button(f"✅ Seleccionar Opción {i+1}", key=f"pick_{i}"):
            st.session_state["selected"] = {"data": c, "safety": badge}
            st.success("Opción seleccionada. Abrí **4) Resultados**, **5) Comparar & Explicar** o **6) Pareto & Export**.")

# ----------------------------- Explicación en criollo (popover global) -----------------------------
top = cands[0] if cands else None
pop = st.popover("🧠 ¿Por qué estas recetas pintan bien? (explicación en criollo)")
with pop:
    bullets = []
    bullets.append("• Sumamos puntos si **rigidez/estanqueidad** se acercan a lo que pediste.")
    bullets.append("• Restamos si se pasa en **agua/energía/tiempo** de la tripulación.")
    if top:
        cats = " ".join(map(str, top.get("source_categories", []))).lower()
        flg = " ".join(map(str, top.get("source_flags", []))).lower()
        if any(k in cats or k in flg for k in ["pouches", "multilayer", "foam", "eva", "ctb", "nitrile", "wipe"]):
            bullets.append("• Bonus porque priorizamos **basura problemática** (la que más molesta en la base).")
        if top.get("regolith_pct", 0) > 0:
            bullets.append("• Usa **MGS-1** (regolito) como carga mineral → ISRU: menos dependencia de la Tierra.")
    st.markdown("\n".join(bullets))

# ----------------------------- Glosario -----------------------------
st.markdown('<div class="hr-micro"></div>', unsafe_allow_html=True)
with st.expander("📚 Glosario ultra rápido", expanded=False):
    st.markdown(
        "- **ISRU**: *In-Situ Resource Utilization*. Usar recursos del lugar (en Marte, el **regolito** MGS-1).\n"
        "- **P02 – Press & Heat Lamination**: “plancha” y “fusiona” multicapa para dar forma.\n"
        "- **P03 – Sinter with MGS-1**: mezcla con regolito y sinteriza → piezas rígidas.\n"
        "- **P04 – CTB Reconfig**: reusar/transformar bolsas EVA/CTB con herrajes.\n"
        "- **Score**: qué tanto ‘cierra’ la opción según objetivo y límites de recursos/tiempo."
    )
st.info("Tip: generá varias opciones y pasá a **4) Resultados**, **5) Comparar** y **6) Pareto & Export** para cerrar tu plan.")
