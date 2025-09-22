# --- path guard para Streamlit Cloud ---
import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# ---------------------------------------

from datetime import datetime

import altair as alt
import pandas as pd
import streamlit as st

from app.modules.generator import generate_candidates
from app.modules.io import load_waste_df, load_process_df
from app.modules.ml_models import MODEL_REGISTRY
from app.modules.process_planner import choose_process
from app.modules.safety import check_safety, safety_badge
from app.modules.ui_blocks import inject_css

st.set_page_config(page_title="Rex-AI • Generador", page_icon="🤖", layout="wide")

inject_css()

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
    .opt-card {border-radius:18px; padding:18px 20px; background: rgba(13,17,23,0.65); border:1px solid rgba(148,163,184,0.2);}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero-gen">
      <h1>🤖 Generador asistido por IA</h1>
      <p>Rex-AI explora combinaciones de residuos NASA, optimiza parámetros con Ax/BoTorch y explica cada predicción con bandas de confianza e importancias de features.</p>
      <div class="chipline">
        <span>Pasos guiados</span>
        <span>RandomForest + XGBoost + TabTransformer</span>
        <span>Confianza 95%</span>
        <span>Comparación heurística vs IA</span>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

layout = st.container()

with layout:
    target = st.session_state.get("target")
    if not target:
        st.warning("Configura primero el objetivo en **2 · Target Designer** para habilitar el generador.")
        st.stop()

    waste_df = load_waste_df()
    proc_df = load_process_df()
    proc_filtered = choose_process(
        target["name"], proc_df,
        scenario=target.get("scenario"),
        crew_time_low=target.get("crew_time_low", False)
    )
    if proc_filtered is None or proc_filtered.empty:
        proc_filtered = proc_df.copy()

    col_control, col_ai = st.columns([1.3, 0.9])
    with col_control:
        st.markdown("### 🎛️ Configuración")
        n_candidates = st.slider("Recetas a explorar", 3, 12, 6)
        opt_evals = st.slider("Iteraciones de optimización (Ax/BoTorch)", 0, 60, 18,
                               help="Rex-AI ejecuta un loop bayesiano para mejorar score sin violar límites de recursos.")
        crew_low = target.get("crew_time_low", False)
        st.caption("Los resultados privilegian %s" % ("tiempo de tripulación" if crew_low else "un balance general"))
        run = st.button("Generar recomendaciones", type="primary", use_container_width=True)
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
        if MODEL_REGISTRY.metadata.get("random_forest"):
            rf_metrics = MODEL_REGISTRY.metadata["random_forest"].get("metrics", {})
            overall = rf_metrics.get("overall", {})
            if overall:
                st.caption(f"MAE promedio: {overall.get('mae', '—'):.3f} · RMSE: {overall.get('rmse', '—'):.3f} · R²: {overall.get('r2', '—'):.3f}")

    if run:
        result = generate_candidates(
            waste_df,
            proc_filtered,
            target,
            n=n_candidates,
            crew_time_low=target.get("crew_time_low", False),
            optimizer_evals=opt_evals,
        )
        if isinstance(result, tuple):
            cands, history = result
    if isinstance(result, tuple):
        cands, history = result
    else:
        cands, history = result, pd.DataFrame()
    st.session_state["candidates"] = cands
    st.session_state["optimizer_history"] = history

st.markdown('<div class="hr-micro"></div>', unsafe_allow_html=True)

# -------------------- Si no hay candidatos aún --------------------
cands = st.session_state.get("candidates", [])
history_df = st.session_state.get("optimizer_history", pd.DataFrame())
if not cands:
    st.info("Todavía no hay candidatos. Configurá el número y presioná **Generar opciones**. "
            "Recomendación: asegurate de que tu inventario tenga pouches, espumas, EVA/CTB, textiles o nitrilo; "
            "y que el catálogo incluya P02/P03/P04.")
    with st.expander("¿Qué hace el generador (en criollo)?", expanded=False):
        st.markdown("""
- **Mira tus residuos** (con foco en los problemáticos de NASA).
- **Elige un proceso** coherente (laminar, sinter con regolito, reconfigurar CTB, etc.).
- **Predice** propiedades y recursos de la receta.
- **Puntúa** balanceando objetivos y costos.
- **Muestra trazabilidad** para que se vea qué basura se valorizó.
""")
    st.stop()

# -------------------- Render de candidatos con UX explicativa --------------------
def _res_bar(current: float, limit: float) -> float:
    if limit is None or float(limit) <= 0:
        return 0.0
    return max(0.0, min(1.0, current/float(limit)))

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
        chart_data = valid_hist.set_index("iteration")["hypervolume"].to_frame()
        chart_data["dominancia"] = valid_hist.set_index("iteration")["dominance_ratio"]
        st.line_chart(chart_data)

st.subheader("Resultados del generador")
st.caption("Cada ‘Opción’ es una combinación concreta de residuos + proceso, con predicción de propiedades y consumo de recursos. "
           "Usá los expanders para ver detalles y trazabilidad NASA.")

for i, c in enumerate(cands):
    p = c["props"]
    header = f"Opción {i+1} — Score {c['score']} — Proceso {c['process_id']} {c['process_name']}"
    with st.expander(header, expanded=(i == 0)):
        # Línea de badges
        badges = []
        if c.get("regolith_pct", 0) > 0:
            badges.append("⛰️ ISRU: +MGS-1")
        # Heurística “problemático presente”
        src_cats = " ".join(map(str, c.get("source_categories", []))).lower()
        src_flags = " ".join(map(str, c.get("source_flags", []))).lower()
        problem_present = any([
            "pouches" in src_cats, "multilayer" in src_flags,
            "foam" in src_cats, "ctb" in src_flags, "eva" in src_cats,
            "nitrile" in src_cats, "wipe" in src_flags
        ])
        if problem_present:
            badges.append("♻️ Valorización de problemáticos")
        if badges:
            st.markdown(" ".join([f'<span class="badge">{b}</span>' for b in badges]), unsafe_allow_html=True)

        # Resumen técnico
        colA, colB = st.columns([1.1, 1])
        with colA:
            st.markdown("**🧪 Materiales**")
            st.write(", ".join(c["materials"]))
            st.markdown("**⚖️ Pesos en mezcla**")
            st.write(c["weights"])

            st.markdown("**🔬 Predicción (demo)**")
            colA1, colA2, colA3 = st.columns(3)
            colA1.metric("Rigidez", f"{p.rigidity:.2f}")
            colA2.metric("Estanqueidad", f"{p.tightness:.2f}")
            colA3.metric("Masa final", f"{p.mass_final_kg:.2f} kg")
            source = getattr(p, "source", "heuristic")
            if source.startswith("rexai"):
                meta = c.get("ml_prediction", {}).get("metadata", {})
                trained_at = meta.get("trained_at", "?")
                latent = c.get("latent_vector", [])
                latent_note = "" if not latent else f" · Vector latente {len(latent)}D Rex-AI"
                st.caption(f"Predicción por modelo ML (**{source}**, entrenado {trained_at}){latent_note}.")
            else:
                st.caption("Predicción heurística basada en reglas.")

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

        # Seguridad (badges)
        st.markdown("**🛡️ Seguridad**")
        flags = check_safety(c["materials"], c["process_name"], c["process_id"])
        badge = safety_badge(flags)
        if badge["level"] == "Riesgo":
            pill("Riesgo", "risk"); st.warning(badge["detail"])
        else:
            cands, history = result, pd.DataFrame()
        st.session_state["candidates"] = cands
        st.session_state["optimizer_history"] = history

    candidates = st.session_state.get("candidates", [])
    history_df = st.session_state.get("optimizer_history", pd.DataFrame())

    if not candidates:
        st.info("Sin recetas todavía. Ajustá los controles y presioná **Generar recomendaciones**.")
    else:
        st.markdown("### 🔍 Recomendaciones con trazabilidad IA")
        for idx, cand in enumerate(candidates, start=1):
            props = cand["props"]
            heur = cand.get("heuristic_props", props)
            ci = cand.get("confidence_interval") or {}
            uncertainty = cand.get("uncertainty") or {}
            comparisons = cand.get("model_variants") or {}
            metadata = cand.get("ml_prediction", {}).get("metadata", {})
            importance = cand.get("feature_importance") or []
            history_label = metadata.get("trained_at", "—")
            with st.container():
                st.markdown("""
                    <div class="candidate">
                      <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:8px;">
                        <h4>Opción #{idx} · Score {score:.3f}</h4>
                        <span class="badge-ai">Modelo: {model} · Entrenado: {trained}</span>
                      </div>
                """.format(
                    idx=idx,
                    score=cand["score"],
                    model=cand.get("prediction_source", "heuristic"),
                    trained=history_label,
                ), unsafe_allow_html=True)

                grid = st.columns(5)
                labels = [
                    ("Rigidez", props.rigidity, heur.rigidity, ci.get("rigidez")),
                    ("Estanqueidad", props.tightness, heur.tightness, ci.get("estanqueidad")),
                    ("Energía (kWh)", props.energy_kwh, heur.energy_kwh, ci.get("energy_kwh")),
                    ("Agua (L)", props.water_l, heur.water_l, ci.get("water_l")),
                    ("Crew (min)", props.crew_min, heur.crew_min, ci.get("crew_min")),
                ]
                for col, (label, val_ml, val_h, interval) in zip(grid, labels):
                    delta = val_ml - val_h
                    with col:
                        st.markdown(f"<div class='candidate-grid'><div><strong>{val_ml:.3f}</strong><span>{label}</span></div></div>", unsafe_allow_html=True)
                        st.markdown(f"<div class='delta'>Heurística: {val_h:.3f} · Δ {delta:+.3f}</div>", unsafe_allow_html=True)
                        if interval:
                            st.markdown(f"<div class='confidence'>CI 95% [{interval[0]:.3f}, {interval[1]:.3f}]</div>", unsafe_allow_html=True)
                if uncertainty:
                    st.caption("Desviación (modelo): " + ", ".join(f"{k} {v:.3f}" for k, v in uncertainty.items()))

                if importance:
                    df_imp = pd.DataFrame(importance, columns=["feature", "value"]).head(6)
                    chart = alt.Chart(df_imp).mark_bar(color="#38bdf8").encode(
                        x=alt.X("value", title="Contribución"),
                        y=alt.Y("feature", sort="-x", title="Feature"),
                    ).properties(height=180)
                    st.altair_chart(chart, use_container_width=True)

                if comparisons:
                    st.caption("Modelos alternativos:")
                    comp_df = pd.DataFrame(comparisons).T
                    st.dataframe(comp_df.style.format("{:.3f}"), use_container_width=True)

                st.caption("Materiales: " + ", ".join(cand["materials"]))
                st.caption("Fuente NASA IDs: " + ", ".join(cand.get("source_ids", [])))

                col_select, col_flags = st.columns([0.3, 0.7])
                with col_select:
                    if st.button(f"Seleccionar opción #{idx}", key=f"select_{idx}"):
                        flags = check_safety(cand["materials"], cand["process_name"], cand["process_id"])
                        badge = safety_badge(flags)
                        st.session_state["selected"] = {"data": cand, "safety": badge}
                        st.success("Receta enviada a Resultados.")
                with col_flags:
                    flags = check_safety(cand["materials"], cand["process_name"], cand["process_id"])
                    badge = safety_badge(flags)
                    st.info(f"Seguridad: {badge['level']} · {badge['detail']}")
                st.markdown("</div>", unsafe_allow_html=True)

    if history_df is not None and not history_df.empty:
        st.markdown("### 📈 Evolución del optimizador bayesiano")
        history_df = history_df.fillna(method="ffill")
        chart = alt.Chart(history_df).transform_fold(
            ["hypervolume", "dominance_ratio"],
            as_=["metric", "value"]
        ).mark_line().encode(
            x=alt.X("iteration:Q", title="Iteración"),
            y=alt.Y("value:Q", title="Valor"),
            color="metric:N",
            tooltip=["iteration", "metric", alt.Tooltip("value", format=".3f")],
        ).properties(height=280)
        st.altair_chart(chart, use_container_width=True)

=======

    candidates = st.session_state.get("candidates", [])
    history_df = st.session_state.get("optimizer_history", pd.DataFrame())

    if not candidates:
        st.info("Sin recetas todavía. Ajustá los controles y presioná **Generar recomendaciones**.")
    else:
        st.markdown("### 🔍 Recomendaciones con trazabilidad IA")
        for idx, cand in enumerate(candidates, start=1):
            props = cand["props"]
            heur = cand.get("heuristic_props", props)
            ci = cand.get("confidence_interval") or {}
            uncertainty = cand.get("uncertainty") or {}
            comparisons = cand.get("model_variants") or {}
            metadata = cand.get("ml_prediction", {}).get("metadata", {})
            importance = cand.get("feature_importance") or []
            history_label = metadata.get("trained_at", "—")
            with st.container():
                st.markdown("""
                    <div class="candidate">
                      <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:8px;">
                        <h4>Opción #{idx} · Score {score:.3f}</h4>
                        <span class="badge-ai">Modelo: {model} · Entrenado: {trained}</span>
                      </div>
                """.format(
                    idx=idx,
                    score=cand["score"],
                    model=cand.get("prediction_source", "heuristic"),
                    trained=history_label,
                ), unsafe_allow_html=True)

                grid = st.columns(5)
                labels = [
                    ("Rigidez", props.rigidity, heur.rigidity, ci.get("rigidez")),
                    ("Estanqueidad", props.tightness, heur.tightness, ci.get("estanqueidad")),
                    ("Energía (kWh)", props.energy_kwh, heur.energy_kwh, ci.get("energy_kwh")),
                    ("Agua (L)", props.water_l, heur.water_l, ci.get("water_l")),
                    ("Crew (min)", props.crew_min, heur.crew_min, ci.get("crew_min")),
                ]
                for col, (label, val_ml, val_h, interval) in zip(grid, labels):
                    delta = val_ml - val_h
                    with col:
                        st.markdown(f"<div class='candidate-grid'><div><strong>{val_ml:.3f}</strong><span>{label}</span></div></div>", unsafe_allow_html=True)
                        st.markdown(f"<div class='delta'>Heurística: {val_h:.3f} · Δ {delta:+.3f}</div>", unsafe_allow_html=True)
                        if interval:
                            st.markdown(f"<div class='confidence'>CI 95% [{interval[0]:.3f}, {interval[1]:.3f}]</div>", unsafe_allow_html=True)
                if uncertainty:
                    st.caption("Desviación (modelo): " + ", ".join(f"{k} {v:.3f}" for k, v in uncertainty.items()))

                if importance:
                    df_imp = pd.DataFrame(importance, columns=["feature", "value"]).head(6)
                    chart = alt.Chart(df_imp).mark_bar(color="#38bdf8").encode(
                        x=alt.X("value", title="Contribución"),
                        y=alt.Y("feature", sort="-x", title="Feature"),
                    ).properties(height=180)
                    st.altair_chart(chart, use_container_width=True)

                if comparisons:
                    st.caption("Modelos alternativos:")
                    comp_df = pd.DataFrame(comparisons).T
                    st.dataframe(comp_df.style.format("{:.3f}"), use_container_width=True)

                st.caption("Materiales: " + ", ".join(cand["materials"]))
                st.caption("Fuente NASA IDs: " + ", ".join(cand.get("source_ids", [])))

                col_select, col_flags = st.columns([0.3, 0.7])
                with col_select:
                    if st.button(f"Seleccionar opción #{idx}", key=f"select_{idx}"):
                        flags = check_safety(cand["materials"], cand["process_name"], cand["process_id"])
                        badge = safety_badge(flags)
                        st.session_state["selected"] = {"data": cand, "safety": badge}
                        st.success("Receta enviada a Resultados.")
                with col_flags:
                    flags = check_safety(cand["materials"], cand["process_name"], cand["process_id"])
                    badge = safety_badge(flags)
                    st.info(f"Seguridad: {badge['level']} · {badge['detail']}")
                st.markdown("</div>", unsafe_allow_html=True)

    if history_df is not None and not history_df.empty:
        st.markdown("### 📈 Evolución del optimizador bayesiano")
        history_df = history_df.fillna(method="ffill")
        chart = alt.Chart(history_df).transform_fold(
            ["hypervolume", "dominance_ratio"],
            as_=["metric", "value"]
        ).mark_line().encode(
            x=alt.X("iteration:Q", title="Iteración"),
            y=alt.Y("value:Q", title="Valor"),
            color="metric:N",
            tooltip=["iteration", "metric", alt.Tooltip("value", format=".3f")],
        ).properties(height=280)
        st.altair_chart(chart, use_container_width=True)
        # Botón de selección
        if st.button(f"✅ Seleccionar Opción {i+1}", key=f"pick_{i}"):
            st.session_state["selected"] = {"data": c, "safety": badge}
            st.success("Opción seleccionada. Abrí **4) Resultados**, **5) Comparar & Explicar** o **6) Pareto & Export**.")

        # Explicación en criollo (mini narrativa) — evitar anidar expanders
pop = st.popover("🧠 ¿Por qué esta receta pinta bien? (explicación en criollo)")
with pop:
    bullets = []
    bullets.append("• Sumamos puntos si **rigidez/estanqueidad** se acercan a lo que pediste.")
    bullets.append("• Restamos si se pasa en **agua/energía/tiempo** de la tripulación.")
    if 'problem_present' in locals() and problem_present:
        bullets.append("• Bonus porque esta opción **se come basura problemática** (¡la que más molesta en la base!).")
    if 'c' in locals() and c.get('regolith_pct', 0) > 0:
        bullets.append("• Usa **MGS-1** (regolito) como carga mineral → eso es ISRU puro: menos dependencia de la Tierra.")
    st.markdown("\n".join(bullets))

# -------------------- Pie de guía / glosario --------------------
st.markdown('<div class="hr-micro"></div>', unsafe_allow_html=True)
with st.expander("📚 Glosario ultra rápido", expanded=False):
    st.markdown("""
- **ISRU**: *In-Situ Resource Utilization*. Usar recursos del lugar (en Marte, el **regolito** MGS-1).
- **P02 – Press & Heat Lamination**: “plancha” y “fusiona” multicapa para dar forma.
- **P03 – Sinter with MGS-1**: mezcla con regolito y sinteriza → piezas rígidas, útiles para interiores.
- **P04 – CTB Reconfig**: reusar/transformar bolsas EVA/CTB con herrajes.
- **Score**: cuánto “cierra” la opción según tu objetivo y límites de recursos/tiempo.
""")
st.info("Sugerencia: generá varias opciones y pasá a **4) Resultados**, **5) Comparar** y **6) Pareto & Export** para cerrar tu plan.")

