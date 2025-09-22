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

