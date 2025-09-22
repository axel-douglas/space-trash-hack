# --- path guard para Streamlit Cloud ---
import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# ---------------------------------------

import altair as alt
import pandas as pd
import streamlit as st

from app.modules.explain import score_breakdown

st.set_page_config(page_title="Rex-AI • Resultados", page_icon="📊", layout="wide")

selected = st.session_state.get("selected")
target = st.session_state.get("target")
if not selected or not target:
    st.warning("Seleccioná una receta en **3 · Generador**.")
    st.stop()

cand = selected["data"]
props = cand["props"]
heur = cand.get("heuristic_props", props)
ci = cand.get("confidence_interval") or {}
uncertainty = cand.get("uncertainty") or {}
comparisons = cand.get("model_variants") or {}
importance = cand.get("feature_importance") or []
metadata = cand.get("ml_prediction", {}).get("metadata", {})
latent = cand.get("latent_vector", [])
regolith_pct = cand.get("regolith_pct", 0.0)
materials = cand.get("materials", [])
score = cand.get("score", 0.0)
safety = selected.get("safety", {"level": "—", "detail": ""})

st.markdown(
    """
    <style>
    .hero-res {padding:28px 30px; border-radius:26px; background: linear-gradient(135deg, rgba(20,184,166,0.18), rgba(14,165,233,0.08)); border:1px solid rgba(45,212,191,0.32);}
    .hero-res h1 {margin-bottom:0.2rem;}
    .hero-res p {margin:0; opacity:0.82; max-width:720px;}
    .metrics {display:grid; grid-template-columns: repeat(auto-fit,minmax(190px,1fr)); gap:14px; margin:18px 0;}
    .metrics div {background:rgba(13,17,23,0.68); border:1px solid rgba(148,163,184,0.22); border-radius:18px; padding:14px 16px;}
    .metrics span {display:block; font-size:0.8rem; opacity:0.7;}
    .metrics strong {display:block; font-size:1.35rem; margin-top:4px;}
    .delta {font-size:0.82rem; opacity:0.75; margin-top:4px;}
    .card {background:rgba(13,17,23,0.65); border:1px solid rgba(148,163,184,0.22); border-radius:20px; padding:20px 22px; margin-top:18px;}
    .badge {display:inline-flex; gap:6px; align-items:center; padding:5px 12px; border-radius:999px; border:1px solid rgba(148,163,184,0.28); font-size:0.78rem;}
    .chips {display:flex; flex-wrap:wrap; gap:8px; margin-top:8px;}
    .chips span {padding:4px 10px; border-radius:999px; border:1px solid rgba(148,163,184,0.25); font-size:0.78rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    f"""
    <div class="hero-res">
      <h1>📊 Resultado seleccionado · Score {score:.3f}</h1>
      <p>Proceso <strong>{cand['process_id']} · {cand['process_name']}</strong>. La IA Rex-AI proporciona predicciones con trazabilidad NASA, bandas de confianza y comparación contra heurísticas originales.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

cols = st.columns(5)
labels = [
    ("Rigidez", props.rigidity, heur.rigidity, ci.get("rigidez")),
    ("Estanqueidad", props.tightness, heur.tightness, ci.get("estanqueidad")),
    ("Energía (kWh)", props.energy_kwh, heur.energy_kwh, ci.get("energy_kwh")),
    ("Agua (L)", props.water_l, heur.water_l, ci.get("water_l")),
    ("Crew (min)", props.crew_min, heur.crew_min, ci.get("crew_min")),
]
for col, (label, val_ml, val_h, interval) in zip(cols, labels):
    with col:
        st.markdown("<div class='metrics'><div><span>{}</span><strong>{:.3f}</strong></div></div>".format(label, val_ml), unsafe_allow_html=True)
        st.markdown(f"<div class='delta'>Heurística: {val_h:.3f} · Δ {val_ml - val_h:+.3f}</div>", unsafe_allow_html=True)
        if interval:
            st.caption(f"CI 95%: [{interval[0]:.3f}, {interval[1]:.3f}]")
if uncertainty:
    st.caption("Desviaciones modelo: " + ", ".join(f"{k} {v:.3f}" for k, v in uncertainty.items()))

with st.container():
    st.markdown("### 🧬 Contribuciones de features (RandomForest)")
    if importance:
        df_imp = pd.DataFrame(importance, columns=["feature", "value"])
        chart = alt.Chart(df_imp).mark_bar(color="#34d399").encode(
            x=alt.X("value", title="Contribución"),
            y=alt.Y("feature", sort="-x", title="Feature"),
            tooltip=["feature", alt.Tooltip("value", format=".3f")],
        ).properties(height=260)
        st.altair_chart(chart, use_container_width=True)
    else:
        st.info("Sin metadata de importancia disponible para este modelo.")

with st.container():
    st.markdown("### 🧾 Comparativa heurística vs IA")
    df_compare = pd.DataFrame(
        {
            "Métrica": ["Rigidez", "Estanqueidad", "Energía", "Agua", "Crew"],
            "Heurística": [heur.rigidity, heur.tightness, heur.energy_kwh, heur.water_l, heur.crew_min],
            "IA Rex-AI": [props.rigidity, props.tightness, props.energy_kwh, props.water_l, props.crew_min],
        }
    )
    st.dataframe(df_compare.style.format({"Heurística": "{:.3f}", "IA Rex-AI": "{:.3f}"}), use_container_width=True)
    if comparisons:
        st.caption("Modelos secundarios (XGBoost / TabTransformer):")
        st.dataframe(pd.DataFrame(comparisons).T.style.format("{:.3f}"), use_container_width=True)

st.markdown("### 🎯 Score anatomy")
parts = score_breakdown(props, target, crew_time_low=target.get("crew_time_low", False))
chart_parts = alt.Chart(parts).mark_bar(color="#60a5fa").encode(
    x=alt.X("component", sort=None, title="Componente"),
    y=alt.Y("contribution", title="Aporte"),
    tooltip=["component", alt.Tooltip("contribution", format=".3f")],
).properties(height=280)
st.altair_chart(chart_parts, use_container_width=True)

with st.container():
    st.markdown("### 🛰️ Contexto y trazabilidad")
    st.markdown(
        """
        <div class="card">
          <div class="chips">
            <span>Seguridad: {safety}</span>
            <span>Regolito MGS-1: {regolith}%</span>
            <span>Entrenado: {trained}</span>
            <span>Muestras: {samples}</span>
          </div>
          <p style="margin-top:12px;">Materiales: {materials}</p>
          <p>Fuente IDs NASA: {ids}</p>
          <p>Latent vector (autoencoder): {latent}</p>
        </div>
        """.format(
            safety=f"{safety['level']} · {safety['detail']}",
            regolith=int(regolith_pct * 100),
            trained=metadata.get("trained_at", "—"),
            samples=metadata.get("n_samples", "—"),
            materials=", ".join(materials),
            ids=", ".join(cand.get("source_ids", [])),
            latent=", ".join(f"{v:.2f}" for v in latent[:8]) if latent else "—",
        ),
        unsafe_allow_html=True,
    )

st.markdown("### 📥 Export quick facts")
st.json(
    {
        "process": {"id": cand["process_id"], "name": cand["process_name"]},
        "materials": cand["materials"],
        "weights": cand.get("weights", []),
        "predictions": {
            "rigidez": props.rigidity,
            "estanqueidad": props.tightness,
            "energy_kwh": props.energy_kwh,
            "water_l": props.water_l,
            "crew_min": props.crew_min,
        },
        "confidence_interval": ci,
        "uncertainty": uncertainty,
        "model_metadata": metadata,
        "score": score,
    },
)
