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
    src = getattr(props, "source", "heuristic")
    if src.startswith("rexai"):
        trained_at = metadata.get("trained_at", "?")
        st.caption(f"Predicciones por modelo Rex-AI (**{src}**, entrenado {trained_at}).")
    else:
        st.caption("Predicciones heurísticas basadas en reglas NASA.")

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
with topR:
    # Estado de seguridad (pill + popover de explicación)
    level = badge.get("level","OK").lower()
    cls = "ok" if "ok" in level else ("risk" if "riesgo" in level or "risk" in level else "warn")
    st.markdown(f'<span class="pill {cls}">Seguridad: {badge.get("level","OK")}</span>', unsafe_allow_html=True)
    pop = st.popover("¿Qué chequeamos?")
    with pop:
        st.write(badge.get("detail", "Sin observaciones."))
        st.caption("Validaciones: PFAS/microplásticos evitados, sin incineración, flags NASA (EVA/CTB, multilayers, nitrilo).")

st.markdown("---")

# ======== KPIs con contexto ========
k1,k2,k3,k4,k5 = st.columns(5)
with k1:
    st.markdown('<div class="kpi"><h3>Score total</h3><div class="v">{:.2f}</div><div class="hint">Función + Recursos + Bono problemáticos</div></div>'.format(score), unsafe_allow_html=True)
with k2:
    st.markdown('<div class="kpi"><h3>Rigidez</h3><div class="v">{:.2f}</div><div class="hint">Objetivo: {:.2f}</div></div>'.format(props.rigidity, float(target["rigidity"])), unsafe_allow_html=True)
with k3:
    st.markdown('<div class="kpi"><h3>Estanqueidad</h3><div class="v">{:.2f}</div><div class="hint">Objetivo: {:.2f}</div></div>'.format(props.tightness, float(target["tightness"])), unsafe_allow_html=True)
with k4:
    st.markdown('<div class="kpi"><h3>Energía (kWh)</h3><div class="v">{:.2f}</div><div class="hint">Máx: {:.2f}</div></div>'.format(props.energy_kwh, float(target["max_energy_kwh"])), unsafe_allow_html=True)
with k5:
    st.markdown('<div class="kpi"><h3>Agua (L)</h3><div class="v">{:.2f}</div><div class="hint">Máx: {:.2f}</div></div>'.format(props.water_l, float(target["max_water_l"])), unsafe_allow_html=True)

c1, c2, c3 = st.columns([1,1,1])
with c1:
    st.markdown('<div class="kpi"><h3>Crew-time (min)</h3><div class="v">{:.0f}</div><div class="hint">Máx: {:.0f}</div></div>'.format(props.crew_min, float(target["max_crew_min"])), unsafe_allow_html=True)
with c2:
    st.markdown('<div class="kpi"><h3>Masa final (kg)</h3><div class="v">{:.2f}</div><div class="hint">Post-proceso / mermas</div></div>'.format(props.mass_final_kg), unsafe_allow_html=True)
with c3:
    # Mini ayuda para crew-time-low
    if target.get("crew_time_low", False):
        st.markdown('<div class="kpi"><h3>Modo</h3><div class="v">Crew-time Low</div><div class="hint">Más peso al tiempo de tripulación</div></div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="kpi"><h3>Modo</h3><div class="v">Balanceado</div><div class="hint">Trade-off estándar</div></div>', unsafe_allow_html=True)

# ======== Tabs principales: (1) Score anatomy (2) Flujo Sankey (3) Checklist (4) Trazabilidad ========
tab1, tab2, tab3, tab4 = st.tabs(["🧩 Anatomía del Score", "🔀 Flujo del proceso (Sankey)", "🛠️ Checklist & Próximos pasos", "🛰️ Trazabilidad NASA"])

# --- TAB 1: Anatomía del Score ---
with tab1:
    st.markdown("## 🧩 Anatomía del Score <span class='sub'>(explicabilidad)</span>", unsafe_allow_html=True)
    parts = score_breakdown(props, target, crew_time_low=target.get("crew_time_low", False))
    # Asegurar índice correcto
    if isinstance(parts, pd.DataFrame) and "component" in parts.columns and "contribution" in parts.columns:
        fig_bar = go.Figure(
            data=[go.Bar(
                x=parts["component"],
                y=parts["contribution"],
                text=[f"{v:.2f}" for v in parts["contribution"]],
                textposition="outside"
            )]
        )
        fig_bar.update_layout(
            margin=dict(l=10,r=10,t=10,b=10),
            yaxis_title="Aporte al Score",
            xaxis_title="Componente",
            height=360
        )
        st.plotly_chart(fig_bar, use_container_width=True)
    else:
        st.info("No se pudo construir el desglose. Verifica `score_breakdown`.")

    # Popover didáctico
    pop1 = st.popover("¿Qué estoy viendo?")
    with pop1:
        st.markdown("""
- Cada barra es una **pieza del puntaje**:
  - **Función**: qué tan cerca está tu receta de la *rigidez* y *estanqueidad* objetivo.
  - **Recursos**: te premia por **bajo consumo** de energía/agua y **poco tiempo** de tripulación.
  - **Seguridad base**: piso de seguridad (las banderas duras se validan aparte).
- Si activaste *Crew-time Low*, la barra de **tiempo** pesa más.
""")

# --- TAB 2: Flujo Sankey ---
with tab2:
    st.markdown("## 🔀 Flujo de materiales → proceso → producto", unsafe_allow_html=True)

    labels = materials + [cand["process_name"], "Producto"]
    src = list(range(len(materials)))
    tgt = [len(materials)] * len(materials)
    # Si vienen pesos con regolito + redondeos, normalizamos para que la suma sea 1
    weights = cand.get("weights", [])
    if weights and abs(sum(weights) - 1.0) > 1e-6:
        s = sum(weights)
        weights = [w/s for w in weights]
    vals = [round(w*100, 1) for w in weights] if weights else [100/len(src)]*len(src)

    # proceso -> producto
    src += [len(materials)]
    tgt += [len(materials) + 1]
    vals += [100.0]

    fig = go.Figure(data=[go.Sankey(
        node=dict(label=labels, pad=20, thickness=18),
        link=dict(source=src, target=tgt, value=vals)
    )])
    fig.update_layout(margin=dict(l=10,r=10,t=10,b=10), height=420)
    st.plotly_chart(fig, use_container_width=True)

    pop2 = st.popover("¿Cómo leerlo?")
    with pop2:
        st.markdown("""
- **Izquierda**: residuos seleccionados (con sus **pesos relativos**).
- **Centro**: el **proceso** (p.ej., Laminar o *Sinter with MGS-1* si hay regolito).
- **Derecha**: el **producto** final.
- Si ves **MGS-1_regolith** en materiales, significa **ISRU**: aprovechamos regolito como carga mineral.
""")

# --- TAB 3: Checklist & Próximos pasos ---
with tab3:
    st.markdown("## 🛠️ Checklist de fabricación")
    st.markdown(f"""
1. **Preparar/Triturar**: acondicionar materiales (**{', '.join(materials)}**).
2. **Ejecutar proceso**: **{cand['process_id']} {cand['process_name']}** con parámetros estándar del hábitat.
3. **Enfriar & post-proceso**: verificar bordes, ajuste y *fit*.
4. **Registrar feedback**: rigidez percibida, facilidad de uso, y problemas (bordes, olor, slip, etc.).
    """)

    st.markdown("### ⏱️ Recursos estimados")
    cA,cB,cC = st.columns(3)
    cA.metric("Energía", f"{props.energy_kwh:.2f} kWh")
    cB.metric("Agua", f"{props.water_l:.2f} L")
    cC.metric("Crew-time", f"{props.crew_min:.0f} min")

    pop3 = st.popover("¿Por qué importa?")
    with pop3:
        st.markdown("""
- En Marte **no hay camión de la basura**: cada minuto de tripulación y cada litro de agua cuenta.
- Minimizar recursos mantiene la **operación sostenible** y deja margen para otras tareas científicas.
""")

# --- TAB 4: Trazabilidad NASA ---
with tab4:
    st.markdown("## 🛰️ Trazabilidad NASA (inputs → plan)")
    # IDs / categorías / flags para auditar que usamos lo problemático
    st.markdown("**IDs usados:** " + ", ".join(cand.get("source_ids", []) or ["—"]))
    st.markdown("**Categorías:** " + ", ".join(map(str, cand.get("source_categories", []) or ["—"])))
    st.markdown("**Flags:** " + ", ".join(map(str, cand.get("source_flags", []) or ["—"])))
    st.caption("Esto permite demostrar que estamos atacando pouches multilayer, espumas ZOTEK, EVA/CTB, nitrilo, etc.")
    feat = cand.get("features", {})
    if feat:
        feat_view = {
            "Masa total (kg)": feat.get("total_mass_kg"),
            "Densidad (kg/m³)": feat.get("density_kg_m3"),
            "Humedad": feat.get("moisture_frac"),
            "Dificultad": feat.get("difficulty_index"),
            "Recupero gas": feat.get("gas_recovery_index"),
            "Reuso logístico": feat.get("logistics_reuse_index"),
            "SiO₂ (regolito)": feat.get("oxide_sio2"),
            "FeOT (regolito)": feat.get("oxide_feot"),
        }
        st.markdown("**Features NASA/ML**")
        st.dataframe(pd.DataFrame([feat_view]), hide_index=True, use_container_width=True)
    latent_view = latent or feat.get("latent_vector")
    if latent_view:
        st.info(
            f"Vector latente Rex-AI de {len(latent_view)} dimensiones listo para clustering o búsqueda de recetas similares."
        )

# ======== Bloque final de educación rápida ========
st.markdown("---")
edu = st.popover("ℹ️ Entender estos trade-offs (explicación simple)")
with edu:
    st.markdown("""
- **Score**: es como el *promedio ponderado* de todo lo que te importa (función + recursos + seguridad).
- **Rigidez/Estanqueidad**: si tu contenedor se **deforma** o **pierde** porosidad, no sirve; por eso están arriba del todo.
- **Energía/Agua/Crew**: si te pasás de los límites objetivo, **penaliza**. Marte **no perdona** derroches.
- **Sankey**: te muestra **qué entra**, **cómo se procesa**, **qué sale**. Ayuda a “ver” si el plan es coherente.
- **MGS-1**: si aparece, es **ISRU** (usar lo que hay en Marte). Menos dependencia de la Tierra, más puntos por sostenibilidad.
""")
