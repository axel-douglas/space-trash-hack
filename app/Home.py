# app/Home.py
# ───────────────────────── path guard ─────────────────────────
import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# ──────────────────────────────────────────────────────────────

from datetime import datetime

import streamlit as st

from app.modules.ml_models import MODEL_REGISTRY

st.set_page_config(
    page_title="Rex-AI • Mission Copilot",
    page_icon="🛰️",
    layout="wide",
)

st.markdown(
    """
    <style>
    :root {
      --bg: #0b0d12;
      --card: rgba(18,21,29,0.72);
      --stroke: rgba(255,255,255,0.08);
      --accent: #60a5fa;
      --ink: #f8fafc;
      --muted: rgba(226,232,240,0.68);
    }
    body {background: var(--bg);}
    .block-container {padding: 2.6rem 3rem 3rem 3rem;}
    .hero {
      border-radius: 28px;
      padding: 36px 42px;
      background: linear-gradient(135deg, rgba(59,130,246,0.22), rgba(59,130,246,0.05)),
                  linear-gradient(180deg, rgba(15,23,42,0.9), rgba(15,23,42,0.72));
      border: 1px solid rgba(96,165,250,0.32);
      color: var(--ink);
      position: relative;
      overflow: hidden;
    }
    .hero:after {
      content:""; position:absolute; inset:-120px; background: radial-gradient(circle at top right, rgba(96,165,250,0.35), transparent 55%);
      pointer-events:none;
    }
    .hero h1 {font-size: 2.25rem; margin-bottom: 12px; letter-spacing: 0.02em;}
    .hero p {font-size: 1.04rem; max-width: 720px; color: var(--muted);}
    .chip-row {display:flex; gap:8px; margin-top: 18px; flex-wrap:wrap;}
    .chip {
      padding:6px 14px; border-radius:999px; font-size:0.82rem; font-weight:600;
      background: rgba(15,23,42,0.6); border: 1px solid rgba(96,165,250,0.35); color: var(--ink);
    }
    .ghost-card {
      margin-top: 38px; display:grid; gap:18px; grid-template-columns: repeat(auto-fit,minmax(260px,1fr));
    }
    .ghost-card > div {
      padding:20px 22px; border-radius:20px; border:1px solid var(--stroke);
      background: var(--card); color: var(--ink);
    }
    .ghost-card h3 {margin-bottom:6px; font-size:1.05rem;}
    .ghost-card p {color: var(--muted); font-size:0.94rem; margin:0;}
    .stepper {display:grid; grid-template-columns: repeat(auto-fit,minmax(180px,1fr)); gap:14px; margin: 32px 0 18px;}
    .step {border-radius:18px; border:1px solid var(--stroke); padding:16px 18px; background: rgba(13,17,23,0.6);}
    .step span {display:inline-flex; width:32px; height:32px; border-radius:999px; align-items:center; justify-content:center; background: rgba(96,165,250,0.24); color: var(--ink); font-weight:700; margin-bottom:10px;}
    .step h4 {margin:0 0 6px 0;}
    .step p {color: var(--muted); font-size:0.9rem; margin:0;}
    .badge {padding:4px 10px; border-radius:999px; font-size:0.78rem; background:rgba(96,165,250,0.16); color:var(--ink); margin-right:6px;}
    .metric-grid {display:grid; grid-template-columns: repeat(auto-fit,minmax(210px,1fr)); gap:16px; margin-top: 18px;}
    .metric {border-radius:18px; padding:18px 20px; background:rgba(15,23,42,0.6); border:1px solid var(--stroke); color:var(--ink);}
    .metric h5 {margin:0; font-size:0.92rem; color:var(--muted);}
    .metric strong {font-size:1.4rem; display:block; margin-top:6px;}
    .timeline {margin-top: 28px;}
    .timeline h3 {margin-bottom: 12px;}
    .timeline ul {list-style:none; padding:0; margin:0;}
    .timeline li {margin-bottom: 14px; padding-left: 18px; position:relative; color:var(--muted);}
    .timeline li::before {content:"•"; position:absolute; left:0; color:var(--accent); font-size:1.3rem;}
    .cta-row {display:flex; gap:12px; flex-wrap:wrap; margin-top:28px;}
    .cta-row button {background: var(--accent); color: #0f172a; border-radius: 999px; padding:10px 18px; font-weight:700; border:none;}
    .sidebar .sidebar-content {background:rgba(15,23,42,0.88);}
    </style>
    """,
    unsafe_allow_html=True,
)

trained_at = MODEL_REGISTRY.metadata.get("trained_at")
try:
    trained_dt = datetime.fromisoformat(trained_at) if trained_at else None
except Exception:
    trained_dt = None
n_samples = MODEL_REGISTRY.metadata.get("n_samples")
model_name = MODEL_REGISTRY.metadata.get("model_name", "rexai-rf-ensemble")

st.markdown(
    """
    <div class="hero">
      <h1>Rex-AI es tu copiloto de reciclaje en Marte</h1>
      <p>
        Convierte flujos de basura no-metabólica y regolito MGS-1 en hardware útil.
        La plataforma guía a la tripulación paso a paso, combinando datos reales de NASA/UCF con
        modelos de aprendizaje automático que priorizan seguridad, trazabilidad y eficiencia.
      </p>
      <div class="chip-row">
        <span class="chip">RandomForest + XGBoost + TabTransformer</span>
        <span class="chip">Embeddings con Autoencoder</span>
        <span class="chip">Optimización Bayesiana (Ax/BoTorch)</span>
        <span class="chip">Explicabilidad con bandas de confianza</span>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("### Ruta de misión (guided flow)")
st.markdown(
    """
    <div class="stepper">
      <div class="step"><span>1</span><h4>Inventario</h4><p>Normalizá residuos NASA, marcá flags problemáticos (multilayer, EVA, nitrilo) y valida masas.</p></div>
      <div class="step"><span>2</span><h4>Target</h4><p>Seleccioná el producto final, definí rigidez/estanqueidad objetivo y límites de agua, energía y crew.</p></div>
      <div class="step"><span>3</span><h4>Generador</h4><p>Rex-AI mezcla ítems, sugiere proceso, ejecuta optimización bayesiana y explica cada predicción.</p></div>
      <div class="step"><span>4</span><h4>Resultados</h4><p>Visualizá trade-offs, confianza 95%, comparación heurística vs IA y exportá el plan de fabricación.</p></div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("### Pila de IA lista para demo")
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("#### Demo actual")
    st.markdown(
        "- **RandomForest multisalida** (scikit-learn) → rigidez, estanqueidad, recursos.\n"
        "- Incertidumbre = desviación estándar entre árboles + residuales.\n"
        "- Autoencoder genera vectores latentes para similitudes y clustering."
    )
with col2:
    st.markdown("#### Wow effect hackathon")
    st.markdown(
        "- Ensemble XGBoost y TabTransformer sugieren alternativas y explican interacciones complejas.\n"
        "- Optimización Ax/BoTorch orienta nuevas recetas sin ensayo-error humano."
    )
with col3:
    st.markdown("#### Roadmap futuro")
    st.markdown(
        "- PINNs para procesos térmicos con regolito.\n"
        "- GNNs para compatibilidad material.\n"
        "- Federated Learning entre hábitats de la red Artemis." )

st.markdown("### Estado del modelo Rex-AI")
trained_label = trained_dt.strftime("%d %b %Y %H:%M UTC") if trained_dt else "sin metadata"
ready = "✅ Modelo listo" if MODEL_REGISTRY.ready else "⚠️ Entrená localmente"
with st.container():
    st.markdown(
        f"""
        <div class="metric-grid">
          <div class="metric"><h5>Estado</h5><strong>{ready}</strong><p>Nombre: {model_name}</p></div>
          <div class="metric"><h5>Entrenado</h5><strong>{trained_label}</strong><p>Muestras: {n_samples or '—'}</p></div>
          <div class="metric"><h5>Feature space</h5><strong>{len(MODEL_REGISTRY.feature_names)}</strong><p>Ingeniería fisicoquímica + procesos NASA</p></div>
          <div class="metric"><h5>Incertidumbre</h5><strong>Desv. árboles + residuales</strong><p>CI 95% expuesto en la UI</p></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("### Cómo navegar ahora")
st.markdown(
    """
    <div class="ghost-card">
      <div><h3>1. Inventario NASA</h3><p>Trabajá sobre <code>data/waste_inventory_sample.csv</code> o subí tu CSV con las columnas normalizadas.</p></div>
      <div><h3>2. Objetivo</h3><p>Seleccioná presets (container, utensil, tool, interior) o define manualmente tus límites.</p></div>
      <div><h3>3. Generador con IA</h3><p>Activa la optimización bayesiana, revisa contribuciones de features y compara heurística vs modelo.</p></div>
      <div><h3>4. Reportar</h3><p>Exportá recetas, Sankey de materiales y feedback/impact para continuar entrenando Rex-AI.</p></div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("### Siguiente acción")
cta_col1, cta_col2, cta_col3, cta_col4 = st.columns(4)
with cta_col1:
    if st.button("🧱 Inventario", use_container_width=True):
        st.switch_page("pages/1_Inventory_Builder.py")
with cta_col2:
    if st.button("🎯 Target", use_container_width=True):
        st.switch_page("pages/2_Target_Designer.py")
with cta_col3:
    if st.button("🤖 Generador", use_container_width=True):
        st.switch_page("pages/3_Generator.py")
with cta_col4:
    if st.button("📊 Resultados", use_container_width=True):
        st.switch_page("pages/4_Results_and_Tradeoffs.py")

st.markdown("---")
st.markdown(
    """
    <div class="timeline">
      <h3>¿Qué demuestra esta demo hoy?</h3>
      <ul>
        <li>Pipeline reproducible: <code>python -m app.modules.model_training</code> genera dataset, RandomForest, XGBoost, TabTransformer y autoencoder.</li>
        <li>Predicciones con trazabilidad: cada receta incluye IDs, categorías, flags y metadatos de entrenamiento.</li>
        <li>Explicabilidad integrada: barras de contribución, comparativa heurística vs IA y bandas de confianza 95%.</li>
        <li>Optimización asistida: Ax/BoTorch prioriza combinaciones que maximizan score sin violar límites de recursos.</li>
      </ul>
    </div>
    """,
    unsafe_allow_html=True,
)
