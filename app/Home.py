# app/Home.py
import _bootstrap  # noqa: F401

from datetime import datetime
import streamlit as st
from app.modules.ml_models import get_model_registry
from app.modules.navigation import set_active_step
from app.modules.ui_blocks import load_theme

st.set_page_config(
    page_title="Rex-AI • Mission Copilot",
    page_icon="🛰️",
    layout="wide",
)

set_active_step("brief")

load_theme()

model_registry = get_model_registry()

# ──────────── Estilos (oscuro, cards y métricas) ────────────
st.markdown(
    """
    <style>
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
    .metric-grid {display:grid; grid-template-columns: repeat(auto-fit,minmax(210px,1fr)); gap:16px; margin-top: 18px;}
    .metric {border-radius:18px; padding:18px 20px; background:rgba(15,23,42,0.6); border:1px solid var(--stroke); color:var(--ink);}
    .metric h5 {margin:0; font-size:0.92rem; color:var(--muted);}
    .metric strong {font-size:1.4rem; display:block; margin-top:6px;}
    .timeline {margin-top: 28px;}
    .timeline h3 {margin-bottom: 12px;}
    .timeline ul {list-style:none; padding:0; margin:0;}
    .timeline li {margin-bottom: 14px; padding-left: 18px; position:relative; color:var(--muted);}
    .timeline li::before {content:"•"; position:absolute; left:0; color: var(--accent); font-size:1.3rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ──────────── Lectura segura de metadata del modelo ────────────
trained_at_raw = model_registry.metadata.get("trained_at")
trained_label_value = (
    model_registry.metadata.get("trained_label")
    or model_registry.metadata.get("trained_on")
)

try:
    trained_dt = datetime.fromisoformat(trained_at_raw) if trained_at_raw else None
except Exception:
    trained_dt = None

trained_at_display = (
    trained_dt.strftime("%d %b %Y %H:%M UTC") if trained_dt else "sin metadata"
)

trained_combo = model_registry.trained_label()
if trained_at_display == "sin metadata" and trained_combo and trained_combo != "—":
    trained_at_display = trained_combo

if not trained_label_value and trained_combo and trained_combo != "—":
    trained_label_value = trained_combo.split(" · ", 1)[0]

trained_label_value = trained_label_value or "—"

n_samples = model_registry.metadata.get("n_samples")
model_name = model_registry.metadata.get("model_name", "rexai-rf-ensemble")
feature_count = len(getattr(model_registry, "feature_names", []) or [])

# ──────────── Hero ────────────
st.markdown(
    """
    <div class="hero">
      <h1>Rex-AI es tu copiloto de reciclaje en Marte</h1>
      <p>
        Convierte flujos de basura no-metabólica y regolito MGS-1 en hardware útil.
        La plataforma guía a la tripulación paso a paso, combinando datos reales con
        modelos que priorizan seguridad, trazabilidad y eficiencia.
      </p>
      <div class="chip-row">
        <span class="chip">RandomForest multisalida</span>
        <span class="chip">Comparadores: XGBoost / Tabular</span>
        <span class="chip">Bandas de confianza 95%</span>
        <span class="chip">Trazabilidad completa</span>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ──────────── Ruta guiada ────────────
st.markdown("### Ruta de misión (guided flow)")
st.markdown(
    """
    <div class="stepper">
      <div class="step"><span>1</span><h4>Inventario</h4><p>Normalizá residuos y marcá flags problemáticos (multilayer, EVA, nitrilo).</p></div>
      <div class="step"><span>2</span><h4>Target</h4><p>Elegí producto final y límites de agua, energía y crew.</p></div>
      <div class="step"><span>3</span><h4>Generador</h4><p>Rex-AI mezcla ítems, sugiere proceso y explica cada predicción.</p></div>
      <div class="step"><span>4</span><h4>Resultados</h4><p>Trade-offs, confianza 95%, comparación heurística vs IA y export.</p></div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ──────────── Pila/estado del modelo ────────────
st.markdown("### Estado del modelo Rex-AI")
ready = "✅ Modelo listo" if model_registry.ready else "⚠️ Entrená localmente"

st.markdown(
    f"""
    <div class="metric-grid">
      <div class="metric"><h5>Estado</h5><strong>{ready}</strong><p>Nombre: {model_name}</p></div>
      <div class="metric"><h5>Entrenado</h5><strong>{trained_at_display}</strong><p>Procedencia: {trained_label_value}</p><p>Muestras: {n_samples or '—'}</p></div>
      <div class="metric"><h5>Feature space</h5><strong>{feature_count}</strong><p>Ingeniería fisicoquímica + proceso</p></div>
      <div class="metric"><h5>Incertidumbre</h5><strong>{model_registry.uncertainty_label()}</strong><p>CI 95% en UI</p></div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ──────────── Cómo navegar ────────────
st.markdown("### Cómo navegar ahora")
st.markdown(
    """
    <div class="ghost-card">
      <div><h3>1. Inventario NASA</h3><p>Trabajá sobre <code>data/waste_inventory_sample.csv</code> o subí tu CSV normalizado.</p></div>
      <div><h3>2. Objetivo</h3><p>Usá presets (container, utensil, tool, interior) o definí límites manuales.</p></div>
      <div><h3>3. Generador con IA</h3><p>Revisá contribuciones de features y compara heurística vs modelo.</p></div>
      <div><h3>4. Reportar</h3><p>Exportá recetas, Sankey y feedback/impact para seguir entrenando Rex-AI.</p></div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ──────────── CTA navegación ────────────
st.info(
    "Usá el **Mission HUD** superior para saltar entre pasos o presioná las teclas `1-9` "
    "para navegar más rápido por el flujo guiado."
)

# ──────────── Qué demuestra hoy ────────────
st.markdown("---")
st.markdown(
    """
    <div class="timeline">
      <h3>¿Qué demuestra esta demo hoy?</h3>
      <ul>
        <li>Pipeline reproducible: <code>python -m app.modules.model_training</code> genera dataset y el RandomForest multisalida.</li>
        <li>Predicciones con trazabilidad: cada receta incluye IDs, categorías, flags y metadatos de entrenamiento.</li>
        <li>Explicabilidad integrada: contribuciones por feature y bandas de confianza 95%.</li>
        <li>Comparación heurística vs IA y export listo para experimentación.</li>
      </ul>
    </div>
    """,
    unsafe_allow_html=True,
)
