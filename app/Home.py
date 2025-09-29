# app/Home.py
import _bootstrap  # noqa: F401

from datetime import datetime
from pathlib import Path

import streamlit as st

from app.modules.luxe_components import (
    BriefingCard,
    TimelineMilestone,
    guided_demo,
    mission_briefing,
    orbital_timeline,
)
from app.modules.ml_models import get_model_registry
from app.modules.ui_blocks import load_theme

st.set_page_config(
    page_title="Rex-AI • Mission Copilot",
    page_icon="🛰️",
    layout="wide",
)

load_theme()

model_registry = get_model_registry()

# ──────────── Estilos (oscuro, cards y métricas) ────────────
st.markdown(
    """
    <style>
    .mission-grid {display:grid; gap:18px; margin-top:26px; grid-template-columns: repeat(auto-fit,minmax(260px,1fr));}
    .mission-grid > div {padding:22px; border-radius:22px; border:1px solid var(--stroke); background:rgba(13,17,23,0.72); color:var(--ink);}
    .mission-grid h3 {margin-bottom:8px; font-size:1.06rem;}
    .mission-grid p {margin:0; color:var(--muted); font-size:0.94rem;}
    .metric-grid {display:grid; grid-template-columns: repeat(auto-fit,minmax(210px,1fr)); gap:16px; margin-top: 18px;}
    .metric {border-radius:18px; padding:18px 20px; background:rgba(15,23,42,0.6); border:1px solid var(--stroke); color:var(--ink); transition: border 300ms ease, box-shadow 300ms ease;}
    .metric.highlight {border-color: rgba(56,189,248,0.6); box-shadow:0 0 0 2px rgba(56,189,248,0.18);}
    .metric h5 {margin:0; font-size:0.92rem; color:var(--muted);}
    .metric strong {font-size:1.4rem; display:block; margin-top:6px;}
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

# ──────────── Hero interactivo ────────────
mission_briefing(
    title="Mission Briefing • Rex-AI en órbita marciana",
    tagline="Sincronizá sensores, crew y modelo para reciclar basura orbital en hardware vital.",
    video_path=Path(__file__).resolve().parent / "static" / "mission_briefing_loop.mp4",
    cards=[
        BriefingCard(
            title="Crew Ops + IA",
            body="La cabina recibe datos del inventario NASA, restricciones de crew-time y energía en tiempo real.",
            accent="#38bdf8",
        ),
        BriefingCard(
            title="Trazabilidad total",
            body="Cada decisión enlaza features, flags de riesgo y la receta final exportable a ingeniería.",
            accent="#a855f7",
        ),
        BriefingCard(
            title="Seguridad primero",
            body="Bandas de confianza, monitoreo de toxicidad EVA y comparadores heurísticos siempre visibles.",
            accent="#f97316",
        ),
    ],
    steps=[
        ("Calibrá el inventario", "Normalizá residuos, detectá flags EVA y estructuras multi-layer."),
        ("Seleccioná objetivo", "Define límites de agua, energía y logística con presets marcianos."),
        ("Generá y valida", "Rex-AI mezcla, explica contribuciones y exporta procesos listos para la tripulación."),
    ],
)

# ──────────── Ruta guiada ────────────
st.markdown("### Ruta de misión (guided flow)")

demo_steps = [
    TimelineMilestone(
        label="Inventario en vivo",
        description="Ingerí CSV NASA, normalizá unidades y marca riesgos EVA desde la cabina.",
        icon="🧱",
    ),
    TimelineMilestone(
        label="Target marciano",
        description="Seleccioná producto final, límites de agua y energía, o usa presets homologados.",
        icon="🎯",
    ),
    TimelineMilestone(
        label="Generador IA",
        description="Explorá mezclas óptimas, revisá contribuciones y bandas de confianza en segundos.",
        icon="🤖",
    ),
    TimelineMilestone(
        label="Resultados y export",
        description="Compará heurísticas vs IA, exportá recetas y registra feedback para retraining.",
        icon="📊",
    ),
]

active_demo_step = guided_demo(steps=demo_steps, step_duration=6.5)

# ──────────── Pila/estado del modelo ────────────
st.markdown("### Estado del modelo Rex-AI")
ready = "✅ Modelo listo" if model_registry.ready else "⚠️ Entrená localmente"

metric_blocks = [
    (
        "Estado",
        ready,
        f"Nombre: {model_name}",
        None,
        active_demo_step and "Inventario" in active_demo_step.label,
    ),
    (
        "Entrenado",
        trained_at_display,
        f"Procedencia: {trained_label_value}",
        f"Muestras: {n_samples or '—'}",
        active_demo_step and "Target" in active_demo_step.label,
    ),
    (
        "Feature space",
        str(feature_count),
        "Ingeniería fisicoquímica + proceso",
        None,
        active_demo_step and "Generador" in active_demo_step.label,
    ),
    (
        "Incertidumbre",
        model_registry.uncertainty_label(),
        "CI 95% en UI",
        None,
        active_demo_step and "Resultados" in active_demo_step.label,
    ),
]

metric_html = "".join(
    f"""
    <div class='metric{' highlight' if highlight else ''}'>
        <h5>{title}</h5>
        <strong>{value}</strong>
        <p>{line1}</p>
        {f'<p>{line2}</p>' if line2 else ''}
    </div>
    """
    for title, value, line1, line2, highlight in metric_blocks
)

st.markdown(
    f"""
    <div class="metric-grid">
      {metric_html}
    </div>
    """,
    unsafe_allow_html=True,
)

# ──────────── Cómo navegar ────────────
st.markdown("### Cómo navegar ahora")
st.markdown(
    """
    <div class="mission-grid">
      <div><h3>1. Inventario NASA</h3><p>Trabajá sobre <code>data/waste_inventory_sample.csv</code> o subí tu CSV normalizado.</p></div>
      <div><h3>2. Objetivo</h3><p>Usá presets (container, utensil, tool, interior) o definí límites manuales.</p></div>
      <div><h3>3. Generador con IA</h3><p>Revisá contribuciones de features y compará heurística vs modelo.</p></div>
      <div><h3>4. Reportar</h3><p>Exportá recetas, Sankey y feedback/impact para seguir entrenando Rex-AI.</p></div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ──────────── CTA navegación ────────────
st.markdown("### Siguiente acción")
c1, c2, c3, c4 = st.columns(4)
with c1:
    if st.button("🧱 Inventario", use_container_width=True):
        st.switch_page("pages/1_Inventory_Builder.py")
with c2:
    if st.button("🎯 Target", use_container_width=True):
        st.switch_page("pages/2_Target_Designer.py")
with c3:
    if st.button("🤖 Generador", use_container_width=True):
        st.switch_page("pages/3_Generator.py")
with c4:
    if st.button("📊 Resultados", use_container_width=True):
        st.switch_page("pages/4_Results_and_Tradeoffs.py")

# ──────────── Qué demuestra hoy ────────────
st.markdown("---")
st.markdown("### ¿Qué demuestra esta demo hoy?")

orbital_timeline(
    [
        TimelineMilestone(
            label="Pipeline reproducible",
            description="<code>python -m app.modules.model_training</code> genera dataset y RandomForest multisalida listo.",
            icon="🛠️",
        ),
        TimelineMilestone(
            label="Trazabilidad de recetas",
            description="Cada receta conserva IDs, categorías, flags de riesgo y metadatos de entrenamiento.",
            icon="🛰️",
        ),
        TimelineMilestone(
            label="Explicabilidad integrada",
            description="Contribuciones por feature, bandas 95% y comparador heurístico vs IA en UI.",
            icon="🧠",
        ),
        TimelineMilestone(
            label="Export y feedback",
            description="Entrega recetas, Sankey y feedback listos para continuar el retraining marciano.",
            icon="📦",
        ),
    ]
)
