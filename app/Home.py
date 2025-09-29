# app/Home.py
import _bootstrap  # noqa: F401

from datetime import datetime

import streamlit as st

from app.modules.luxe_components import (
    GlassCard,
    GlassStack,
    MetricGalaxy,
    MetricItem,
    TeslaHero,
)
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
TeslaHero(
    title="Rex-AI es tu copiloto de reciclaje en Marte",
    subtitle=(
        "Convierte flujos de basura no-metabólica y regolito MGS-1 en hardware útil. "
        "La plataforma guía a la tripulación paso a paso, combinando datos reales "
        "con modelos que priorizan seguridad, trazabilidad y eficiencia."
    ),
    chips=[
        {"label": "RandomForest multisalida", "tone": "accent"},
        {"label": "Comparadores: XGBoost / Tabular", "tone": "info"},
        {"label": "Bandas de confianza 95%", "tone": "accent"},
        {"label": "Trazabilidad completa", "tone": "info"},
    ],
    icon="🛰️",
    gradient="linear-gradient(135deg, rgba(59,130,246,0.28), rgba(14,165,233,0.08))",
    glow="rgba(96,165,250,0.45)",
    density="roomy",
    parallax_icons=[
        {"icon": "🛰️", "top": "8%", "left": "74%", "size": "4.8rem", "speed": "22s"},
        {"icon": "🪐", "top": "62%", "left": "80%", "size": "5.2rem", "speed": "28s"},
        {"icon": "✨", "top": "20%", "left": "12%", "size": "3.2rem", "speed": "18s"},
    ],
).render()

# ──────────── Ruta guiada ────────────
st.markdown("### Ruta de misión (guided flow)")
GlassStack(
    cards=[
        GlassCard(
            title="1 · Inventario",
            body="Normalizá residuos y marcá flags problemáticos (multilayer, EVA, nitrilo).",
            icon="🧱",
            footer="Dataset NASA + crew flags",
        ),
        GlassCard(
            title="2 · Target",
            body="Elegí producto final y límites de agua, energía y crew para la misión.",
            icon="🎯",
            footer="Presets o límites manuales",
        ),
        GlassCard(
            title="3 · Generador",
            body="Rex-AI mezcla ítems, sugiere proceso y explica cada predicción en vivo.",
            icon="🤖",
            footer="ML + heurística cooperativa",
        ),
        GlassCard(
            title="4 · Resultados",
            body="Trade-offs, confianza 95%, comparación heurística vs IA y export final.",
            icon="📊",
            footer="Listo para experimentos",
        ),
    ],
    columns_min="15rem",
    density="compact",
).render()

# ──────────── Pila/estado del modelo ────────────
st.markdown("### Estado del modelo Rex-AI")
ready = "✅ Modelo listo" if model_registry.ready else "⚠️ Entrená localmente"

MetricGalaxy(
    metrics=[
        MetricItem(
            label="Estado",
            value=ready,
            caption=f"Nombre: {model_name}",
            icon="🛰️",
        ),
        MetricItem(
            label="Entrenado",
            value=trained_at_display,
            caption=f"Procedencia: {trained_label_value} · Muestras: {n_samples or '—'}",
            icon="🧪",
        ),
        MetricItem(
            label="Feature space",
            value=str(feature_count),
            caption="Ingeniería fisicoquímica + proceso",
            icon="🧬",
        ),
        MetricItem(
            label="Incertidumbre",
            value=model_registry.uncertainty_label(),
            caption="CI 95% expuesta en UI",
            icon="📈",
        ),
    ],
    density="cozy",
).render()

# ──────────── Cómo navegar ────────────
st.markdown("### Cómo navegar ahora")
GlassStack(
    cards=[
        GlassCard(
            title="1. Inventario NASA",
            body="Trabajá sobre <code>data/waste_inventory_sample.csv</code> o subí tu CSV normalizado.",
            icon="📦",
        ),
        GlassCard(
            title="2. Objetivo",
            body="Usá presets (container, utensil, tool, interior) o definí límites manuales.",
            icon="🎛️",
        ),
        GlassCard(
            title="3. Generador con IA",
            body="Revisá contribuciones de features y compará heurística vs modelo.",
            icon="🤝",
        ),
        GlassCard(
            title="4. Reportar",
            body="Exportá recetas, Sankey y feedback/impact para seguir entrenando Rex-AI.",
            icon="📤",
        ),
    ],
    columns_min="15rem",
    density="cozy",
).render()

# ──────────── CTA navegación ────────────
st.info(
    "Usá el **Mission HUD** superior para saltar entre pasos o presioná las teclas `1-9` "
    "para navegar más rápido por el flujo guiado."
)

# ──────────── Qué demuestra hoy ────────────
st.markdown("---")
GlassStack(
    cards=[
        GlassCard(
            title="¿Qué demuestra esta demo hoy?",
            body=(
                "<ul>"
                "<li>Pipeline reproducible: <code>python -m app.modules.model_training</code> genera dataset y el RandomForest multisalida.</li>"
                "<li>Predicciones con trazabilidad: cada receta incluye IDs, categorías, flags y metadatos de entrenamiento.</li>"
                "<li>Explicabilidad integrada: contribuciones por feature y bandas de confianza 95%.</li>"
                "<li>Comparación heurística vs IA y export listo para experimentación.</li>"
                "</ul>"
            ),
            icon="🛰️",
        ),
    ],
    columns_min="26rem",
    density="roomy",
).render()
