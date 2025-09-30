# app/Home.py
import _bootstrap  # noqa: F401
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

from app.modules.luxe_components import (
    CarouselItem,
    CarouselRail,
    HeroFlowStage,
    MetricGalaxy,
    MinimalHero,
    MissionBoard,
    MissionMetrics,
    guided_demo,
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


@st.cache_data
def load_inventory_sample() -> pd.DataFrame | None:
    sample_path = Path("data") / "waste_inventory_sample.csv"
    if not sample_path.exists():
        return None
    try:
        return pd.read_csv(sample_path)
    except Exception:
        return None


def format_mass(value: float | int | None) -> str:
    if value is None:
        return "—"
    if value >= 1000:
        return f"{value/1000:.1f} t"
    return f"{value:.0f} kg"


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
ready = "✅ Modelo listo" if model_registry.ready else "⚠️ Entrená localmente"

mission_stages = [
    HeroFlowStage(
        key="inventory",
        order=1,
        name="Inventario",
        hero_headline="Calibrá el inventario",
        hero_copy="Normalizá residuos, detectá flags EVA y estructuras multi-layer.",
        card_body=(
            "Normalizá residuos, marcá flags (multilayer, EVA, nitrilo) y trabajá sobre "
            "<code>data/waste_inventory_sample.csv</code> o tu CSV limpio."
        ),
        compact_card_body=(
            "Normalizá residuos y flags EVA/multilayer sobre <code>data/waste_inventory_sample.csv</code>."
        ),
        icon="🧱",
        timeline_label="Inventario en vivo",
        timeline_description="Ingerí CSV NASA, normalizá unidades y marca riesgos EVA desde la cabina.",
        footer="Dataset NASA + crew flags",
    ),
    HeroFlowStage(
        key="target",
        order=2,
        name="Target",
        hero_headline="Seleccioná objetivo",
        hero_copy="Define límites de agua, energía y logística con presets marcianos.",
        card_body=(
            "Elegí producto final, límites de agua/energía y presets marcianos (container, utensil, tool, interior)."
        ),
        compact_card_body="Elegí producto y límites con presets marcianos certificados.",
        icon="🎯",
        timeline_label="Target marciano",
        timeline_description="Seleccioná producto final, límites de agua y energía, o usa presets homologados.",
        footer="Presets o límites manuales",
    ),
    HeroFlowStage(
        key="generator",
        order=3,
        name="Generador",
        hero_headline="Generá y valida",
        hero_copy="Rex-AI mezcla, explica contribuciones y exporta procesos listos para la tripulación.",
        card_body=(
            "Rex-AI mezcla ítems, compara heurística vs modelo y explica cada contribución en vivo."
        ),
        compact_card_body="Mezclá ítems, compará heurística vs IA y revisá contribuciones al instante.",
        icon="🤖",
        timeline_label="Generador IA",
        timeline_description="Explorá mezclas óptimas, revisá contribuciones y bandas de confianza en segundos.",
        footer="ML + heurística cooperativa",
    ),
    HeroFlowStage(
        key="results",
        order=4,
        name="Resultados",
        hero_headline="Reportá y exportá",
        hero_copy="Trade-offs, confianza 95% y comparativa heurística listos para ingeniería.",
        card_body=(
            "Trade-offs, bandas 95%, comparación heurística vs IA y export Sankey/feedback listos para ingeniería."
        ),
        compact_card_body="Revisá trade-offs, bandas 95% y export Sankey/feedback final.",
        icon="📊",
        timeline_label="Resultados y export",
        timeline_description="Compará heurísticas vs IA, exportá recetas y registra feedback para retraining.",
        footer="Listo para experimentos",
    ),
]

mission_metrics = [
    {
        "key": "status",
        "label": "Estado",
        "value": ready,
        "details": [f"Modelo <code>{model_name}</code>"],
        "caption": f"Nombre: {model_name}",
        "icon": "🛰️",
        "stage_key": "inventory",
        "tone": "accent",
    },
    {
        "key": "training",
        "label": "Entrenamiento",
        "value": trained_at_display,
        "details": [
            f"Origen: {trained_label_value}",
            f"Muestras: {n_samples or '—'}",
        ],
        "caption": f"Procedencia: {trained_label_value} · Muestras: {n_samples or '—'}",
        "icon": "🧪",
        "stage_key": "target",
        "tone": "info",
    },
    {
        "key": "feature_space",
        "label": "Feature space",
        "value": str(feature_count),
        "details": ["Fisicoquímica + proceso"],
        "caption": "Ingeniería fisicoquímica + proceso",
        "icon": "🧬",
        "stage_key": "generator",
    },
    {
        "key": "uncertainty",
        "label": "Incertidumbre",
        "value": model_registry.uncertainty_label(),
        "details": ["CI 95% visible en UI"],
        "caption": "CI 95% expuesta en UI",
        "icon": "📈",
        "stage_key": "results",
    },
]

hero_col, metrics_col = st.columns([2.8, 1.2], gap="large")
with hero_col:
    hero_scene = MinimalHero(
        title="Rex-AI orquesta el reciclaje orbital y marciano",
        subtitle=(
            "Un loop autónomo que mezcla regolito MGS-1, polímeros EVA y residuos de carga "
            "para fabricar piezas listas para misión. El copiloto gestiona riesgos, "
            "energía y trazabilidad sin perder contexto."
        ),
        chips=[
            {"label": "RandomForest multisalida", "tone": "accent"},
            {"label": "Comparadores heurísticos", "tone": "info"},
            {"label": "Crew telemetry ready", "tone": "accent"},
        ],
        icon="🛰️",
        gradient="linear-gradient(135deg, rgba(59,130,246,0.28), rgba(14,165,233,0.08))",
        glow="rgba(96,165,250,0.45)",
        density="roomy",
        metrics=mission_metrics,
        flow=mission_stages,
    )
    hero_scene.render()
with metrics_col:
    metrics_placeholder = st.empty()
    board_placeholder = st.empty()

mission_metric_payload = hero_scene.metrics_payload()
mission_metrics_component = MissionMetrics.from_payload(
    mission_metric_payload,
    title="Panel de misión",
    animate=False,
)
mission_board_payload = [
    {
        "key": "inventory",
        "title": "Inventario",
        "description": "Normalizá residuos NASA y marcá flags EVA o multilayer.",
        "href": "./?page=1_Inventory_Builder",
        "icon": "🧱",
    },
    {
        "key": "target",
        "title": "Target",
        "description": "Define objetivo, límites de agua/energía y presets marcianos.",
        "href": "./?page=2_Target_Designer",
        "icon": "🎯",
    },
    {
        "key": "generator",
        "title": "Generador",
        "description": "Compara recetas IA vs heurística y valida contribuciones.",
        "href": "./?page=3_Generator",
        "icon": "🤖",
    },
    {
        "key": "results",
        "title": "Resultados",
        "description": "Exportá trade-offs, bandas 95% y Sankey para ingeniería.",
        "href": "./?page=4_Results_and_Tradeoffs",
        "icon": "📊",
    },
]
mission_board_component = MissionBoard.from_payload(
    mission_board_payload,
    title="Próxima acción",
    reveal=False,
)
metrics_placeholder.markdown(
    mission_metrics_component.markup(with_board=True),
    unsafe_allow_html=True,
)
board_placeholder.markdown(
    mission_board_component.markup(),
    unsafe_allow_html=True,
)

# ──────────── Laboratorio profundo ────────────
st.markdown(
    """
    <section class="home-section" id="laboratorio-profundo">
      <div class="home-section__header">
        <span class="home-section__icon">🧪</span>
        <h2>Laboratorio profundo</h2>
      </div>
      <p class="home-section__lead">Radiografiamos el inventario NASA, destacamos masas críticas y exponemos hipótesis de proceso en paneles compactos.</p>
    </section>
    """,
    unsafe_allow_html=True,
)

inventory_df = load_inventory_sample()

category_items = []
if inventory_df is not None and not inventory_df.empty:
    category_summary = (
        inventory_df.groupby("category")[["mass_kg", "volume_l"]]
        .sum()
        .sort_values("mass_kg", ascending=False)
        .head(6)
    )
    for category, row in category_summary.iterrows():
        category_items.append(
            CarouselItem(
                title=category,
                value=format_mass(row["mass_kg"]),
                description=f"Volumen: {row['volume_l']:.0f} L",
            )
        )

if category_items:
    CarouselRail(
        items=category_items,
        data_track="categorias",
        reveal=False,
    ).render()

info_cards: list[str] = [
    """
    <article class="home-card">
      <h4>Ruta guiada de misión</h4>
      <ol class="home-card__list">
        <li>Inventario: normalizá residuos y marca flags EVA, multilayer y nitrilo.</li>
        <li>Target: define producto, límites de agua, energía y crew-time.</li>
        <li>Generador: Rex-AI mezcla ítems, sugiere procesos y explica cada paso.</li>
        <li>Resultados: trade-offs, confianza 95% y comparativa heurística.</li>
      </ol>
    </article>
    """
]

if inventory_df is not None:
    sample_materials = (
        inventory_df[["material", "material_family", "moisture_pct", "difficulty_factor"]]
        .head(4)
        .to_dict(orient="records")
    )
    if sample_materials:
        list_items = "".join(
            f"<li><strong>{item['material']}</strong> · {item['material_family']} · humedad {item['moisture_pct']}% · dificultad {item['difficulty_factor']}</li>"
            for item in sample_materials
        )
        info_cards.append(
            f"""
            <article class="home-card">
              <h4>Determinantes fisicoquímicos</h4>
              <ul class="home-card__list">{list_items}</ul>
            </article>
            """
        )

    flagged = (
        inventory_df["flags"]
        .dropna()
        .loc[lambda series: series.astype(str).str.len() > 0]
        .head(4)
        .tolist()
    )
    if flagged:
        bullet_items = "".join(
            f"<li>{flag}</li>" for flag in flagged if isinstance(flag, str)
        )
        info_cards.append(
            f"""
            <article class="home-card">
              <h4>Flags operativos activos</h4>
              <ul class="home-card__list">{bullet_items}</ul>
            </article>
            """
        )

if info_cards:
    st.markdown(
        f"<div class=\"home-card-stack\">{''.join(info_cards)}</div>",
        unsafe_allow_html=True,
    )
# ──────────── Ruta guiada ────────────
st.markdown("### Ruta de misión (guided flow)")

demo_steps = hero_scene.timeline_milestones()
active_demo_step = guided_demo(steps=demo_steps, step_duration=6.5)

active_stage_key = (
    hero_scene.stage_key_for_label(active_demo_step.label)
    if active_demo_step
    else None
)
metrics_placeholder.markdown(
    mission_metrics_component.markup(
        highlight_key=active_stage_key,
        with_board=True,
    ),
    unsafe_allow_html=True,
)
board_placeholder.markdown(
    mission_board_component.markup(highlight_key=active_stage_key),
    unsafe_allow_html=True,
)

# ──────────── Métricas de misión ────────────
MetricGalaxy(
    metrics=hero_scene.metric_items(),
    density="cozy",
).render()

st.info(
    "Usá el **Mission HUD** superior para saltar entre pasos o presioná las teclas `1-9` "
    "para navegar más rápido por el flujo guiado."
)
