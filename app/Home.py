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
    MetricItem,
    MissionBoard,
    MissionMetrics,
    TeslaHero,
    TimelineMilestone,
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
        hero_headline="Prepará el inventario",
        hero_copy="Normalizá residuos y registrá flags EVA o multilayer.",
        card_body=(
            "Normalizá residuos en <code>data/waste_inventory_sample.csv</code> o en tu CSV y "
            "registrá flags EVA, multilayer o nitrilo."
        ),
        compact_card_body=(
            "Normalizá residuos y flags EVA o multilayer en <code>data/waste_inventory_sample.csv</code>."
        ),
        icon="🧱",
        timeline_label="Inventario en vivo",
        timeline_description="Cargá CSV NASA, normalizá unidades y marcá riesgos EVA.",
        footer="Dataset NASA y flags de crew",
    ),
    HeroFlowStage(
        key="target",
        order=2,
        name="Target",
        hero_headline="Definí el objetivo",
        hero_copy="Configurá límites de agua, energía y crew-time con presets marcianos.",
        card_body=(
            "Elegí producto final, límites de agua y energía y presets marcianos (container, utensil, tool, interior)."
        ),
        compact_card_body="Elegí producto y límites con presets marcianos certificados.",
        icon="🎯",
        timeline_label="Target marciano",
        timeline_description="Seleccioná producto, límites de agua y energía o usá presets homologados.",
        footer="Presets y límites manuales",
    ),
    HeroFlowStage(
        key="generator",
        order=3,
        name="Generador",
        hero_headline="Generá y validá",
        hero_copy="Combiná residuos, compará IA vs heurística y verificá contribuciones.",
        card_body=(
            "Rex-AI mezcla ítems, contrasta heurística con modelo y detalla cada contribución en vivo."
        ),
        compact_card_body="Mezclá ítems, compará IA vs heurística y revisá contribuciones al instante.",
        icon="🤖",
        timeline_label="Generador IA",
        timeline_description="Explorá mezclas, revisá contribuciones y bandas de confianza en segundos.",
        footer="ML y heurística cooperativa",
    ),
    HeroFlowStage(
        key="results",
        order=4,
        name="Resultados",
        hero_headline="Reportá y exportá",
        hero_copy="Compartí trade-offs, confianza 95% y comparativas para ingeniería.",
        card_body=(
            "Trade-offs, bandas 95%, comparación heurística vs IA y export de Sankey o feedback listos para ingeniería."
        ),
        compact_card_body="Revisá trade-offs, bandas 95% y exportá Sankey o feedback final.",
        icon="📊",
        timeline_label="Resultados y export",
        timeline_description="Compará heurística e IA, exportá recetas y registrá feedback para retraining.",
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
    TeslaHero(
        title="Rex-AI coordina el reciclaje orbital y marciano",
        subtitle=(
            "8 astronautas generan 12.6 t de residuos en misión y Rex-AI los convierte en "
            "equipamiento listo. Automatiza mezclas con regolito MGS-1 del cráter Jezero, "
            "polímeros EVA y residuos de carga útil para entregar piezas auditables y trazables."
        ),
        chips=[
            {"label": "Hook: 8 astronautas → 12.6 t", "tone": "warning"},
            {"label": "Playbook • Residence Renovations", "tone": "accent"},
            {"label": "Playbook • Daring Discoveries", "tone": "accent"},
            {"label": "RandomForest multisalida", "tone": "info"},
        ],
        icon="🛰️",
        gradient="linear-gradient(135deg, rgba(59,130,246,0.28), rgba(14,165,233,0.08))",
        glow="rgba(96,165,250,0.45)",
        density="roomy",
        variant="minimal",
    ).render()
with metrics_col:
    metrics_placeholder = st.empty()
    board_placeholder = st.empty()

mission_metric_payload = []
for metric in mission_metrics:
    normalized = dict(metric)
    if "label" in normalized:
        normalized["label"] = str(normalized["label"])
    if "value" in normalized:
        normalized["value"] = str(normalized["value"])
    mission_metric_payload.append(normalized)
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
        "description": "Definí objetivo, límites de agua y energía y presets marcianos.",
        "href": "./?page=2_Target_Designer",
        "icon": "🎯",
    },
    {
        "key": "generator",
        "title": "Generador",
        "description": "Compará recetas IA y heurística y validá contribuciones.",
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
timeline_milestones = [
    TimelineMilestone(
        label=stage.timeline_label,
        description=stage.timeline_description,
        icon=stage.icon,
    )
    for stage in mission_stages
]
stage_by_label = {stage.timeline_label: stage.key for stage in mission_stages}
hero_metric_items = [
    MetricItem(
        label=str(metric.get("label", "")),
        value=str(metric.get("value", "")),
        caption=metric.get("caption"),
        delta=metric.get("delta"),
        icon=metric.get("icon"),
        tone=metric.get("tone"),
    )
    for metric in mission_metrics
]
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
      <p class="home-section__lead">Analizamos el inventario NASA, destacamos masas críticas y mostramos hipótesis de proceso en paneles compactos.</p>
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

# Tarjetas de escenarios con inputs/outputs clave
scenario_cards = [
    {
        "name": "Residence Renovations",
        "inputs": [
            "Marcos y CTB de aluminio reutilizados",
            "Espumas ZOTEK/bubble wrap y films MLI",
            "Opcional: regolito MGS-1 para refuerzos",
        ],
        "outputs": [
            "Estanterías y particiones modulares",
            "Paneles aislantes laminados para habitat",
        ],
        "why": (
            "Maximiza puntos al transformar masa estructural pesada en mejoras de habitabilidad "
            "con bajo crew-time y alto puntaje de resiliencia térmica."
        ),
    },
    {
        "name": "Cosmic Celebrations",
        "inputs": [
            "Textiles limpios y wipes de poliéster/nylon",
            "Films multicapa encapsulados",
            "Herrajes CTB o clips reutilizables",
        ],
        "outputs": [
            "Utilería y decoración segura sin agua",
            "Elementos modulares para morale boost",
        ],
        "why": (
            "Maximiza puntos morales y de bajo consumo al priorizar procesos secos de rápido "
            "ensamblaje y energía mínima."
        ),
    },
    {
        "name": "Daring Discoveries",
        "inputs": [
            "Carbono residual clasificado",
            "Meshes metálicas/poliméricas",
            "Polímeros y MGS-1 para sinterizado",
        ],
        "outputs": [
            "Componentes rígidos para ciencia y filtros",
            "Superficies reforzadas anti-impacto",
        ],
        "why": (
            "Maximiza puntos científicos al entregar piezas de alta rigidez y trazabilidad "
            "que habilitan experimentos críticos con mínima merma."
        ),
    },
]

for card in scenario_cards:
    inputs_html = "".join(f"<li>{item}</li>" for item in card["inputs"])
    outputs_html = "".join(f"<li>{item}</li>" for item in card["outputs"])
    info_cards.append(
        f"""
        <article class="home-card">
          <h4>{card['name']}</h4>
          <p><strong>Inputs clave</strong></p>
          <ul class="home-card__list">{inputs_html}</ul>
          <p><strong>Outputs estrella</strong></p>
          <ul class="home-card__list">{outputs_html}</ul>
          <p class="home-card__note">¿Por qué maximiza puntos? {card['why']}</p>
        </article>
        """
    )

if info_cards:
    st.markdown(
        f"<div class=\"home-card-stack\">{''.join(info_cards)}</div>",
        unsafe_allow_html=True,
    )
# ──────────── Ruta guiada ────────────
st.markdown("### Ruta de misión")

demo_steps = timeline_milestones
active_demo_step = guided_demo(steps=demo_steps, step_duration=6.5)

active_stage_key = (
    stage_by_label.get(active_demo_step.label)
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
    metrics=hero_metric_items,
    density="cozy",
).render()

st.info(
    "Usá el **Mission HUD** superior para saltar entre pasos o presioná las teclas `1-9` "
    "para navegar rápido por el flujo guiado."
)
st.caption(
    "Trash → Tools → Survival: cada feedback acelera el salto de residuo a herramienta "
    "y de herramienta a supervivencia marciana."
)
