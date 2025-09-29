# app/Home.py
import _bootstrap  # noqa: F401
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

from app.modules.luxe_components import (
    ActionCard,
    ActionDeck,
    BriefingCard,
    CarouselItem,
    CarouselRail,
    GlassCard,
    GlassStack,
    HeroFlowStage,
    MetricGalaxy,
    MissionMetrics,
    TeslaHero,
    TimelineMilestone,
    guided_demo,
    orbital_timeline,
)
from app.modules.ml_models import get_model_registry
from app.modules.navigation import set_active_step
from app.modules.ui_blocks import enable_reveal_animation, futuristic_button, load_theme

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
ready = "✅ Modelo listo" if model_registry.ready else "⚠️ Entrená localmente"

mission_stages = [
    HeroFlowStage(
        key="inventory",
        order=1,
        name="Inventario",
        hero_headline="Calibrá el inventario",
        hero_copy="Normalizá residuos, detectá flags EVA y estructuras multi-layer.",
        card_body="Normalizá residuos y marcá flags problemáticos (multilayer, EVA, nitrilo).",
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
        card_body="Elegí producto final y límites de agua, energía y crew para la misión.",
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
        card_body="Rex-AI mezcla ítems, sugiere proceso y explica cada predicción en vivo.",
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
        card_body="Trade-offs, confianza 95%, comparación heurística vs IA y export final.",
        icon="📊",
        timeline_label="Resultados y export",
        timeline_description="Compará heurísticas vs IA, exportá recetas y registra feedback para retraining.",
        footer="Listo para experimentos",
    ),
]

briefing_cards = [
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
    hero_scene = TeslaHero.with_briefing(
        title="Rex-AI orquesta el reciclaje orbital y marciano",
        subtitle=(
            "Un loop autónomo que mezcla regolito MGS-1, polímeros EVA y residuos de carga "
            "para fabricar piezas listas para misión. El copiloto gestiona riesgos, "
            "energía y trazabilidad sin perder contexto."
        ),
        tagline="Sincronizá sensores, crew y modelo para reciclar basura orbital en hardware vital.",
        video_url="https://cdn.coverr.co/videos/coverr-into-the-blue-nebula-9071/1080p.mp4",
        chips=[
            {"label": "RandomForest multisalida", "tone": "accent"},
            {"label": "Comparadores: XGBoost / Tabular", "tone": "info"},
            {"label": "Bandas de confianza 95%", "tone": "accent"},
            {"label": "Telemetría NASA · Crew safe", "tone": "info"},
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
        flow=mission_stages,
        briefing_video_path=Path(__file__).resolve().parent / "static" / "mission_briefing_loop.mp4",
        briefing_cards=briefing_cards,
        metrics=mission_metrics,
    )
with metrics_col:
    metrics_placeholder = st.empty()

mission_metric_payload = hero_scene.metrics_payload()
mission_metrics_component = MissionMetrics.from_payload(
    mission_metric_payload,
    title="Panel de misión",
)
metrics_placeholder.markdown(
    mission_metrics_component.markup(),
    unsafe_allow_html=True,
)

# ──────────── Laboratorio profundo ────────────
st.markdown(
    """
    <section class="lab-block reveal" id="laboratorio-profundo">
      <div class="section-title"><span class="icon">🧪</span><h2>Laboratorio profundo</h2></div>
      <p>Radiografiamos el inventario NASA, destacamos masas críticas y exponemos hipótesis de proceso en paneles compactos.</p>
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
    CarouselRail(items=category_items, data_track="categorias").render()

col_lab_a, col_lab_b = st.columns([1.6, 1], gap="large")

with col_lab_a:
    st.markdown(
        """
        <div class="lab-grid">
          <div class="drawer reveal">
            <h4>Ruta guiada de misión</h4>
            <ol>
              <li>Inventario: normalizá residuos y marca flags EVA, multilayer y nitrilo.</li>
              <li>Target: define producto, límites de agua, energía y crew-time.</li>
              <li>Generador: Rex-AI mezcla ítems, sugiere procesos y explica cada paso.</li>
              <li>Resultados: trade-offs, confianza 95% y comparativa heurística.</li>
            </ol>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col_lab_b:
    composition_toggle = st.toggle(
        "Mostrar composición científica de la receta base",
        value=False,
        key="toggle_composicion",
    )

    if composition_toggle and inventory_df is not None:
        sample_materials = (
            inventory_df[
                ["material", "material_family", "moisture_pct", "difficulty_factor"]
            ]
            .head(5)
            .to_dict(orient="records")
        )
        list_items = "".join(
            f"<li><strong>{item['material']}</strong> · {item['material_family']} · humedad {item['moisture_pct']}% · dificultad {item['difficulty_factor']}</li>"
            for item in sample_materials
        )
        st.markdown(
            f"""
            <div class="drawer reveal">
              <h4>Determinantes fisicoquímicos</h4>
              <ul>{list_items}</ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

scenario_toggle = st.toggle(
    "Mostrar banderas críticas y telemetría",
    value=False,
    key="toggle_flags",
)
# ──────────── Ruta guiada ────────────
st.markdown("### Ruta de misión (guided flow)")

demo_steps = hero_scene.timeline_milestones()
active_demo_step = guided_demo(steps=demo_steps, step_duration=6.5)
GlassStack(
    cards=hero_scene.glass_cards(),
    columns_min="15rem",
    density="compact",
).render()

active_stage_key = (
    hero_scene.stage_key_for_label(active_demo_step.label)
    if active_demo_step
    else None
)
metrics_placeholder.markdown(
    mission_metrics_component.markup(highlight_key=active_stage_key),
    unsafe_allow_html=True,
)

if scenario_toggle and inventory_df is not None:
    flagged = inventory_df["flags"].dropna().head(6).tolist()
    bullet_items = "".join(
        f"<li>{flag}</li>" for flag in flagged if isinstance(flag, str)
    )
    st.markdown(
        f"""
        <div class="drawer reveal">
          <h4>Flags operativos activos</h4>
          <ul>{bullet_items}</ul>
          <div class="timeline">
            <ul>
              <li>Pipeline reproducible: <code>python -m app.modules.model_training</code> genera dataset + RandomForest.</li>
              <li>Trazabilidad completa: cada receta incorpora IDs, categorías y metadatos.</li>
              <li>Explicabilidad integrada: contribuciones por feature y bandas de confianza 95%.</li>
              <li>Comparativa heurística vs IA lista para export.</li>
            </ul>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ──────────── Acciones siguientes ────────────
st.markdown(
    """
    <section class="next-block reveal" id="acciones-siguientes">
      <div class="section-title"><span class="icon">🚀</span><h2>Acciones siguientes</h2></div>
      <p>Mantené el contexto del laboratorio mientras ejecutás exports, simulaciones y reportes clave.</p>
    </section>
    """,
    unsafe_allow_html=True,
)

cta_col1, cta_col2 = st.columns(2, gap="large")
with cta_col1:
    ActionDeck(
        cards=[
            ActionCard(
                title="Exportar receta y telemetría",
                body="Descargá reportes con Sankey, contribuciones y feedback para seguimiento.",
                icon="📤",
            )
        ],
        columns_min="18rem",
    ).render()

    export_state_key = "home_cta_export_state"
    if st.session_state.get(export_state_key) == "loading":
        st.session_state[export_state_key] = "success"

    export_state = st.session_state.setdefault(export_state_key, "idle")
    if futuristic_button(
        "Exportar\nreceta y telemetría",
        key="home_cta_export",
        icon="📤",
        state=export_state,
        loading_label="Generando reporte…",
        success_label="Reporte enviado",
        help_text="Descargá Sankey, contribuciones y feedback para seguimiento.",
    ):
        st.session_state[export_state_key] = "loading"
        st.switch_page("pages/4_Results_and_Tradeoffs.py")

with cta_col2:
    ActionDeck(
        cards=[
            ActionCard(
                title="Simular escenarios",
                body="Prueba configuraciones de energía, crew y materiales para stress tests.",
                icon="🧮",
            )
        ],
        columns_min="18rem",
    ).render()

    sim_state_key = "home_cta_simulation_state"
    if st.session_state.get(sim_state_key) == "loading":
        st.session_state[sim_state_key] = "success"

    sim_state = st.session_state.setdefault(sim_state_key, "idle")
    if futuristic_button(
        "Simular\nescenarios",
        key="home_cta_simulation",
        icon="🧮",
        state=sim_state,
        loading_label="Lanzando simulación…",
        success_label="Escenarios listos",
        help_text="Prueba configuraciones de energía, crew y materiales para stress tests.",
    ):
        st.session_state[sim_state_key] = "loading"
        st.switch_page("pages/2_Target_Designer.py")

st.markdown(
    mission_metrics_component.markup(
        layout="grid",
        highlight_key=active_stage_key,
        detail_limit=2,
        show_title=False,
    ),
    unsafe_allow_html=True,
)

ActionDeck(
    cards=[
        ActionCard(
            title="1. Inventario NASA",
            body="Trabajá sobre <code>data/waste_inventory_sample.csv</code> o subí tu CSV normalizado.",
        ),
        ActionCard(
            title="2. Objetivo",
            body="Usá presets (container, utensil, tool, interior) o definí límites manuales.",
        ),
        ActionCard(
            title="3. Generador con IA",
            body="Revisá contribuciones de features y compará heurística vs modelo.",
        ),
        ActionCard(
            title="4. Reportar",
            body="Exportá recetas, Sankey y feedback/impact para seguir entrenando Rex-AI.",
        ),
    ],
    columns_min="15rem",
    density="cozy",
).render()

ActionDeck(
    cards=[
        ActionCard(
            title="Construir inventario",
            body="Normalizá residuos NASA y etiquetá flags EVA, multilayer y nitrilo.",
            icon="🧱",
        ),
        ActionCard(
            title="Generador IA vs heurística",
            body="Compara recetas propuestas, trade-offs y bandas de confianza.",
            icon="🤖",
        ),
    ],
    columns_min="14rem",
    density="cozy",
).render()

# ──────────── CTA navegación ────────────
st.markdown("### Siguiente acción")
c1, c2, c3, c4 = st.columns(4)
with c1:
    if futuristic_button(
        "🧱 Inventario",
        key="home_nav_inventory",
        help_text="Subí o limpiá tu CSV base",
        loading_label="Abriendo…",
        success_label="Inventario listo",
        status_hints={
            "idle": "",
            "loading": "Preparando pantalla",
            "success": "UI cargada",
            "error": "",
        },
    ):
        st.switch_page("pages/1_Inventory_Builder.py")
with c2:
    if futuristic_button(
        "🎯 Target",
        key="home_nav_target",
        help_text="Configura límites y objetivos",
        loading_label="Abriendo…",
        success_label="Target listo",
        status_hints={
            "idle": "",
            "loading": "Cargando diseñador",
            "success": "UI cargada",
            "error": "",
        },
    ):
        st.switch_page("pages/2_Target_Designer.py")
with c3:
    if futuristic_button(
        "🤖 Generador",
        key="home_nav_generator",
        help_text="Corre IA o heurísticas",
        loading_label="Abriendo…",
        success_label="Generador listo",
        status_hints={
            "idle": "",
            "loading": "Cargando modelo",
            "success": "UI cargada",
            "error": "",
        },
    ):
        st.switch_page("pages/3_Generator.py")
with c4:
    if futuristic_button(
        "📊 Resultados",
        key="home_nav_results",
        help_text="Analiza trade-offs y export",
        loading_label="Abriendo…",
        success_label="Resultados listos",
        status_hints={
            "idle": "",
            "loading": "Abriendo dashboards",
            "success": "UI cargada",
            "error": "",
        },
    ):
        st.switch_page("pages/4_Results_and_Tradeoffs.py")
cta_buttons = st.columns(2)
with cta_buttons[0]:
    inventory_state_key = "home_cta_inventory_state"
    if st.session_state.get(inventory_state_key) == "loading":
        st.session_state[inventory_state_key] = "success"
    inventory_state = st.session_state.setdefault(inventory_state_key, "idle")
    if futuristic_button(
        "Abrir\ninventario",
        key="cta_inventory",
        icon="🧱",
        state=inventory_state,
        loading_label="Abriendo inventario…",
        success_label="Inventario listo",
        width="full",
    ):
        st.session_state[inventory_state_key] = "loading"
        st.switch_page("pages/1_Inventory_Builder.py")
with cta_buttons[1]:
    generator_state_key = "home_cta_generator_state"
    if st.session_state.get(generator_state_key) == "loading":
        st.session_state[generator_state_key] = "success"
    generator_state = st.session_state.setdefault(generator_state_key, "idle")
    if futuristic_button(
        "Abrir\ngenerador",
        key="cta_generator",
        icon="🤖",
        state=generator_state,
        loading_label="Activando Rex-AI…",
        success_label="Generador listo",
        width="full",
    ):
        st.session_state[generator_state_key] = "loading"
        st.switch_page("pages/3_Generator.py")

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
# ──────────── Animación de aparición por scroll ────────────
enable_reveal_animation()
MetricGalaxy(
    metrics=hero_scene.metric_items(),
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
