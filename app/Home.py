# app/Home.py
import _bootstrap  # noqa: F401

from datetime import datetime
from pathlib import Path
from textwrap import dedent

import pandas as pd
import streamlit as st

from app.modules.luxe_components import (
    BriefingCard,
    TimelineMilestone,
    guided_demo,
    mission_briefing,
    orbital_timeline,
    GlassCard,
    GlassStack,
    MetricGalaxy,
    MetricItem,
    TeslaHero,
)
from app.modules.ml_models import get_model_registry
from app.modules.navigation import set_active_step
from app.modules.ui_blocks import futuristic_button, load_theme

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


# ──────────── Estilos y animaciones globales ────────────
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
      background: var(--chip-bg); border: 1px solid var(--chip-border); color: var(--chip-ink);
    }
    .ghost-card {
      margin-top: 38px; display:grid; gap:18px; grid-template-columns: repeat(auto-fit,minmax(260px,1fr));
    }
    .ghost-card > div {
      padding:20px 22px; border-radius:20px; border:1px solid var(--border-soft);
      background: var(--surface-card); color: var(--ink);
    }
    .ghost-card h3 {margin-bottom:6px; font-size:1.05rem;}
    .ghost-card p {color: var(--muted); font-size:0.94rem; margin:0;}
    .stepper {display:grid; grid-template-columns: repeat(auto-fit,minmax(180px,1fr)); gap:14px; margin: 32px 0 18px;}
    .step {border-radius:18px; border:1px solid var(--step-border); padding:16px 18px; background: var(--step-bg);}
    .step span {display:inline-flex; width:32px; height:32px; border-radius:999px; align-items:center; justify-content:center; background: var(--step-icon-bg); color: var(--ink); font-weight:700; margin-bottom:10px;}
    .step h4 {margin:0 0 6px 0;}
    .step p {color: var(--muted); font-size:0.9rem; margin:0;}
    .metric-grid {display:grid; grid-template-columns: repeat(auto-fit,minmax(210px,1fr)); gap:16px; margin-top: 18px;}
    .metric {border-radius:18px; padding:18px 20px; background:var(--metric-bg); border:1px solid var(--metric-border); color:var(--ink); box-shadow: var(--metric-shadow);}
    .metric h5 {margin:0; font-size:0.92rem; color:var(--muted);}
    .metric strong {font-size:1.4rem; display:block; margin-top:6px;}
    </style>
    """,
    dedent(
        """
        <style>
        .main > div {padding-top: 1.5rem;}
        section {width: 100%;}
        .overview-block,
        .lab-block,
        .next-block {margin-bottom: 4rem;}
        .reveal {opacity: 0; transform: translateY(42px); transition: opacity 0.8s ease, transform 0.8s ease;}
        .reveal.is-visible {opacity: 1; transform: translateY(0);}
        .tesla-hero {position: relative; border-radius: 28px; overflow: hidden; min-height: 360px; display:flex; align-items:flex-end;}
        .tesla-hero__bg {position:absolute; inset:0; width:100%; height:100%; object-fit:cover; filter: saturate(1.3) brightness(0.7);}
        .tesla-hero__veil {position:absolute; inset:0; background: linear-gradient(180deg, rgba(15,23,42,0.05) 10%, rgba(2,6,23,0.85) 100%);}
        .tesla-hero__content {position:relative; padding: 48px; max-width: 760px;}
        .tesla-hero__content h1 {font-size: 2.6rem; margin-bottom: 16px; letter-spacing: 0.01em;}
        .tesla-hero__content p {font-size: 1.05rem; color: var(--muted); margin-bottom: 18px;}
        .chip-row {display:flex; gap:8px; flex-wrap:wrap;}
        .chip {padding:6px 16px; border-radius:999px; font-size:0.82rem; font-weight:600; background: rgba(12,28,64,0.75); border:1px solid rgba(96,165,250,0.35); color: var(--ink); backdrop-filter: blur(12px);}
        .sticky-panel {position: sticky; top: 84px; border-radius: 22px; border: 1px solid rgba(148,163,184,0.18); padding: 24px 26px; background: rgba(10,18,36,0.82); backdrop-filter: blur(18px);}
        .sticky-panel h3 {margin-bottom: 18px;}
        .sticky-panel .metric {border-radius:16px; padding:14px 16px; background: rgba(15,23,42,0.65); border:1px solid rgba(148,163,184,0.12); margin-bottom: 14px;}
        .sticky-panel .metric:last-child {margin-bottom: 0;}
        .sticky-panel .metric h5 {margin:0; font-size:0.85rem; text-transform:uppercase; letter-spacing:0.08em; color:rgba(148,163,184,0.9);}
        .sticky-panel .metric strong {display:block; margin-top:4px; font-size:1.3rem;}
        .sticky-panel .metric p {margin:0; color: var(--muted); font-size:0.85rem;}
        .section-title {display:flex; align-items:center; gap:12px; margin-bottom: 14px;}
        .section-title span.icon {width:34px; height:34px; border-radius:12px; display:flex; align-items:center; justify-content:center; background: rgba(59,130,246,0.12); color: var(--accent); font-size:1.1rem;}
        .lab-grid {display:grid; gap:24px; grid-template-columns: minmax(0,1fr);}
        .carousel {display:flex; gap:16px; overflow-x:auto; padding-bottom: 6px; scroll-snap-type: x mandatory;}
        .carousel::-webkit-scrollbar {height:6px;}
        .carousel::-webkit-scrollbar-thumb {background: rgba(59,130,246,0.45); border-radius:999px;}
        .carousel-card {min-width: 240px; scroll-snap-align: start; border-radius: 18px; border: 1px solid rgba(148,163,184,0.16); padding:18px; background: rgba(12,18,34,0.75);}
        .carousel-card h4 {margin:0 0 6px 0; font-size:1.02rem;}
        .carousel-card .value {font-size:1.3rem; font-weight:700;}
        .carousel-card p {margin:4px 0 0 0; font-size:0.85rem; color: var(--muted);}
        .drawer {border-radius: 18px; border: 1px solid rgba(148,163,184,0.2); padding:18px 20px; background: rgba(10,18,36,0.7); margin-top:16px;}
        .drawer h4 {margin-top:0;}
        .drawer ul {margin:0; padding-left: 20px; color: var(--muted);}
        .timeline {margin-top: 18px; border-left: 2px solid rgba(96,165,250,0.25); padding-left: 18px;}
        .timeline li {margin-bottom: 14px; color: var(--muted);}
        .cta-grid {display:grid; gap:22px; grid-template-columns: repeat(auto-fit, minmax(240px,1fr));}
        .cta-card {border-radius: 22px; border:1px solid rgba(96,165,250,0.28); padding:22px 24px; background: linear-gradient(160deg, rgba(59,130,246,0.16), rgba(15,23,42,0.78)); position:relative;}
        .cta-card strong {display:block; font-size:1.18rem; margin-bottom:8px;}
        .cta-card p {margin:0; color: var(--muted);}
        .cta-card button {margin-top:18px; width:100%;}
        .cta-card .icon {position:absolute; top:16px; right:18px; font-size:1.4rem; opacity:0.8;}
        @media (max-width: 992px) {
          .tesla-hero__content {padding:32px 24px;}
          .sticky-panel {position:relative; top:auto; margin-top:24px;}
        }
        </style>
        """
    ),
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
ready = "✅ Modelo listo" if model_registry.ready else "⚠️ Entrená localmente"

# ──────────── Overview cinematográfico ────────────
hero_col, metrics_col = st.columns([2.8, 1.2], gap="large")
with hero_col:
    TeslaHero(
        title="Rex-AI orquesta el reciclaje orbital y marciano",
        subtitle=(
            "Un loop autónomo que mezcla regolito MGS-1, polímeros EVA y residuos de carga "
            "para fabricar piezas listas para misión. El copiloto gestiona riesgos, "
            "energía y trazabilidad sin perder contexto."
        ),
        chips=[
            "RandomForest multisalida",
            "Comparadores XGBoost / Tabular",
            "Bandas de confianza 95%",
            "Telemetría NASA · Crew safe",
        ],
        video_url="https://cdn.coverr.co/videos/coverr-into-the-blue-nebula-9071/1080p.mp4",
    ).render()

with metrics_col:
    st.markdown(
        f"""
        <aside class="sticky-panel reveal" id="sticky-metrics">
          <h3>Panel de misión</h3>
          <div class="metric">
            <h5>Estado</h5>
            <strong>{ready}</strong>
            <p>Modelo <code>{model_name}</code></p>
          </div>
          <div class="metric">
            <h5>Entrenamiento</h5>
            <strong>{trained_at_display}</strong>
            <p>Origen: {trained_label_value}</p>
            <p>Muestras: {n_samples or '—'}</p>
          </div>
          <div class="metric">
            <h5>Feature space</h5>
            <strong>{feature_count}</strong>
            <p>Fisicoquímica + proceso</p>
          </div>
          <div class="metric">
            <h5>Incertidumbre</h5>
            <strong>{model_registry.uncertainty_label()}</strong>
            <p>CI 95% visible en UI</p>
          </div>
        </aside>
        """,
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

category_cards = ""
if inventory_df is not None and not inventory_df.empty:
    category_summary = (
        inventory_df.groupby("category")[["mass_kg", "volume_l"]]
        .sum()
        .sort_values("mass_kg", ascending=False)
        .head(6)
    )
    for category, row in category_summary.iterrows():
        category_cards += (
            f"<div class='carousel-card'>"
            f"<h4>{category}</h4>"
            f"<div class='value'>{format_mass(row['mass_kg'])}</div>"
            f"<p>Volumen: {row['volume_l']:.0f} L</p>"
            f"</div>"
        )

if category_cards:
    st.markdown(
        f"""
        <div class="carousel reveal" data-carousel="categorias">
          {category_cards}
        </div>
        """,
        unsafe_allow_html=True,
    )

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
    st.markdown(
        """
        <div class="cta-grid">
          <div class="cta-card reveal">
            <span class="icon">📤</span>
            <strong>Exportar receta y telemetría</strong>
            <p>Descargá reportes con Sankey, contribuciones y feedback para seguimiento.</p>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if st.button("📤 Exportar", use_container_width=True):
        st.switch_page("pages/4_Results_and_Tradeoffs.py")

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
with cta_col2:
    st.markdown(
        """
        <div class="cta-grid">
          <div class="cta-card reveal">
            <span class="icon">🧮</span>
            <strong>Simular escenarios</strong>
            <p>Prueba configuraciones de energía, crew y materiales para stress tests.</p>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if st.button("🧮 Simular escenarios", use_container_width=True):
        st.switch_page("pages/2_Target_Designer.py")

st.markdown(
    """
    <div class="mission-grid">
      <div><h3>1. Inventario NASA</h3><p>Trabajá sobre <code>data/waste_inventory_sample.csv</code> o subí tu CSV normalizado.</p></div>
      <div><h3>2. Objetivo</h3><p>Usá presets (container, utensil, tool, interior) o definí límites manuales.</p></div>
      <div><h3>3. Generador con IA</h3><p>Revisá contribuciones de features y compará heurística vs modelo.</p></div>
      <div><h3>4. Reportar</h3><p>Exportá recetas, Sankey y feedback/impact para seguir entrenando Rex-AI.</p></div>
    <div class="cta-grid" style="margin-top: 12px;">
      <div class="cta-card reveal">
        <span class="icon">🧱</span>
        <strong>Construir inventario</strong>
        <p>Normalizá residuos NASA y etiquetá flags EVA, multilayer y nitrilo.</p>
      </div>
      <div class="cta-card reveal">
        <span class="icon">🤖</span>
        <strong>Generador IA vs heurística</strong>
        <p>Compara recetas propuestas, trade-offs y bandas de confianza.</p>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

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
    if st.button("🧱 Abrir inventario", use_container_width=True, key="cta_inventory"):
        st.switch_page("pages/1_Inventory_Builder.py")
with cta_buttons[1]:
    if st.button("🤖 Abrir generador", use_container_width=True, key="cta_generator"):
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
# ──────────── Animación de aparición por scroll ────────────
st.markdown(
    """
    <script>
      const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
          if (entry.isIntersecting) {
            entry.target.classList.add('is-visible');
          }
        });
      }, {threshold: 0.2});

      document.querySelectorAll('.reveal').forEach((element) => {
        observer.observe(element);
      });
    </script>
    """,
    unsafe_allow_html=True,
)
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
