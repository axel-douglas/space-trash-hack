# app/Home.py
import _bootstrap  # noqa: F401

from datetime import datetime
from pathlib import Path
from textwrap import dedent

import pandas as pd
import streamlit as st

from app.modules.ml_models import get_model_registry
from app.modules.ui_blocks import load_theme

st.set_page_config(
    page_title="Rex-AI • Mission Copilot",
    page_icon="🛰️",
    layout="wide",
)

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


def tesla_hero(title: str, subtitle: str, chips: list[str], video_url: str) -> None:
    chip_markup = "".join(f"<span class='chip'>{chip}</span>" for chip in chips)
    st.markdown(
        f"""
        <section class="overview-block reveal" id="overview-cinematic">
          <div class="tesla-hero">
            <video class="tesla-hero__bg" autoplay muted loop playsinline>
              <source src="{video_url}" type="video/mp4" />
            </video>
            <div class="tesla-hero__veil"></div>
            <div class="tesla-hero__content">
              <h1>{title}</h1>
              <p>{subtitle}</p>
              <div class="chip-row">{chip_markup}</div>
            </div>
          </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def format_mass(value: float | int | None) -> str:
    if value is None:
        return "—"
    if value >= 1000:
        return f"{value/1000:.1f} t"
    return f"{value:.0f} kg"


# ──────────── Estilos y animaciones globales ────────────
st.markdown(
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

ready = "✅ Modelo listo" if model_registry.ready else "⚠️ Entrená localmente"

# ──────────── Overview cinematográfico ────────────
hero_col, metrics_col = st.columns([2.8, 1.2], gap="large")
with hero_col:
    tesla_hero(
        title="Rex-AI orquesta el reciclaje orbital y marciano",
        subtitle=(
            "Un loop autónomo que mezcla regolito MGS-1, polímeros EVA y residuos de carga "
            "para fabricar piezas listas para misión. El copiloto gestiona riesgos, "
            "energía y trazabilidad sin perder contexto.")
        ,
        chips=[
            "RandomForest multisalida",
            "Comparadores XGBoost / Tabular",
            "Bandas de confianza 95%",
            "Telemetría NASA · Crew safe",
        ],
        video_url="https://cdn.coverr.co/videos/coverr-into-the-blue-nebula-9071/1080p.mp4",
    )

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

cta_buttons = st.columns(2)
with cta_buttons[0]:
    if st.button("🧱 Abrir inventario", use_container_width=True, key="cta_inventory"):
        st.switch_page("pages/1_Inventory_Builder.py")
with cta_buttons[1]:
    if st.button("🤖 Abrir generador", use_container_width=True, key="cta_generator"):
        st.switch_page("pages/3_Generator.py")

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
