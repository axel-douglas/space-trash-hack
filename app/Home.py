# app/Home.py
import _bootstrap  # noqa: F401
from datetime import datetime, timezone
from pathlib import Path
import textwrap

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


_SCENARIO_CONFIG: dict[str, dict[str, object]] = {
    "residences": {
        "label": "Residences",
        "badge_tone": "residences",
        "priority": 0,
    },
    "daring": {
        "label": "Daring",
        "badge_tone": "daring",
        "priority": 1,
    },
}

_CATEGORY_SCENARIO_MAP = {
    "foam": "residences",
    "foam packaging": "residences",
    "packaging": "residences",
    "food packaging": "residences",
    "structural elements": "residences",
    "eva waste": "daring",
    "fabrics": "daring",
    "gloves": "daring",
}

_SCENARIO_KEYWORDS = {
    "residences": ("foam", "pack", "alumin", "struct"),
    "daring": ("carbon", "mesh", "eva", "fabric", "glove"),
}


def map_category_to_scenario(category: str) -> str:
    normalized = (category or "").strip().lower()
    if not normalized:
        return "residences"
    if normalized in _CATEGORY_SCENARIO_MAP:
        return _CATEGORY_SCENARIO_MAP[normalized]
    for scenario, keywords in _SCENARIO_KEYWORDS.items():
        if any(keyword in normalized for keyword in keywords):
            return scenario
    return "residences"


def _tone_rank(tone: str | None) -> int:
    order = {"positive": 0, "info": 1, "warning": 2, "danger": 3}
    return order.get(tone or "", -1)


def _tone_max(current: str, candidate: str) -> str:
    return candidate if _tone_rank(candidate) > _tone_rank(current) else current


def training_health_summary(
    trained_dt: datetime | str | None, n_samples: int | None
) -> tuple[str, list[str]]:
    """Return tone and bullet points describing training freshness."""

    tone = "positive"
    notes: list[str] = []

    normalized_dt: datetime | None = None

    if not trained_dt:
        notes.append("Sin fecha de entrenamiento registrada")
        tone = "danger"
    else:
        parsed_note: str | None = None
        if isinstance(trained_dt, datetime):
            normalized_dt = trained_dt
        elif isinstance(trained_dt, str):
            iso_str = trained_dt.strip()
            if iso_str.endswith("Z"):
                iso_str = iso_str[:-1] + "+00:00"
            try:
                normalized_dt = datetime.fromisoformat(iso_str)
            except ValueError:
                parsed_note = "Formato de fecha no reconocido"
        else:
            parsed_note = f"Tipo de fecha inesperado: {type(trained_dt).__name__}"

        if normalized_dt is None:
            tone = _tone_max(tone, "warning")
            notes.append(parsed_note or "Fecha de entrenamiento no interpretable")
        else:
            try:
                if normalized_dt.tzinfo is None:
                    normalized_dt = normalized_dt.replace(tzinfo=timezone.utc)
                else:
                    normalized_dt = normalized_dt.astimezone(timezone.utc)
                age_days = max((datetime.now(timezone.utc) - normalized_dt).days, 0)
                notes.append(f"Edad del modelo: {age_days} días")
                if age_days > 180:
                    tone = "danger"
                    notes.append("⚠️ Reentrená: supera 6 meses")
                elif age_days > 90:
                    tone = _tone_max(tone, "warning")
                    notes.append("Sugerido reentrenar en <90 días")
                elif age_days <= 30:
                    notes.append("Entrenamiento reciente (<30 días)")
            except Exception as exc:  # pragma: no cover - defensive
                tone = _tone_max(tone, "warning")
                notes.append(f"No se pudo normalizar fecha: {exc}")

    sample_count = int(n_samples or 0)
    if sample_count <= 0:
        tone = "danger"
        notes.append("Sin muestras declaradas")
    else:
        notes.append(f"Muestras: {sample_count:,}")
        if sample_count < 400:
            tone = _tone_max(tone, "warning")
            notes.append("Amplía dataset: <400 muestras")
        elif sample_count >= 1000:
            notes.append("Cobertura sólida (≥1k)")

    return tone, notes


def uncertainty_health_summary(metadata: dict[str, object]) -> tuple[str, list[str]]:
    """Compute tone and highlights from residual_std metadata."""

    residual_std = metadata.get("residual_std")
    if not isinstance(residual_std, dict) or not residual_std:
        return "danger", ["Sin residuales reportados"]

    def _get_float(key: str, default: float = 0.0) -> float:
        try:
            return float(residual_std.get(key, default))
        except (TypeError, ValueError):
            return default

    energy_std = _get_float("energy_kwh")
    water_std = _get_float("water_l")
    crew_std = _get_float("crew_min")
    rigidity_std = _get_float("rigidez")
    tightness_std = _get_float("estanqueidad")

    tone = "positive"
    if any(value == 0 for value in (energy_std, water_std, crew_std)):
        tone = _tone_max(tone, "warning")

    if energy_std > 22000 or crew_std > 6000 or water_std > 220:
        tone = "danger"
    elif energy_std > 16000 or crew_std > 4500 or water_std > 160:
        tone = _tone_max(tone, "warning")

    if rigidity_std > 0.45 or tightness_std > 0.12:
        tone = _tone_max(tone, "warning")

    notes = [
        f"σ energía: {energy_std:.0f} kWh",
        f"σ agua: {water_std:.0f} L",
        f"σ crew: {crew_std:.0f} min",
        f"σ rigidez: {rigidity_std:.3f}",
        f"σ estanqueidad: {tightness_std:.3f}",
    ]

    return tone, notes


def compute_inventory_totals(df: pd.DataFrame | None) -> dict[str, float]:
    if df is None or df.empty:
        return {}

    mass = pd.to_numeric(df.get("mass_kg"), errors="coerce").fillna(0.0)
    moisture = pd.to_numeric(df.get("moisture_pct"), errors="coerce").fillna(0.0) / 100.0
    difficulty = (
        pd.to_numeric(df.get("difficulty_factor"), errors="coerce")
        .fillna(1.0)
        .clip(lower=1.0, upper=3.0)
    )

    base_energy = 0.12  # kWh/kg para operaciones base (shredder)
    max_energy = 0.70   # kWh/kg para operaciones complejas (sinterizado)
    energy_per_kg = base_energy + (difficulty - 1.0) / 2.0 * (max_energy - base_energy)

    totals = {
        "mass_kg": float(mass.sum()),
        "water_l": float((mass * moisture).sum()),
        "energy_kwh": float((mass * energy_per_kg).sum()),
    }
    return totals


def classify_inventory_tone(value: float, warn: float, danger: float) -> str:
    if value >= danger:
        return "danger"
    if value >= warn:
        return "warning"
    return "positive"


def compute_delta_strings(
    key: str,
    current: float,
    baseline: dict[str, float] | None,
    unit: str,
    *,
    precision: int = 0,
) -> tuple[str | None, str]:
    if baseline is None or key not in baseline:
        return None, "Sin histórico"

    diff = current - baseline[key]
    tolerance = 1.0 if precision == 0 else 0.1
    if abs(diff) < tolerance:
        return None, "Sin cambios vs. baseline"

    arrow = "↑" if diff > 0 else "↓"
    formatted = f"{abs(diff):.{precision}f}"
    label = f"{arrow} {formatted}{unit}".rstrip()
    detail = f"{label} vs. baseline guardado"
    return label, detail


def describe_baseline_caption(state: dict | None) -> str:
    if not state:
        return "Baseline pendiente: guardá inventario para registrar histórico."
    saved_at = state.get("saved_at") if isinstance(state, dict) else None
    if isinstance(saved_at, datetime):
        timestamp = saved_at.astimezone(timezone.utc).strftime("%d %b %Y %H:%M UTC")
        return f"Baseline desde último save_waste_df ({timestamp})."
    return "Baseline según último save_waste_df disponible."


def format_water(value: float | None) -> str:
    if value is None:
        return "—"
    if value >= 1000:
        return f"{value/1000:.2f} m³"
    return f"{value:.0f} L"


def format_energy(value: float | None) -> str:
    if value is None:
        return "—"
    if value >= 1000:
        return f"{value/1000:.2f} MWh"
    return f"{value:.0f} kWh"


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

hero_chips = [
    {"label": "Hook: 8 astronautas → 12.6 t", "tone": "warning"},
    {
        "label": "Playbook • Residence Renovations (volumen alto)",
        "tone": "accent",
    },
    {
        "label": "Playbook • Daring Discoveries (reuso de carbono)",
        "tone": "accent",
    },
    {"label": "RandomForest multisalida", "tone": "info"},
]

target_state = st.session_state.get("target")
if isinstance(target_state, dict):
    scenario_label = target_state.get("scenario")
    if scenario_label:
        hero_chips.append({"label": f"Escenario • {scenario_label}", "tone": "positive"})

inventory_session_df = st.session_state.get("inventory_data")
inventory_count = 0
if isinstance(inventory_session_df, pd.DataFrame):
    inventory_count = int(inventory_session_df.shape[0])
inventory_loaded = inventory_count > 0

if inventory_loaded:
    inventory_reference_df: pd.DataFrame | None = inventory_session_df
else:
    inventory_reference_df = load_inventory_sample()

inventory_status = "OK" if inventory_loaded else "Pendiente"
if inventory_loaded:
    item_label = "ítem" if inventory_count == 1 else "ítems"
    normalized_label = "normalizado" if inventory_count == 1 else "normalizados"
    inventory_subtitle = f"{inventory_count} {item_label} {normalized_label}"
else:
    inventory_subtitle = "Cargá y normalizá tu inventario base."

target_ready = isinstance(target_state, dict) and bool(target_state)
target_name = ""
if target_ready:
    target_name = str(target_state.get("name") or target_state.get("product") or "Objetivo listo")
target_status = "OK" if target_ready else ("Pendiente" if inventory_loaded else "Alerta")
target_subtitle = (
    f"Objetivo {target_name} calibrado" if target_ready else "Definí objetivo y límites energéticos."
)
if not inventory_loaded and not target_ready:
    target_subtitle = "Cargá inventario antes de definir el target."

candidates_state = st.session_state.get("candidates")
try:
    candidates_count = len(candidates_state) if candidates_state is not None else 0
except TypeError:
    candidates_count = 0
has_candidates = candidates_count > 0

generator_error_msg = st.session_state.get("generator_button_error")
if not generator_error_msg and st.session_state.get("generator_button_state") == "error":
    generator_error_msg = "Revisá los parámetros del generador."

generator_status: str
if generator_error_msg:
    generator_status = "Alerta"
    generator_subtitle = textwrap.shorten(str(generator_error_msg), width=72, placeholder="…")
elif has_candidates:
    generator_status = "OK"
    label = "candidato" if candidates_count == 1 else "candidatos"
    listo_label = "listo" if candidates_count == 1 else "listos"
    generator_subtitle = f"{candidates_count} {label} {listo_label}"
else:
    generator_status = "Pendiente" if target_ready else "Alerta"
    generator_subtitle = (
        "Ejecutá el generador IA." if target_ready else "Configura el target antes de generar."
    )

selected_candidate = st.session_state.get("selected")
results_ready = bool(selected_candidate)

if results_ready:
    results_status = "OK"
    results_subtitle = "Candidata lista para reportar y exportar."
elif has_candidates:
    results_status = "Pendiente"
    results_subtitle = "Seleccioná una candidata para comparar y exportar."
else:
    results_status = "Alerta"
    results_subtitle = (
        "Corregí el generador antes de exportar." if generator_error_msg else "Generá recetas antes de reportar."
    )

baseline_state = st.session_state.get("_inventory_baseline")
if not isinstance(baseline_state, dict):
    baseline_state = None

baseline_totals: dict[str, float] | None = None
if baseline_state is not None:
    baseline_df_candidate = baseline_state.get("df")
    if isinstance(baseline_df_candidate, pd.DataFrame):
        baseline_totals = compute_inventory_totals(baseline_df_candidate)

inventory_totals = compute_inventory_totals(inventory_reference_df)

try:
    n_samples_int = int(n_samples) if n_samples is not None else None
except (TypeError, ValueError):
    n_samples_int = None

training_tone, training_notes = training_health_summary(trained_dt, n_samples_int)
uncertainty_tone, uncertainty_notes = uncertainty_health_summary(model_registry.metadata)

mission_metrics: list[dict[str, object]] = []

status_tone = "positive" if model_registry.ready else "danger"
mission_metrics.append(
    {
        "key": "status",
        "label": "Estado",
        "value": ready,
        "details": [f"Modelo <code>{model_name}</code>"],
        "caption": f"Nombre: {model_name}",
        "icon": "🛰️",
        "stage_key": "inventory",
        "tone": status_tone,
    }
)

mission_metrics.append(
    {
        "key": "training",
        "label": "Entrenamiento",
        "value": trained_at_display,
        "details": [f"Origen: {trained_label_value}", *training_notes],
        "caption": "Control de frescura y cobertura",
        "icon": "🧪",
        "stage_key": "target",
        "tone": training_tone,
    }
)

mission_metrics.append(
    {
        "key": "feature_space",
        "label": "Feature space",
        "value": str(feature_count),
        "details": ["Fisicoquímica + proceso"],
        "caption": "Ingeniería fisicoquímica + proceso",
        "icon": "🧬",
        "stage_key": "generator",
        "tone": "info",
    }
)

mission_metrics.append(
    {
        "key": "uncertainty",
        "label": "Incertidumbre",
        "value": model_registry.uncertainty_label(),
        "details": uncertainty_notes,
        "caption": "CI 95% expuesta en UI",
        "icon": "📈",
        "stage_key": "results",
        "tone": uncertainty_tone,
    }
)

baseline_caption = describe_baseline_caption(baseline_state)

if inventory_totals:
    mass_total = inventory_totals.get("mass_kg", 0.0)
    water_total = inventory_totals.get("water_l", 0.0)
    energy_total = inventory_totals.get("energy_kwh", 0.0)

    mass_delta, mass_delta_detail = compute_delta_strings(
        "mass_kg", mass_total, baseline_totals, " kg"
    )
    water_delta, water_delta_detail = compute_delta_strings(
        "water_l", water_total, baseline_totals, " L", precision=1
    )
    energy_delta, energy_delta_detail = compute_delta_strings(
        "energy_kwh", energy_total, baseline_totals, " kWh"
    )

    mass_details = ["Masa total normalizada"]
    if mass_delta_detail:
        mass_details.insert(0, mass_delta_detail)
    mass_details.insert(0, baseline_caption)
    water_details = ["Basado en humedad declarada"]
    if water_delta_detail:
        water_details.insert(0, water_delta_detail)
    water_details.insert(0, baseline_caption)
    energy_details = ["Escalado por factor de dificultad"]
    if energy_delta_detail:
        energy_details.insert(0, energy_delta_detail)
    energy_details.insert(0, baseline_caption)

    mission_metrics.extend(
        [
            {
                "key": "inventory_mass",
                "label": "Masa total",
                "value": format_mass(mass_total),
                "details": mass_details,
                "icon": "🧱",
                "stage_key": "inventory",
                "tone": classify_inventory_tone(mass_total, 5000, 8000),
                "delta": mass_delta,
            },
            {
                "key": "inventory_water",
                "label": "Agua estimada",
                "value": format_water(water_total),
                "details": water_details,
                "icon": "💧",
                "stage_key": "target",
                "tone": classify_inventory_tone(water_total, 600, 1200),
                "delta": water_delta,
            },
            {
                "key": "inventory_energy",
                "label": "Energía estimada",
                "value": format_energy(energy_total),
                "details": energy_details,
                "icon": "⚡",
                "stage_key": "generator",
                "tone": classify_inventory_tone(energy_total, 2500, 4000),
                "delta": energy_delta,
            },
        ]
    )
else:
    fallback_details = [baseline_caption, "Cargá inventario para estimaciones"]
    mission_metrics.extend(
        [
            {
                "key": "inventory_mass",
                "label": "Masa total",
                "value": "—",
                "details": fallback_details,
                "icon": "🧱",
                "stage_key": "inventory",
                "tone": "warning",
            },
            {
                "key": "inventory_water",
                "label": "Agua estimada",
                "value": "—",
                "details": fallback_details,
                "icon": "💧",
                "stage_key": "target",
                "tone": "warning",
            },
            {
                "key": "inventory_energy",
                "label": "Energía estimada",
                "value": "—",
                "details": fallback_details,
                "icon": "⚡",
                "stage_key": "generator",
                "tone": "warning",
            },
        ]
    )

st.session_state["_inventory_totals"] = inventory_totals

hero_col, metrics_col = st.columns([2.8, 1.2], gap="large")
with hero_col:
    TeslaHero(
        title="Rex-AI coordina el reciclaje orbital y marciano",
        subtitle=(
            "8 astronautas generan 12.6 t de residuos en misión y Rex-AI los convierte en "
            "equipamiento listo. Automatiza mezclas con regolito MGS-1 del cráter Jezero, "
            "polímeros EVA y residuos de carga útil para entregar piezas auditables y trazables."
        ),
        chips=hero_chips,
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
    if normalized.get("delta") is not None:
        normalized["delta"] = str(normalized["delta"])
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
        "status": inventory_status,
        "subtitle": inventory_subtitle,
    },
    {
        "key": "target",
        "title": "Target",
        "description": "Definí objetivo, límites de agua y energía y presets marcianos.",
        "href": "./?page=2_Target_Designer",
        "icon": "🎯",
        "status": target_status,
        "subtitle": target_subtitle,
    },
    {
        "key": "generator",
        "title": "Generador",
        "description": "Compará recetas IA y heurística y validá contribuciones.",
        "href": "./?page=3_Generator",
        "icon": "🤖",
        "status": generator_status,
        "subtitle": generator_subtitle,
    },
    {
        "key": "results",
        "title": "Resultados",
        "description": "Exportá trade-offs, bandas 95% y Sankey para ingeniería.",
        "href": "./?page=4_Results_and_Tradeoffs",
        "icon": "📊",
        "status": results_status,
        "subtitle": results_subtitle,
    },
]
mission_board_component = MissionBoard.from_payload(
    mission_board_payload,
    title="Próxima acción",
    reveal=True,
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

inventory_df = inventory_reference_df

category_items: list[CarouselItem] = []
if inventory_df is not None and not inventory_df.empty:
    total_mass = float(inventory_df["mass_kg"].sum() or 0)
    category_summary = (
        inventory_df.groupby("category")[["mass_kg", "volume_l"]]
        .sum()
        .sort_values("mass_kg", ascending=False)
        .head(6)
    )
    category_cards: list[dict[str, object]] = []
    for category, row in category_summary.iterrows():
        scenario_key = map_category_to_scenario(category)
        scenario_config = _SCENARIO_CONFIG.get(
            scenario_key, _SCENARIO_CONFIG["residences"]
        )
        mass_value = float(row["mass_kg"])
        volume_value = float(row["volume_l"])
        share = (mass_value / total_mass) if total_mass else 0.0
        share_pct = share * 100
        description_parts = [f"Volumen: {volume_value:.0f} L"]
        if share_pct >= 1:
            description_parts.append(f"{share_pct:.0f}% de la masa total")
        elif share_pct > 0:
            description_parts.append(f"{share_pct:.1f}% de la masa total")
        category_cards.append(
            {
                "priority": int(scenario_config["priority"]),
                "mass": mass_value,
                "item": CarouselItem(
                    title=category,
                    value=format_mass(mass_value),
                    description=" • ".join(description_parts),
                    badge=str(scenario_config["label"]),
                    badge_tone=str(scenario_config["badge_tone"]),
                    highlight=share >= 0.2,
                ),
            }
        )
    category_items = [
        entry["item"]
        for entry in sorted(
            category_cards,
            key=lambda entry: (entry["priority"], -entry["mass"]),
        )
    ]

if category_items:
    CarouselRail(
        items=category_items,
        data_track="categorias",
        reveal=False,
    ).render()

risk_watchlist: list[dict[str, object]] = []
risk_flag_map = {
    "pfas": {
        "label": "PFAS",
        "keywords": ["pfas", "fluoro", "ptfe", "fep"],
    },
    "microplastics": {
        "label": "Microplásticos",
        "keywords": ["micro", "foam", "pellet"],
    },
    "incineration": {
        "label": "Incineración",
        "keywords": ["inciner", "combust", "burn"],
    },
}

if inventory_df is not None and not inventory_df.empty:
    flags_series = inventory_df["flags"].fillna("").astype(str)
    for risk_key, config in risk_flag_map.items():
        mask = pd.Series(False, index=flags_series.index)
        for keyword in config["keywords"]:
            if not keyword:
                continue
            mask |= flags_series.str.contains(keyword, case=False, na=False)
        count = int(mask.sum())
        materials = (
            inventory_df.loc[mask, "material"].head(3).dropna().astype(str).tolist()
            if count
            else []
        )
        risk_watchlist.append(
            {
                "key": risk_key,
                "label": config["label"],
                "count": count,
                "status": "warning" if count else "ok",
                "status_label": "Advertencia" if count else "OK",
                "materials": materials,
            }
        )

if risk_watchlist:
    pill_items = []
    for item in risk_watchlist:
        materials_html = (
            f"<span class=\"pill-materials\">{', '.join(item['materials'])}</span>"
            if item["materials"]
            else "<span class=\"pill-materials pill-materials--empty\">Sin materiales críticos detectados</span>"
        )
        pill_items.append(
            """
            <div class="pill-stack__item">
              <span class="pill {status}">
                <strong>{label}</strong>
                <span class="pill-count">{count}</span>
                <span class="pill-status">{status_label}</span>
              </span>
              {materials_html}
            </div>
            """.format(
                status=item["status"],
                label=item["label"],
                count=item["count"],
                status_label=item["status_label"],
                materials_html=materials_html,
            )
        )

    checklist_link = (
        "<a class=\"pill-stack__link\" href=\"./?page=8_Feedback_and_Impact\">"
        "Abrir checklist Feedback & Impact</a>"
    )
    st.markdown(
        """
        <section class="home-card home-card--pills">
          <header class="pill-stack__header">
            <h4>Vigilancia de riesgos</h4>
            {checklist_link}
          </header>
          <div class="pill-stack">
            {pill_items}
          </div>
        </section>
        """.format(
            pill_items="".join(pill_items),
            checklist_link=checklist_link,
        ),
        unsafe_allow_html=True,
    )

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
