# app/pages/6_Pareto_and_Export.py
import _bootstrap  # noqa: F401

import io
import json
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import altair as alt
from plotly.colors import sample_colorscale

from app.modules.analytics import pareto_front
from app.modules.explain import compare_table
from app.modules.exporters import candidate_to_csv, candidate_to_json
from app.modules.navigation import render_breadcrumbs, set_active_step
from app.modules.safety import check_safety, safety_badge  # recalcular badge al seleccionar
from app.modules.ui_blocks import futuristic_button, load_theme
from app.modules.io import load_waste_df
from app.modules.page_data import build_export_kpi_table

# ⚠️ PRIMERA llamada
st.set_page_config(page_title="Pareto & Export", page_icon="📤", layout="wide")

current_step = set_active_step("export")

load_theme()

render_breadcrumbs(current_step)

# Columnas externas (polímeros / aluminio)
POLYMER_NUMERIC_COLUMNS = (
    "pc_density_density_g_per_cm3",
    "pc_mechanics_tensile_strength_mpa",
    "pc_mechanics_modulus_gpa",
    "pc_thermal_glass_transition_c",
    "pc_ignition_ignition_temperature_c",
    "pc_ignition_burn_time_min",
)

ALUMINIUM_NUMERIC_COLUMNS = (
    "aluminium_tensile_strength_mpa",
    "aluminium_yield_strength_mpa",
    "aluminium_elongation_pct",
)

POLYMER_LABEL_MAP = {
    "density_g_cm3": "ρ ref (g/cm³)",
    "tensile_mpa": "σₜ ref (MPa)",
    "modulus_gpa": "E ref (GPa)",
    "glass_c": "Tg (°C)",
    "ignition_c": "Ignición (°C)",
    "burn_min": "Burn (min)",
}

ALUMINIUM_LABEL_MAP = {
    "tensile_mpa": "σₜ Al (MPa)",
    "yield_mpa": "σᵧ Al (MPa)",
    "elongation_pct": "ε Al (%)",
}


def _collect_external_profiles(candidate: dict, inventory: pd.DataFrame) -> dict[str, dict[str, float]]:
    if not isinstance(candidate, dict) or inventory.empty:
        return {}

    ids = {str(value).strip() for value in candidate.get("source_ids", []) if str(value).strip()}
    if not ids:
        return {}

    mask = pd.Series(False, index=inventory.index)
    if "id" in inventory.columns:
        mask |= inventory["id"].astype(str).isin(ids)
    if "_source_id" in inventory.columns:
        mask |= inventory["_source_id"].astype(str).isin(ids)

    subset = inventory.loc[mask]
    if subset.empty:
        return {}

    payload: dict[str, dict[str, float]] = {}

    def _aggregate(section_columns: tuple[str, ...], mapping: dict[str, str]) -> dict[str, float] | None:
        relevant = [column for column in section_columns if column in subset.columns]
        if not relevant:
            return None

        rows = subset[relevant].apply(pd.to_numeric, errors="coerce")
        if not rows.notna().any().any():
            return None

        section_metrics: dict[str, float] = {}
        for column in relevant:
            series = pd.to_numeric(subset[column], errors="coerce")
            if not series.notna().any():
                continue
            value = float(series.mean())
            if column == "pc_density_density_g_per_cm3":
                section_metrics.setdefault("density_g_cm3", value)
            elif column == "pc_mechanics_tensile_strength_mpa":
                section_metrics.setdefault("tensile_mpa", value)
            elif column == "pc_mechanics_modulus_gpa":
                section_metrics.setdefault("modulus_gpa", value)
            elif column == "pc_thermal_glass_transition_c":
                section_metrics.setdefault("glass_c", value)
            elif column == "pc_ignition_ignition_temperature_c":
                section_metrics.setdefault("ignition_c", value)
            elif column == "pc_ignition_burn_time_min":
                section_metrics.setdefault("burn_min", value)
            elif column == "aluminium_tensile_strength_mpa":
                section_metrics.setdefault("tensile_mpa", value)
            elif column == "aluminium_yield_strength_mpa":
                section_metrics.setdefault("yield_mpa", value)
            elif column == "aluminium_elongation_pct":
                section_metrics.setdefault("elongation_pct", value)

        if not section_metrics:
            return None

        ordered_metrics: dict[str, float] = {}
        for key in mapping.keys():
            if key in section_metrics:
                ordered_metrics[key] = section_metrics[key]
        for key, value in section_metrics.items():
            ordered_metrics.setdefault(key, value)
        return ordered_metrics

    polymer_metrics = _aggregate(POLYMER_NUMERIC_COLUMNS, POLYMER_LABEL_MAP)
    if polymer_metrics:
        payload["polymer"] = polymer_metrics

    aluminium_metrics = _aggregate(ALUMINIUM_NUMERIC_COLUMNS, ALUMINIUM_LABEL_MAP)
    if aluminium_metrics:
        payload["aluminium"] = aluminium_metrics

    return payload


# ======== estado requerido ========
cands  = st.session_state.get("candidates", [])
target = st.session_state.get("target", None)
state_sel = st.session_state.get("selected", None)

st.session_state.setdefault("export_history", [])
st.session_state.setdefault("export_wizard_step", 1)
st.session_state.setdefault("selected_export_format", "Plan JSON")
st.session_state.setdefault("last_export_payload", None)
st.session_state.setdefault("selected_option_number", None)
st.session_state.setdefault("export_payload_cache", {})

selected_candidate = state_sel["data"] if state_sel else None
safety_flags = state_sel["safety"] if state_sel else None

if selected_candidate and not st.session_state.get("selected_option_number"):
    try:
        matched_idx = next(idx for idx, cand in enumerate(cands, start=1) if cand is selected_candidate)
        st.session_state["selected_option_number"] = matched_idx
    except StopIteration:
        pass

safety_summary = safety_badge(safety_flags) if safety_flags else {"level": "Sin datos", "detail": "Seleccioná un candidato."}

if not cands or not target:
    st.warning("Generá opciones en **3) Generador** primero.")
    st.stop()

inventory_df = load_waste_df()



def render_safety_badges_html(flags) -> str:
    if not flags:
        return "<span class='safety-badge'>Sin evaluación</span>"
    badges = []
    checklist = [
        (not getattr(flags, "pfas", False), "PFAS sweep"),
        (not getattr(flags, "microplastics", False), "Microplásticos"),
        (not getattr(flags, "incineration", False), "Incineración"),
    ]
    for ok, label in checklist:
        state = "OK" if ok else "Revisar"
        css = "safety-badge ok" if ok else "safety-badge alert"
        badges.append(f"<span class='{css}'>{label}: {state}</span>")
    return "".join(badges)

# ======== HERO ========
st.title("📤 Pareto & Export")
st.caption(
    "Explorá el trade-off Energía ↔ Agua ↔ Crew con datos reales de tus candidatos y exportá el plan enlazado al objetivo definido en 2) Target."
)
st.markdown(
    "- Paso 1: Explorá el frente Pareto con los filtros inferiores.\n"
    "- Paso 2: Seleccioná la opción priorizada por la misión.\n"
    "- Paso 3: Exportá el plan para compartirlo con la tripulación."
)

# ======== tabla base (derivada de candidates reales) ========
df_raw = compare_table(cands, target, crew_time_low=target.get("crew_time_low", False)).copy()

# Enlazar métricas externas por Opción
external_profiles_by_option: dict[int, dict[str, dict[str, float]]] = {}
for idx, candidate in enumerate(cands, start=1):
    profiles = _collect_external_profiles(candidate, inventory_df)
    if profiles:
        external_profiles_by_option[idx] = profiles


def _metric_for(option_value: object, section: str, key: str) -> float:
    try:
        option_number = int(option_value)
    except (TypeError, ValueError):
        return float("nan")

    section_payload = external_profiles_by_option.get(option_number, {}).get(section, {})
    value = section_payload.get(key)
    if value is None:
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


external_columns = [
    ("polymer", "density_g_cm3", "ρ ref (g/cm³)", 3),
    ("polymer", "tensile_mpa", "σₜ ref (MPa)", 1),
    ("polymer", "modulus_gpa", "E ref (GPa)", 2),
    ("polymer", "glass_c", "Tg (°C)", 1),
    ("polymer", "ignition_c", "Ignición (°C)", 0),
    ("polymer", "burn_min", "Burn (min)", 1),
    ("aluminium", "tensile_mpa", "σₜ Al (MPa)", 1),
    ("aluminium", "yield_mpa", "σᵧ Al (MPa)", 1),
    ("aluminium", "elongation_pct", "ε Al (%)", 1),
]

option_series = pd.to_numeric(df_raw.get("Opción"), errors="coerce") if "Opción" in df_raw else pd.Series(dtype=float)
for section, key, label, _precision in external_columns:
    df_raw[label] = option_series.map(lambda opt: _metric_for(opt, section, key))

# Normalización robusta de nombres
rename_map = {}
for col in df_raw.columns:
    low = col.lower().strip()
    if low in ["energia (kwh)", "energía (kwh)", "energia kwh"]: rename_map[col] = "Energía (kWh)"
    if low in ["agua (l)", "agua l", "agua"]: rename_map[col] = "Agua (L)"
    if low in ["crew (min)", "crew min", "crew"]: rename_map[col] = "Crew (min)"
    if low in ["masa (kg)", "masa kg", "kg"]: rename_map[col] = "Masa (kg)"
    if low in ["opción","opcion"]: rename_map[col] = "Opción"
    if low == "materiales": rename_map[col] = "Materiales"
    if low == "proceso": rename_map[col] = "Proceso"
    if low == "score": rename_map[col] = "Score"
df_raw.rename(columns=rename_map, inplace=True)

# Tipos y saneo
for k in ["Score","Energía (kWh)","Agua (L)","Crew (min)","Masa (kg)"]:
    if k in df_raw: df_raw[k] = pd.to_numeric(df_raw[k], errors="coerce")
if "Materiales" in df_raw:
    df_raw["Materiales"] = df_raw["Materiales"].apply(
        lambda v: ", ".join(v) if isinstance(v, (list,tuple)) else (str(v) if pd.notna(v) else "")
    )

df_plot = df_raw.dropna(subset=["Energía (kWh)","Agua (L)","Crew (min)","Score"]).copy()

# ======== KPIs ========
kpi_df = build_export_kpi_table(df_plot)
st.subheader("Indicadores del lote")
kpi_style = kpi_df.style.format({"Valor": "{:.3f}"})
try:
    kpi_style = kpi_style.hide(axis="index")
except AttributeError:
    kpi_style = kpi_style.hide_index()
st.dataframe(kpi_style, use_container_width=True)

kpi_chart = alt.Chart(kpi_df).mark_bar(color="#38bdf8").encode(
    x=alt.X("Valor:Q", title="Valor"),
    y=alt.Y("KPI:N", sort="-x"),
    tooltip=["KPI", alt.Tooltip("Valor", format=".3f")],
).properties(height=220)
st.altair_chart(kpi_chart, use_container_width=True)

# ======== What-If de límites ========
st.markdown("### 🎛️ What-If (filtro visual)")

what_if_presets = {
    "Residence": {
        "label": "Residence Renovations (volumen alto)",
        "button_label": "Residence",
        "energy": 1.8,
        "water": 0.28,
        "crew": 36.0,
        "insight": "Favorece piezas voluminosas optimizando agua y minutos de tripulación.",
    },
    "Daring": {
        "label": "Daring Discoveries (reuso de carbono)",
        "button_label": "Daring",
        "energy": 1.1,
        "water": 0.15,
        "crew": 28.0,
        "insight": "Maximiza el reuso de carbono con operaciones de baja energía y agua controlada.",
    },
}

def _safe_float(value, fallback: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback

target_signature = (
    round(_safe_float(target.get("max_energy_kwh", 0.0)), 3),
    round(_safe_float(target.get("max_water_l", 0.0)), 3),
    round(_safe_float(target.get("max_crew_min", 0.0)), 1),
    str(target.get("scenario", "")),
)

if st.session_state.get("__what_if_signature") != target_signature:
    st.session_state["__what_if_energy"] = _safe_float(target.get("max_energy_kwh", 0.0))
    st.session_state["__what_if_water"] = _safe_float(target.get("max_water_l", 0.0))
    st.session_state["__what_if_crew"] = _safe_float(target.get("max_crew_min", 0.0))
    st.session_state["__what_if_signature"] = target_signature
    st.session_state["__what_if_active_preset"] = None


def _apply_what_if_preset(preset_key: str) -> None:
    preset_cfg = what_if_presets[preset_key]
    st.session_state["__what_if_energy"] = _safe_float(preset_cfg["energy"], 0.0)
    st.session_state["__what_if_water"] = _safe_float(preset_cfg["water"], 0.0)
    st.session_state["__what_if_crew"] = _safe_float(preset_cfg["crew"], 0.0)
    st.session_state["__what_if_active_preset"] = preset_key


preset_cols = st.columns([1.3, 1, 1])
with preset_cols[0]:
    st.caption(
        "Activá límites sugeridos NASA según el foco del escenario y ajustá manualmente si lo necesitás."
    )
for col, preset_key in zip(preset_cols[1:], ("Residence", "Daring")):
    with col:
        cfg = what_if_presets[preset_key]
        if st.button(cfg["button_label"], use_container_width=True):
            _apply_what_if_preset(preset_key)
        st.caption(
            f"≤ {cfg['energy']:.2f} kWh · {cfg['water']:.2f} L · {cfg['crew']:.0f} min"
        )

f1, f2, f3 = st.columns(3)
with f1:
    lim_e = st.number_input(
        "Límite de Energía (kWh)",
        min_value=0.0,
        max_value=999.0,
        value=_safe_float(
            st.session_state.get("__what_if_energy", target.get("max_energy_kwh", 0.0))
        ),
        step=0.1,
        key="__what_if_energy",
    )
with f2:
    lim_w = st.number_input(
        "Límite de Agua (L)",
        min_value=0.0,
        max_value=999.0,
        value=_safe_float(
            st.session_state.get("__what_if_water", target.get("max_water_l", 0.0))
        ),
        step=0.1,
        key="__what_if_water",
    )
with f3:
    lim_c = st.number_input(
        "Límite de Crew (min)",
        min_value=0.0,
        max_value=999.0,
        value=_safe_float(
            st.session_state.get("__what_if_crew", target.get("max_crew_min", 0.0))
        ),
        step=1.0,
        key="__what_if_crew",
    )

current_limits = {
    "energy": _safe_float(lim_e),
    "water": _safe_float(lim_w),
    "crew": _safe_float(lim_c),
}

matched_preset = None
for preset_name, preset_cfg in what_if_presets.items():
    if (
        abs(current_limits["energy"] - _safe_float(preset_cfg["energy"])) < 1e-6
        and abs(current_limits["water"] - _safe_float(preset_cfg["water"])) < 1e-6
        and abs(current_limits["crew"] - _safe_float(preset_cfg["crew"])) < 1e-6
    ):
        matched_preset = preset_name
        break

st.session_state["__what_if_active_preset"] = matched_preset

mask_ok = (
    (df_plot["Energía (kWh)"] <= current_limits["energy"])
    & (df_plot["Agua (L)"] <= current_limits["water"])
    & (df_plot["Crew (min)"] <= current_limits["crew"])
)
df_view = df_plot.copy()
df_view["Dentro_límites"] = np.where(mask_ok, "Dentro de límites", "Excede límites")

# ======== Frontera de Pareto ========
try:
    front_idx = pareto_front(df_plot)
    front_mask = df_plot.index.isin(front_idx)
except Exception:
    # fallback estable si el usuario sube columnas raras
    front_mask = df_plot["Score"].rank(ascending=False, method="first") <= 5
df_view["Pareto"] = np.where(front_mask, "Pareto", "No Pareto")
df_view["ScorePos"] = np.clip(df_view["Score"].fillna(0.0), 0.01, None)
table_pareto = df_view[df_view["Pareto"] == "Pareto"].sort_values("Score", ascending=False)
pareto_options = table_pareto["Opción"].astype(int).tolist() if "Opción" in table_pareto else []

selected_option_number = st.session_state.get("selected_option_number")

# ======== Sidebar – Flight Plans & Previews ========
sidebar = st.sidebar
sidebar.markdown("<div class='sidebar-flight'><h2>🛫 Flight Plans</h2></div>", unsafe_allow_html=True)

flash_event = st.session_state.pop("flight_flash", None)
if flash_event:
    sidebar.markdown(
        f"<div class='flight-alert animate'>Flight plan #{flash_event.get('option', '—')} listo para export. Revisá la Nebula preview.</div>",
        unsafe_allow_html=True,
    )

if table_pareto.empty:
    sidebar.info("Generá candidatos para visualizar planes de vuelo.")
else:
    for _, row in table_pareto.head(4).iterrows():
        option_number = int(row.get("Opción", 0))
        energy = row.get("Energía (kWh)", 0.0)
        water = row.get("Agua (L)", 0.0)
        crew = row.get("Crew (min)", 0.0)
        score_val = row.get("Score", 0.0)
        rho_val = row.get("ρ ref (g/cm³)")
        tensile_val = row.get("σₜ ref (MPa)")
        tensile_al_val = row.get("σₜ Al (MPa)")

        def _fmt(value: object, precision: int, suffix: str) -> str:
            try:
                return f"{float(value):.{precision}f}{suffix}"
            except (TypeError, ValueError):
                return "—"

        active_cls = " active" if selected_option_number and option_number == int(selected_option_number) else ""
        sidebar.markdown(
            f"<div class='flight-card{active_cls}'>"
            f"<strong>Plan #{option_number}</strong>"
            f"<span>Score: {score_val:.2f}</span>"
            f"<span>Energía: {energy:.2f} kWh · Agua: {water:.2f} L · Crew: {crew:.1f} min</span>"
            f"<span>ρ ref {_fmt(rho_val, 3, ' g/cm³')} · σₜ {_fmt(tensile_val, 1, ' MPa')} · σₜ Al {_fmt(tensile_al_val, 1, ' MPa')}</span>"
            "</div>",
            unsafe_allow_html=True,
        )

if selected_candidate:
    props = selected_candidate.get("props")
    materials = ", ".join(selected_candidate.get("materials", [])[:3])
    if selected_candidate.get("materials") and len(selected_candidate["materials"]) > 3:
        materials += "…"
    profile_payload: dict[str, dict[str, float]] = {}
    try:
        if selected_option_number:
            profile_payload = external_profiles_by_option.get(int(selected_option_number), {}) or {}
    except (TypeError, ValueError):
        profile_payload = {}

    def _format_profile_section(metrics: dict[str, float]) -> str:
        if not metrics:
            return "—"
        formatted: list[str] = []
        for key, label in (
            ("density_g_cm3", "ρ ref"),
            ("tensile_mpa", "σₜ"),
            ("modulus_gpa", "E"),
            ("glass_c", "Tg"),
            ("ignition_c", "Ign"),
            ("burn_min", "Burn"),
            ("yield_mpa", "σᵧ"),
            ("elongation_pct", "ε"),
        ):
            value = metrics.get(key)
            if value is None:
                continue
            suffix = ""
            if key == "density_g_cm3":
                suffix = " g/cm³"
            elif key in {"tensile_mpa", "yield_mpa"}:
                suffix = " MPa"
            elif key in {"glass_c", "ignition_c"}:
                suffix = " °C"
            elif key == "burn_min":
                suffix = " min"
            elif key == "elongation_pct":
                suffix = " %"
            try:
                formatted.append(f"{label} {float(value):.2f}{suffix}")
            except (TypeError, ValueError):
                continue
        return " · ".join(formatted) if formatted else "—"

    polymer_preview = _format_profile_section(profile_payload.get("polymer", {}))
    aluminium_preview = _format_profile_section(profile_payload.get("aluminium", {}))
    sidebar.markdown(
        "<h3 style='margin-top:14px;'>Nebula preview</h3>",
        unsafe_allow_html=True,
    )
    sidebar.markdown(
        "<div class='safety-badges'>" + render_safety_badges_html(safety_flags) + "</div>",
        unsafe_allow_html=True,
    )
    if props:
        sidebar.markdown(
            """
            <div class='nebula-preview'>
              <strong>{label}</strong><br/>
              Proceso: {proc}<br/>
              Materiales: {mats}<br/>
              Score: {score:.2f} · Energía: {energy:.2f} kWh · Agua: {water:.2f} L · Crew: {crew:.1f} min
              <br/>Polímero: {poly}<br/>Aluminio: {alum}
            </div>
            """.format(
                label=f"Plan #{selected_option_number or '—'}",
                proc=f"{selected_candidate.get('process_id', '')} {selected_candidate.get('process_name', '')}".strip(),
                mats=materials or "—",
                score=selected_candidate.get("score", 0.0),
                energy=getattr(props, "energy_kwh", 0.0),
                water=getattr(props, "water_l", 0.0),
                crew=getattr(props, "crew_min", 0.0),
                poly=polymer_preview,
                alum=aluminium_preview,
            ),
            unsafe_allow_html=True,
        )

    fmt_choice = st.session_state.get("selected_export_format", "Plan JSON")
    sidebar.caption(f"Formato seleccionado: {fmt_choice}")
    try:
        if fmt_choice == "Plan JSON" and safety_flags:
            preview = candidate_to_json(selected_candidate, target, safety_flags).decode("utf-8")
            preview_lines = preview.splitlines()
            sidebar.code("\n".join(preview_lines[:10]) + ("\n…" if len(preview_lines) > 10 else ""), language="json")
        elif fmt_choice == "Resumen CSV":
            csv_text = candidate_to_csv(selected_candidate).decode("utf-8")
            preview_lines = csv_text.splitlines()
            sidebar.code("\n".join(preview_lines[:8]) + ("\n…" if len(preview_lines) > 8 else ""), language="csv")
        elif fmt_choice == "Pareto CSV":
            sidebar.dataframe(table_pareto.head(6), use_container_width=True, hide_index=True)
    except Exception as preview_error:
        sidebar.warning(f"No se pudo generar preview: {preview_error}")


tab_pareto, tab_trials, tab_objectives, tab_export = st.tabs(
    ["🌌 Pareto Explorer", "🔮 Predicciones de ensayo (demo)", "🎯 Objetivos por eje", "📦 Export Center"]
)

# ---------- TAB 1: Pareto Explorer ----------
with tab_pareto:
    st.markdown('<h3 class="section-title">Explorador 3D</h3>', unsafe_allow_html=True)
    usable = df_view.dropna(subset=["Energía (kWh)","Agua (L)","Crew (min)","ScorePos"]).copy()

    if usable.empty:
        st.info("No hay suficientes datos para graficar.")
    else:
        active_preset = st.session_state.get("__what_if_active_preset")
        explanation_lines: list[str] = []
        for preset_key, preset_cfg in what_if_presets.items():
            preset_mask = (
                (df_view["Pareto"] == "Pareto")
                & (df_view["Energía (kWh)"] <= _safe_float(preset_cfg["energy"]))
                & (df_view["Agua (L)"] <= _safe_float(preset_cfg["water"]))
                & (df_view["Crew (min)"] <= _safe_float(preset_cfg["crew"]))
            )
            subset = df_view[preset_mask].sort_values("Score", ascending=False)
            if subset.empty:
                detail = (
                    "sin candidatos Pareto dentro de los límites sugeridos "
                    f"(≤ {preset_cfg['energy']:.2f} kWh · {preset_cfg['water']:.2f} L · {preset_cfg['crew']:.0f} min)."
                )
            else:
                top_row = subset.iloc[0]
                option_raw = top_row.get("Opción")
                option_label = (
                    f"Plan #{int(option_raw)}" if pd.notna(option_raw) else "Plan destacado"
                )
                detail = (
                    f"{option_label} lidera con Score {top_row['Score']:.2f}, "
                    f"{top_row['Energía (kWh)']:.2f} kWh, {top_row['Agua (L)']:.2f} L y "
                    f"{top_row['Crew (min)']:.1f} min. {preset_cfg['insight']}"
                )
                if subset.shape[0] > 1:
                    detail += f" ({subset.shape[0]} planes dentro de límites.)"
            marker = "⭐" if preset_key == active_preset else "•"
            explanation_lines.append(
                f"- {marker} **{preset_cfg['label']}**: {detail}"
            )

        st.markdown("#### Dominio Pareto por escenario")
        st.markdown("\n".join(explanation_lines))
        st.caption("⭐ indica el preset aplicado en los filtros What-If.")

        if "Opción" in usable:
            usable["Opción"] = pd.to_numeric(usable["Opción"], errors="coerce")

        pareto_points = usable[usable["Pareto"] == "Pareto"].copy()
        other_points = usable[usable["Pareto"] != "Pareto"].copy()

        def _color_scale(values: pd.Series, scale: str) -> list[str]:
            if values.empty:
                return []
            vals = values.fillna(values.mean() if not values.empty else 0.0)
            vmin, vmax = float(vals.min()), float(vals.max())
            if abs(vmax - vmin) < 1e-9:
                norm = [0.5] * len(vals)
            else:
                norm = ((vals - vmin) / (vmax - vmin)).clip(0.0, 1.0).tolist()
            return sample_colorscale(scale, norm)

        def _safe_series(df: pd.DataFrame, key: str) -> pd.Series:
            if key in df:
                return df[key]
            return pd.Series(np.nan, index=df.index)

        def _hover_matrix(df: pd.DataFrame) -> np.ndarray:
            if df.empty:
                return np.empty((0, 5), dtype=object)
            option_series = pd.to_numeric(_safe_series(df, "Opción"), errors="coerce")
            score_series = pd.to_numeric(_safe_series(df, "Score"), errors="coerce")
            rho_series = pd.to_numeric(_safe_series(df, "ρ ref (g/cm³)"), errors="coerce")
            tensile_series = pd.to_numeric(_safe_series(df, "σₜ ref (MPa)"), errors="coerce")
            tensile_al_series = pd.to_numeric(_safe_series(df, "σₜ Al (MPa)"), errors="coerce")

            def _fmt(values: pd.Series, precision: int, suffix: str = "") -> np.ndarray:
                formatted: list[str] = []
                for value in values:
                    if pd.isna(value):
                        formatted.append("—")
                    else:
                        formatted.append(f"{value:.{precision}f}{suffix}")
                return np.array(formatted, dtype=object)

            option_text = np.where(option_series.notna(), option_series.fillna(0).astype(int).astype(str), "—")
            return np.column_stack(
                [
                    option_text,
                    _fmt(score_series, 2),
                    _fmt(rho_series, 3, " g/cm³"),
                    _fmt(tensile_series, 1, " MPa"),
                    _fmt(tensile_al_series, 1, " MPa"),
                ]
            )

        fig3d = go.Figure()

        if not other_points.empty:
            fig3d.add_trace(
                go.Scatter3d(
                    x=other_points["Energía (kWh)"],
                    y=other_points["Agua (L)"],
                    z=other_points["Crew (min)"],
                    mode="markers",
                    name="Candidatos",
                    marker=dict(
                        size=6,
                        color=_color_scale(other_points["Score"], "Viridis"),
                        opacity=0.45,
                        line=dict(width=1.2, color="rgba(148,163,184,0.4)"),
                        symbol="circle",
                    ),
                    hovertemplate=(
                        "<b>Opción %{customdata[0]}</b><br>Score %{customdata[1]}<br>"
                        "ρ ref %{customdata[2]}<br>σₜ %{customdata[3]}<br>σₜ Al %{customdata[4]}<extra></extra>"
                    ),
                    customdata=_hover_matrix(other_points),
                )
            )

        if not pareto_points.empty:
            pareto_colors = _color_scale(pareto_points["Score"], "IceFire")
            fig3d.add_trace(
                go.Scatter3d(
                    x=pareto_points["Energía (kWh)"],
                    y=pareto_points["Agua (L)"],
                    z=pareto_points["Crew (min)"],
                    mode="markers",
                    name="Pareto Prime",
                    marker=dict(
                        size=11,
                        color=pareto_colors,
                        opacity=0.98,
                        symbol="diamond",
                        line=dict(width=3, color="rgba(240,249,255,0.9)"),
                        lighting=dict(ambient=0.62, diffuse=0.9, specular=0.88, roughness=0.2, fresnel=0.25),
                        lightposition=dict(x=200, y=120, z=140),
                    ),
                    hovertemplate=(
                        "<b>Plan %{customdata[0]}</b><br>Score %{customdata[1]}<br>"
                        "ρ ref %{customdata[2]} · σₜ %{customdata[3]} · σₜ Al %{customdata[4]}<br>"
                        "Energía %{x:.2f} kWh<br>Agua %{y:.2f} L<br>Crew %{z:.1f} min<extra></extra>"
                    ),
                    customdata=_hover_matrix(pareto_points),
                )
            )

            score_min = float(pareto_points["ScorePos"].min()) if not pareto_points.empty else 0.0
            score_max = float(pareto_points["ScorePos"].max()) if not pareto_points.empty else 1.0
            size_span = max(score_max - score_min, 0.01)
            halo_sizes = 18 + 20 * (pareto_points["ScorePos"] - score_min) / size_span
            fig3d.add_trace(
                go.Scatter3d(
                    x=pareto_points["Energía (kWh)"],
                    y=pareto_points["Agua (L)"],
                    z=pareto_points["Crew (min)"],
                    mode="markers",
                    name="Nebula halo",
                    marker=dict(
                        size=halo_sizes,
                        color="rgba(125,211,252,0.18)",
                        opacity=0.22,
                    ),
                    hoverinfo="skip",
                    showlegend=False,
                )
            )

            # Nebula cloud around Pareto points
            x_vals = pareto_points["Energía (kWh)"].to_numpy()
            y_vals = pareto_points["Agua (L)"].to_numpy()
            z_vals = pareto_points["Crew (min)"].to_numpy()
            x_span = max(float(usable["Energía (kWh)"].max() - usable["Energía (kWh)"].min()), 1e-3)
            y_span = max(float(usable["Agua (L)"].max() - usable["Agua (L)"].min()), 1e-3)
            z_span = max(float(usable["Crew (min)"].max() - usable["Crew (min)"].min()), 1e-3)
            nebula_points = []
            for option, xv, yv, zv in zip(pareto_points.get("Opción", []), x_vals, y_vals, z_vals):
                rng = np.random.default_rng(int(option * 997) if not pd.isna(option) else 42)
                spread = np.array([x_span, y_span, z_span]) * 0.04
                cloud = rng.normal(loc=[xv, yv, zv], scale=np.maximum(spread, 1e-3), size=(24, 3))
                nebula_points.append(cloud)
            if nebula_points:
                nebula = np.vstack(nebula_points)
                fig3d.add_trace(
                    go.Scatter3d(
                        x=nebula[:, 0],
                        y=nebula[:, 1],
                        z=nebula[:, 2],
                        mode="markers",
                        marker=dict(size=3, opacity=0.18, color="rgba(148,197,255,0.18)"),
                        hoverinfo="skip",
                        showlegend=False,
                    )
                )

        # Highlight selected candidate in scene
        if selected_option_number:
            selected_trace = usable[usable.get("Opción") == int(selected_option_number)] if "Opción" in usable else pd.DataFrame()
            if not selected_trace.empty:
                fig3d.add_trace(
                    go.Scatter3d(
                        x=selected_trace["Energía (kWh)"],
                        y=selected_trace["Agua (L)"],
                        z=selected_trace["Crew (min)"],
                        mode="markers",
                        name="Seleccionado",
                        marker=dict(
                            size=14,
                            color="rgba(74,222,128,0.95)",
                            line=dict(width=4, color="rgba(255,255,255,0.95)"),
                            opacity=1.0,
                            symbol="circle",
                        ),
                        hovertemplate=(
                            "<b>Seleccionado %{customdata[0]}</b><br>"
                            "Score %{customdata[1]}<br>ρ ref %{customdata[2]} · σₜ %{customdata[3]} · σₜ Al %{customdata[4]}"
                            "<extra></extra>"
                        ),
                        customdata=_hover_matrix(selected_trace),
                    )
                )

        # Illuminated axes
        x_min, x_max = float(usable["Energía (kWh)"].min()), float(usable["Energía (kWh)"].max())
        y_min, y_max = float(usable["Agua (L)"].min()), float(usable["Agua (L)"].max())
        z_min, z_max = float(usable["Crew (min)"].min()), float(usable["Crew (min)"].max())
        axis_lines = [
            ([x_min, x_max], [y_min, y_min], [z_min, z_min]),
            ([x_min, x_min], [y_min, y_max], [z_min, z_min]),
            ([x_min, x_min], [y_min, y_min], [z_min, z_max]),
        ]
        for idx, (xs, ys, zs) in enumerate(axis_lines):
            fig3d.add_trace(
                go.Scatter3d(
                    x=xs,
                    y=ys,
                    z=zs,
                    mode="lines",
                    line=dict(color="rgba(148,197,255,0.85)", width=6),
                    hoverinfo="skip",
                    showlegend=False,
                    name=f"axis-{idx}",
                )
            )

        fig3d.update_layout(
            height=540,
            margin=dict(l=0, r=0, b=0, t=24),
            paper_bgcolor="rgba(4,7,20,1)",
            scene=dict(
                xaxis=dict(
                    title="Energía (kWh)",
                    backgroundcolor="rgba(8,12,35,0.92)",
                    gridcolor="rgba(96,165,250,0.12)",
                    zerolinecolor="rgba(148,197,255,0.6)",
                    showbackground=True,
                    showspikes=True,
                    spikecolor="rgba(125,211,252,0.8)",
                    spikethickness=2,
                    tickfont=dict(color="#cbd5f5"),
                    titlefont=dict(color="#bae6fd"),
                ),
                yaxis=dict(
                    title="Agua (L)",
                    backgroundcolor="rgba(6,11,32,0.9)",
                    gridcolor="rgba(96,165,250,0.12)",
                    zerolinecolor="rgba(148,197,255,0.6)",
                    showbackground=True,
                    showspikes=True,
                    spikecolor="rgba(56,189,248,0.75)",
                    spikethickness=2,
                    tickfont=dict(color="#cbd5f5"),
                    titlefont=dict(color="#bae6fd"),
                ),
                zaxis=dict(
                    title="Crew (min)",
                    backgroundcolor="rgba(5,9,28,0.9)",
                    gridcolor="rgba(96,165,250,0.12)",
                    zerolinecolor="rgba(148,197,255,0.6)",
                    showbackground=True,
                    showspikes=True,
                    spikecolor="rgba(14,165,233,0.8)",
                    spikethickness=2,
                    tickfont=dict(color="#cbd5f5"),
                    titlefont=dict(color="#bae6fd"),
                ),
                camera=dict(eye=dict(x=1.65, y=1.72, z=1.45)),
                dragmode="orbit",
                aspectmode="cube",
            ),
            legend=dict(
                bgcolor="rgba(8,12,35,0.82)",
                font=dict(color="#e0f2fe"),
                orientation="h",
                yanchor="bottom",
                y=0.01,
                x=0.02,
            ),
        )

        st.plotly_chart(
            fig3d,
            use_container_width=True,
            config={"displaylogo": False, "modeBarButtonsToRemove": ["resetCameraDefault3d"], "scrollZoom": True},
        )

    st.markdown("""
<div class="legend">
<b>Cómo leerlo (criollo):</b> querés puntos <b>abajo/izquierda</b> (menos energía/agua) y <b>adelante</b> (menos crew).
La capa “Pareto” marca los que no pueden mejorarse en un eje sin empeorar otro.
</div>
""", unsafe_allow_html=True)

    density_series = pd.to_numeric(df_view.get("ρ ref (g/cm³)"), errors="coerce").dropna()
    polymer_tensile_series = pd.to_numeric(df_view.get("σₜ ref (MPa)"), errors="coerce").dropna()
    aluminium_tensile_series = pd.to_numeric(df_view.get("σₜ Al (MPa)"), errors="coerce").dropna()
    if not density_series.empty or not polymer_tensile_series.empty or not aluminium_tensile_series.empty:
        st.markdown('<h4 class="section-title">Propiedades externas (NASA)</h4>', unsafe_allow_html=True)
        metric_cols = st.columns(3)
        if not density_series.empty:
            with metric_cols[0]:
                density_fig = px.histogram(
                    pd.DataFrame({"ρ ref (g/cm³)": density_series}),
                    x="ρ ref (g/cm³)",
                    nbins=14,
                    title="Densidad",
                )
                density_fig.update_layout(height=280, margin=dict(l=10, r=10, t=40, b=10))
                st.plotly_chart(density_fig, use_container_width=True)
        if not polymer_tensile_series.empty:
            with metric_cols[1]:
                tensile_fig = px.histogram(
                    pd.DataFrame({"σₜ ref (MPa)": polymer_tensile_series}),
                    x="σₜ ref (MPa)",
                    nbins=14,
                    title="σₜ polímero",
                )
                tensile_fig.update_layout(height=280, margin=dict(l=10, r=10, t=40, b=10))
                st.plotly_chart(tensile_fig, use_container_width=True)
        if not aluminium_tensile_series.empty:
            with metric_cols[2]:
                tensile_al_fig = px.histogram(
                    pd.DataFrame({"σₜ Al (MPa)": aluminium_tensile_series}),
                    x="σₜ Al (MPa)",
                    nbins=14,
                    title="σₜ aluminio",
                )
                tensile_al_fig.update_layout(height=280, margin=dict(l=10, r=10, t=40, b=10))
                st.plotly_chart(tensile_al_fig, use_container_width=True)
        st.caption(
            "Distribuciones calculadas sobre candidatos filtrados. Úsalas para balancear densidad vs. resistencia en la selección."
        )

    st.markdown('<h4 class="section-title">Tabla — Frontera de Pareto</h4>', unsafe_allow_html=True)
    if not table_pareto.empty:
        columns_to_show = [
            "Opción",
            "Score",
            "Proceso",
            "Materiales",
            "Energía (kWh)",
            "Agua (L)",
            "Crew (min)",
            "ρ ref (g/cm³)",
            "σₜ ref (MPa)",
            "σₜ Al (MPa)",
        ]
        available_columns = [column for column in columns_to_show if column in table_pareto.columns]
        st.dataframe(
            table_pareto[available_columns],
            use_container_width=True, hide_index=True
        )
    else:
        st.info("No hay puntos en la frontera con datos completos.")

    st.markdown('<h4 class="section-title">Seleccionar candidato</h4>', unsafe_allow_html=True)
    opciones = table_pareto["Opción"].astype(int).tolist()
    if opciones:
        pick_opt = st.selectbox("Elegí Opción #", opciones, index=0, key="pick_from_pareto")
        select_state_key = "pareto_select_state"
        select_trigger_key = "pareto_select_trigger"
        select_feedback_key = "pareto_select_feedback"
        button_state = st.session_state.get(select_state_key, "idle")
        if futuristic_button(
            "✅ Usar como seleccionado",
            key="pareto_select_button",
            state=button_state,
            width="full",
            help_text="Habilita export JSON/CSV en la pestaña",
            loading_label="Sincronizando…",
            success_label="Listo para exportar",
            error_label="Revisar opción",
            enable_vibration=True,
            status_hints={
                "idle": "",
                "loading": "Checando seguridad",
                "success": "Disponible en Export Center",
                "error": "Revisá la opción seleccionada",
            },
            mode="cinematic",
        ):
            st.session_state[select_state_key] = "loading"
            st.session_state[select_trigger_key] = True
            st.session_state["pareto_select_value"] = st.session_state.get("pick_from_pareto", pick_opt)
            st.session_state.pop(select_feedback_key, None)
            st.experimental_rerun()
    else:
        st.info("No hay puntos en la frontera con datos completos.")

    if st.session_state.get("pareto_select_trigger"):
        current_selection = st.session_state.get(
            "pareto_select_value", st.session_state.get("pick_from_pareto")
        )
        try:
            pick_value = int(current_selection) if current_selection is not None else None
        except (TypeError, ValueError):
            pick_value = None
        message_level = "error"
        message_text = "No se pudo interpretar la opción seleccionada."
        if pick_value is not None:
            idx = pick_value - 1
    if pareto_options:
        default_index = 0
        if selected_option_number and int(selected_option_number) in pareto_options:
            default_index = pareto_options.index(int(selected_option_number))
        pick_opt = st.selectbox("Elegí Opción #", pareto_options, index=default_index, key="pick_from_pareto")
        if st.button("✅ Usar como seleccionado"):
            idx = int(pick_opt) - 1
            if 0 <= idx < len(cands):
                selected = cands[idx]
                flags = check_safety(selected["materials"], selected["process_name"], selected["process_id"])
                st.session_state["selected"] = {"data": selected, "safety": flags}
                message_level = "success"
                message_text = (
                    f"Candidato #{pick_value} seleccionado. Abrí **4) Resultados** o **5) Comparar & Explicar**."
                )
                st.session_state["selected_option_number"] = pick_opt
                st.session_state["flight_flash"] = {"option": pick_opt}
                st.session_state["export_wizard_step"] = 1
                st.session_state["last_export_payload"] = None
                st.success(f"Candidato #{pick_opt} seleccionado. Abrí **4) Resultados** o **5) Comparar & Explicar**.")
            else:
                message_level = "warning"
                message_text = "Opción fuera de rango respecto a la lista de candidates."
        st.session_state["pareto_select_feedback"] = (message_level, message_text)
        st.session_state["pareto_select_state"] = "success" if message_level == "success" else "error"
        st.session_state["pareto_select_trigger"] = False
        st.session_state.pop("pareto_select_value", None)

    feedback_state = st.session_state.get("pareto_select_feedback")
    if feedback_state:
        level, message = feedback_state
        if level == "success":
            st.success(message)
        elif level == "warning":
            st.warning(message)
        elif level == "error":
            st.error(message)

# ---------- TAB 2: Predicciones de ensayo (demo conectada a datos) ----------
with tab_trials:
    st.markdown('<h3 class="section-title">Score predictions — barras de confianza</h3>', unsafe_allow_html=True)
    st.caption("Usa los **scores reales** y les aplica un ±CI porcentual para visualizar la variabilidad esperable (demo).")

    ci_pct = st.slider("Intervalo de confianza (± % de Score)", 5, 50, 20, step=5)
    top_n  = st.slider("Top-N por Score", 3, max(3, len(df_view)), min(8, len(df_view)))

    df_trials = df_view.sort_values("Score", ascending=False).head(top_n).copy()
    if df_trials.empty:
        st.info("No hay candidatos suficientes para graficar.")
    else:
        yerr = (df_trials["Score"].abs() * (ci_pct/100.0)).clip(lower=0.05)
        density_vals = pd.to_numeric(df_trials.get("ρ ref (g/cm³)"), errors="coerce")
        tensile_vals = pd.to_numeric(df_trials.get("σₜ ref (MPa)"), errors="coerce")
        tensile_al_vals = pd.to_numeric(df_trials.get("σₜ Al (MPa)"), errors="coerce")

        def _format_hover(values: pd.Series, precision: int, suffix: str = "") -> np.ndarray:
            formatted: list[str] = []
            for value in values:
                if pd.isna(value):
                    formatted.append("—")
                else:
                    formatted.append(f"{value:.{precision}f}{suffix}")
            return np.array(formatted, dtype=object)

        customdata = np.column_stack(
            [
                df_trials["Opción"].astype(str).to_numpy(),
                _format_hover(density_vals, 3, " g/cm³"),
                _format_hover(tensile_vals, 1, " MPa"),
                _format_hover(tensile_al_vals, 1, " MPa"),
            ]
        )

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df_trials["Opción"].astype(str),
            y=df_trials["Score"],
            error_y=dict(type='data', array=yerr, thickness=1.2, width=4),
            mode="markers",
            marker=dict(size=10),
            name="Predicted trial score",
            customdata=customdata,
            hovertemplate=(
                "<b>Opción %{customdata[0]}</b><br>Score %{y:.2f}<br>"
                "ρ ref %{customdata[1]}<br>σₜ %{customdata[2]}<br>σₜ Al %{customdata[3]}<extra></extra>"
            ),
        ))
        fig.update_layout(yaxis_title="Score ± CI", xaxis_title="Opción", height=420, margin=dict(l=10,r=10,t=10,b=10))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
<div class="legend"><b>Interpretación:</b> si dos opciones se solapan mucho en su CI, tal vez requieras otra señal (p. ej., menos agua) para decidir.
</div>
""", unsafe_allow_html=True)

# ---------- TAB 3: Objetivos por eje ----------
with tab_objectives:
    st.markdown('<h3 class="section-title">Métricas por componente del objetivo</h3>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    if "Energía (kWh)" in df_view:
        with col1:
            st.markdown("**Energía (kWh)**")
            e_fig = px.histogram(df_view, x="Energía (kWh)")
            e_fig.update_layout(height=280, margin=dict(l=10,r=10,t=10,b=10))
            st.plotly_chart(e_fig, use_container_width=True)
    if "Agua (L)" in df_view:
        with col2:
            st.markdown("**Agua (L)**")
            w_fig = px.histogram(df_view, x="Agua (L)")
            w_fig.update_layout(height=280, margin=dict(l=10,r=10,t=10,b=10))
            st.plotly_chart(w_fig, use_container_width=True)
    if "Crew (min)" in df_view:
        with col3:
            st.markdown("**Crew (min)**")
            c_fig = px.histogram(df_view, x="Crew (min)")
            c_fig.update_layout(height=280, margin=dict(l=10,r=10,t=10,b=10))
            st.plotly_chart(c_fig, use_container_width=True)

    st.markdown("""
<div class="legend"><b>Ejemplo:</b> si tu objetivo prioriza tiempo de tripulación,
mirá la cola izquierda del histograma de <i>Crew (min)</i> y elegí opciones con menor valor.
</div>
""", unsafe_allow_html=True)

# ---------- TAB 4: Export Center ----------
with tab_export:
    st.markdown('<h3 class="section-title">🧭 Mission Export Assistant</h3>', unsafe_allow_html=True)

    if not selected_candidate:
        st.info("Seleccioná primero un plan en la pestaña **Pareto Explorer** para habilitar el asistente.")
    else:
        st.markdown(
            "<div class='safety-badges'>" + render_safety_badges_html(safety_flags) + "</div>",
            unsafe_allow_html=True,
        )
        st.caption(f"Estado de seguridad: {safety_summary['level']} — {safety_summary['detail']}")

        step_labels = ["1️⃣ Formato", "2️⃣ Previsualizar", "3️⃣ Confirmar"]
        current_step = st.session_state.get("export_wizard_step", 1)
        step_choice = st.radio(
            "Asistente de exportación",
            step_labels,
            index=max(0, min(len(step_labels) - 1, current_step - 1)),
            horizontal=True,
            key="export_step_radio",
        )
        current_step = step_labels.index(step_choice) + 1
        st.session_state["export_wizard_step"] = current_step

        format_options = ["Plan JSON", "Resumen CSV", "Pareto CSV"]
        if st.session_state["selected_export_format"] not in format_options:
            st.session_state["selected_export_format"] = "Plan JSON"

        def generate_payload(fmt: str):
            if fmt == "Plan JSON":
                if not safety_flags:
                    raise ValueError("Se requiere evaluación de seguridad para exportar JSON.")
                data = candidate_to_json(selected_candidate, target, safety_flags)
                filename = f"flight_plan_{int(selected_option_number or 0):02d}.json"
                mime = "application/json"
            elif fmt == "Resumen CSV":
                data = candidate_to_csv(selected_candidate)
                filename = f"candidate_{int(selected_option_number or 0):02d}_summary.csv"
                mime = "text/csv"
            elif fmt == "Pareto CSV":
                dataset = table_pareto if not table_pareto.empty else df_view
                data = dataset.to_csv(index=False).encode("utf-8")
                filename = "pareto_frontier.csv"
                mime = "text/csv"
            else:
                raise ValueError(f"Formato no soportado: {fmt}")
            return data, mime, filename

        with st.container():
            st.markdown("<div class='wizard-container'>", unsafe_allow_html=True)
            if current_step == 1:
                st.markdown(
                    """
                    <div class='wizard-panel'>
                      <h4>Paso 1 — Elegí tu carga útil</h4>
                      <p>Seleccioná el formato con el que vas a compartir el plan. Podés moverte de paso cuando quieras.</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                fmt_index = format_options.index(st.session_state.get("selected_export_format", "Plan JSON"))
                fmt_choice = st.radio(
                    "Formato de export",
                    format_options,
                    index=fmt_index,
                    key="export_format_selector",
                )
                st.session_state["selected_export_format"] = fmt_choice
                if st.button("Siguiente ➡️", key="wizard_next_1", use_container_width=True):
                    st.session_state["export_wizard_step"] = 2
                    current_step = 2
            elif current_step == 2:
                fmt_choice = st.session_state.get("selected_export_format", "Plan JSON")
                st.markdown(
                    """
                    <div class='wizard-panel'>
                      <h4>Paso 2 — Nebula preview</h4>
                      <p>Verificá los datos renderizados con el formato seleccionado antes de autorizar la exportación.</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                try:
                    payload_bytes, _, _ = generate_payload(fmt_choice)
                    if fmt_choice == "Plan JSON":
                        st.json(json.loads(payload_bytes.decode("utf-8")))
                    elif fmt_choice == "Resumen CSV":
                        preview_df = pd.read_csv(io.StringIO(payload_bytes.decode("utf-8")))
                        st.dataframe(preview_df, use_container_width=True, hide_index=True)
                    elif fmt_choice == "Pareto CSV":
                        st.dataframe(table_pareto if not table_pareto.empty else df_view, use_container_width=True, hide_index=True)
                except Exception as preview_error:
                    st.warning(f"No se pudo generar la previsualización: {preview_error}")

                col_back, col_next = st.columns([1, 1])
                with col_back:
                    if st.button("⬅️ Volver", key="wizard_back_2", use_container_width=True):
                        st.session_state["export_wizard_step"] = 1
                        current_step = 1
                with col_next:
                    if st.button("Continuar ➡️", key="wizard_next_2", use_container_width=True):
                        st.session_state["export_wizard_step"] = 3
                        current_step = 3
            else:
                fmt_choice = st.session_state.get("selected_export_format", "Plan JSON")
                st.markdown(
                    """
                    <div class='wizard-panel'>
                      <h4>Paso 3 — Checklist & confirmación</h4>
                      <p>Confirmá la exportación desde la consola translúcida. Podés volver atrás para ajustar.</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.markdown("<div class='translucent-panel'>", unsafe_allow_html=True)
                st.markdown(
                    f"""
                    <h4 style='margin-top:0;'>Checklist operativo</h4>
                    <ul>
                      <li>Formato elegido: <b>{fmt_choice}</b></li>
                      <li>Plan vinculado: <b>#{selected_option_number or '—'}</b> — {selected_candidate.get('process_name', 'sin proceso')}</li>
                      <li>Seguridad: <b>{safety_summary['level']}</b> · {safety_summary['detail']}</li>
                    </ul>
                    """,
                    unsafe_allow_html=True,
                )
                st.markdown(
                    "<div class='safety-badges'>" + render_safety_badges_html(safety_flags) + "</div>",
                    unsafe_allow_html=True,
                )
                col_confirm, col_back = st.columns([1.2, 1])
                confirm_clicked = False
                with col_confirm:
                    confirm_clicked = st.button("🚀 Generar paquete", key="wizard_confirm", use_container_width=True)
                with col_back:
                    if st.button("⬅️ Ajustar", key="wizard_back_3", use_container_width=True):
                        st.session_state["export_wizard_step"] = 2
                        current_step = 2

                if confirm_clicked:
                    try:
                        payload_bytes, mime, filename = generate_payload(fmt_choice)
                        timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
                        st.session_state["last_export_payload"] = {
                            "data": payload_bytes,
                            "mime": mime,
                            "filename": filename,
                            "format": fmt_choice,
                            "timestamp": timestamp,
                        }
                        history = st.session_state.get("export_history", [])
                        history.insert(
                            0,
                            {
                                "timestamp": timestamp,
                                "plan": f"#{selected_option_number or '—'}",
                                "format": fmt_choice,
                                "safety": safety_summary["level"],
                            },
                        )
                        st.session_state["export_history"] = history[:12]
                        st.success(f"Paquete {fmt_choice} listo para descargar.")
                    except Exception as export_error:
                        st.warning(f"No se pudo generar el paquete: {export_error}")

                payload = st.session_state.get("last_export_payload")
                if payload and payload.get("format") == fmt_choice:
                    st.download_button(
                        "⬇️ Descargar misión",
                        data=payload["data"],
                        file_name=payload["filename"],
                        mime=payload["mime"],
                        key="export_download_button",
                    )

                st.markdown("</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("### 📜 Historial de exportaciones")
        history = st.session_state.get("export_history", [])
        if history:
            hist_df = pd.DataFrame(history)
            ordered_cols = [c for c in ["timestamp", "plan", "format", "safety"] if c in hist_df.columns]
            if ordered_cols:
                hist_df = hist_df[ordered_cols]
            st.markdown(
                "<div class='history-table'>" + hist_df.to_html(index=False, escape=False) + "</div>",
                unsafe_allow_html=True,
            )
        else:
            st.caption("Aún no generaste exportaciones en esta sesión.")
