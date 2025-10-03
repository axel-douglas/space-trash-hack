"""Consolidated feedback capture and mission impact tracking."""

from datetime import datetime

from app.bootstrap import ensure_streamlit_path

ensure_streamlit_path(__file__)

import pandas as pd
import streamlit as st

from app.modules.impact import (
    FeedbackEntry,
    ImpactEntry,
    append_feedback,
    append_impact,
    load_feedback_df,
    load_impact_df,
    parse_extra_blob,
    summarize_impact,
)
from app.modules.navigation import render_breadcrumbs, set_active_step
from app.modules.page_data import build_feedback_summary_table
from app.modules.ui_blocks import initialise_frontend, layout_stack, load_theme


st.set_page_config(page_title="Feedback & Impact", page_icon="📝", layout="wide")
initialise_frontend()

current_step = set_active_step("feedback")

load_theme()

render_breadcrumbs(current_step)


def _expand_extra_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty or "extra" not in df.columns:
        return df
    extras = pd.DataFrame([parse_extra_blob(value) for value in df["extra"]])
    merged = pd.concat([df.drop(columns=["extra"]), extras], axis=1)
    return merged


def _format_metric(value: float | int | None, precision: int = 2) -> str:
    if value is None:
        return "—"
    return f"{value:.{precision}f}"


target = st.session_state.get("target")
selected_state = st.session_state.get("selected")
selected_candidate = selected_state.get("data") if isinstance(selected_state, dict) else None
props = selected_candidate.get("props") if isinstance(selected_candidate, dict) else None
selected_option_number = st.session_state.get("selected_option_number")

impact_df = load_impact_df() or pd.DataFrame()
feedback_df = load_feedback_df() or pd.DataFrame()
expanded_impact_df = _expand_extra_columns(impact_df)
expanded_feedback_df = _expand_extra_columns(feedback_df)

impact_summary = summarize_impact(impact_df if not impact_df.empty else pd.DataFrame())
feedback_summary_df = build_feedback_summary_table(expanded_feedback_df)

with layout_stack():
    st.title("📝 Feedback & Impact")
    st.caption("Registra corridas reales, consolida el feedback y cuantificá el impacto en la misión.")

metric_cols = st.columns(4)
metric_cols[0].metric("Corridas registradas", impact_summary.get("runs", 0))
metric_cols[1].metric("Kg totales", _format_metric(impact_summary.get("kg")))
metric_cols[2].metric("kWh totales", _format_metric(impact_summary.get("kwh")))
metric_cols[3].metric("Crew total (min)", _format_metric(impact_summary.get("crew_min"), precision=0))

if not feedback_summary_df.empty:
    st.subheader("Impacto del feedback")
    st.dataframe(feedback_summary_df, use_container_width=True, hide_index=True)
else:
    st.caption("Aún no se registró feedback cuantitativo.")

st.subheader("Corridas registradas")
if expanded_impact_df.empty:
    st.info("Todavía no hay registros de impacto. Usá el formulario para agregar la primera corrida.")
else:
    impact_preview = expanded_impact_df.sort_values("ts_iso", ascending=False).head(20)
    st.dataframe(impact_preview, use_container_width=True, hide_index=True)

st.subheader("Feedback recientes")
if expanded_feedback_df.empty:
    st.caption("Sin feedback registrado por el momento.")
else:
    feedback_preview = expanded_feedback_df.sort_values("ts_iso", ascending=False).head(20)
    st.dataframe(feedback_preview, use_container_width=True, hide_index=True)

st.markdown("---")

with st.form("impact_form"):
    st.subheader("Registrar impacto de una corrida")
    scenario = st.selectbox(
        "Escenario",
        [target.get("scenario", "-")] if isinstance(target, dict) else ["-"],
    )
    default_mass = float(getattr(props, "mass_final_kg", 0.0) or 0.0)
    default_energy = float(getattr(props, "energy_kwh", 0.0) or 0.0)
    default_water = float(getattr(props, "water_l", 0.0) or 0.0)
    default_crew = float(getattr(props, "crew_min", 0.0) or 0.0)
    col_mass, col_energy, col_water, col_crew = st.columns(4)
    mass_value = col_mass.number_input("Masa final (kg)", min_value=0.0, value=default_mass, step=0.05)
    energy_value = col_energy.number_input("Energía (kWh)", min_value=0.0, value=default_energy, step=0.01)
    water_value = col_water.number_input("Agua (L)", min_value=0.0, value=default_water, step=0.01)
    crew_value = col_crew.number_input("Crew (min)", min_value=0.0, value=default_crew, step=1.0)
    impact_notes = st.text_area("Notas de la corrida", "")
    submit_impact = st.form_submit_button("Guardar impacto")
    if submit_impact:
        candidate_materials = (selected_candidate.get("materials") if isinstance(selected_candidate, dict) else []) or []
        candidate_weights = (selected_candidate.get("weights") if isinstance(selected_candidate, dict) else []) or []
        process_id = str(selected_candidate.get("process_id", "")) if isinstance(selected_candidate, dict) else ""
        process_name = str(selected_candidate.get("process_name", "")) if isinstance(selected_candidate, dict) else ""
        score_value = selected_candidate.get("score", 0.0) if isinstance(selected_candidate, dict) else 0.0
        entry = ImpactEntry(
            ts_iso=datetime.utcnow().isoformat() + "Z",
            scenario=str(scenario),
            target_name=str(target.get("name", "-")) if isinstance(target, dict) else "-",
            materials="|".join(map(str, candidate_materials)),
            weights="|".join(map(str, candidate_weights)),
            process_id=process_id,
            process_name=process_name,
            mass_final_kg=float(mass_value),
            energy_kwh=float(energy_value),
            water_l=float(water_value),
            crew_min=float(crew_value),
            score=float(score_value or 0.0),
            extra={"notes": impact_notes},
        )
        append_impact(entry)
        st.success("Impacto registrado.")
        st.rerun()

with st.form("feedback_form"):
    st.subheader("Registrar feedback operativo")
    astronaut = st.text_input("Operador o equipo", value="")
    overall = st.slider("Satisfacción general", 0, 10, 8)
    rigidity_ok = st.checkbox("Rigidez dentro de lo esperado", value=True)
    ease_ok = st.checkbox("Proceso fácil de ejecutar", value=True)
    issues_text = st.text_area("Issues detectados", "")
    notes_text = st.text_area("Notas adicionales", "")
    submit_feedback = st.form_submit_button("Guardar feedback")
    if submit_feedback:
        entry = FeedbackEntry(
            ts_iso=datetime.utcnow().isoformat() + "Z",
            astronaut=astronaut or "-",
            scenario=str(target.get("scenario", "-")) if isinstance(target, dict) else "-",
            target_name=str(target.get("name", "-")) if isinstance(target, dict) else "-",
            option_idx=int(selected_option_number or 0),
            rigidity_ok=bool(rigidity_ok),
            ease_ok=bool(ease_ok),
            issues=issues_text,
            notes=notes_text,
            extra={
                "feedback_overall": overall,
                "feedback_rigidity": 10 if rigidity_ok else 5,
                "feedback_ease": 10 if ease_ok else 5,
            },
        )
        append_feedback(entry)
        st.success("Feedback registrado.")
        st.rerun()
