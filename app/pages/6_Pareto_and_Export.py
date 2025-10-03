import sys
from pathlib import Path

if not __package__:
    repo_root = Path(__file__).resolve().parents[2]
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)

from app.bootstrap import ensure_streamlit_entrypoint

ensure_streamlit_entrypoint(__file__)

__doc__ = """Streamlined Pareto exploration and export centre."""

import math
from typing import Iterable

import streamlit as st

from app.modules.analytics import pareto_front
from app.modules.exporters import candidate_to_csv, candidate_to_json
from app.modules.io import (
    MissingDatasetError,
    format_missing_dataset_message,
    load_waste_df,
)
from app.modules.navigation import render_breadcrumbs, set_active_step
from app.modules.page_data import (
    build_candidate_export_table,
    build_export_kpi_table,
    build_material_summary_table,
)
from app.modules.safety import check_safety, safety_badge
from app.modules.ui_blocks import (
    action_button,
    configure_page,
    initialise_frontend,
    layout_stack,
    render_brand_header,
)
from app.modules.utils import safe_int

configure_page(page_title="Pareto & Export", page_icon="📤")
initialise_frontend()

current_step = set_active_step("export")

render_brand_header()

render_breadcrumbs(current_step)


def _safe_float(value: object, fallback: float | None = None) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return fallback
    if math.isnan(number):
        return fallback
    return number


candidates: Iterable[dict] = st.session_state.get("candidates", [])
target = st.session_state.get("target")
selected_state = st.session_state.get("selected")

if not candidates or not target:
    st.warning("Generá candidatos en **3 · Generador** y definí un objetivo en **2 · Target Designer**.")
    st.stop()

selected_candidate = selected_state["data"] if isinstance(selected_state, dict) else None
selected_badge = selected_state.get("safety") if isinstance(selected_state, dict) else None
selected_option_number = safe_int(st.session_state.get("selected_option_number"), default=0)

try:
    inventory_df = load_waste_df()
except MissingDatasetError as error:
    st.error(format_missing_dataset_message(error))
    st.stop()
export_df = build_candidate_export_table(candidates, inventory_df)

with layout_stack():
    st.title("📤 Pareto & Export")
    st.caption(
        "Explorá la frontera de Pareto con límites reales y exportá el plan priorizado "
        "para compartirlo con la tripulación."
    )

kpi_df = build_export_kpi_table(export_df)
if not kpi_df.empty:
    kpi_lookup = {row["KPI"]: row["Valor"] for _, row in kpi_df.iterrows()}
    metrics = [
        ("Opciones válidas", kpi_lookup.get("Opciones válidas")),
        ("Score máximo", kpi_lookup.get("Score máximo")),
        ("Mín. Agua", kpi_lookup.get("Mín. Agua")),
        ("Mín. Energía", kpi_lookup.get("Mín. Energía")),
    ]
    metric_columns = st.columns(len(metrics))
    for column, (label, value) in zip(metric_columns, metrics):
        display = "—" if value is None or (isinstance(value, float) and math.isnan(value)) else f"{value:.3f}"
        column.metric(label, display)

limits_container = st.container()
with limits_container:
    st.subheader("Límites activos")
    defaults = {
        "energy": _safe_float(target.get("max_energy_kwh"), fallback=_safe_float(export_df["Energía (kWh)"].max())),
        "water": _safe_float(target.get("max_water_l"), fallback=_safe_float(export_df["Agua (L)"].max())),
        "crew": _safe_float(target.get("max_crew_min"), fallback=_safe_float(export_df["Crew (min)"].max())),
    }
    col_energy, col_water, col_crew = st.columns(3)
    with col_energy:
        limit_energy = st.number_input(
            "Máximo de energía (kWh)",
            min_value=0.0,
            value=defaults["energy"] or 0.0,
            step=0.05,
            key="export_limit_energy",
        )
    with col_water:
        limit_water = st.number_input(
            "Máximo de agua (L)",
            min_value=0.0,
            value=defaults["water"] or 0.0,
            step=0.05,
            key="export_limit_water",
        )
    with col_crew:
        limit_crew = st.number_input(
            "Máximo de crew (min)",
            min_value=0.0,
            value=defaults["crew"] or 0.0,
            step=1.0,
            key="export_limit_crew",
        )

filtered_df = export_df.copy()
filtered_df = filtered_df[
    (filtered_df["Energía (kWh)"].le(limit_energy) | filtered_df["Energía (kWh)"].isna())
    & (filtered_df["Agua (L)"].le(limit_water) | filtered_df["Agua (L)"].isna())
    & (filtered_df["Crew (min)"].le(limit_crew) | filtered_df["Crew (min)"].isna())
]

pareto_indices: list[int] = []
if not filtered_df.empty:
    required = ["Energía (kWh)", "Agua (L)", "Crew (min)", "Score"]
    if all(column in filtered_df.columns for column in required):
        pareto_indices = pareto_front(filtered_df[required])
filtered_df["En Pareto"] = filtered_df.index.isin(pareto_indices)

option_numbers: list[int] = []
for option in filtered_df["Opción"].tolist():
    parsed_option = safe_int(option, default=None)
    if parsed_option is not None and parsed_option > 0:
        option_numbers.append(parsed_option)
if option_numbers:
    default_index = 0
    if selected_option_number in option_numbers:
        default_index = option_numbers.index(selected_option_number)
    chosen_option = st.selectbox(
        "Seleccioná el plan prioritario",
        option_numbers,
        index=default_index,
        key="export_active_option",
    )
    if chosen_option != selected_option_number:
        idx = chosen_option - 1
        if 0 <= idx < len(candidates):
            candidate = candidates[idx]
            materials = [str(item) for item in candidate.get("materials", [])]
            process_name = str(candidate.get("process_name") or "")
            process_id = str(candidate.get("process_id") or "")
            flags = check_safety(materials, process_name, process_id)
            badge = safety_badge(flags)
            badge.update(
                {
                    "pfas": bool(getattr(flags, "pfas", False)),
                    "microplastics": bool(getattr(flags, "microplastics", False)),
                    "incineration": bool(getattr(flags, "incineration", False)),
                }
            )
            st.session_state["selected"] = {"data": candidate, "safety": badge}
            selected_candidate = candidate
            selected_badge = badge
        st.session_state["selected_option_number"] = chosen_option
        selected_option_number = chosen_option

selected_series = filtered_df["Opción"].apply(lambda value: safe_int(value, default=None))
if selected_option_number > 0:
    filtered_df["Seleccionado"] = selected_series.eq(selected_option_number).fillna(False)
else:
    filtered_df["Seleccionado"] = False

st.subheader("Opciones dentro de límites")
st.dataframe(
    filtered_df,
    use_container_width=True,
    hide_index=True,
)

materials_df = build_material_summary_table(candidates)
if not materials_df.empty:
    st.subheader("Top residuos aportados")
    st.dataframe(materials_df, use_container_width=True, hide_index=True)

if selected_candidate and isinstance(selected_badge, dict):
    st.subheader("Estado del plan seleccionado")
    option_label = safe_int(selected_option_number, default=0)
    option_text = f"#{option_label}" if option_label else "—"
    st.markdown(
        f"**Opción {option_text}** — {selected_candidate.get('process_name', 'Proceso')}"
    )
    st.caption(f"Seguridad: {selected_badge.get('level', '—')} · {selected_badge.get('detail', '')}")
else:
    st.info("Seleccioná un plan para habilitar la exportación.")

st.markdown("---")

if selected_candidate:
    st.subheader("Exportar")
    badge_payload = selected_badge or {}
    json_data = candidate_to_json(selected_candidate, target, badge_payload)
    csv_data = candidate_to_csv(selected_candidate)
    pareto_payload = filtered_df.to_csv(index=False).encode("utf-8")
    col_json, col_csv, col_pareto = st.columns(3)
    with col_json:
        action_button(
            "⬇️ Plan JSON",
            key="export_plan_json",
            download_data=json_data,
            download_file_name=f"flight_plan_{safe_int(selected_option_number, default=0):02d}.json",
            download_mime="application/json",
            state="idle",
        )
    with col_csv:
        action_button(
            "⬇️ Resumen CSV",
            key="export_candidate_csv",
            download_data=csv_data,
            download_file_name=f"candidate_{safe_int(selected_option_number, default=0):02d}_summary.csv",
            download_mime="text/csv",
            state="idle",
        )
    with col_pareto:
        action_button(
            "⬇️ Tabla filtrada",
            key="export_filtered_table",
            download_data=pareto_payload,
            download_file_name="pareto_filtrado.csv",
            download_mime="text/csv",
            state="idle",
        )
else:
    st.caption("Los botones de exportación se activan al elegir un plan prioritario.")
