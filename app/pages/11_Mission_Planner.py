import sys
from pathlib import Path

if not __package__:
    repo_root = Path(__file__).resolve().parents[2]
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)

from app.bootstrap import ensure_streamlit_entrypoint

ensure_streamlit_entrypoint(__file__)

__doc__ = """Interactive mission planner combining process, inventory and policy signals."""

import pandas as pd
import streamlit as st

from app.modules import mission_planner
from app.modules.navigation import render_breadcrumbs, set_active_step
from app.modules.process_planner import SCENARIO_HINTS
from app.modules.ui_blocks import (
    configure_page,
    initialise_frontend,
    layout_stack,
    render_brand_header,
)


configure_page(page_title="Mission Planner", page_icon="🛰️")
initialise_frontend()

current_step = set_active_step("mission_planner")

render_brand_header()

render_breadcrumbs(current_step, extra=["Mission Planner"])

inventory = mission_planner.load_inventory()
scenario_options = [""] + sorted(SCENARIO_HINTS)
has_id_column = "id" in inventory.columns
inventory_by_id = inventory.set_index("id") if has_id_column else pd.DataFrame()

with layout_stack():
    st.title("🛰️ Mission Planner")
    st.caption(
        "Seleccioná lotes de materiales, ajustá objetivos operativos y generá rutas logísticas optimizadas."
    )

    selector_cols = st.columns((2, 2, 1.4, 1.4))
    scenario = selector_cols[0].selectbox(
        "Escenario operativo",
        options=scenario_options,
        format_func=lambda value: "Escenario base" if value == "" else value.title(),
    )
    crew_bias = selector_cols[1].slider(
        "Disponibilidad de crew",
        0,
        10,
        5,
        help="Valores bajos priorizan procesos con menor tiempo de crew por lote.",
    )
    energy_limit = selector_cols[2].slider(
        "Límite energía (kWh/kg)",
        0.0,
        1.5,
        0.7,
        step=0.05,
    )
    crew_limit = selector_cols[3].slider(
        "Límite crew (min/lote)",
        0,
        60,
        24,
        step=2,
    )

    material_ids = inventory.get("id", pd.Series(dtype=str)).tolist()
    default_selection = material_ids[:4]
    selected_ids = st.multiselect(
        "Materiales disponibles",
        options=material_ids,
        default=default_selection,
        format_func=lambda value: (
            f"{value} · {inventory_by_id.loc[value, 'material']}"
            if has_id_column and value in inventory_by_id.index
            else value
        ),
    )

    if has_id_column:
        selected = inventory[inventory["id"].isin(selected_ids)].copy()
    else:
        selected = inventory.copy()
        if selected_ids:
            selected = selected.head(len(selected_ids))
    if selected.empty and not inventory.empty:
        selected = inventory.head(3).copy()
        st.info("No se seleccionaron materiales, se utilizarán los primeros del inventario de referencia.")

    lot_size = st.slider(
        "Tamaño de lote (nº materiales)",
        min_value=1,
        max_value=max(1, len(selected)),
        value=min(3, max(1, len(selected))),
    )

    objective = st.radio(
        "Objetivo primario",
        options=("min_energy", "max_rigidity"),
        format_func=lambda value: "Minimizar energía" if value == "min_energy" else "Maximizar rigidez",
        horizontal=True,
    )

    target_cols = st.columns(3)
    max_energy_total = target_cols[0].number_input(
        "Energía total máx. (kWh)",
        min_value=10.0,
        value=250.0,
        step=10.0,
    )
    max_water_total = target_cols[1].number_input(
        "Agua total máx. (L)",
        min_value=0.0,
        value=30.0,
        step=5.0,
    )
    max_crew_total = target_cols[2].number_input(
        "Crew total máx. (min)",
        min_value=10.0,
        value=480.0,
        step=10.0,
    )

assignments = mission_planner.recommend_processes(
    selected,
    scenario=scenario or None,
    crew_time_low=crew_bias <= 4,
    max_energy_kwh_per_kg=energy_limit if energy_limit > 0 else None,
    max_crew_min_per_batch=crew_limit if crew_limit > 0 else None,
)
assignments_df = mission_planner.assignments_to_dataframe(assignments)

manifest = mission_planner.build_manifest(selected)
scored_manifest, alerts = mission_planner.evaluate_policy_signals(manifest)

target_limits = {
    "max_energy_kwh": float(max_energy_total),
    "max_water_l": float(max_water_total),
    "max_crew_min": float(max_crew_total),
}

pareto_df, pareto_candidates = mission_planner.optimize_assignments(
    assignments,
    scored_manifest,
    lot_size=lot_size,
    objective=objective,
    target_limits=target_limits,
)

sankey = mission_planner.build_sankey(assignments)

with layout_stack():
    st.subheader("Asignación de procesos sugeridos")
    if assignments_df.empty:
        st.warning("No se encontraron procesos compatibles con las restricciones seleccionadas.")
    else:
        st.dataframe(assignments_df, use_container_width=True, hide_index=True)

    st.subheader("Logística prevista")
    if sankey is None:
        st.caption("Añadí materiales para visualizar la ruta logística consolidada.")
    else:
        st.plotly_chart(sankey, use_container_width=True)

    st.subheader("Optimización del lote")
    if pareto_df.empty:
        st.caption("Ajustá el tamaño del lote u objetivos para explorar combinaciones óptimas.")
    else:
        st.dataframe(pareto_df, use_container_width=True, hide_index=True)
        top_candidate = pareto_candidates[0]
        products = top_candidate.get("products", [])
        if products:
            st.success(
                "Productos habilitados: "
                + ", ".join(products)
                + ". Estos outputs pueden alimentar composites, paneles estructurales y kits modulares futuros."
            )

    st.subheader("Alertas de política y sustitución")
    if not alerts:
        st.caption("Sin alertas relevantes para los materiales seleccionados.")
    else:
        for alert in alerts:
            st.warning(alert)

