"""Mission overview entrypoint consolidating mission status panels."""

from datetime import datetime
from pathlib import Path
from typing import Iterable

import streamlit as st

from app.modules.mission_overview import (
    compute_mission_summary,
    load_inventory_overview,
    render_material_summary,
    render_mission_objective,
    render_model_health,
    summarize_model_state,
)
from app.modules.ml_models import get_model_registry
from app.modules.navigation import render_breadcrumbs, render_stepper, set_active_step
from app.modules.ui_blocks import initialise_frontend, load_theme


# ⚠️ Debe ser la PRIMERA llamada de Streamlit en la página
st.set_page_config(page_title="Mission Overview", page_icon="🛰️", layout="wide")
initialise_frontend()

current_step = set_active_step("overview")

load_theme(show_hud=False)

render_breadcrumbs(current_step)
render_stepper(current_step)

st.title("0) Mission Overview")

inventory_df = load_inventory_overview()
mission_metrics = compute_mission_summary(inventory_df)
render_mission_objective(mission_metrics)

registry = get_model_registry()
metadata = getattr(registry, "metadata", {}) if registry else {}
model_summary = summarize_model_state(metadata)
render_model_health(model_summary)

if inventory_df is None or inventory_df.empty:
    st.info("No se encontró inventario para la misión actual.")
else:
    categories_column = inventory_df.get("category")
    unique_categories: list[str] = []
    if categories_column is not None:
        categories: Iterable[str] = (
            str(value).strip() for value in categories_column if str(value).strip()
        )
        unique_categories = sorted({category for category in categories})

    problematic_column = inventory_df.get("_problematic")
    problematic = 0
    if problematic_column is not None:
        problematic = int(problematic_column.astype(bool).sum())

    if unique_categories:
        categories_label = ", ".join(unique_categories[:5])
        if len(unique_categories) > 5:
            categories_label += "…"
        st.caption(f"Categorías: {categories_label}")
    st.caption(f"Problemáticos detectados: {problematic}")

    last_modified: datetime | None = None
    data_path = Path("data/waste_inventory_sample.csv")
    if data_path.exists():
        last_modified = datetime.fromtimestamp(data_path.stat().st_mtime)
    if last_modified:
        st.caption(last_modified.strftime("Actualizado: %Y-%m-%d %H:%M"))

    render_material_summary(inventory_df, max_rows=20)
