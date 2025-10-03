from app.bootstrap import ensure_streamlit_entrypoint

ensure_streamlit_entrypoint(__file__)

__doc__ = """Streamlit entrypoint that mirrors the mission overview page."""

from typing import Iterable

import streamlit as st

from app.modules import mission_overview
from app.modules.ml_models import get_model_registry
from app.modules.navigation import render_breadcrumbs, render_stepper, set_active_step
from app.modules.ui_blocks import initialise_frontend, load_theme
from app.modules.io import (
    MissingDatasetError,
    format_missing_dataset_message,
    get_last_modified,
)
from app.modules.paths import DATA_ROOT


def render_page() -> None:
    """Render the combined home + mission overview experience."""

    # ⚠️ Debe ser la PRIMERA llamada de Streamlit en la página
    st.set_page_config(page_title="Mission Overview", page_icon="🛰️", layout="wide")
    initialise_frontend()

    current_step = set_active_step("overview")

    load_theme(show_hud=False)

    render_breadcrumbs(current_step)
    render_stepper(current_step)

    st.title("0) Mission Overview")

    try:
        inventory_df = mission_overview.load_inventory_overview()
    except MissingDatasetError as error:
        st.error(format_missing_dataset_message(error))
        st.stop()
    mission_metrics = mission_overview.compute_mission_summary(inventory_df)
    mission_overview.render_mission_objective(mission_metrics)

    registry = get_model_registry()
    metadata = getattr(registry, "metadata", {}) if registry else {}
    model_summary = mission_overview.summarize_model_state(metadata)
    mission_overview.render_model_health(model_summary)

    if inventory_df is None or inventory_df.empty:
        st.info("No se encontró inventario para la misión actual.")
        return

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

    data_path = DATA_ROOT / "waste_inventory_sample.csv"
    last_modified = get_last_modified(data_path)
    if last_modified:
        st.caption(last_modified.strftime("Actualizado: %Y-%m-%d %H:%M"))

    mission_overview.render_material_summary(inventory_df, max_rows=20)


if __name__ == "__main__":  # pragma: no cover - Streamlit entrypoint
    render_page()
