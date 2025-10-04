import sys
from pathlib import Path

if not __package__:
    repo_root = Path(__file__).resolve().parents[2]
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)

from app.bootstrap import ensure_streamlit_entrypoint

ensure_streamlit_entrypoint(__file__)

import altair as alt
import pandas as pd
import streamlit as st

from app.modules import data_sources as ds
from app.modules.generator.service import GeneratorService
from app.modules.ui_blocks import configure_page, initialise_frontend, layout_block, render_brand_header

configure_page(page_title="Rex-AI • Mezclador espectral", page_icon="🌈")
initialise_frontend()
render_brand_header()

st.title("🌈 Mezclador espectral FTIR")
st.caption("Subí una firma objetivo y Rex-AI propondrá una mezcla viable usando el stock disponible.")

try:
    bundle = ds.load_material_reference_bundle()
except Exception as error:  # pragma: no cover - fallback for missing datasets
    st.error(f"No fue posible cargar las curvas de referencia: {error}")
    st.stop()

if not bundle.spectral_curves:
    st.info("El bundle de referencia no incluye curvas FTIR para mezclar todavía.")
    st.stop()

spectral_keys = sorted(bundle.spectral_curves.keys())
generator = GeneratorService()

with layout_block("layout-stack"):
    uploaded_file = st.file_uploader(
        "Arrastrá un CSV con la firma FTIR objetivo",
        type=["csv"],
        accept_multiple_files=False,
    )

    selected_materials = st.multiselect(
        "Seleccioná materiales base",
        spectral_keys,
        default=spectral_keys,
        help="Las curvas provienen del bundle de referencia Zenodo.",
    )

    stock_entries: list[dict[str, float | str]] = []
    if selected_materials:
        st.markdown("#### Stock disponible")
        columns = st.columns(max(1, min(3, len(selected_materials))))
        for column, key in zip(columns, selected_materials, strict=False):
            stock_entries.append(
                {
                    "spectral_key": key,
                    "kg": column.number_input(
                        f"{key}",
                        min_value=0.0,
                        value=1.0,
                        step=0.1,
                        help="Cantidad máxima disponible en kg.",
                        key=f"spectral-stock-{key}",
                    ),
                }
            )
    stock_df = pd.DataFrame(stock_entries) if stock_entries else pd.DataFrame(columns=["spectral_key", "kg"])

    max_components = 1
    if selected_materials:
        max_components = st.slider(
            "Máx. componentes en la mezcla",
            min_value=1,
            max_value=len(selected_materials),
            value=min(3, len(selected_materials)),
        )
    max_fraction = st.slider(
        "Fracción total máxima",
        min_value=0.1,
        max_value=1.0,
        value=1.0,
        step=0.05,
    )

    result_payload: dict[str, object] | None = None
    if uploaded_file is not None and selected_materials:
        try:
            target_df = pd.read_csv(uploaded_file)
        except Exception as error:
            st.error(f"No se pudo leer el CSV: {error}")
            target_df = pd.DataFrame()
        if not target_df.empty:
            try:
                constraints = {"max_components": max_components, "max_fraction": max_fraction}
                result_payload = generator.propose_spectral_mix(
                    target_curve=target_df,
                    stock_df=stock_df,
                    constraints=constraints,
                )
            except Exception as error:  # pragma: no cover - surfaced to the UI
                st.error(f"El solver no pudo encontrar una mezcla: {error}")
        else:
            st.info("El CSV no contiene filas con datos numéricos.")
    elif uploaded_file is None:
        st.info("Cargá un CSV para obtener la mezcla propuesta.")
    elif not selected_materials:
        st.info("Seleccioná al menos un material base.")

if not result_payload:
    st.stop()

coefficients = pd.DataFrame(result_payload.get("coefficients", []))
synthetic_curve = result_payload.get("synthetic_curve")
target_curve = result_payload.get("target_curve")
error_metrics = result_payload.get("error", {})

st.subheader("Comparativa espectral")
if isinstance(synthetic_curve, pd.DataFrame) and isinstance(target_curve, pd.DataFrame):
    target_df = target_curve.rename(columns={"intensity": "value"})
    target_df["serie"] = "Objetivo"
    synthetic_df = synthetic_curve.rename(columns={"synthetic_intensity": "value"})[["wavenumber_cm_1", "synthetic_intensity"]]
    synthetic_df = synthetic_df.rename(columns={"synthetic_intensity": "value"})
    synthetic_df["serie"] = "Mezcla"
    chart_df = pd.concat([target_df[["wavenumber_cm_1", "value", "serie"]], synthetic_df], ignore_index=True)
    chart = (
        alt.Chart(chart_df)
        .mark_line()
        .encode(
            x=alt.X("wavenumber_cm_1", title="Número de onda (cm⁻¹)"),
            y=alt.Y("value", title="Intensidad"),
            color=alt.Color("serie", title="Curva"),
        )
        .interactive()
    )
    st.altair_chart(chart, use_container_width=True)
else:
    st.warning("No se pudieron graficar las curvas generadas.")

if not coefficients.empty:
    st.subheader("Receta recomendada")
    stock_limits = {}
    if not stock_df.empty and stock_df["kg"].sum() > 0:
        total_stock = float(stock_df["kg"].sum())
        stock_limits = {
            row["spectral_key"]: float(row["kg"]) / total_stock
            for _, row in stock_df.iterrows()
            if float(total_stock) > 0
        }
    coefficients["stock_limit"] = coefficients["material"].map(stock_limits).fillna(1.0)
    coefficients = coefficients.rename(columns={"material": "Material", "fraction": "Fracción", "stock_limit": "Máx. logística"})
    coefficients["Máx. logística"] = coefficients["Máx. logística"].map(lambda x: f"{x:.2f}")
    coefficients["Fracción"] = coefficients["Fracción"].map(lambda x: f"{x:.3f}")
    st.dataframe(coefficients, hide_index=True)

if error_metrics:
    st.subheader("Errores de ajuste")
    metrics_df = pd.DataFrame(
        {
            "Métrica": ["MAE", "RMSE", "Error máximo"],
            "Valor": [
                f"{float(error_metrics.get('mae', float('nan'))):.4f}",
                f"{float(error_metrics.get('rmse', float('nan'))):.4f}",
                f"{float(error_metrics.get('max_abs', float('nan'))):.4f}",
            ],
        }
    )
    st.table(metrics_df)
