import sys
from pathlib import Path

from app.bootstrap import ensure_streamlit_entrypoint

ensure_streamlit_entrypoint(__file__)

import io
from typing import Any

import pandas as pd
import streamlit as st

from app.modules.generator import GeneratorService
from app.modules.ui_blocks import (
    configure_page,
    initialise_frontend,
    micro_divider,
    render_brand_header,
)

configure_page(page_title="Rex-AI • Aduana inteligente", page_icon="🛃")
initialise_frontend()
render_brand_header()

st.header("Aduana inteligente")
st.write(
    "Sube el manifiesto de carga para evaluar compatibilidad material, penalizaciones operativas y oportunidades de sustitución."
)

service = GeneratorService()

_TEMPLATE_COLUMNS = [
    "item",
    "category",
    "mass_kg",
    "tg_loss_pct",
    "ega_loss_pct",
    "water_l_per_kg",
    "energy_kwh_per_kg",
]


def _render_template_download() -> None:
    template_df = pd.DataFrame(
        [
            {
                "item": "HDPE packaging film",
                "category": "Packaging",
                "mass_kg": 12.5,
                "tg_loss_pct": 4.0,
                "ega_loss_pct": 0.5,
                "water_l_per_kg": 0.1,
                "energy_kwh_per_kg": 0.9,
            },
            {
                "item": "Nomex insulation",
                "category": "Structural elements",
                "mass_kg": 8.2,
                "tg_loss_pct": 2.5,
                "ega_loss_pct": 0.2,
                "water_l_per_kg": 0.0,
                "energy_kwh_per_kg": 0.4,
            },
        ],
        columns=_TEMPLATE_COLUMNS,
    )
    csv_buffer = io.StringIO()
    template_df.to_csv(csv_buffer, index=False)
    st.download_button(
        "Descargar plantilla CSV",
        csv_buffer.getvalue().encode("utf-8"),
        file_name="manifiesto_plantilla.csv",
        mime="text/csv",
        use_container_width=True,
    )


def _traffic_color(score: float) -> str:
    if score >= 0.75:
        return "#22c55e"
    if score >= 0.5:
        return "#facc15"
    return "#ef4444"


def _traffic_label(score: float) -> str:
    if score >= 0.75:
        return "Alto"
    if score >= 0.5:
        return "Medio"
    return "Bajo"


def _render_metric(label: str, score: float, help_text: str | None = None) -> None:
    color = _traffic_color(score)
    status = _traffic_label(score)
    container = st.container()
    with container:
        st.markdown(
            f"<div style='border-radius:12px;padding:1rem;background:{color};color:white;'>"
            f"<div style='font-size:0.85rem;text-transform:uppercase;opacity:0.85;'>{label}</div>"
            f"<div style='font-size:1.8rem;font-weight:700;'>{score:.2f}</div>"
            f"<div style='font-size:0.9rem;'>Nivel {status}</div>"
            "</div>",
            unsafe_allow_html=True,
        )
        if help_text:
            st.caption(help_text)


_render_template_download()

uploaded_file = st.file_uploader("Manifiesto (CSV)", type=["csv"], accept_multiple_files=False)
include_pdf = st.checkbox("Generar Material Passport en PDF", value=False)

analysis_state: dict[str, Any] | None = st.session_state.get("policy_analysis")

if st.button("Evaluar manifiesto", use_container_width=True) and uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    with st.spinner("Analizando manifiesto con heurísticas de política..."):
        analysis_state = service.analyze_manifest(data, include_pdf=include_pdf)
        st.session_state["policy_analysis"] = analysis_state

if analysis_state is None:
    st.info("Carga un manifiesto y presiona \"Evaluar manifiesto\" para obtener recomendaciones.")
    st.stop()

scored_manifest = analysis_state["scored_manifest"]
recommendations = analysis_state["policy_recommendations"]
compatibility = analysis_state["compatibility_records"]
passport = analysis_state["material_passport"]
artifacts = analysis_state["artifacts"]

summary_cols = st.columns(3)
with summary_cols[0]:
    _render_metric("Puntaje promedio", float(passport.get("mean_material_utility_score", 0.0)))
with summary_cols[1]:
    total_mass = float(passport.get("total_mass_kg", 0.0))
    st.metric("Masa total (kg)", f"{total_mass:.1f}")
with summary_cols[2]:
    st.metric("Total ítems", int(passport.get("total_items", 0)))

micro_divider()

st.subheader("Detalle de ítems evaluados")
st.dataframe(
    scored_manifest[[
        "item",
        "material_key",
        "material_utility_score",
        "spectral_score",
        "mechanical_score",
        "penalty_factor",
        "match_confidence",
    ]].sort_values("material_utility_score", ascending=False),
    use_container_width=True,
)

micro_divider()

st.subheader("Recomendaciones de política")
if recommendations.empty:
    st.success("No se identificaron acciones prioritarias: todos los ítems superan el umbral de utilidad.")
else:
    st.dataframe(recommendations, use_container_width=True)

micro_divider()

st.subheader("Trazabilidad de compatibilidad")
if compatibility.empty:
    st.write("Sin datos de compatibilidad asociados al manifiesto.")
else:
    st.dataframe(compatibility, use_container_width=True)

micro_divider()

st.subheader("Material Passport")
st.json(passport)

st.markdown("### Descargas")
col_a, col_b, col_c = st.columns(3)
with col_a:
    policy_path = artifacts.get("policy_recommendations_csv")
    if isinstance(policy_path, Path) and policy_path.exists():
        st.download_button(
            "Descargar recomendaciones (CSV)",
            policy_path.read_bytes(),
            file_name=policy_path.name,
            mime="text/csv",
            use_container_width=True,
        )
with col_b:
    compat_path = artifacts.get("compatibility_matrix_parquet")
    if isinstance(compat_path, Path) and compat_path.exists():
        st.download_button(
            "Descargar compatibilidad (Parquet)",
            compat_path.read_bytes(),
            file_name=compat_path.name,
            mime="application/octet-stream",
            use_container_width=True,
        )
with col_c:
    passport_path = artifacts.get("material_passport_json")
    if isinstance(passport_path, Path) and passport_path.exists():
        st.download_button(
            "Descargar Material Passport (JSON)",
            passport_path.read_bytes(),
            file_name=passport_path.name,
            mime="application/json",
            use_container_width=True,
        )

if include_pdf:
    pdf_path = artifacts.get("material_passport_pdf")
    if isinstance(pdf_path, Path) and pdf_path.exists():
        st.download_button(
            "Descargar Material Passport (PDF)",
            pdf_path.read_bytes(),
            file_name=pdf_path.name,
            mime="application/pdf",
            use_container_width=True,
        )
