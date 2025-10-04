from pathlib import Path

from app.bootstrap import ensure_streamlit_entrypoint

ensure_streamlit_entrypoint(__file__)

from typing import Any, Mapping

import pandas as pd
import streamlit as st

from app.modules.generator import GeneratorService
from app.modules.manifest_loader import (
    build_manifest_template,
    load_manifest_from_upload,
    manifest_template_csv_bytes,
    run_policy_analysis,
)
from app.modules.mars_control_center import (
    MarsControlCenterService,
    summarize_artifacts,
)
from app.modules.ui_blocks import (
    configure_page,
    initialise_frontend,
    micro_divider,
    render_brand_header,
)


configure_page(page_title="Rex-AI • Mars Control Center", page_icon="🛰️")
initialise_frontend()
render_brand_header(tagline="Mars Control Center · Interplanetary Recycling")


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
    with st.container():
        st.markdown(
            (
                "<div style='border-radius:12px;padding:1rem;background:{color};color:white;'>"
                "<div style='font-size:0.85rem;text-transform:uppercase;opacity:0.85;'>{label}</div>"
                "<div style='font-size:1.8rem;font-weight:700;'>{value:.2f}</div>"
                "<div style='font-size:0.9rem;'>Nivel {status}</div>"
                "</div>"
            ).format(color=color, label=label, value=score, status=status),
            unsafe_allow_html=True,
        )
        if help_text:
            st.caption(help_text)


st.title("Centro de control marciano")
st.markdown(
    """
    Orquesta la operación completa de la misión: vuelos de logística, inventario
    orbital, decisiones asistidas por IA y planificación de procesos. Cada
    sección consume los servicios de telemetría recién desplegados para mantener
    la vista táctica de los equipos de operaciones.
    """
)

generator_service = GeneratorService()
telemetry_service = MarsControlCenterService()

analysis_state: dict[str, Any] | None = st.session_state.get("policy_analysis")
manifest_df: pd.DataFrame | None = st.session_state.get("uploaded_manifest_df")

tabs = st.tabs(
    [
        "🛰️ Flight Radar / Mapa",
        "📦 Inventario vivo",
        "🤖 Decisiones IA",
        "🗺️ Planner",
        "🎛️ Modo Demo",
    ]
)


with tabs[0]:
    st.subheader("Flight Radar · logística interplanetaria")
    passport: Mapping[str, Any] | None = None
    if analysis_state:
        passport = analysis_state.get("material_passport")

    radar_df = telemetry_service.flight_radar_snapshot(passport)
    if radar_df.empty:
        st.info("Aún no hay vuelos registrados. Cargá un manifiesto para sincronizar la carga.")
    else:
        map_df = radar_df.rename(columns={"latitude": "lat", "longitude": "lon"})
        map_df = map_df[["lat", "lon", "vehicle", "phase", "payload_kg"]]
        st.caption("Posiciones aproximadas (lat/lon) en coordenadas marcianas normalizadas.")
        st.map(map_df, size="payload_kg")
        micro_divider()
        st.dataframe(
            radar_df,
            use_container_width=True,
            column_config={
                "vehicle": st.column_config.TextColumn("Vehículo"),
                "phase": st.column_config.TextColumn("Fase actual"),
                "altitude_km": st.column_config.NumberColumn("Altitud (km)", format="%.1f km"),
                "eta_minutes": st.column_config.NumberColumn("ETA (min)", format="%d min"),
                "payload_kg": st.column_config.NumberColumn("Carga útil (kg)", format="%.1f kg"),
            },
            hide_index=True,
        )
        st.caption(
            "Sincronizá con logística para alinear ventanas de descenso y despliegue en superficie."
        )


with tabs[1]:
    st.subheader("Inventario vivo")
    try:
        inventory_df, metrics = telemetry_service.inventory_snapshot()
    except Exception as exc:
        st.error(f"No se pudo cargar el inventario en vivo: {exc}")
    else:
        metric_cols = st.columns(4)
        metric_cols[0].metric("Masa total (kg)", f"{metrics.get('mass_kg', 0.0):.1f}")
        metric_cols[1].metric("Volumen (m³)", f"{metrics.get('volume_m3', 0.0):.3f}")
        metric_cols[2].metric("Agua recuperable (L)", f"{metrics.get('water_l', 0.0):.1f}")
        metric_cols[3].metric("Energía estimada (kWh)", f"{metrics.get('energy_kwh', 0.0):.1f}")

        problematic = int(metrics.get("problematic_count", 0))
        st.caption(
            "Residuos problemáticos detectados: "
            f"{problematic}. Coordiná protocolos especiales según severidad."
        )

        micro_divider()
        st.dataframe(
            inventory_df,
            use_container_width=True,
            hide_index=True,
        )


with tabs[2]:
    st.subheader("Decisiones IA y reporting")
    st.caption(
        "Cargá el manifiesto actualizado para que Rex-AI evalúe compatibilidad, penalizaciones y artefactos de reporting."
    )

    template_bytes = manifest_template_csv_bytes()
    template_df = build_manifest_template()
    col_template, col_preview = st.columns(2)
    with col_template:
        st.download_button(
            "Descargar plantilla CSV",
            template_bytes,
            file_name="manifiesto_plantilla.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with col_preview:
        st.dataframe(
            template_df,
            column_config={col: st.column_config.Column(col.replace("_", " ").title()) for col in template_df.columns},
            hide_index=True,
            use_container_width=True,
        )

    uploaded_file = st.file_uploader(
        "Manifiesto (CSV)",
        type=["csv"],
        accept_multiple_files=False,
        key="manifest_uploader",
    )
    include_pdf = st.checkbox("Generar Material Passport en PDF", value=False)

    if st.button("Evaluar manifiesto", use_container_width=True):
        if uploaded_file is None:
            st.warning("Subí un archivo CSV para iniciar el análisis.")
        else:
            manifest_df = load_manifest_from_upload(uploaded_file)
            with st.spinner("Analizando manifiesto con heurísticas de política..."):
                analysis_state = run_policy_analysis(
                    generator_service, manifest_df, include_pdf=include_pdf
                )
            st.session_state["policy_analysis"] = analysis_state
            st.session_state["uploaded_manifest_df"] = manifest_df

    if not analysis_state:
        st.info(
            "Esperando un manifiesto. Cuando se procese uno verás métricas en vivo, recomendaciones y descargas."
        )
    else:
        summary = telemetry_service.summarize_decisions(analysis_state)

        cols = st.columns(3)
        with cols[0]:
            _render_metric("Puntaje promedio", summary["score"])
        with cols[1]:
            st.metric("Masa total (kg)", f"{summary['total_mass']:.1f}")
        with cols[2]:
            st.metric("Total ítems", f"{summary['item_count']}")

        micro_divider()
        manifest_table = summary["manifest"]
        if isinstance(manifest_table, pd.DataFrame) and not manifest_table.empty:
            st.markdown("**Detalle de ítems evaluados**")
            st.dataframe(
                manifest_table,
                use_container_width=True,
                hide_index=True,
            )

        recommendations = summary["recommendations"]
        micro_divider()
        st.markdown("**Recomendaciones de política**")
        if isinstance(recommendations, pd.DataFrame) and not recommendations.empty:
            st.dataframe(recommendations, use_container_width=True)
        else:
            st.success(
                "No se identificaron acciones prioritarias: todos los ítems superan el umbral de utilidad."
            )

        compatibility = summary["compatibility"]
        micro_divider()
        st.markdown("**Trazabilidad de compatibilidad**")
        if isinstance(compatibility, pd.DataFrame) and not compatibility.empty:
            st.dataframe(compatibility, use_container_width=True)
        else:
            st.caption("Sin datos de compatibilidad asociados al manifiesto.")

        passport = analysis_state.get("material_passport", {})
        micro_divider()
        st.markdown("**Material Passport**")
        st.json(passport)

        micro_divider()
        st.markdown("### Descargas")
        artifacts = summarize_artifacts(analysis_state)
        col_a, col_b, col_c = st.columns(3)
        policy_path = artifacts.get("policy_recommendations_csv")
        if isinstance(policy_path, Path) and policy_path.exists():
            with col_a:
                st.download_button(
                    "Recomendaciones (CSV)",
                    policy_path.read_bytes(),
                    file_name=policy_path.name,
                    mime="text/csv",
                    use_container_width=True,
                )
        compat_path = artifacts.get("compatibility_matrix_parquet")
        if isinstance(compat_path, Path) and compat_path.exists():
            with col_b:
                st.download_button(
                    "Compatibilidad (Parquet)",
                    compat_path.read_bytes(),
                    file_name=compat_path.name,
                    mime="application/octet-stream",
                    use_container_width=True,
                )
        passport_path = artifacts.get("material_passport_json")
        if isinstance(passport_path, Path) and passport_path.exists():
            with col_c:
                st.download_button(
                    "Material Passport (JSON)",
                    passport_path.read_bytes(),
                    file_name=passport_path.name,
                    mime="application/json",
                    use_container_width=True,
                )

        pdf_path = artifacts.get("material_passport_pdf")
        if isinstance(pdf_path, Path) and pdf_path.exists():
            st.download_button(
                "Material Passport (PDF)",
                pdf_path.read_bytes(),
                file_name=pdf_path.name,
                mime="application/pdf",
                use_container_width=True,
            )


with tabs[3]:
    st.subheader("Planner operacional")
    if not analysis_state:
        st.info("Cargá un manifiesto para que el planner recomiende procesos prioritarios.")
    else:
        summary = telemetry_service.summarize_decisions(analysis_state)
        planner_df = telemetry_service.build_planner_schedule(summary.get("manifest"))
        if planner_df.empty:
            st.caption("Sin procesos asignados todavía. Ajustá el manifiesto para generar recomendaciones.")
        else:
            st.caption(
                "Top procesos sugeridos por residuo crítico (ordenados por masa declarada)."
            )
            st.dataframe(
                planner_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "item": st.column_config.TextColumn("Residuo"),
                    "category": st.column_config.TextColumn("Categoría"),
                    "process_id": st.column_config.TextColumn("Proceso"),
                    "match_score": st.column_config.NumberColumn("Score", format="%.2f"),
                    "match_reason": st.column_config.TextColumn("Racional"),
                },
            )


with tabs[4]:
    st.subheader("Modo Demo")
    st.caption(
        "Activá este modo cuando presentes la misión: mantiene un guion sintético sincronizado con la telemetría."
    )

    if st.button("Regenerar guion", use_container_width=True):
        st.session_state["demo_script"] = telemetry_service.demo_script()

    script = st.session_state.get("demo_script")
    if not script:
        script = telemetry_service.demo_script()
        st.session_state["demo_script"] = script

    for step in script:
        with st.container():
            st.markdown(f"**{step['timestamp']}** · {step['action']}")
            st.caption(step["notes"])

