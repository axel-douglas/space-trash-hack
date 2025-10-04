from pathlib import Path

from app.bootstrap import ensure_streamlit_entrypoint

ensure_streamlit_entrypoint(__file__)

from typing import Any, Mapping

import pandas as pd
import plotly.graph_objects as go
import pydeck as pdk
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
from app.modules.mission_overview import _format_metric, render_mission_objective
from app.modules.ui_blocks import (
    badge_group,
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

    manifest_signature = "baseline"
    if passport:
        manifest_signature = (
            f"{passport.get('generated_at', 'baseline')}"
            f":{passport.get('total_items', 0)}:{passport.get('total_mass_kg', 0)}"
        )
    elif isinstance(manifest_df, pd.DataFrame) and not manifest_df.empty:
        manifest_signature = f"uploaded:{manifest_df.shape[0]}:{','.join(manifest_df.columns)}"

    flights_df: pd.DataFrame | None = st.session_state.get("flight_operations_table")
    previous_signature = st.session_state.get("flight_operations_signature")
    if flights_df is None or previous_signature != manifest_signature:
        flights_df = telemetry_service.flight_operations_overview(
            passport,
            manifest_df=manifest_df,
            analysis_state=analysis_state,
        )
        st.session_state["flight_operations_table"] = flights_df
        st.session_state["flight_operations_signature"] = manifest_signature
        st.session_state["flight_operations_last_decisions"] = {
            row["flight_id"]: row["ai_decision"]
            for row in flights_df.to_dict(orient="records")
        }
        st.session_state.setdefault("flight_operations_recent_events", [])
        st.session_state.setdefault("flight_operations_recent_changes", [])

    if flights_df is None or flights_df.empty:
        st.info("Aún no hay vuelos registrados. Cargá un manifiesto para sincronizar la carga.")
    else:
        control_cols = st.columns([2, 1])
        with control_cols[0]:
            auto_tick = st.toggle(
                "Tick automático cada 20 s",
                value=st.session_state.get("mars_auto_tick_toggle", False),
                key="mars_auto_tick_toggle",
            )
        with control_cols[1]:
            manual_tick = st.button("Avanzar simulación", use_container_width=True)

        tick_triggered = bool(manual_tick)
        if auto_tick:
            tick_count = st.autorefresh(
                interval=20000,
                limit=None,
                key="mars_auto_tick_counter",
            )
            previous_count = st.session_state.get("mars_auto_tick_prev", 0)
            if tick_count > previous_count:
                st.session_state["mars_auto_tick_prev"] = tick_count
                if tick_count > 0:
                    tick_triggered = True

        previous_decisions: Mapping[str, str] = st.session_state.get(
            "flight_operations_last_decisions", {}
        )
        if tick_triggered:
            flights_df, events, changed_flights = telemetry_service.advance_timeline(
                flights_df,
                manifest_df=manifest_df,
                analysis_state=analysis_state,
                previous_decisions=previous_decisions,
            )
            st.session_state["flight_operations_table"] = flights_df
            st.session_state["flight_operations_last_decisions"] = {
                row["flight_id"]: row["ai_decision"]
                for row in flights_df.to_dict(orient="records")
            }
            st.session_state["flight_operations_recent_events"] = events
            st.session_state["flight_operations_recent_changes"] = list(changed_flights)
        else:
            events = st.session_state.get("flight_operations_recent_events", [])
            changed_flights = set(
                st.session_state.get("flight_operations_recent_changes", [])
            )

        map_payload = telemetry_service.build_map_payload(flights_df)
        capsule_data = map_payload["capsules"]
        zone_data = map_payload["zones"]
        geometry = map_payload["geometry"]

        layers: list[pdk.Layer] = []
        if geometry and isinstance(geometry, Mapping) and geometry.get("features"):
            layers.append(
                pdk.Layer(
                    "GeoJsonLayer",
                    geometry,
                    id="jezero-boundary",
                    stroked=True,
                    filled=False,
                    get_line_color=[180, 198, 231],
                    line_width_min_pixels=2,
                )
            )
        if isinstance(zone_data, pd.DataFrame) and not zone_data.empty:
            layers.append(
                pdk.Layer(
                    "ScatterplotLayer",
                    zone_data,
                    id="zones",
                    get_position="[longitude, latitude]",
                    get_radius="radius_m",
                    get_fill_color="[color_r, color_g, color_b]",
                    get_line_color="[color_r, color_g, color_b]",
                    pickable=True,
                    stroked=True,
                    opacity=0.25,
                    radius_units="meters",
                )
            )
        if isinstance(capsule_data, pd.DataFrame) and not capsule_data.empty:
            layers.append(
                pdk.Layer(
                    "ScatterplotLayer",
                    capsule_data,
                    id="capsules",
                    get_position="[longitude, latitude]",
                    get_radius="marker_radius_m",
                    get_fill_color="[category_color_r, category_color_g, category_color_b]",
                    get_line_color="[status_color_r, status_color_g, status_color_b]",
                    radius_units="meters",
                    pickable=True,
                    stroked=True,
                    auto_highlight=True,
                )
            )

        tooltip = {
            "html": (
                "<div style='font-size:14px;font-weight:600;'>{vehicle}</div>"
                "<div>{status}</div>"
                "<div>ETA: {eta_minutes} min</div>"
                "<div>Materiales: {materials_tooltip}</div>"
                "<div>Espectro: {material_spectrum}</div>"
                "<div>Densidad: {density} g/cm³ · Compatibilidad: {compatibility}</div>"
                "<div>{tooltip}</div>"
            ),
            "style": {"backgroundColor": "#0f172a", "color": "white"},
        }

        view_state = pdk.ViewState(
            latitude=18.43,
            longitude=77.58,
            zoom=9.1,
            pitch=45,
            bearing=25,
        )

        st.pydeck_chart(
            pdk.Deck(layers=layers, initial_view_state=view_state, tooltip=tooltip),
            use_container_width=True,
        )
        st.caption("Mapa operacional de Jezero: cápsulas, zonas clave y perímetro de seguridad.")

        micro_divider()
        display_df = flights_df[
            [
                "flight_id",
                "vehicle",
                "status_label",
                "eta_minutes",
                "key_materials_display",
                "ai_decision",
                "decision_indicator",
            ]
        ].rename(
            columns={
                "flight_id": "Vuelo",
                "vehicle": "Vehículo",
                "status_label": "Estado",
                "eta_minutes": "ETA (min)",
                "key_materials_display": "Materiales clave",
                "ai_decision": "Decisión IA",
                "decision_indicator": "Decisión ∆",
            }
        )

        def _status_style(series: pd.Series) -> list[str]:
            colors = flights_df.loc[series.index, "status_color"]
            return [
                f"background-color: {color}; color: white; font-weight:600" for color in colors
            ]

        def _decision_style(series: pd.Series) -> list[str]:
            flags = flights_df.loc[series.index, "decision_indicator"].astype(str)
            return [
                "background-color: #facc15; color: #1f2937; font-weight:700" if flag else ""
                for flag in flags
            ]

        styled_df = (
            display_df.style.apply(_status_style, subset=["Estado"])
            .apply(_decision_style, subset=["Decisión ∆"])
            .format({"ETA (min)": "{:,.0f}"})
        )

        st.dataframe(
            styled_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "ETA (min)": st.column_config.NumberColumn("ETA (min)", format="%d min"),
                "Materiales clave": st.column_config.TextColumn(
                    "Materiales clave",
                    help="Top materiales declarados por masa para la cápsula.",
                ),
                "Decisión IA": st.column_config.TextColumn(
                    "Decisión IA",
                    help="Última directriz activa para la misión.",
                ),
                "Decisión ∆": st.column_config.TextColumn(
                    "∆",
                    help="Indicador de cambios recientes en decisiones automáticas.",
                ),
            },
        )

        editor_df = flights_df[
            [
                "flight_id",
                "vehicle",
                "status_badge",
                "eta_minutes",
                "ai_decision",
                "key_materials_display",
                "material_spectrum",
                "material_density",
                "compatibility_index",
            ]
        ].rename(
            columns={
                "flight_id": "Vuelo",
                "vehicle": "Vehículo",
                "status_badge": "Estado",
                "eta_minutes": "ETA (min)",
                "ai_decision": "Decisión IA",
                "key_materials_display": "Materiales clave",
                "material_spectrum": "Espectro",
                "material_density": "Densidad (g/cm³)",
                "compatibility_index": "Compatibilidad",
            }
        )

        editor_result = st.data_editor(
            editor_df,
            hide_index=True,
            use_container_width=True,
            column_config={
                "Vuelo": st.column_config.TextColumn("Vuelo", disabled=True),
                "Vehículo": st.column_config.TextColumn("Vehículo", disabled=True),
                "Estado": st.column_config.TextColumn("Estado", disabled=True),
                "ETA (min)": st.column_config.NumberColumn("ETA (min)", format="%d min", disabled=True),
                "Materiales clave": st.column_config.TextColumn(
                    "Materiales clave",
                    disabled=True,
                ),
                "Espectro": st.column_config.TextColumn("Espectro", disabled=True),
                "Densidad (g/cm³)": st.column_config.NumberColumn(
                    "Densidad (g/cm³)", format="%.2f", disabled=True
                ),
                "Compatibilidad": st.column_config.NumberColumn(
                    "Compatibilidad", format="%.2f", disabled=True
                ),
                "Decisión IA": st.column_config.TextColumn(
                    "Decisión IA",
                    help="Podés forzar una decisión manual que se mantendrá hasta el próximo tick.",
                ),
            },
            key="flight_ops_editor",
        )

        if not editor_result.equals(editor_df):
            for idx in editor_result.index:
                new_value = editor_result.loc[idx, "Decisión IA"]
                flights_df.at[idx, "ai_decision"] = new_value
            flights_df["decision_changed"] = False
            flights_df["decision_indicator"] = ""
            st.session_state["flight_operations_table"] = flights_df
            st.session_state["flight_operations_last_decisions"] = {
                row["flight_id"]: row["ai_decision"]
                for row in flights_df.to_dict(orient="records")
            }
            st.session_state["flight_operations_recent_changes"] = []
            st.success("Decisiones actualizadas manualmente.")

        micro_divider()

        timeline_df = telemetry_service.timeline_history()
        if isinstance(timeline_df, pd.DataFrame) and not timeline_df.empty:
            st.markdown("#### Timeline de eventos logísticos")
            st.dataframe(
                timeline_df[[
                    "tick",
                    "category",
                    "title",
                    "details",
                    "capsule_id",
                    "mass_delta",
                ]],
                hide_index=True,
                use_container_width=True,
                column_config={
                    "tick": st.column_config.NumberColumn("Tick"),
                    "category": st.column_config.TextColumn("Categoría"),
                    "title": st.column_config.TextColumn("Evento"),
                    "details": st.column_config.TextColumn("Detalles"),
                    "capsule_id": st.column_config.TextColumn("Cápsula"),
                    "mass_delta": st.column_config.NumberColumn("Δ Masa (kg)", format="%.1f"),
                },
            )

        if events:
            st.markdown("**Eventos recientes**")
            for event in events:
                tick = event.get("tick")
                title = event.get("title")
                category = event.get("category")
                st.markdown(f"• Tick {tick}: {title} ({category})")
                details = event.get("details")
                if details:
                    st.caption(details)


with tabs[1]:
    st.subheader("Inventario vivo")
    try:
        inventory_df, metrics, category_payload = telemetry_service.inventory_snapshot()
    except Exception as exc:
        st.error(f"No se pudo cargar el inventario en vivo: {exc}")
    else:
        render_mission_objective(metrics)

        problematic = int(metrics.get("problematic_count", 0))
        st.caption(
            "Residuos problemáticos detectados: "
            f"{problematic}. Coordiná protocolos especiales según severidad."
        )

        category_stats = category_payload.get("categories")
        flows_df = category_payload.get("flows")
        palette = category_payload.get("group_palette", {})
        group_labels = category_payload.get("material_groups", {})
        destination_info = category_payload.get("destinations", {})

        has_breakdown = isinstance(category_stats, pd.DataFrame) and not category_stats.empty

        if has_breakdown:
            micro_divider()
            st.markdown("**Flujos circulares por categoría**")

            badge_emojis = {
                "polimeros": "🟦",
                "metales": "🟧",
                "textiles": "🟩",
                "espumas": "🟪",
                "mixtos": "⬜",
            }
            legend_labels: list[str] = []
            for key in ("polimeros", "metales", "textiles", "espumas", "mixtos"):
                label = group_labels.get(key, key.title())
                color = palette.get(key)
                emoji = badge_emojis.get(key, "•")
                legend_labels.append(
                    f"{emoji} {label}{f' · {color}' if color else ''}"
                )
            badge_group(legend_labels)
            st.caption("Colores sincronizados con el bubble chart según familia de material.")

            sankey_col, bubble_col = st.columns((1.1, 1))

            if isinstance(flows_df, pd.DataFrame) and not flows_df.empty:
                categories = category_stats["category"].astype(str).tolist()
                destination_keys = list(destination_info.keys())
                node_labels = categories + [
                    destination_info[key]["display"] for key in destination_keys
                ]
                node_colors = [
                    palette.get(group, "#94a3b8")
                    for group in category_stats["material_group"].astype(str)
                ] + [destination_info[key]["color"] for key in destination_keys]

                link_sources: list[int] = []
                link_targets: list[int] = []
                link_values: list[float] = []
                link_custom: list[list[Any]] = []

                for _, flow_row in flows_df.iterrows():
                    destination_key = str(flow_row.get("destination_key"))
                    if destination_key not in destination_keys:
                        continue
                    category = str(flow_row.get("category"))
                    try:
                        source_index = categories.index(category)
                        target_index = len(categories) + destination_keys.index(destination_key)
                    except ValueError:
                        continue
                    mass_value = float(flow_row.get("mass_kg", 0.0))
                    if mass_value <= 0:
                        continue
                    link_sources.append(source_index)
                    link_targets.append(target_index)
                    link_values.append(mass_value)
                    display = destination_info[destination_key]["label"]
                    link_custom.append([category, display, mass_value])

                if link_values:
                    sankey_fig = go.Figure(
                        data=[
                            go.Sankey(
                                arrangement="snap",
                                node=dict(
                                    pad=16,
                                    thickness=18,
                                    label=node_labels,
                                    color=node_colors,
                                ),
                                link=dict(
                                    source=link_sources,
                                    target=link_targets,
                                    value=link_values,
                                    customdata=link_custom,
                                    hovertemplate=(
                                        "<b>%{customdata[0]}</b> → <b>%{customdata[1]}</b>"
                                        "<br>Masa: %{customdata[2]:,.1f} kg<extra></extra>"
                                    ),
                                ),
                            )
                        ]
                    )
                    sankey_fig.update_layout(
                        margin=dict(t=30, b=10, l=10, r=10),
                        font=dict(color="#e2e8f0"),
                        paper_bgcolor="rgba(12,18,28,1)",
                    )
                    sankey_col.plotly_chart(sankey_fig, use_container_width=True)
                else:
                    sankey_col.info("No hay datos suficientes para el flujo Sankey.")
            else:
                sankey_col.info("No hay datos suficientes para el flujo Sankey.")

            stock_sizes = category_stats["stock_mass_kg"].astype(float).fillna(0.0)
            max_stock = float(stock_sizes.max()) if not stock_sizes.empty else 0.0
            if max_stock <= 0:
                marker_sizes = [15.0 for _ in stock_sizes]
                size_reference = 1.0
            else:
                min_size = max_stock * 0.1
                marker_sizes = stock_sizes.clip(lower=min_size).tolist()
                size_reference = 2.0 * max(marker_sizes) / (32.0**2)

            bubble_colors = [
                palette.get(group, "#94a3b8")
                for group in category_stats["material_group"].astype(str)
            ]
            bubble_custom = [
                [
                    float(row.stock_mass_kg),
                    float(row.water_l),
                    float(row.energy_kwh),
                    float(row.cross_contamination_risk),
                ]
                for row in category_stats.itertuples()
            ]

            bubble_fig = go.Figure(
                data=[
                    go.Scatter3d(
                        x=category_stats["water_l"].astype(float),
                        y=category_stats["energy_kwh"].astype(float),
                        z=category_stats["purity_index"].astype(float),
                        text=category_stats["category"].astype(str),
                        mode="markers",
                        customdata=bubble_custom,
                        marker=dict(
                            size=marker_sizes,
                            sizemode="area",
                            sizeref=size_reference,
                            opacity=0.85,
                            color=bubble_colors,
                            line=dict(color="rgba(255,255,255,0.55)", width=1),
                        ),
                        hovertemplate=(
                            "<b>%{text}</b><br>Stock estratégico: %{customdata[0]:,.1f} kg"
                            "<br>Agua recuperable: %{x:,.1f} L"
                            "<br>Energía estimada: %{y:,.1f} kWh"
                            "<br>Pureza: %{z:.1f}% · Contaminación: %{customdata[3]:.1f}%"
                            "<extra></extra>"
                        ),
                    )
                ]
            )
            bubble_fig.update_layout(
                margin=dict(t=30, b=10, l=0, r=0),
                paper_bgcolor="rgba(12,18,28,1)",
                plot_bgcolor="rgba(12,18,28,1)",
                font=dict(color="#e2e8f0"),
                scene=dict(
                    xaxis=dict(
                        title="Agua recuperable (L)",
                        backgroundcolor="rgba(15,23,42,0.18)",
                        gridcolor="rgba(148,163,184,0.3)",
                    ),
                    yaxis=dict(
                        title="Energía estimada (kWh)",
                        backgroundcolor="rgba(15,23,42,0.18)",
                        gridcolor="rgba(148,163,184,0.3)",
                    ),
                    zaxis=dict(
                        title="Índice de pureza (%)",
                        range=[0, 100],
                        backgroundcolor="rgba(15,23,42,0.18)",
                        gridcolor="rgba(148,163,184,0.3)",
                    ),
                ),
            )
            bubble_col.plotly_chart(bubble_fig, use_container_width=True)

            top_categories = (
                category_stats.sort_values("stock_mass_kg", ascending=False).head(3)
            )
            if not top_categories.empty:
                st.markdown("**Indicadores clave por stock estratégico**")
                indicator_cols = st.columns(len(top_categories))
                for column, (_, row) in zip(indicator_cols, top_categories.iterrows()):
                    column.metric(
                        str(row["category"]),
                        _format_metric(float(row["stock_mass_kg"]), "kg"),
                        delta=(
                            f"Pureza {row['purity_index']:.0f}% · Contam. "
                            f"{row['cross_contamination_risk']:.0f}%"
                        ),
                    )
                    column.caption(
                        " · ".join(
                            [
                                _format_metric(float(row["energy_kwh"]), "kWh"),
                                _format_metric(float(row["water_l"]), "L"),
                            ]
                        )
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

