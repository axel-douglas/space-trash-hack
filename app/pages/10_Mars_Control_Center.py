from __future__ import annotations

from datetime import datetime
import hashlib
import io
import json
from pathlib import Path
import html

from app.bootstrap import ensure_streamlit_entrypoint

ensure_streamlit_entrypoint(__file__)

from typing import Any, Mapping

import pandas as pd
import plotly.graph_objects as go
import pydeck as pdk
import streamlit as st

from app.modules import data_sources as ds
from app.modules import mars_control

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


_MANUAL_DECISIONS_KEY = "mars_decision_actions"
_BATCH_RESULTS_KEY = "mars_manifest_batch_results"
_BATCH_SIGNATURE_KEY = "mars_manifest_batch_signature"
_SCORE_THRESHOLDS = {"spectral": 0.65, "mechanical": 0.6}
_ACTION_PRESETS = {
    "accept": {"label": "Aceptar plan Rex-AI", "badge": "🟢 Aceptado"},
    "reject": {"label": "Rechazar acción propuesta", "badge": "🔴 Rechazado"},
    "reprioritize": {"label": "Repriorizar envío crítico", "badge": "🟠 Repriorizar"},
}

_VERIFIED_MATERIALS: dict[str, dict[str, Any]] = {
    "HDPE": {
        "label": "HDPE · Polietileno de alta densidad",
        "visual_score": 0.92,
        "note": "Mediciones reales de laboratorio proxy. Confianza elevada.",
    },
    "PVDF": {
        "label": "PVDF · Fluoruro de polivinilideno",
        "visual_score": 0.88,
        "note": "Material con espectro FTIR verificado. Score visual reforzado.",
    },
}
_VERIFIED_DEFAULT_NOTE = (
    "Material con mediciones reales cargadas en la base proxy: el sistema confía más en su score."
)


@st.cache_resource(show_spinner=False)
def _load_reference_bundle() -> ds.MaterialReferenceBundle:
    return ds.load_material_reference_bundle()


def _manifest_signature(manifest_df: pd.DataFrame | None) -> str:
    if manifest_df is None or manifest_df.empty:
        return "empty"
    try:
        payload = manifest_df.to_csv(index=False).encode("utf-8")
    except Exception:
        payload = json.dumps(manifest_df.to_dict(orient="records"), sort_keys=True).encode("utf-8")
    return hashlib.sha1(payload).hexdigest()


def _store_flight_snapshot(flights_df: pd.DataFrame) -> None:
    st.session_state["flight_operations_table"] = flights_df
    st.session_state["flight_operations_last_decisions"] = {
        row["flight_id"]: row["ai_decision"]
        for row in flights_df.to_dict(orient="records")
        if row.get("flight_id")
    }


def _apply_manual_overrides(flights_df: pd.DataFrame | None) -> pd.DataFrame | None:
    if flights_df is None or flights_df.empty:
        return flights_df
    overrides = st.session_state.get(_MANUAL_DECISIONS_KEY, {})
    if not overrides:
        return flights_df
    updated = flights_df.copy()
    for manifest_ref, payload in overrides.items():
        if not isinstance(payload, Mapping):
            continue
        mask = updated["manifest_ref"].astype(str) == str(manifest_ref)
        if not mask.any():
            continue
        label = payload.get("label")
        badge = payload.get("badge", "⚙️ Manual")
        timestamp = payload.get("timestamp")
        if label:
            updated.loc[mask, "ai_decision"] = label
        if timestamp:
            updated.loc[mask, "ai_decision_timestamp"] = timestamp
        updated.loc[mask, "decision_indicator"] = badge
        updated.loc[mask, "decision_changed"] = True
    return updated


def _normalise_material_token(value: Any) -> str:
    return str(value or "").strip().upper()


def _match_verified_material(row: Mapping[str, Any]) -> tuple[str | None, dict[str, Any] | None]:
    tokens = " ".join(
        filter(
            None,
            (
                _normalise_material_token(row.get("material_key")),
                _normalise_material_token(row.get("item")),
                _normalise_material_token(row.get("material")),
            ),
        )
    )
    for token, meta in _VERIFIED_MATERIALS.items():
        if token and token in tokens:
            return token, meta
    return None, None


def _annotate_verified_materials(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    updated = df.copy()
    visual_scores: list[float] = []
    labels: list[str] = []
    notes: list[str] = []
    for _, row in updated.iterrows():
        token, meta = _match_verified_material(row)
        base_score = float(row.get("material_utility_score") or 0.0)
        if meta:
            visual_score = max(base_score, float(meta.get("visual_score", base_score)))
            visual_scores.append(visual_score)
            labels.append(str(meta.get("label") or token))
            notes.append(str(meta.get("note") or _VERIFIED_DEFAULT_NOTE))
        else:
            visual_scores.append(base_score)
            labels.append("")
            notes.append("")
    updated["visual_material_score"] = visual_scores
    updated["verified_material_label"] = labels
    updated["verified_confidence_note"] = notes
    return updated


def _register_manual_action(
    manifest_ref: str,
    action_key: str,
    *,
    label: str | None = None,
    badge: str | None = None,
) -> None:
    presets = _ACTION_PRESETS.get(action_key, {})
    payload = {
        "action": action_key,
        "label": label or presets.get("label") or action_key.title(),
        "badge": badge or presets.get("badge", "⚙️ Manual"),
        "timestamp": datetime.utcnow().isoformat(),
    }
    overrides = st.session_state.setdefault(_MANUAL_DECISIONS_KEY, {})
    overrides[str(manifest_ref)] = payload
    st.session_state[_MANUAL_DECISIONS_KEY] = overrides

    flights_df: pd.DataFrame | None = st.session_state.get("flight_operations_table")
    if isinstance(flights_df, pd.DataFrame):
        updated = _apply_manual_overrides(flights_df)
        _store_flight_snapshot(updated)


def _demo_event_severity(severity: str | None) -> str:
    normalized = str(severity or "info").lower()
    if normalized in {"critical", "alert", "danger", "severe"}:
        return "critical"
    if normalized in {"warning", "caution", "warn"}:
        return "warning"
    return "info"


def _format_demo_timestamp(event: mars_control.DemoEvent) -> str:
    if event.emitted_at is None:
        return "En cola"
    try:
        return f"{event.emitted_at.strftime('%H:%M:%S')} UTC"
    except Exception:
        return str(event.emitted_at)


def _render_demo_event_card(event: mars_control.DemoEvent) -> str:
    severity = _demo_event_severity(event.severity)
    icon = html.escape(event.icon or "🛰️")
    category = html.escape(event.category.title())
    timestamp = html.escape(_format_demo_timestamp(event))
    metadata_html = ""
    if event.metadata:
        tags = []
        for key, value in event.metadata.items():
            key_label = html.escape(str(key).replace("_", " ").title())
            value_label = html.escape(str(value))
            tags.append(
                f"<span class='demo-event-card__tag'><strong>{key_label}</strong>: {value_label}</span>"
            )
        metadata_html = (
            "<div class='demo-event-card__meta-tags'>" + " · ".join(tags) + "</div>"
        )
    title = html.escape(event.title)
    message = html.escape(event.message)
    return (
        "<div class='demo-event-card demo-event-card--"
        + severity
        + "'>"
        + f"<div class='demo-event-card__icon'>{icon}</div>"
        + "<div class='demo-event-card__content'>"
        + "<div class='demo-event-card__header'>"
        + f"<span class='demo-event-card__category'>{category}</span>"
        + f"<span class='demo-event-card__timestamp'>{timestamp}</span>"
        + "</div>"
        + f"<div class='demo-event-card__title'>{title}</div>"
        + f"<div class='demo-event-card__message'>{message}</div>"
        + metadata_html
        + "</div></div>"
    )


def _render_demo_ticker(events: list[mars_control.DemoEvent]) -> str:
    if not events:
        return ""
    items: list[str] = []
    for event in events:
        severity = _demo_event_severity(event.severity)
        icon = html.escape(event.icon or "🛰️")
        title = html.escape(event.title)
        timestamp = html.escape(
            event.emitted_at.strftime("%H:%M:%S") if event.emitted_at else "—"
        )
        items.append(
            "<div class='demo-event-ticker__item demo-event-ticker__item--"
            + severity
            + "'>"
            + f"<span class='demo-event-ticker__icon'>{icon}</span>"
            + f"<span class='demo-event-ticker__text'>{title}</span>"
            + f"<span class='demo-event-ticker__time'>{timestamp}</span>"
            + "</div>"
        )
    return "<div class='demo-event-ticker'>" + "".join(items) + "</div>"


def _ensure_manifest_batch(
    service: GeneratorService, manifest_df: pd.DataFrame | None
) -> list[dict[str, Any]]:
    if manifest_df is None or manifest_df.empty:
        st.session_state.pop(_BATCH_RESULTS_KEY, None)
        st.session_state.pop(_BATCH_SIGNATURE_KEY, None)
        return []

    signature = _manifest_signature(manifest_df)
    if st.session_state.get(_BATCH_SIGNATURE_KEY) != signature:
        batch = mars_control.score_manifest_batch(service, [manifest_df])
        st.session_state[_BATCH_RESULTS_KEY] = batch
        st.session_state[_BATCH_SIGNATURE_KEY] = signature
    return st.session_state.get(_BATCH_RESULTS_KEY, [])


def _resolve_spectral_curve(
    material_key: str | None,
    item_name: str | None,
) -> tuple[str | None, pd.DataFrame | None, Mapping[str, Any]]:
    bundle = _load_reference_bundle()
    alias_map = bundle.alias_map
    spectral_curves = bundle.spectral_curves
    metadata = bundle.metadata

    candidates = [material_key, item_name]
    for candidate in candidates:
        if not candidate:
            continue
        text = str(candidate)
        if text in spectral_curves:
            return text, spectral_curves[text], metadata.get(text, {})
        slug = ds.slugify(ds.normalize_item(text))
        canonical = alias_map.get(slug)
        if canonical and canonical in spectral_curves:
            return canonical, spectral_curves[canonical], metadata.get(canonical, {})
    return None, None, {}


def _synthetic_spectral_curve(
    spectral_score: float, mechanical_score: float
) -> pd.DataFrame:
    import numpy as np

    wavenumbers = np.linspace(500, 4000, 40)
    base = 0.45 + (1.0 - spectral_score) * 0.35
    modulation = 0.1 + (1.0 - mechanical_score) * 0.25
    transmittance = 100.0 * (base + modulation * np.sin(np.linspace(0, 3.5, 40)))
    frame = pd.DataFrame(
        {
            "wavenumber_cm_1": wavenumbers,
            "transmittance_pct": transmittance.clip(lower=5.0, upper=95.0),
        }
    )
    return frame


def _score_radar_chart(spectral: float, mechanical: float) -> go.Figure:
    thresholds = [
        _SCORE_THRESHOLDS["spectral"],
        _SCORE_THRESHOLDS["mechanical"],
    ]
    values = [spectral, mechanical]
    categories = ["Espectral", "Mecánico"]

    def _close(payload: list[float]) -> list[float]:
        return payload + payload[:1]

    fig = go.Figure()
    fig.add_trace(
        go.Scatterpolar(
            r=_close(values),
            theta=_close(categories),
            fill="toself",
            name="Score",
            line=dict(color="#38bdf8"),
            fillcolor="rgba(56, 189, 248, 0.3)",
        )
    )
    fig.add_trace(
        go.Scatterpolar(
            r=_close(thresholds),
            theta=_close(categories),
            fill="toself",
            name="Umbral",
            line=dict(color="#f97316", dash="dash"),
            fillcolor="rgba(249, 115, 22, 0.1)",
        )
    )
    fig.update_layout(
        showlegend=False,
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        margin=dict(l=0, r=0, t=20, b=20),
        height=220,
    )
    return fig


def _compute_severity(row: Mapping[str, Any]) -> float:
    spectral = max(float(row.get("spectral_score", 0.0) or 0.0), 0.0)
    mechanical = max(float(row.get("mechanical_score", 0.0) or 0.0), 0.0)
    utility = max(float(row.get("material_utility_score", 0.0) or 0.0), 0.0)
    gaps = [
        max(0.0, _SCORE_THRESHOLDS["spectral"] - spectral),
        max(0.0, _SCORE_THRESHOLDS["mechanical"] - mechanical),
        max(0.0, 0.5 - utility),
    ]
    return max(gaps)


def _standardise_spectral_curve(curve: pd.DataFrame) -> pd.DataFrame:
    if curve is None or curve.empty:
        return pd.DataFrame(columns=["wavenumber_cm_1", "transmittance_pct"])

    working = curve.copy()
    if "transmittance_pct" in working.columns:
        working["transmittance_pct"] = pd.to_numeric(
            working["transmittance_pct"], errors="coerce"
        )
    else:
        value_column = None
        for candidate in ("absorbance_norm_1um", "absorbance", "intensity", "signal"):
            if candidate in working.columns:
                value_column = candidate
                break
        if value_column is None:
            for column in working.columns:
                if column != "wavenumber_cm_1":
                    value_column = column
                    break
        values = pd.to_numeric(working.get(value_column), errors="coerce")
        if value_column and "absorb" in value_column:
            min_val = float(values.min()) if not values.isna().all() else 0.0
            max_val = float(values.max()) if not values.isna().all() else 1.0
            span = max(max_val - min_val, 1e-6)
            working["transmittance_pct"] = (1.0 - (values - min_val) / span) * 100.0
        else:
            working["transmittance_pct"] = values

    working["transmittance_pct"] = working["transmittance_pct"].interpolate().fillna(0.0)
    working = working.loc[:, [col for col in working.columns if col in {"wavenumber_cm_1", "transmittance_pct"}]]
    return working.dropna(subset=["wavenumber_cm_1", "transmittance_pct"])


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


st.title("🛰️ Centro de control marciano")
st.markdown(
    """
    Consolida vuelos, inventario, decisiones automáticas y planificación diaria
    en una sola consola. Cada pestaña se alimenta de telemetría en tiempo real
    para que operaciones priorice acciones críticas y documente resultados.
    """
)

generator_service = GeneratorService()
telemetry_service = MarsControlCenterService()

analysis_state: dict[str, Any] | None = st.session_state.get("policy_analysis")
manifest_df: pd.DataFrame | None = st.session_state.get("uploaded_manifest_df")

st.session_state.setdefault(_MANUAL_DECISIONS_KEY, {})

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
        flights_df = _apply_manual_overrides(flights_df)
        _store_flight_snapshot(flights_df)
        st.session_state["flight_operations_signature"] = manifest_signature
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
            flights_df = _apply_manual_overrides(flights_df)
            _store_flight_snapshot(flights_df)
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

        passport = analysis_state.get("material_passport", {})

        manifest_source_df: pd.DataFrame | None = None
        if isinstance(manifest_df, pd.DataFrame) and not manifest_df.empty:
            manifest_source_df = manifest_df
        else:
            candidate_manifest = analysis_state.get("manifest")
            if isinstance(candidate_manifest, pd.DataFrame):
                manifest_source_df = candidate_manifest
            elif isinstance(candidate_manifest, Mapping):
                manifest_source_df = pd.DataFrame(candidate_manifest)

        batch_results = _ensure_manifest_batch(generator_service, manifest_source_df)
        if batch_results:
            batch_entry = batch_results[0]
            scored_manifest = batch_entry.get("scored_manifest", pd.DataFrame())
            compatibility = batch_entry.get("compatibility", pd.DataFrame())
            recommendations = batch_entry.get("policy_recommendations", pd.DataFrame())
        else:
            scored_manifest = summary.get("manifest", pd.DataFrame())
            compatibility = summary.get("compatibility", pd.DataFrame())
            recommendations = summary.get("recommendations", pd.DataFrame())

        flights_df = st.session_state.get("flight_operations_table")
        if (
            (flights_df is None or flights_df.empty)
            and manifest_source_df is not None
            and not manifest_source_df.empty
        ):
            flights_df = telemetry_service.flight_operations_overview(
                passport,
                manifest_df=manifest_source_df,
                analysis_state=analysis_state,
            )
            flights_df = _apply_manual_overrides(flights_df)
            _store_flight_snapshot(flights_df)

        working_manifest = (
            scored_manifest.copy() if isinstance(scored_manifest, pd.DataFrame) else pd.DataFrame()
        )
        verified_manifest_count = 0
        verified_manifest_labels: list[str] = []
        if not working_manifest.empty:
            for column in ("spectral_score", "mechanical_score", "material_utility_score", "mass_kg"):
                if column in working_manifest.columns:
                    working_manifest[column] = pd.to_numeric(
                        working_manifest[column], errors="coerce"
                    ).fillna(0.0)
                else:
                    working_manifest[column] = 0.0
            working_manifest["severity"] = working_manifest.apply(_compute_severity, axis=1)
            working_manifest = _annotate_verified_materials(working_manifest)
            if "verified_material_label" in working_manifest.columns:
                verified_mask = working_manifest["verified_material_label"].astype(str).str.len() > 0
                verified_manifest_count = int(verified_mask.sum())
                verified_manifest_labels = sorted(
                    {
                        str(label)
                        for label in working_manifest.loc[verified_mask, "verified_material_label"].tolist()
                        if str(label)
                    }
                )
        else:
            working_manifest["visual_material_score"] = []
            working_manifest["verified_material_label"] = []
            working_manifest["verified_confidence_note"] = []

        flight_lookup: dict[str, Mapping[str, Any]] = {}
        if isinstance(flights_df, pd.DataFrame) and not flights_df.empty:
            for record in flights_df.to_dict(orient="records"):
                manifest_ref = str(record.get("manifest_ref"))
                if manifest_ref:
                    flight_lookup[manifest_ref] = record

        if not working_manifest.empty:
            def _resolve_shipment(row: pd.Series) -> str:
                item_text = str(row.get("item") or "").lower()
                material_text = str(row.get("material_key") or "").lower()
                for manifest_ref, record in flight_lookup.items():
                    materials = record.get("key_materials") or []
                    for material in materials:
                        token = str(material).lower()
                        if token and (token in item_text or token in material_text):
                            return manifest_ref
                if flight_lookup:
                    return next(iter(flight_lookup.keys()))
                return "manifest-alpha"

            working_manifest["shipment_ref"] = working_manifest.apply(_resolve_shipment, axis=1)
        else:
            working_manifest["shipment_ref"] = "manifest-alpha"

        shipments: list[dict[str, Any]] = []
        decisions_records: list[dict[str, Any]] = []

        if not working_manifest.empty:
            grouped = working_manifest.groupby("shipment_ref", dropna=False)
            for manifest_ref, group in grouped:
                manifest_key = str(manifest_ref or "manifest-alpha")
                flight_info = flight_lookup.get(manifest_key, {})
                severity = float(group["severity"].max()) if not group.empty else 0.0
                critical = group[group["severity"] > 0.0].sort_values(
                    "material_utility_score"
                )
                focus_pool = critical if not critical.empty else group
                focus_row = focus_pool.sort_values("material_utility_score").iloc[0]
                compatibility_score = float(group["material_utility_score"].mean())
                spectral_avg = float(group["spectral_score"].mean())
                mechanical_avg = float(group["mechanical_score"].mean())
                critical_count = int((group["severity"] > 0.0).sum())

                rec_subset = pd.DataFrame()
                if (
                    isinstance(recommendations, pd.DataFrame)
                    and not recommendations.empty
                    and "item_name" in recommendations.columns
                ):
                    rec_subset = recommendations[recommendations["item_name"].isin(
                        group["item"].astype(str)
                    )]

                comp_subset = pd.DataFrame()
                if (
                    isinstance(compatibility, pd.DataFrame)
                    and not compatibility.empty
                    and "material_key" in compatibility.columns
                ):
                    comp_subset = compatibility[compatibility["material_key"].isin(
                        group["material_key"].astype(str)
                    )]

                spectral_key, spectral_curve, spectral_meta = _resolve_spectral_curve(
                    focus_row.get("material_key"), focus_row.get("item")
                )
                synthetic_curve = False
                if spectral_curve is None or spectral_curve.empty:
                    spectral_curve = _synthetic_spectral_curve(
                        float(focus_row.get("spectral_score") or 0.0),
                        float(focus_row.get("mechanical_score") or 0.0),
                    )
                    synthetic_curve = True

                overrides = st.session_state.get(_MANUAL_DECISIONS_KEY, {})
                manual_payload = overrides.get(manifest_key) or overrides.get(
                    str(manifest_key), {}
                )
                order_label = manual_payload.get("label") or flight_info.get(
                    "ai_decision"
                ) or "Monitoreo nominal"
                order_badge = (
                    manual_payload.get("badge")
                    or flight_info.get("decision_indicator")
                    or "Orden Rex-AI"
                )

                if rec_subset is not None and not rec_subset.empty:
                    top_rec = rec_subset.sort_values(
                        "recommended_score", ascending=False
                    ).iloc[0]
                    recommendation_text = (
                        f"{top_rec.get('action')} → {top_rec.get('recommended_material_key')}"
                    )
                else:
                    recommendation_text = "Sin cambios"

                visual_score = float(
                    focus_row.get("visual_material_score")
                    or focus_row.get("material_utility_score")
                    or 0.0
                )
                verified_label = str(focus_row.get("verified_material_label") or "")
                detected_text = (
                    f"{focus_row.get('item')} · Score verificado {visual_score:.2f}"
                    if verified_label
                    else f"{focus_row.get('item')} · Score {visual_score:.2f}"
                )
                compatibility_text = f"{compatibility_score:.2f}"

                radar_fig = _score_radar_chart(spectral_avg, mechanical_avg)

                spectral_curve = _standardise_spectral_curve(spectral_curve).sort_values(
                    "wavenumber_cm_1", ascending=False
                )
                spectral_caption = (
                    "Curva sintetizada a partir de los puntajes de compatibilidad: "
                    "el bundle FTIR no expone este material."
                    if synthetic_curve
                    else f"FTIR de referencia · {spectral_meta.get('material', spectral_key)}"
                )

                badges = [
                    f"Detectado · {detected_text}",
                    f"Compatibilidad · {compatibility_text}",
                    f"Sugerencia Rex-AI · {recommendation_text}",
                    f"{order_badge} · {order_label}",
                ]
                if verified_label:
                    confidence_note = str(
                        focus_row.get("verified_confidence_note") or _VERIFIED_DEFAULT_NOTE
                    )
                    badges.insert(1, f"Confianza · {verified_label}")
                    badges.append(f"Mediciones · {confidence_note}")

                shipments.append(
                    {
                        "manifest_ref": manifest_key,
                        "flight": flight_info,
                        "severity": severity,
                        "critical_count": critical_count,
                        "radar_fig": radar_fig,
                        "spectral_curve": spectral_curve,
                        "spectral_caption": spectral_caption,
                        "critical_table": (
                            critical[[
                                "item",
                                "material_key",
                                "material_utility_score",
                                "visual_material_score",
                                "spectral_score",
                                "mechanical_score",
                                "mass_kg",
                                "verified_material_label",
                            ]].head(6)
                            if not critical.empty
                            else group[[
                                "item",
                                "material_key",
                                "material_utility_score",
                                "visual_material_score",
                                "spectral_score",
                                "mechanical_score",
                                "mass_kg",
                                "verified_material_label",
                            ]]
                            .sort_values("visual_material_score", ascending=False)
                            .head(6)
                        ),
                        "compatibility_subset": comp_subset,
                        "badges": badges,
                        "order_label": order_label,
                        "order_badge": order_badge,
                    }
                )

                decisions_records.append(
                    {
                        "manifest_ref": manifest_key,
                        "flight_id": flight_info.get("flight_id"),
                        "vehicle": flight_info.get("vehicle"),
                        "critical_items": critical_count,
                        "worst_item": focus_row.get("item"),
                        "mean_score": compatibility_score,
                        "manual_action": manual_payload.get("action"),
                        "order_label": order_label,
                        "severity": severity,
                    }
                )

        shipments.sort(
            key=lambda payload: (-payload.get("severity", 0.0), payload.get("critical_count", 0))
        )

        if verified_manifest_count:
            verified_labels_text = ", ".join(verified_manifest_labels)
            st.success(
                "🔒 Materiales verificados con mediciones reales: "
                f"{verified_labels_text}. Se muestran con scores positivos reforzados "
                "porque el bundle tiene mediciones físicas proxy."
            )

        micro_divider()
        st.markdown("### Prioridades por envío")
        if shipments:
            for shipment in shipments:
                flight_info = shipment["flight"]
                flight_label = flight_info.get("flight_id") or shipment["manifest_ref"]
                with st.container():
                    header_cols = st.columns([3, 1])
                    header_cols[0].markdown(f"**{flight_label} · {shipment['manifest_ref']}**")
                    badge_group(shipment["badges"], parent=header_cols[0])
                    header_cols[1].metric(
                        "Severidad",
                        f"{shipment['severity']:.2f}",
                        delta=f"{shipment['critical_count']} críticos",
                    )

                    chart_cols = st.columns([1, 1])
                    chart_cols[0].plotly_chart(
                        shipment["radar_fig"], use_container_width=True
                    )
                    spectral_fig = go.Figure(
                        data=[
                            go.Scatter(
                                x=shipment["spectral_curve"]["wavenumber_cm_1"],
                                y=shipment["spectral_curve"]["transmittance_pct"],
                                mode="lines",
                                line=dict(color="#facc15"),
                            )
                        ]
                    )
                    spectral_fig.update_layout(
                        margin=dict(t=10, b=0, l=0, r=0),
                        xaxis_title="Número de onda (cm⁻¹)",
                        yaxis_title="Transmittancia (%)",
                        height=220,
                    )
                    chart_cols[0].plotly_chart(spectral_fig, use_container_width=True)
                    chart_cols[0].caption(shipment["spectral_caption"])

                    detail_df = shipment["critical_table"]
                    chart_cols[1].markdown("**Ítems críticos**")
                    chart_cols[1].dataframe(
                        detail_df,
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            "visual_material_score": st.column_config.NumberColumn(
                                "Score verificado (proxy)",
                                format="%.2f",
                                help="Score mostrado con refuerzo de confianza cuando hay mediciones reales.",
                            ),
                            "material_utility_score": st.column_config.NumberColumn(
                                "Score", format="%.2f"
                            ),
                            "spectral_score": st.column_config.NumberColumn(
                                "Espectral", format="%.2f"
                            ),
                            "mechanical_score": st.column_config.NumberColumn(
                                "Mecánico", format="%.2f"
                            ),
                            "mass_kg": st.column_config.NumberColumn("Masa (kg)", format="%.2f"),
                            "verified_material_label": st.column_config.TextColumn(
                                "Material verificado",
                                help="Etiqueta del material con mediciones físicas registradas.",
                            ),
                        },
                    )

                    if isinstance(shipment["compatibility_subset"], pd.DataFrame) and not shipment[
                        "compatibility_subset"
                    ].empty:
                        chart_cols[1].markdown("**Compatibilidad documentada**")
                        bullets = []
                        for _, row in shipment["compatibility_subset"].iterrows():
                            bullets.append(
                                f"- `{row.get('material_key')}` ↔ `{row.get('partner_key')}` · {row.get('rule')}"
                            )
                        chart_cols[1].markdown("\n".join(bullets))
                    else:
                        chart_cols[1].caption("Sin trazas de compatibilidad adicionales.")

                    action_cols = st.columns(3)
                    if action_cols[0].button(
                        "Aceptar", key=f"accept_{shipment['manifest_ref']}"
                    ):
                        _register_manual_action(shipment["manifest_ref"], "accept")
                        st.success("Orden manual registrada.")
                    if action_cols[1].button(
                        "Rechazar", key=f"reject_{shipment['manifest_ref']}"
                    ):
                        _register_manual_action(shipment["manifest_ref"], "reject")
                        st.warning("La orden fue rechazada manualmente.")
                    if action_cols[2].button(
                        "Repriorizar", key=f"reprioritize_{shipment['manifest_ref']}"
                    ):
                        _register_manual_action(shipment["manifest_ref"], "reprioritize")
                        st.info("El envío fue marcado para repriorización.")
        else:
            st.success(
                "El lote evaluado no contiene alertas críticas: Rex-AI mantiene monitoreo nominal."
            )

        if decisions_records:
            export_df = pd.DataFrame(decisions_records)
            csv_buffer = io.StringIO()
            export_df.to_csv(csv_buffer, index=False)
            json_payload = json.dumps(
                export_df.to_dict(orient="records"), ensure_ascii=False, indent=2
            )
            export_cols = st.columns(2)
            with export_cols[0]:
                st.download_button(
                    "Descargar decisiones (CSV)",
                    csv_buffer.getvalue().encode("utf-8"),
                    file_name="decisiones_mars_control.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
            with export_cols[1]:
                st.download_button(
                    "Descargar decisiones (JSON)",
                    json_payload.encode("utf-8"),
                    file_name="decisiones_mars_control.json",
                    mime="application/json",
                    use_container_width=True,
                )

        micro_divider()
        st.markdown("**Detalle de ítems evaluados**")
        if not working_manifest.empty:
            st.caption(
                "Los materiales verificados se muestran con un score positivo reforzado porque tienen mediciones reales cargadas (proxy)."
            )
            st.dataframe(
                working_manifest,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "visual_material_score": st.column_config.NumberColumn(
                        "Score verificado (proxy)", format="%.2f"
                    ),
                    "verified_material_label": st.column_config.TextColumn(
                        "Material verificado",
                        help="Identifica la receta con datos validados.",
                    ),
                    "verified_confidence_note": st.column_config.TextColumn(
                        "Notas de confianza", help="Detalle de la medición física utilizada."
                    ),
                },
            )
        else:
            st.caption("Sin datos de manifiesto evaluados en esta corrida.")

        micro_divider()
        st.markdown("**Recomendaciones de política**")
        if isinstance(recommendations, pd.DataFrame) and not recommendations.empty:
            st.dataframe(recommendations, use_container_width=True)
        else:
            st.success(
                "No se identificaron acciones prioritarias: todos los ítems superan el umbral de utilidad."
            )

        micro_divider()
        st.markdown("**Trazabilidad de compatibilidad**")
        if isinstance(compatibility, pd.DataFrame) and not compatibility.empty:
            st.dataframe(compatibility, use_container_width=True)
        else:
            st.caption("Sin datos de compatibilidad asociados al manifiesto.")

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

    loop_cols = st.columns([3, 2, 2])
    default_auto = st.session_state.get("demo_event_auto", False)
    default_interval = int(st.session_state.get("demo_event_interval", 20))
    with loop_cols[0]:
        auto_loop = st.checkbox(
            "Loop automático",
            value=default_auto,
            key="demo_event_auto_checkbox",
            help="Genera un evento demo cada n segundos",
        )
    with loop_cols[1]:
        interval_seconds = st.slider(
            "Intervalo (s)",
            min_value=5,
            max_value=60,
            value=default_interval,
            step=5,
            key="demo_event_interval_slider",
        )
    with loop_cols[2]:
        reset_script = st.button("Reiniciar script", use_container_width=True)

    trigger_next = st.button("Emitir siguiente evento", use_container_width=True)

    st.session_state["demo_event_interval"] = interval_seconds

    if reset_script:
        mars_control.reset_demo_events()
        st.session_state.pop("demo_last_event", None)
        st.success("Script demo reiniciado")
        st.experimental_rerun()

    new_event: mars_control.DemoEvent | None = None
    if trigger_next:
        new_event = mars_control.generate_demo_event(interval_seconds, force=True)

    if auto_loop:
        st.session_state["demo_event_auto"] = True
        st.autorefresh(
            interval=int(interval_seconds * 1000),
            limit=None,
            key="demo_event_autorefresh",
        )
        new_event = new_event or mars_control.generate_demo_event(interval_seconds)
    else:
        st.session_state["demo_event_auto"] = False

    history = mars_control.get_demo_event_history(limit=8)
    latest_event = new_event or (history[0] if history else None)
    if latest_event:
        st.session_state["demo_last_event"] = latest_event

    last_event: mars_control.DemoEvent | None = st.session_state.get(
        "demo_last_event"
    )

    if last_event:
        st.markdown(_render_demo_event_card(last_event), unsafe_allow_html=True)
        if last_event.audio_bytes:
            st.audio(last_event.audio_bytes, format="audio/wav")
        elif last_event.audio_path:
            st.audio(last_event.audio_path, format="audio/wav")
    else:
        st.info(
            "Aún no se emitieron eventos demo. Activá el loop o generá uno manualmente."
        )

    if history:
        ticker_events = list(reversed(history))
        st.markdown(_render_demo_ticker(ticker_events), unsafe_allow_html=True)

        log_rows: list[dict[str, str]] = []
        for event in history:
            metadata = " · ".join(
                f"{key}: {value}" for key, value in event.metadata.items()
            )
            log_rows.append(
                {
                    "Hora": _format_demo_timestamp(event),
                    "Evento": event.title,
                    "Detalle": event.message,
                    "Nivel": event.severity.upper(),
                    "Metadata": metadata,
                }
            )
        st.dataframe(
            pd.DataFrame(log_rows),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.caption("El ticker se completará a medida que se emitan eventos del guion demo.")

    micro_divider()
    st.markdown("#### Inyectar manifiesto demo")
    st.caption(
        "Seleccioná un manifiesto preconfigurado para ver cómo las decisiones IA cambian en vivo."
    )

    catalogue = mars_control.demo_manifest_catalogue()
    if catalogue:
        options = {entry["label"]: entry for entry in catalogue if entry.get("label")}
        if not options:
            options = {entry["key"]: entry for entry in catalogue}
        labels = list(options.keys())
        default_label = st.session_state.get("demo_manifest_selected", labels[0])
        selection = st.selectbox(
            "Seleccioná un manifiesto de prueba",
            options=labels,
            index=labels.index(default_label) if default_label in labels else 0,
        )
        st.session_state["demo_manifest_selected"] = selection
        selected_entry = options[selection]
        if selected_entry.get("description"):
            st.caption(selected_entry["description"])
        preview_df = pd.DataFrame(selected_entry.get("rows", []))
        st.dataframe(preview_df, use_container_width=True, hide_index=True)

        if st.button(
            "Inyectar manifiesto demo",
            use_container_width=True,
            key="inject_demo_manifest_button",
        ):
            manifest_df = mars_control.load_demo_manifest(selected_entry["key"])
            with st.spinner("Generando decisiones Rex-AI para el manifiesto demo..."):
                analysis = run_policy_analysis(
                    generator_service, manifest_df, include_pdf=False
                )
            st.session_state["policy_analysis"] = analysis
            st.session_state["uploaded_manifest_df"] = manifest_df
            st.success(
                "Manifiesto demo procesado. Revisá Decisiones IA y Flight Radar para ver los cambios."
            )
            st.experimental_rerun()
    else:
        st.warning("No se encontraron manifiestos demo preconfigurados.")

    with st.expander("Guion demo predefinido"):
        script_entries = mars_control.demo_event_script()
        script_df = pd.DataFrame(
            [
                {
                    "Evento": entry.title,
                    "Categoría": entry.category,
                    "Nivel": entry.severity,
                    "Mensaje": entry.message,
                }
                for entry in script_entries
            ]
        )
        st.dataframe(script_df, use_container_width=True, hide_index=True)

