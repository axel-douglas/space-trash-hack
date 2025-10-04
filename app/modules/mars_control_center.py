"""Telemetry services backing the Mars Control Center page."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import html
import math
from typing import Any, Final, Mapping, MutableMapping, Sequence

import pandas as pd

from app.modules.io import load_process_df
from app.modules.mars_control import (
    MarsLogisticsData,
    SimulationEvent,
    aggregate_inventory_by_category,
    load_jezero_bitmap,
    load_jezero_ortho_bitmap,
    load_jezero_slope_bitmap,
    load_jezero_geodata,
    load_logistics_baseline,
    apply_simulation_tick,
    iterate_events,
)
from app.modules.mission_overview import compute_mission_summary, load_inventory_overview
from app.modules.process_planner import choose_process

TICK_MINUTES: Final[int] = 15

_STATUS_META: Final[dict[str, dict[str, str]]] = {
    "boarding": {"label": "Embarque", "color": "#facc15", "badge": "🟡 Embarque"},
    "en_route": {"label": "En vuelo", "color": "#38bdf8", "badge": "🟦 En vuelo"},
    "launch-ready": {"label": "Listo para lanzamiento", "color": "#22d3ee", "badge": "🟦 Preparación"},
    "ready": {"label": "Listo", "color": "#22d3ee", "badge": "🟦 Preparación"},
    "landed": {"label": "Aterrizado", "color": "#22c55e", "badge": "🟢 Aterrizado"},
    "hold": {"label": "En espera", "color": "#f97316", "badge": "🟠 Espera"},
    "delayed": {"label": "Demorado", "color": "#ef4444", "badge": "🔴 Demorado"},
    "default": {"label": "Sin estado", "color": "#94a3b8", "badge": "⚪ Sin estado"},
}

_CATEGORY_COLORS: Final[dict[str, list[int]]] = {
    "orbital_corridor": [56, 189, 248],
    "surface": [34, 197, 94],
    "resource_zone": [248, 113, 113],
    "landing_zone": [251, 191, 36],
    "science_zone": [129, 140, 248],
    "default": [148, 163, 184],
}

_CAPSULE_COORDINATES: Final[dict[str, dict[str, float | str]]] = {
    "ares_cargo_7": {
        "latitude": 18.38,
        "longitude": 77.58,
        "altitude_km": 420.0,
        "category": "orbital_corridor",
    },
    "skylift_3": {
        "latitude": 18.46,
        "longitude": 77.64,
        "altitude_km": 1085.0,
        "category": "orbital_corridor",
    },
    "dusthopper": {
        "latitude": 18.39,
        "longitude": 77.56,
        "altitude_km": 0.12,
        "category": "surface",
    },
    "default": {
        "latitude": 18.38,
        "longitude": 77.58,
        "altitude_km": 0.0,
        "category": "surface",
    },
}

_MANIFEST_MATERIAL_HINTS: Final[dict[str, list[str]]] = {
    "manifest-alpha": [
        "Aleaciones Ti-Fe",
        "Repuestos impresos 3D",
        "Paneles fotónicos",
    ],
    "manifest-beta": [
        "Polímeros fluorados",
        "Consumibles EVA",
        "Resinas dieléctricas",
    ],
    "manifest-recycle": [
        "Regolito tamizado",
        "Polvo metálico",
        "Polímeros recuperados",
    ],
}

_MANIFEST_DOSSIER_HINTS: Final[dict[str, dict[str, float | str]]] = {
    "manifest-alpha": {
        "spectrum": "Aleaciones Ti-Fe · polímeros avanzados",
        "density": 1.82,
        "compatibility": 0.83,
    },
    "manifest-beta": {
        "spectrum": "Polímeros fluorados · materiales EVA",
        "density": 1.35,
        "compatibility": 0.78,
    },
    "manifest-recycle": {
        "spectrum": "Regolito tratado · compuestos metálicos",
        "density": 1.95,
        "compatibility": 0.71,
    },
    "default": {
        "spectrum": "Compuesto mixto",
        "density": 1.6,
        "compatibility": 0.7,
    },
}

_ZONE_DATA: Final[list[dict[str, Any]]] = [
    {
        "name": "Base Jezero",
        "latitude": 18.39,
        "longitude": 77.58,
        "category": "landing_zone",
        "radius_m": 900.0,
        "spectrum": "Acero reforzado · vidrio marciano",
        "density": 2.4,
        "compatibility": 0.91,
    },
    {
        "name": "Delta Oeste",
        "latitude": 18.62,
        "longitude": 77.44,
        "category": "resource_zone",
        "radius_m": 1200.0,
        "spectrum": "Sedimentos hidratados",
        "density": 1.7,
        "compatibility": 0.76,
    },
    {
        "name": "Nodo Orbital Phobos",
        "latitude": 18.1,
        "longitude": 77.9,
        "category": "orbital_corridor",
        "radius_m": 1600.0,
        "spectrum": "Aleaciones Al-Li · compuestos carbono",
        "density": 2.1,
        "compatibility": 0.84,
    },
]


@dataclass(frozen=True)
class FlightTelemetry:
    """Snapshot of a logistics flight used in the radar visualisations."""

    vehicle: str
    phase: str
    latitude: float
    longitude: float
    altitude_km: float
    eta_minutes: int
    payload_kg: float


class MarsControlCenterService:
    """High level orchestrator that aggregates live mission telemetry."""

    def __init__(self) -> None:
        self._process_df: pd.DataFrame | None = None
        self._logistics_data: MarsLogisticsData | None = None

    # ------------------------------------------------------------------
    # Flight radar & logistics map
    # ------------------------------------------------------------------

    def _ensure_logistics(self) -> MarsLogisticsData:
        if self._logistics_data is None:
            self._logistics_data = load_logistics_baseline()
        return self._logistics_data

    def _status_token(self, status: str) -> dict[str, str]:
        status_key = str(status or "").lower()
        return _STATUS_META.get(status_key, _STATUS_META["default"])

    def _hex_to_rgb(self, value: str) -> list[int]:
        candidate = (value or "").lstrip("#")
        if len(candidate) == 3:
            candidate = "".join(ch * 2 for ch in candidate)
        try:
            return [int(candidate[index : index + 2], 16) for index in (0, 2, 4)]
        except ValueError:
            return _CATEGORY_COLORS["default"]

    def _category_color(self, category: str | None) -> list[int]:
        key = str(category or "").lower()
        return _CATEGORY_COLORS.get(key, _CATEGORY_COLORS["default"])

    def _capsule_coordinates(self, capsule_id: str) -> dict[str, float | str]:
        return _CAPSULE_COORDINATES.get(capsule_id, _CAPSULE_COORDINATES["default"])

    def _build_manifest_material_index(
        self,
        manifest_df: pd.DataFrame | None,
        analysis_state: Mapping[str, Any] | None,
    ) -> dict[str, list[str]]:
        index = {key: list(values) for key, values in _MANIFEST_MATERIAL_HINTS.items()}

        candidate_df: pd.DataFrame | None = None
        if analysis_state:
            scored = analysis_state.get("scored_manifest")
            if isinstance(scored, pd.DataFrame) and not scored.empty:
                candidate_df = scored
        if candidate_df is None and isinstance(manifest_df, pd.DataFrame) and not manifest_df.empty:
            candidate_df = manifest_df

        if candidate_df is not None and not candidate_df.empty:
            working = candidate_df.copy()
            if "mass_kg" in working.columns:
                working["mass_kg"] = pd.to_numeric(working["mass_kg"], errors="coerce").fillna(0.0)
                working = working.sort_values("mass_kg", ascending=False)
            label_col = None
            for column in ("material_key", "material", "item", "item_name"):
                if column in working.columns:
                    label_col = column
                    break
            if label_col:
                candidates: list[str] = []
                for value in working[label_col].tolist():
                    if pd.isna(value):
                        continue
                    text = str(value).strip()
                    if not text:
                        continue
                    candidates.append(text)
                    if len(candidates) == 3:
                        break
                if candidates:
                    index["manifest-alpha"] = candidates
        return index

    def _build_material_dossiers(
        self,
        manifest_df: pd.DataFrame | None,
        analysis_state: Mapping[str, Any] | None,
    ) -> dict[str, dict[str, float | str]]:
        dossier = {key: dict(values) for key, values in _MANIFEST_DOSSIER_HINTS.items()}

        candidate_df: pd.DataFrame | None = None
        if analysis_state:
            scored = analysis_state.get("scored_manifest")
            if isinstance(scored, pd.DataFrame) and not scored.empty:
                candidate_df = scored
        if candidate_df is None and isinstance(manifest_df, pd.DataFrame) and not manifest_df.empty:
            candidate_df = manifest_df

        if candidate_df is not None and not candidate_df.empty:
            working = candidate_df.copy()
            working_numeric = pd.to_numeric(working.get("mass_kg"), errors="coerce").fillna(0.0)
            total_mass = float(working_numeric.sum())

            total_volume = 0.0
            if "volume_m3" in working.columns:
                total_volume = float(pd.to_numeric(working["volume_m3"], errors="coerce").fillna(0.0).sum())
            elif "volume_l" in working.columns:
                total_volume = float(pd.to_numeric(working["volume_l"], errors="coerce").fillna(0.0).sum()) / 1000.0

            density = dossier["default"]["density"]
            if total_mass and total_volume:
                density = max(total_mass / total_volume, 0.1)

            compatibility = dossier["default"]["compatibility"]
            if "material_utility_score" in working.columns:
                compatibility = float(
                    pd.to_numeric(working["material_utility_score"], errors="coerce").fillna(0.0).mean() or compatibility
                )

            spectral_fields = [
                "spectral_signature",
                "spectrum",
                "material_family",
                "category",
            ]
            spectrum = None
            for field in spectral_fields:
                if field in working.columns:
                    values = [
                        html.escape(str(value))
                        for value in working[field].astype(str).tolist()
                        if str(value).strip() and not pd.isna(value)
                    ]
                    if values:
                        # Preserve order while deduplicating
                        seen: set[str] = set()
                        unique_values: list[str] = []
                        for value in values:
                            if value in seen:
                                continue
                            seen.add(value)
                            unique_values.append(value)
                            if len(unique_values) == 3:
                                break
                        spectrum = " · ".join(unique_values)
                        break
            if not spectrum:
                spectrum = dossier["default"]["spectrum"]

            dossier["manifest-alpha"] = {
                "spectrum": spectrum,
                "density": round(density, 2),
                "compatibility": round(max(min(compatibility, 1.0), 0.0), 2),
            }

        return dossier

    def _apply_status_tokens(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        tokens = df["status"].apply(self._status_token)
        df["status_label"] = tokens.apply(lambda item: item["label"])
        df["status_color"] = tokens.apply(lambda item: item["color"])
        df["status_badge"] = tokens.apply(lambda item: item["badge"])
        rgb_components = df["status_color"].apply(self._hex_to_rgb)
        df[["status_color_r", "status_color_g", "status_color_b"]] = pd.DataFrame(
            rgb_components.tolist(), index=df.index
        )
        return df

    def _event_payload(self, event: SimulationEvent) -> dict[str, Any]:
        payload = event.to_dict()
        if event.metadata:
            for key, value in event.metadata.items():
                payload[f"metadata_{key}"] = value
        return payload

    def _analysis_directive(self, analysis_state: Mapping[str, Any] | None) -> str | None:
        if not analysis_state:
            return None

        recommendations = analysis_state.get("policy_recommendations")
        if isinstance(recommendations, pd.DataFrame) and not recommendations.empty:
            top = recommendations.iloc[0]
            action = str(top.get("action") or "").strip()
            recommended = str(top.get("recommended_material_key") or "").strip()
            if action and recommended:
                return f"{action.capitalize()}: {recommended}"
            if action:
                return action.capitalize()

        passport = analysis_state.get("material_passport") or {}
        if isinstance(passport, Mapping):
            items = passport.get("recommendations")
            if isinstance(items, list) and items:
                first = items[0]
                if isinstance(first, Mapping):
                    item = str(first.get("item_name") or first.get("current_material_key") or "").strip()
                    recommended = str(first.get("recommended_material_key") or "").strip()
                    if item and recommended:
                        return f"{item} → {recommended}"
        return None

    def _resolve_payload_mass(self, passport: Mapping[str, Any] | None) -> float:
        if not passport:
            return 0.0
        try:
            return float(passport.get("total_mass_kg", 0.0))
        except (TypeError, ValueError):
            return 0.0

    def flight_operations_overview(
        self,
        passport: Mapping[str, Any] | None,
        *,
        manifest_df: pd.DataFrame | None = None,
        analysis_state: Mapping[str, Any] | None = None,
    ) -> pd.DataFrame:
        logistics = self._ensure_logistics()
        capsule_lookup = {capsule.capsule_id: capsule for capsule in logistics.capsules}

        material_index = self._build_manifest_material_index(manifest_df, analysis_state)
        dossier_index = self._build_material_dossiers(manifest_df, analysis_state)
        analysis_directive = self._analysis_directive(analysis_state)

        orders_by_target: dict[str, list] = {}
        for order in logistics.ai_orders:
            if not order.target:
                continue
            orders_by_target.setdefault(order.target, []).append(order)

        rows: list[dict[str, Any]] = []
        now = datetime.now(timezone.utc)
        for flight in logistics.flights:
            manifest_ref = flight.manifest_ref or "manifest-alpha"
            coordinates = self._capsule_coordinates(flight.capsule_id)
            capsule = capsule_lookup.get(flight.capsule_id)
            capsule_name = capsule.name if capsule else flight.capsule_id
            eta_minutes = 0
            if flight.arrival:
                eta_minutes = max(int((flight.arrival - now).total_seconds() // 60), 0)
            elif str(flight.status).lower() in {"en_route", "boarding", "launch-ready", "ready"}:
                eta_minutes = max(TICK_MINUTES * 2, 10)

            orders = orders_by_target.get(manifest_ref, [])
            ai_decision = None
            decision_timestamp = None
            if orders:
                latest = max(
                    orders,
                    key=lambda order: order.issued_at or datetime.min.replace(tzinfo=timezone.utc),
                )
                ai_decision = latest.directive
                decision_timestamp = (
                    latest.issued_at.isoformat() if latest.issued_at else None
                )
            elif analysis_directive:
                ai_decision = analysis_directive
            else:
                ai_decision = f"Monitorear {manifest_ref}"

            materials = material_index.get(
                manifest_ref, material_index.get("manifest-alpha", material_index.get("default", []))
            )
            dossier = dossier_index.get(
                manifest_ref, dossier_index.get("default", {"spectrum": "Compuesto mixto", "density": 1.6, "compatibility": 0.7})
            )

            rows.append(
                {
                    "flight_id": flight.flight_id,
                    "vehicle": capsule_name,
                    "capsule_id": flight.capsule_id,
                    "phase": " → ".join(
                        [
                            part
                            for part in (
                                str(flight.origin or "").strip(),
                                str(flight.destination or "").strip(),
                            )
                            if part
                        ]
                    ),
                    "status": str(flight.status or "unknown").lower(),
                    "latitude": float(coordinates.get("latitude", 18.38)),
                    "longitude": float(coordinates.get("longitude", 77.58)),
                    "altitude_km": float(coordinates.get("altitude_km", 0.0)),
                    "capsule_category": str(coordinates.get("category", "surface")),
                    "eta_minutes": int(eta_minutes),
                    "payload_kg": float(flight.payload_mass_kg or 0.0),
                    "manifest_ref": manifest_ref,
                    "key_materials": materials,
                    "ai_decision": str(ai_decision or "Monitoreo nominal"),
                    "ai_decision_timestamp": decision_timestamp,
                    "material_spectrum": dossier.get("spectrum"),
                    "material_density": float(dossier.get("density", 1.6) or 0.0),
                    "compatibility_index": float(dossier.get("compatibility", 0.7) or 0.0),
                    "decision_changed": False,
                }
            )

        flights_df = pd.DataFrame(rows)
        if flights_df.empty:
            return flights_df

        flights_df = self._apply_status_tokens(flights_df)
        category_colors = flights_df["capsule_category"].apply(self._category_color)
        flights_df[["category_color_r", "category_color_g", "category_color_b"]] = pd.DataFrame(
            category_colors.tolist(), index=flights_df.index
        )
        flights_df["key_materials_display"] = flights_df["key_materials"].apply(
            lambda items: ", ".join(items)
            if isinstance(items, Sequence) and not isinstance(items, (str, bytes))
            else str(items)
        )
        flights_df["materials_tooltip"] = flights_df["key_materials"].apply(
            lambda items: "<br/>".join(html.escape(str(item)) for item in items)
            if isinstance(items, Sequence) and not isinstance(items, (str, bytes))
            else html.escape(str(items))
        )
        flights_df["marker_radius_m"] = flights_df["payload_kg"].apply(
            lambda mass: float(max(600.0, min(mass * 3.5, 4000.0)))
        )
        flights_df["compatibility_index"] = flights_df["compatibility_index"].clip(lower=0.0, upper=1.0)
        flights_df["decision_indicator"] = ""

        return flights_df

    def flight_radar_snapshot(
        self,
        passport: Mapping[str, Any] | None,
    ) -> pd.DataFrame:
        """Return a dataframe with the current flights tracked by mission control."""

        flights_df = self.flight_operations_overview(passport)
        if flights_df.empty:
            return flights_df

        return flights_df[
            [
                "vehicle",
                "phase",
                "latitude",
                "longitude",
                "altitude_km",
                "eta_minutes",
                "payload_kg",
            ]
        ]

    def build_map_payload(
        self,
        flights_df: pd.DataFrame,
        *,
        include_geometry: bool = True,
        include_slope: bool = False,
        include_ortho: bool = False,
        slope_opacity: float | None = None,
        ortho_opacity: float | None = None,
    ) -> dict[str, Any]:
        bitmap = load_jezero_bitmap()
        raw_bounds = bitmap.get("bounds") if isinstance(bitmap, Mapping) else None
        map_bounds = tuple(raw_bounds) if raw_bounds else None
        geometry = load_jezero_geodata() if include_geometry else None

        overlays: dict[str, MutableMapping[str, Any]] = {}
        if include_slope:
            slope_layer = deepcopy(load_jezero_slope_bitmap())
            slope_meta = slope_layer.get("metadata", {}) if isinstance(slope_layer, Mapping) else {}
            default_opacity = float(slope_meta.get("default_opacity", 0.65))
            resolved_opacity = slope_opacity if slope_opacity is not None else default_opacity
            slope_layer["opacity"] = float(max(0.0, min(resolved_opacity, 1.0)))
            slope_layer["label"] = slope_meta.get("label", "Mapa de pendientes")
            overlays["slope_layer"] = slope_layer

        if include_ortho:
            ortho_layer = deepcopy(load_jezero_ortho_bitmap())
            ortho_meta = ortho_layer.get("metadata", {}) if isinstance(ortho_layer, Mapping) else {}
            default_opacity = float(ortho_meta.get("default_opacity", 0.75))
            resolved_opacity = ortho_opacity if ortho_opacity is not None else default_opacity
            ortho_layer["opacity"] = float(max(0.0, min(resolved_opacity, 1.0)))
            ortho_layer["label"] = ortho_meta.get("label", "Ortofoto HiRISE")
            overlays["ortho_layer"] = ortho_layer

        active_overlay_labels = [
            layer.get("label")
            for layer in overlays.values()
            if isinstance(layer, Mapping) and layer.get("label")
        ]

        capsules = pd.DataFrame(columns=[])
        if isinstance(flights_df, pd.DataFrame) and not flights_df.empty:
            capsules = flights_df[
                [
                    "flight_id",
                    "vehicle",
                    "capsule_id",
                    "latitude",
                    "longitude",
                    "altitude_km",
                    "marker_radius_m",
                    "category_color_r",
                    "category_color_g",
                    "category_color_b",
                    "status_color_r",
                    "status_color_g",
                    "status_color_b",
                    "status_label",
                    "status_badge",
                    "eta_minutes",
                    "key_materials_display",
                    "materials_tooltip",
                    "material_spectrum",
                    "material_density",
                    "compatibility_index",
                ]
            ].copy()
            capsules.rename(
                columns={
                    "status_label": "status",
                    "status_badge": "status_badge",
                    "key_materials_display": "materials_display",
                    "material_density": "density",
                    "compatibility_index": "compatibility",
                },
                inplace=True,
            )

        zones = pd.DataFrame(_ZONE_DATA)
        if not zones.empty:
            zone_colors = zones["category"].apply(self._category_color)
            zones[["color_r", "color_g", "color_b"]] = pd.DataFrame(
                zone_colors.tolist(), index=zones.index
            )
            zones["tooltip"] = zones.apply(
                lambda row: (
                    f"<b>{html.escape(str(row['name']))}</b><br/>"
                    f"Espectro: {html.escape(str(row['spectrum']))}<br/>"
                    f"Densidad: {row['density']:.2f} g/cm³<br/>"
                    f"Compatibilidad: {row['compatibility']:.2f}"
                ),
                axis=1,
            )
            zones["vehicle"] = zones["name"]
            zones["status"] = zones["category"].str.replace("_", " ").str.title()
            zones["eta_minutes"] = ""
            zones["materials_tooltip"] = zones["tooltip"]
            zones["material_spectrum"] = zones["spectrum"]
            zones["density"] = zones["density"].astype(float)
            zones["compatibility"] = zones["compatibility"].astype(float)

        return {
            "geometry": geometry,
            "capsules": capsules,
            "zones": zones,
            "bitmap_layer": bitmap,
            "slope_layer": overlays.get("slope_layer"),
            "ortho_layer": overlays.get("ortho_layer"),
            "active_overlay_labels": active_overlay_labels,
            "map_bounds": map_bounds,
            "map_center": bitmap.get("center") if isinstance(bitmap, Mapping) else None,
            "map_view_state": self._view_state_from_bounds(map_bounds, bitmap),
        }

    @staticmethod
    def _view_state_from_bounds(
        bounds: tuple[float, float, float, float] | None,
        bitmap: Mapping[str, Any] | None,
    ) -> dict[str, float]:
        default_state: dict[str, float] = {
            "latitude": 18.43,
            "longitude": 77.58,
            "zoom": 9.1,
            "pitch": 45.0,
            "bearing": 25.0,
        }

        if not bounds:
            return default_state

        min_lon, min_lat, max_lon, max_lat = bounds
        center_lon = (min_lon + max_lon) / 2.0
        center_lat = (min_lat + max_lat) / 2.0
        span_lon = max_lon - min_lon
        span_lat = max_lat - min_lat
        max_span = max(span_lon, span_lat, 1e-6)
        zoom = math.log2(360.0 / max_span)
        zoom = max(6.0, min(zoom, 16.0))

        view_state: dict[str, float] = {
            **default_state,
            "latitude": center_lat,
            "longitude": center_lon,
            "zoom": zoom,
        }

        if isinstance(bitmap, Mapping):
            metadata = bitmap.get("metadata")
            if isinstance(metadata, Mapping):
                width = metadata.get("width_px")
                height = metadata.get("height_px")
                if width and height:
                    view_state["max_zoom"] = max(zoom + 4.0, view_state.get("max_zoom", zoom + 4.0))
                    view_state["min_zoom"] = min(zoom - 2.0, view_state.get("min_zoom", zoom - 2.0))

        return view_state

    # ------------------------------------------------------------------
    # Inventory telemetry
    # ------------------------------------------------------------------

    def inventory_snapshot(self) -> tuple[pd.DataFrame, dict[str, float], dict[str, Any]]:
        """Return the latest inventory dataframe with aggregated metrics."""

        inventory_df = load_inventory_overview()
        metrics = compute_mission_summary(inventory_df)
        category_payload = aggregate_inventory_by_category(inventory_df)
        return inventory_df, metrics, category_payload

    # ------------------------------------------------------------------
    # Timeline & simulation synchronisation
    # ------------------------------------------------------------------

    def jezero_geometry(self, *, refresh: bool = False) -> dict[str, Any]:
        return load_jezero_geodata(refresh=refresh)

    def advance_timeline(
        self,
        flights_df: pd.DataFrame,
        *,
        manifest_df: pd.DataFrame | None = None,
        analysis_state: Mapping[str, Any] | None = None,
        previous_decisions: Mapping[str, str] | None = None,
        session: MutableMapping[str, Any] | None = None,
    ) -> tuple[pd.DataFrame, list[dict[str, Any]], set[str]]:
        if flights_df is None or flights_df.empty:
            events = apply_simulation_tick(session=session)
            return flights_df, [self._event_payload(event) for event in events], set()

        updated = flights_df.copy()
        updated["decision_changed"] = False
        logistics = self._ensure_logistics()
        order_index = {order.order_id: order for order in logistics.ai_orders}

        try:
            inventory_df = load_inventory_overview()
        except Exception:
            inventory_df = None

        events = apply_simulation_tick(
            {"inventory": inventory_df} if inventory_df is not None else None,
            session=session,
        )

        for idx, row in updated.iterrows():
            status = str(row.get("status", "")).lower()
            if status not in {"landed", "completed"}:
                new_eta = max(int(row.get("eta_minutes", 0)) - TICK_MINUTES, 0)
                updated.at[idx, "eta_minutes"] = new_eta
                if new_eta == 0 and status != "landed":
                    updated.at[idx, "status"] = "landed"

        changed_flights: set[str] = set()
        for event in events:
            if event.category == "orders":
                reference = None
                if isinstance(event.metadata, Mapping):
                    reference = event.metadata.get("reference") or event.metadata.get("target")
                order = order_index.get(reference) if reference else None
                manifest_ref = order.target if order and order.target else reference
                directive = order.directive if order else event.title
                if manifest_ref:
                    mask = updated["manifest_ref"] == manifest_ref
                    if mask.any():
                        updated.loc[mask, "ai_decision"] = directive
                        updated.loc[mask, "ai_decision_timestamp"] = datetime.now(timezone.utc).isoformat()
                        changed_flights.update(updated.loc[mask, "flight_id"].tolist())
            elif event.category == "inbound":
                capsule_id = event.capsule_id
                if not capsule_id and isinstance(event.metadata, Mapping):
                    capsule_id = event.metadata.get("capsule_id")
                if capsule_id:
                    mask = updated["capsule_id"] == capsule_id
                    if mask.any():
                        updated.loc[mask, "status"] = "landed"
                        updated.loc[mask, "eta_minutes"] = 0

        updated = self._apply_status_tokens(updated)
        updated["marker_radius_m"] = updated["payload_kg"].apply(
            lambda mass: float(max(600.0, min(mass * 3.5, 4000.0)))
        )

        if previous_decisions:
            for idx, row in updated.iterrows():
                flight_id = row.get("flight_id")
                if not flight_id:
                    continue
                previous = previous_decisions.get(flight_id)
                if previous is not None and str(previous) != str(row.get("ai_decision")):
                    changed_flights.add(flight_id)

        updated["decision_changed"] = updated["flight_id"].isin(changed_flights)
        updated["decision_indicator"] = updated["decision_changed"].apply(
            lambda flag: "⚡ Cambio" if flag else ""
        )

        if manifest_df is not None or analysis_state is not None:
            # Refresh dossiers if new manifest data is available
            dossier_index = self._build_material_dossiers(manifest_df, analysis_state)
            material_index = self._build_manifest_material_index(manifest_df, analysis_state)
            updated["material_spectrum"] = updated["manifest_ref"].apply(
                lambda ref: dossier_index.get(ref, dossier_index.get("default", {})).get("spectrum")
            )
            updated["material_density"] = updated["manifest_ref"].apply(
                lambda ref: float(
                    dossier_index.get(ref, dossier_index.get("default", {"density": 1.6})).get("density", 1.6)
                )
            )
            updated["compatibility_index"] = updated["manifest_ref"].apply(
                lambda ref: float(
                    dossier_index.get(ref, dossier_index.get("default", {"compatibility": 0.7})).get(
                        "compatibility", 0.7
                    )
                )
            ).clip(lower=0.0, upper=1.0)
            updated["key_materials"] = updated["manifest_ref"].apply(
                lambda ref: material_index.get(
                    ref,
                    material_index.get("manifest-alpha", material_index.get("default", [])),
                )
            )
            updated["key_materials_display"] = updated["key_materials"].apply(
                lambda items: ", ".join(items)
                if isinstance(items, Sequence) and not isinstance(items, (str, bytes))
                else str(items)
            )
            updated["materials_tooltip"] = updated["key_materials"].apply(
                lambda items: "<br/>".join(html.escape(str(item)) for item in items)
                if isinstance(items, Sequence) and not isinstance(items, (str, bytes))
                else html.escape(str(items))
            )

        return updated, [self._event_payload(event) for event in events], changed_flights

    def timeline_history(self, *, since_tick: int | None = None) -> pd.DataFrame:
        events = iterate_events(since_tick=since_tick)
        if not events:
            return pd.DataFrame(
                columns=["tick", "category", "title", "details", "capsule_id", "mass_delta"]
            )

        rows = []
        for event in events:
            payload = self._event_payload(event)
            rows.append(
                {
                    "tick": event.tick,
                    "category": event.category,
                    "title": event.title,
                    "details": event.details,
                    "capsule_id": event.capsule_id,
                    "mass_delta": event.delta_mass_kg,
                    "metadata": {key: value for key, value in payload.items() if key.startswith("metadata_")},
                }
            )
        timeline_df = pd.DataFrame(rows)
        return timeline_df.sort_values("tick", ascending=False).reset_index(drop=True)

    # ------------------------------------------------------------------
    # AI decisions & reporting
    # ------------------------------------------------------------------

    def summarize_decisions(self, analysis_state: Mapping[str, Any] | None) -> dict[str, Any]:
        if not analysis_state:
            return {
                "score": 0.0,
                "item_count": 0,
                "total_mass": 0.0,
                "compatibility": pd.DataFrame(),
                "recommendations": pd.DataFrame(),
                "manifest": pd.DataFrame(),
            }

        passport = analysis_state.get("material_passport") or {}
        summary = {
            "score": float(passport.get("mean_material_utility_score", 0.0) or 0.0),
            "item_count": int(passport.get("total_items", 0) or 0),
            "total_mass": float(passport.get("total_mass_kg", 0.0) or 0.0),
            "compatibility": analysis_state.get("compatibility_records", pd.DataFrame()),
            "recommendations": analysis_state.get("policy_recommendations", pd.DataFrame()),
            "manifest": analysis_state.get("scored_manifest", pd.DataFrame()),
        }
        return summary

    # ------------------------------------------------------------------
    # Planner telemetry
    # ------------------------------------------------------------------

    def _ensure_process_df(self) -> pd.DataFrame:
        if self._process_df is None:
            self._process_df = load_process_df()
        return self._process_df

    def build_planner_schedule(
        self,
        manifest: pd.DataFrame | None,
        *,
        max_items: int = 4,
    ) -> pd.DataFrame:
        """Generate a planning table linking manifest items with processes."""

        if manifest is None or manifest.empty:
            return pd.DataFrame(
                columns=["item", "category", "process_id", "match_score", "match_reason"]
            )

        process_df = self._ensure_process_df()
        manifest_copy = manifest.copy()

        if "mass_kg" in manifest_copy.columns:
            manifest_copy["mass_kg"] = pd.to_numeric(
                manifest_copy["mass_kg"], errors="coerce"
            ).fillna(0.0)
            manifest_copy = manifest_copy.sort_values("mass_kg", ascending=False)

        top_manifest = manifest_copy.head(max_items)

        rows: list[dict[str, Any]] = []
        for _, row in top_manifest.iterrows():
            item_label = str(row.get("item") or row.get("material_key") or "")
            category = str(row.get("category") or "")
            planner_df = choose_process(item_label, process_df)
            if planner_df.empty:
                continue
            for _, proc in planner_df.head(3).iterrows():
                rows.append(
                    {
                        "item": item_label,
                        "category": category,
                        "process_id": proc.get("process_id"),
                        "match_score": float(proc.get("match_score", 0.0) or 0.0),
                        "match_reason": proc.get("match_reason", ""),
                    }
                )

        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Demo playbooks
    # ------------------------------------------------------------------

    def demo_script(self) -> list[dict[str, Any]]:
        """Return a lightweight script used in demo mode."""

        now = datetime.now(timezone.utc)
        return [
            {
                "timestamp": now.strftime("%H:%M UTC"),
                "action": "Sincronizar telemetría de vuelos",
                "notes": "Revisá si Ares Cargo 7 ya finalizó la inserción polar.",
            },
            {
                "timestamp": (now + timedelta(minutes=7)).strftime("%H:%M UTC"),
                "action": "Priorizar manifiesto crítico",
                "notes": "Analizá primero los ítems con penalización alta (penalty_factor > 1.1).",
            },
            {
                "timestamp": (now + timedelta(minutes=18)).strftime("%H:%M UTC"),
                "action": "Coordinar con logística",
                "notes": "Cruce los procesos sugeridos con la disponibilidad de crew en Jezero.",
            },
        ]


def summarize_artifacts(analysis_state: Mapping[str, Any] | None) -> Mapping[str, Any]:
    """Expose artifact paths while keeping a consistent structure."""

    if not analysis_state:
        return {}
    artifacts = analysis_state.get("artifacts", {})
    if not isinstance(artifacts, Mapping):
        return {}
    return artifacts


__all__ = [
    "FlightTelemetry",
    "MarsControlCenterService",
    "summarize_artifacts",
]
