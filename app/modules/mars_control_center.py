"""Telemetry services backing the Mars Control Center page."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Mapping

import pandas as pd

from app.modules.io import load_process_df
from app.modules.mission_overview import compute_mission_summary, load_inventory_overview
from app.modules.process_planner import choose_process


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

    # ------------------------------------------------------------------
    # Flight radar & logistics map
    # ------------------------------------------------------------------

    def _resolve_payload_mass(self, passport: Mapping[str, Any] | None) -> float:
        if not passport:
            return 0.0
        try:
            return float(passport.get("total_mass_kg", 0.0))
        except (TypeError, ValueError):
            return 0.0

    def flight_radar_snapshot(
        self,
        passport: Mapping[str, Any] | None,
    ) -> pd.DataFrame:
        """Return a dataframe with the current flights tracked by mission control."""

        total_mass = self._resolve_payload_mass(passport)
        mass_chunks = [0.45, 0.35, 0.2]
        payloads = [round(total_mass * share, 1) for share in mass_chunks]

        vehicles: list[FlightTelemetry] = [
            FlightTelemetry(
                vehicle="Ares Cargo 7",
                phase="Orbita polar marciana",
                latitude=-4.5895,
                longitude=137.4417,
                altitude_km=420.0,
                eta_minutes=18,
                payload_kg=payloads[0],
            ),
            FlightTelemetry(
                vehicle="Skylift 3",
                phase="Trans-mars injection",
                latitude=2.08,
                longitude=-1.52,
                altitude_km=1290.0,
                eta_minutes=96,
                payload_kg=payloads[1],
            ),
            FlightTelemetry(
                vehicle="Dusthopper",
                phase="Ruta base Jezero",
                latitude=18.38,
                longitude=77.58,
                altitude_km=0.12,
                eta_minutes=42,
                payload_kg=payloads[2],
            ),
        ]

        return pd.DataFrame(vehicles)

    # ------------------------------------------------------------------
    # Inventory telemetry
    # ------------------------------------------------------------------

    def inventory_snapshot(self) -> tuple[pd.DataFrame, dict[str, float]]:
        """Return the latest inventory dataframe with aggregated metrics."""

        inventory_df = load_inventory_overview()
        metrics = compute_mission_summary(inventory_df)
        return inventory_df, metrics

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
