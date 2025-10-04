"""Mars logistics orchestration helpers.

The module centralises data structures and reusable helpers used by the Mars
Control Center dashboard.  Public API:

Dataclasses
===========
* :class:`CapsuleSpec`
* :class:`FlightPlan`
* :class:`ProcessSpec`
* :class:`AIOrder`
* :class:`MarsLogisticsData`
* :class:`SimulationEvent`

Utilities
=========
* :func:`load_logistics_baseline` – load curated datasets describing the
  logistics backbone of the mission from ``DATA_ROOT``.
* :func:`load_live_inventory` – thin wrapper that exposes the mission inventory
  with the same semantics as :func:`mission_overview.load_inventory_overview`.
* :func:`compute_mission_summary` – aggregate inventory and logistics metrics to
  feed the dashboard KPIs.
* :func:`score_manifest_batch` – evaluate one or more manifests using
  :class:`~app.modules.generator.service.GeneratorService` and transform the
  results into UI-friendly payloads.
* :func:`summarise_policy_actions` – reduce a policy recommendation dataframe
  into compact statistics for quick rendering.
* :func:`iterate_events` and :func:`apply_simulation_tick` – simple simulation
  engine that synthesises logistics events while keeping state in
  ``st.session_state`` to ensure idempotent execution.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, MutableMapping, Sequence
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from datetime import datetime
import math
from pathlib import Path
from typing import Any, Final

import pandas as pd
import streamlit as st
import yaml

from app.modules import mission_overview
from app.modules.generator import GeneratorService
from app.modules.paths import DATA_ROOT


_DATASET_FILENAME: Final[str] = "mars_logistics.yaml"
_SIM_STATE_KEY: Final[str] = "mars_control_simulation"
_SIM_EVENTS_HISTORY_KEY: Final[str] = "events"
_SIM_TICK_KEY: Final[str] = "tick"
_SIM_LAST_GENERATED_KEY: Final[str] = "last_generated_tick"


@dataclass(slots=True, frozen=True)
class CapsuleSpec:
    """Physical specification of a reusable cargo capsule."""

    capsule_id: str
    name: str
    capacity_kg: float
    status: str
    location: str
    notes: str | None = None


@dataclass(slots=True, frozen=True)
class FlightPlan:
    """Scheduled or in-flight cargo movement."""

    flight_id: str
    capsule_id: str
    origin: str
    destination: str
    departure: datetime | None
    arrival: datetime | None
    status: str
    payload_mass_kg: float
    manifest_ref: str | None = None


@dataclass(slots=True, frozen=True)
class ProcessSpec:
    """Industrial process available at the Mars base."""

    process_id: str
    name: str
    throughput_kg_per_day: float
    energy_kwh_per_kg: float
    crew_hours_per_day: float
    constraint: str | None = None


@dataclass(slots=True, frozen=True)
class AIOrder:
    """Directive emitted by the autonomous logistics assistant."""

    order_id: str
    issued_at: datetime | None
    priority: str
    directive: str
    target: str | None = None
    context: str | None = None


@dataclass(slots=True)
class MarsLogisticsData:
    """Bundled dataset describing the current logistics baseline."""

    flights: list[FlightPlan] = field(default_factory=list)
    capsules: list[CapsuleSpec] = field(default_factory=list)
    processes: list[ProcessSpec] = field(default_factory=list)
    ai_orders: list[AIOrder] = field(default_factory=list)
    event_templates: dict[str, list[dict[str, Any]]] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        """Serialize the logistics data into plain Python primitives."""

        return {
            "flights": [asdict(flight) for flight in self.flights],
            "capsules": [asdict(capsule) for capsule in self.capsules],
            "processes": [asdict(process) for process in self.processes],
            "ai_orders": [asdict(order) for order in self.ai_orders],
            "event_templates": deepcopy(self.event_templates),
        }


@dataclass(slots=True, frozen=True)
class SimulationEvent:
    """Synthetic event emitted by the logistics simulator."""

    tick: int
    category: str
    title: str
    details: str
    delta_mass_kg: float = 0.0
    capsule_id: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "tick": self.tick,
            "category": self.category,
            "title": self.title,
            "details": self.details,
            "delta_mass_kg": self.delta_mass_kg,
            "capsule_id": self.capsule_id,
        }
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload


_DEFAULT_DATASET: Final[dict[str, Any]] = {
    "flights": [
        {
            "flight_id": "MC-042",
            "capsule_id": "ares_cargo_7",
            "origin": "Orbital Hub Phobos",
            "destination": "Base Jezero",
            "departure": "2043-04-12T06:30:00Z",
            "arrival": "2043-04-12T18:05:00Z",
            "status": "en_route",
            "payload_mass_kg": 1240,
            "manifest_ref": "manifest-alpha",
        }
    ],
    "capsules": [
        {
            "capsule_id": "ares_cargo_7",
            "name": "Ares Cargo 7",
            "capacity_kg": 1500,
            "status": "active",
            "location": "Orbit",
        }
    ],
    "processes": [
        {
            "process_id": "regolith_refinery",
            "name": "Refinería de regolito",
            "throughput_kg_per_day": 48,
            "energy_kwh_per_kg": 0.9,
            "crew_hours_per_day": 6,
        }
    ],
    "orders": [
        {
            "order_id": "ia-ops-01",
            "issued_at": "2043-04-12T05:00:00Z",
            "priority": "high",
            "directive": "Priorizar manifiesto crítico",
            "target": "manifest-alpha",
        }
    ],
    "event_templates": {
        "inbound": [
            {
                "title": "Arribo parcial",
                "description": "Carga recibida en la bahía principal.",
                "delta_mass_kg": 120,
                "capsule_id": "ares_cargo_7",
            }
        ],
        "recycling": [
            {
                "title": "Ciclo de reciclaje",
                "description": "Se inicia ciclo estándar.",
                "delta_mass_kg": -40,
                "capsule_id": "ares_cargo_7",
            }
        ],
        "orders": [
            {
                "title": "Órden IA",
                "description": "Revisar sustituciones prioritarias.",
                "reference": "ia-ops-01",
            }
        ],
    },
}


def _read_dataset() -> dict[str, Any]:
    payload = deepcopy(_DEFAULT_DATASET)
    dataset_path = DATA_ROOT / _DATASET_FILENAME
    if dataset_path.is_file():
        try:
            with dataset_path.open("r", encoding="utf-8") as handle:
                loaded = yaml.safe_load(handle) or {}
        except yaml.YAMLError as exc:  # pragma: no cover - defensive logging
            st.warning(f"No se pudo parsear {dataset_path.name}: {exc}")
            loaded = {}
        if isinstance(loaded, Mapping):
            for key, value in loaded.items():
                payload[key] = value
    return payload


def _coerce_datetime(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        candidate = value.strip()
        if not candidate:
            return None
        if candidate.endswith("Z"):
            candidate = candidate[:-1] + "+00:00"
        try:
            return datetime.fromisoformat(candidate)
        except ValueError:
            return None
    return None


def _coerce_float(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    if math.isnan(number):
        return 0.0
    return number


def _build_capsules(raw_capsules: Sequence[Mapping[str, Any]]) -> list[CapsuleSpec]:
    capsules: list[CapsuleSpec] = []
    for entry in raw_capsules:
        capsule_id = str(entry.get("capsule_id", "")).strip()
        if not capsule_id:
            continue
        capsules.append(
            CapsuleSpec(
                capsule_id=capsule_id,
                name=str(entry.get("name", capsule_id)).strip() or capsule_id,
                capacity_kg=_coerce_float(entry.get("capacity_kg")),
                status=str(entry.get("status", "unknown")).strip(),
                location=str(entry.get("location", "unknown")).strip(),
                notes=(str(entry["notes"]).strip() if entry.get("notes") else None),
            )
        )
    return capsules


def _build_flights(raw_flights: Sequence[Mapping[str, Any]]) -> list[FlightPlan]:
    flights: list[FlightPlan] = []
    for entry in raw_flights:
        flight_id = str(entry.get("flight_id", "")).strip()
        capsule_id = str(entry.get("capsule_id", "")).strip()
        if not flight_id or not capsule_id:
            continue
        flights.append(
            FlightPlan(
                flight_id=flight_id,
                capsule_id=capsule_id,
                origin=str(entry.get("origin", "")).strip(),
                destination=str(entry.get("destination", "")).strip(),
                departure=_coerce_datetime(entry.get("departure")),
                arrival=_coerce_datetime(entry.get("arrival")),
                status=str(entry.get("status", "unknown")).strip(),
                payload_mass_kg=_coerce_float(entry.get("payload_mass_kg")),
                manifest_ref=(str(entry["manifest_ref"]).strip() if entry.get("manifest_ref") else None),
            )
        )
    return flights


def _build_processes(raw_processes: Sequence[Mapping[str, Any]]) -> list[ProcessSpec]:
    processes: list[ProcessSpec] = []
    for entry in raw_processes:
        process_id = str(entry.get("process_id", "")).strip()
        if not process_id:
            continue
        processes.append(
            ProcessSpec(
                process_id=process_id,
                name=str(entry.get("name", process_id)).strip() or process_id,
                throughput_kg_per_day=_coerce_float(entry.get("throughput_kg_per_day")),
                energy_kwh_per_kg=_coerce_float(entry.get("energy_kwh_per_kg")),
                crew_hours_per_day=_coerce_float(entry.get("crew_hours_per_day")),
                constraint=(str(entry["constraint"]).strip() if entry.get("constraint") else None),
            )
        )
    return processes


def _build_orders(raw_orders: Sequence[Mapping[str, Any]]) -> list[AIOrder]:
    orders: list[AIOrder] = []
    for entry in raw_orders:
        order_id = str(entry.get("order_id", "")).strip()
        if not order_id:
            continue
        orders.append(
            AIOrder(
                order_id=order_id,
                issued_at=_coerce_datetime(entry.get("issued_at")),
                priority=str(entry.get("priority", "normal")).strip(),
                directive=str(entry.get("directive", "")).strip(),
                target=(str(entry["target"]).strip() if entry.get("target") else None),
                context=(str(entry["context"]).strip() if entry.get("context") else None),
            )
        )
    return orders


_BASELINE_CACHE: MarsLogisticsData | None = None


def load_logistics_baseline(*, refresh: bool = False) -> MarsLogisticsData:
    """Return the curated Mars logistics dataset."""

    global _BASELINE_CACHE
    if _BASELINE_CACHE is not None and not refresh:
        return _BASELINE_CACHE

    raw = _read_dataset()
    flights = _build_flights(raw.get("flights", []))
    capsules = _build_capsules(raw.get("capsules", []))
    processes = _build_processes(raw.get("processes", []))
    ai_orders = _build_orders(raw.get("orders", []))
    event_templates = raw.get("event_templates", {})
    if not isinstance(event_templates, Mapping):
        event_templates = {}
    normalized_templates: dict[str, list[dict[str, Any]]] = {}
    for category, entries in event_templates.items():
        if not isinstance(entries, Sequence):
            continue
        normalized: list[dict[str, Any]] = []
        for entry in entries:
            if not isinstance(entry, Mapping):
                continue
            normalized.append({str(key): value for key, value in entry.items()})
        normalized_templates[str(category)] = normalized

    _BASELINE_CACHE = MarsLogisticsData(
        flights=flights,
        capsules=capsules,
        processes=processes,
        ai_orders=ai_orders,
        event_templates=normalized_templates,
    )
    return _BASELINE_CACHE


def load_live_inventory() -> pd.DataFrame:
    """Expose the latest enriched inventory dataframe."""

    return mission_overview.load_inventory_overview()


def compute_mission_summary(
    inventory: pd.DataFrame | None = None,
    *,
    logistics: MarsLogisticsData | None = None,
) -> dict[str, float]:
    """Aggregate mission KPIs combining inventory and logistics data."""

    if inventory is None:
        inventory = load_live_inventory()
    base_metrics = mission_overview.compute_mission_summary(inventory)

    logistics = logistics or load_logistics_baseline()
    capsule_by_id = {capsule.capsule_id: capsule for capsule in logistics.capsules}
    active_flights = [
        flight
        for flight in logistics.flights
        if flight.status.lower() in {"en_route", "boarding", "launch-ready", "ready"}
    ]
    available_capacity = 0.0
    for flight in active_flights:
        capsule = capsule_by_id.get(flight.capsule_id)
        if capsule is None:
            continue
        available_capacity += max(capsule.capacity_kg - flight.payload_mass_kg, 0.0)

    high_priority_orders = sum(1 for order in logistics.ai_orders if order.priority.lower() == "high")
    total_throughput = sum(process.throughput_kg_per_day for process in logistics.processes)

    base_metrics.update(
        {
            "active_flights": float(len(active_flights)),
            "available_capacity_kg": float(available_capacity),
            "high_priority_orders": float(high_priority_orders),
            "process_throughput_kg_per_day": float(total_throughput),
        }
    )
    return base_metrics


def _resolve_manifest_label(manifest: Any, index: int) -> str:
    if isinstance(manifest, pd.DataFrame) and manifest.attrs.get("name"):
        return str(manifest.attrs["name"])
    if isinstance(manifest, Mapping) and "name" in manifest:
        return str(manifest.get("name"))
    return f"batch-{index + 1}"


def _safe_numeric(series: pd.Series | None) -> pd.Series:
    if series is None:
        return pd.Series(dtype=float)
    return pd.to_numeric(series, errors="coerce").fillna(0.0)


def summarise_policy_actions(policy_df: pd.DataFrame | None) -> dict[str, Any]:
    """Reduce the policy recommendation dataframe into dashboard friendly stats."""

    if policy_df is None or policy_df.empty:
        return {
            "total_actions": 0,
            "actions_by_type": {},
            "mean_score_gain": 0.0,
            "total_quota": 0.0,
            "top_actions": [],
        }

    normalized = policy_df.copy()
    normalized["action"] = normalized.get("action", "unknown").fillna("unknown")
    normalized["current_score"] = _safe_numeric(normalized.get("current_score"))
    normalized["recommended_score"] = _safe_numeric(normalized.get("recommended_score"))
    normalized["recommended_quota"] = _safe_numeric(normalized.get("recommended_quota"))
    normalized["score_gain"] = normalized["recommended_score"] - normalized["current_score"]

    action_counts = (
        normalized["action"].astype(str).str.lower().value_counts().to_dict()
    )
    mean_gain = float(normalized["score_gain"].mean() or 0.0)
    total_quota = float(normalized["recommended_quota"].sum() or 0.0)

    top_slice = (
        normalized.sort_values(["score_gain", "recommended_score"], ascending=False)
        .head(5)
        .to_dict(orient="records")
    )
    top_actions = [
        {
            "item": entry.get("item_name") or entry.get("current_material_key"),
            "recommended_material": entry.get("recommended_material_key"),
            "score_gain": float(entry.get("score_gain", 0.0) or 0.0),
            "quota": float(entry.get("recommended_quota", 0.0) or 0.0),
            "justification": entry.get("justification", ""),
        }
        for entry in top_slice
    ]

    return {
        "total_actions": int(len(normalized)),
        "actions_by_type": action_counts,
        "mean_score_gain": mean_gain,
        "total_quota": total_quota,
        "top_actions": top_actions,
    }


def score_manifest_batch(
    service: GeneratorService,
    manifest_batch: Iterable[
        pd.DataFrame | Mapping[str, Sequence[object]] | Sequence[Mapping[str, object]] | str | Path
    ],
    *,
    include_analysis: bool = False,
) -> list[dict[str, Any]]:
    """Evaluate *manifest_batch* and return serialisable summaries."""

    results: list[dict[str, Any]] = []
    for index, manifest in enumerate(manifest_batch):
        analysis = service.analyze_manifest(manifest)
        scored_manifest = analysis.get("scored_manifest", pd.DataFrame())
        policy_df = analysis.get("policy_recommendations", pd.DataFrame())
        passport = analysis.get("material_passport") or {}

        total_mass = float(passport.get("total_mass_kg") or 0.0)
        if not total_mass and isinstance(scored_manifest, pd.DataFrame):
            total_mass = float(_safe_numeric(scored_manifest.get("mass_kg")).sum())

        mean_score = float(passport.get("mean_material_utility_score") or 0.0)
        if (not mean_score or math.isnan(mean_score)) and isinstance(scored_manifest, pd.DataFrame):
            scores = _safe_numeric(scored_manifest.get("material_utility_score"))
            if not scores.empty:
                mean_score = float(scores.mean())
            else:
                mean_score = 0.0

        item_count = int(passport.get("total_items") or 0)
        if item_count <= 0 and isinstance(scored_manifest, pd.DataFrame):
            item_count = int(scored_manifest.shape[0])

        summary = {
            "label": _resolve_manifest_label(manifest, index),
            "item_count": item_count,
            "total_mass_kg": total_mass,
            "mean_material_score": mean_score,
            "policy_summary": summarise_policy_actions(policy_df),
        }

        payload: dict[str, Any] = {
            "summary": summary,
            "manifest": analysis.get("manifest", pd.DataFrame()),
            "scored_manifest": scored_manifest,
            "policy_recommendations": policy_df,
            "compatibility": analysis.get("compatibility_records", pd.DataFrame()),
            "material_passport": passport,
        }
        if include_analysis:
            payload["analysis"] = analysis
        results.append(payload)

    return results


def _ensure_sim_state(session: MutableMapping[str, Any]) -> MutableMapping[str, Any]:
    if _SIM_STATE_KEY not in session:
        session[_SIM_STATE_KEY] = {
            _SIM_TICK_KEY: 0,
            _SIM_LAST_GENERATED_KEY: 0,
            _SIM_EVENTS_HISTORY_KEY: [],
        }
    return session[_SIM_STATE_KEY]


def _event_from_record(record: Mapping[str, Any]) -> SimulationEvent:
    metadata = record.get("metadata")
    if not isinstance(metadata, Mapping):
        metadata = {}
    return SimulationEvent(
        tick=int(record.get("tick", 0) or 0),
        category=str(record.get("category", "")).strip(),
        title=str(record.get("title", "")).strip(),
        details=str(record.get("details", "")).strip(),
        delta_mass_kg=_coerce_float(record.get("delta_mass_kg")),
        capsule_id=(str(record["capsule_id"]).strip() if record.get("capsule_id") else None),
        metadata=metadata,
    )


def iterate_events(
    *,
    since_tick: int | None = None,
    session: MutableMapping[str, Any] | None = None,
) -> list[SimulationEvent]:
    """Return previously generated simulation events from session state."""

    session = session or st.session_state
    state = _ensure_sim_state(session)
    history = state.get(_SIM_EVENTS_HISTORY_KEY, [])
    events: list[SimulationEvent] = []
    for record in history:
        if not isinstance(record, Mapping):
            continue
        event = _event_from_record(record)
        if since_tick is not None and event.tick <= since_tick:
            continue
        events.append(event)
    return events


def _synthesise_event(
    category: str,
    template: Mapping[str, Any],
    *,
    tick: int,
    summary: Mapping[str, Any],
) -> SimulationEvent:
    base_metadata: dict[str, Any] = {}
    for key, value in template.items():
        if key in {"title", "description", "delta_mass_kg", "capsule_id"}:
            continue
        base_metadata[key] = value

    delta_mass = _coerce_float(template.get("delta_mass_kg"))
    if delta_mass:
        base_metadata["inventory_mass_kg"] = float(summary.get("mass_kg", 0.0)) + delta_mass
        base_metadata["mass_delta"] = delta_mass

    return SimulationEvent(
        tick=tick,
        category=category,
        title=str(template.get("title", "Evento logístico")).strip(),
        details=str(template.get("description", "")).strip(),
        delta_mass_kg=delta_mass,
        capsule_id=(str(template["capsule_id"]).strip() if template.get("capsule_id") else None),
        metadata=base_metadata,
    )


def apply_simulation_tick(
    payload: Mapping[str, Any] | None = None,
    *,
    session: MutableMapping[str, Any] | None = None,
) -> list[SimulationEvent]:
    """Advance the simulation by one tick and persist the generated events."""

    session = session or st.session_state
    state = _ensure_sim_state(session)

    if payload and payload.get("reset"):
        state[_SIM_TICK_KEY] = 0
        state[_SIM_LAST_GENERATED_KEY] = 0
        state[_SIM_EVENTS_HISTORY_KEY] = []

    next_tick = state[_SIM_TICK_KEY] + 1
    if state.get(_SIM_LAST_GENERATED_KEY) == next_tick:
        # Idempotent behaviour: return the previously generated events for this tick.
        return [
            event
            for event in iterate_events(since_tick=next_tick - 1, session=session)
        ]

    logistics = load_logistics_baseline()
    inventory = None
    if payload and isinstance(payload.get("inventory"), pd.DataFrame):
        inventory = payload["inventory"]
    summary = compute_mission_summary(inventory, logistics=logistics)

    categories: Sequence[str]
    if payload and isinstance(payload.get("categories"), Sequence) and not isinstance(
        payload.get("categories"), (str, bytes)
    ):
        categories = [str(item) for item in payload["categories"] if isinstance(item, (str, int))]
        if not categories:
            categories = list(logistics.event_templates.keys())
    else:
        categories = list(logistics.event_templates.keys())

    events: list[SimulationEvent] = []
    for category in categories:
        templates = logistics.event_templates.get(category, [])
        if not templates:
            continue
        index = (next_tick - 1) % len(templates)
        template = templates[index]
        event = _synthesise_event(category, template, tick=next_tick, summary=summary)
        events.append(event)

    if payload and isinstance(payload.get("inject_event"), Mapping):
        injected = _event_from_record({"tick": next_tick, **payload["inject_event"]})
        events.append(injected)

    state[_SIM_TICK_KEY] = next_tick
    state[_SIM_LAST_GENERATED_KEY] = next_tick
    history = state.setdefault(_SIM_EVENTS_HISTORY_KEY, [])
    history.extend(event.to_dict() for event in events)

    return events


__all__ = [
    "CapsuleSpec",
    "FlightPlan",
    "ProcessSpec",
    "AIOrder",
    "MarsLogisticsData",
    "SimulationEvent",
    "load_logistics_baseline",
    "load_live_inventory",
    "compute_mission_summary",
    "score_manifest_batch",
    "summarise_policy_actions",
    "iterate_events",
    "apply_simulation_tick",
]
