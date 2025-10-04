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
* :class:`DemoEvent`

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
* :func:`demo_event_script`, :func:`generate_demo_event` and
  :func:`get_demo_event_history` – helpers for the demo control room ticker.
* :func:`demo_manifest_catalogue` and :func:`load_demo_manifest` – provide
  ready-to-use manifests for testing and demos.
"""

from __future__ import annotations

from base64 import b64encode
from collections.abc import Iterable, Mapping, MutableMapping, Sequence
from copy import deepcopy
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
import io
import json
import math
from pathlib import Path
import shutil
import re
from typing import Any, Final
import wave

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
_DEMO_STATE_KEY: Final[str] = "mars_control_demo"
_DEMO_QUEUE_KEY: Final[str] = "queue"
_DEMO_CURSOR_KEY: Final[str] = "cursor"
_DEMO_LAST_TS_KEY: Final[str] = "last_emitted_at"
_DEMO_HISTORY_KEY: Final[str] = "history"
_STATIC_ROOT: Final[Path] = Path(__file__).resolve().parents[1] / "static"
_STATIC_AUDIO_ROOT: Final[Path] = _STATIC_ROOT / "audio"
_STATIC_MARS_ROOT: Final[Path] = _STATIC_ROOT / "mars"
_JEZERO_GEOJSON_PATH: Final[Path] = _STATIC_ROOT / "geodata" / "jezero.geojson"
_DATASETS_ROOT: Final[Path] = Path(__file__).resolve().parents[2] / "datasets"
_MARS_DATASETS_ROOT: Final[Path] = _DATASETS_ROOT / "mars"
_JEZERO_BITMAP_PATH: Final[Path] = _MARS_DATASETS_ROOT / "Katie_1_-_DLR_Jezero_hi_v2.jpg"
_JEZERO_BITMAP_FALLBACK_PATH: Final[Path] = _MARS_DATASETS_ROOT / "8k_mars.jpg"
_JEZERO_SLOPE_BITMAP_PATH: Final[Path] = _MARS_DATASETS_ROOT / "j03_045994_1986_j03_046060_1986_20m_slope_20m-full.jpg"
_JEZERO_ORTHO_BITMAP_PATH: Final[Path] = _MARS_DATASETS_ROOT / "j03_045994_1986_xn_18n282w_6m_ortho-full.jpg"
_MARS_SCENEGRAPH_FILENAME: Final[str] = "24881_Mars_1_6792.glb"
_JEZERO_DEFAULT_BOUNDS: Final[tuple[float, float, float, float]] = (77.18, 18.05, 78.05, 18.86)

_PERCENTAGE_PATTERN: Final[re.Pattern[str]] = re.compile(r"(\d+(?:\.\d+)?)\s*%")

_MATERIAL_GROUP_LABELS: Final[dict[str, str]] = {
    "polimeros": "Polímeros",
    "metales": "Metales",
    "textiles": "Textiles",
    "espumas": "Espumas técnicas",
    "mixtos": "Mixtos",
}

_MATERIAL_GROUP_COLORS: Final[dict[str, str]] = {
    "polimeros": "#0ea5e9",
    "metales": "#f97316",
    "textiles": "#22c55e",
    "espumas": "#a855f7",
    "mixtos": "#64748b",
}

_DESTINATION_INFO: Final[dict[str, dict[str, str]]] = {
    "recycle": {
        "label": "Reciclaje",
        "display": "♻️ Reciclaje",
        "color": "#22c55e",
    },
    "reuse": {
        "label": "Reutilización",
        "display": "🔁 Reutilización",
        "color": "#facc15",
    },
    "stock": {
        "label": "Stock",
        "display": "📦 Stock estratégico",
        "color": "#38bdf8",
    },
}

_DESTINATION_WEIGHTS: Final[dict[str, dict[str, float]]] = {
    "polimeros": {"recycle": 0.7, "reuse": 0.15, "stock": 0.15},
    "metales": {"recycle": 0.2, "reuse": 0.2, "stock": 0.6},
    "textiles": {"recycle": 0.25, "reuse": 0.55, "stock": 0.2},
    "espumas": {"recycle": 0.8, "reuse": 0.05, "stock": 0.15},
    "mixtos": {"recycle": 0.6, "reuse": 0.25, "stock": 0.15},
}

_DEMO_AUDIO_CLIPS: Final[dict[str, list[tuple[float, float, float]]]] = {
    "mission_ping": [
        (620.0, 0.16, 0.5),
        (0.0, 0.04, 0.0),
        (880.0, 0.2, 0.45),
    ],
    "alert_burst": [
        (320.0, 0.12, 0.6),
        (0.0, 0.04, 0.0),
        (520.0, 0.12, 0.65),
        (0.0, 0.04, 0.0),
        (860.0, 0.16, 0.7),
    ],
    "comms_chime": [
        (660.0, 0.18, 0.4),
        (825.0, 0.18, 0.4),
        (990.0, 0.18, 0.38),
    ],
}

_DEMO_AUDIO_CACHE: dict[str, bytes] = {}


_DEMO_EVENT_PLAYLIST: Final[list[dict[str, Any]]] = [
    {
        "event_id": "mission-approach",
        "category": "mission",
        "title": "Ares Cargo 7 acoplando",
        "message": "La cápsula Ares 7 inicia maniobra de acoplamiento sobre Nodo Phobos.",
        "icon": "🛰️",
        "audio_clip": "mission_ping",
        "metadata": {"capsule_id": "ares_cargo_7", "eta_minutes": 12},
    },
    {
        "event_id": "alert-dust-storm",
        "category": "alert",
        "title": "Tormenta de polvo sobre Delta Oeste",
        "message": "Sensores LIDAR registran ráfagas de 95 km/h. Ajustando rutas de superficie.",
        "severity": "critical",
        "icon": "⚠️",
        "audio_clip": "alert_burst",
        "metadata": {"zone": "Delta Oeste", "impact": "surface_ops"},
    },
    {
        "event_id": "voice-briefing",
        "category": "voice",
        "title": "Mensaje de Rex-AI",
        "message": "Tripulación Delta 3, prioricen separación de polímeros fluorados en línea 2.",
        "icon": "🎙️",
        "audio_clip": "comms_chime",
        "metadata": {"call_sign": "Rex-AI", "channel": "ops"},
    },
    {
        "event_id": "mission-recycle",
        "category": "mission",
        "title": "Lote reciclaje listo",
        "message": "Proceso de sinterizado completado: 420 kg de polvo metálico disponibles.",
        "icon": "♻️",
        "metadata": {"process": "sinterizado", "mass_kg": 420},
    },
]

_DEMO_MANIFESTS: Final[list[dict[str, Any]]] = [
    {
        "key": "ares-resupply",
        "label": "Ares Resupply · Órbita",
        "description": "Carga crítica con repuestos orbitales y compuestos ligeros.",
        "rows": [
            {
                "item": "Aleación Ti-Fe estructural",
                "category": "Structural elements",
                "mass_kg": 14.2,
                "tg_loss_pct": 2.5,
                "ega_loss_pct": 0.4,
                "water_l_per_kg": 0.0,
                "energy_kwh_per_kg": 0.8,
            },
            {
                "item": "Panel fotónico modular",
                "category": "Energy systems",
                "mass_kg": 9.8,
                "tg_loss_pct": 1.8,
                "ega_loss_pct": 0.3,
                "water_l_per_kg": 0.0,
                "energy_kwh_per_kg": 0.6,
            },
            {
                "item": "Repuesto de bomba criogénica",
                "category": "Life support",
                "mass_kg": 6.4,
                "tg_loss_pct": 3.2,
                "ega_loss_pct": 0.6,
                "water_l_per_kg": 0.05,
                "energy_kwh_per_kg": 0.9,
            },
            {
                "item": "Textiles Nomex EVA",
                "category": "Consumables",
                "mass_kg": 4.1,
                "tg_loss_pct": 2.1,
                "ega_loss_pct": 0.5,
                "water_l_per_kg": 0.0,
                "energy_kwh_per_kg": 0.35,
            },
        ],
    },
    {
        "key": "surface-reuse",
        "label": "Superficie · Reuso rápido",
        "description": "Material recuperado para líneas de impresión y sellado.",
        "rows": [
            {
                "item": "Polímeros recuperados",
                "category": "Packaging",
                "mass_kg": 11.6,
                "tg_loss_pct": 5.5,
                "ega_loss_pct": 1.2,
                "water_l_per_kg": 0.08,
                "energy_kwh_per_kg": 0.5,
            },
            {
                "item": "Polvo metálico refinado",
                "category": "Structural elements",
                "mass_kg": 7.9,
                "tg_loss_pct": 4.0,
                "ega_loss_pct": 0.8,
                "water_l_per_kg": 0.02,
                "energy_kwh_per_kg": 1.1,
            },
            {
                "item": "Espuma técnica aislante",
                "category": "Thermal systems",
                "mass_kg": 5.5,
                "tg_loss_pct": 2.4,
                "ega_loss_pct": 0.4,
                "water_l_per_kg": 0.01,
                "energy_kwh_per_kg": 0.7,
            },
            {
                "item": "Compuesto fibra carbono",
                "category": "Structural elements",
                "mass_kg": 6.3,
                "tg_loss_pct": 3.1,
                "ega_loss_pct": 0.7,
                "water_l_per_kg": 0.0,
                "energy_kwh_per_kg": 0.95,
            },
        ],
    },
    {
        "key": "science-swap",
        "label": "Ciencia · Swap laboratorio",
        "description": "Muestra mixta con consumibles de laboratorio y residuos EVA.",
        "rows": [
            {
                "item": "Consumibles laboratorio húmedo",
                "category": "Lab supplies",
                "mass_kg": 3.2,
                "tg_loss_pct": 1.6,
                "ega_loss_pct": 0.2,
                "water_l_per_kg": 0.12,
                "energy_kwh_per_kg": 0.4,
            },
            {
                "item": "Tejido técnico respirable",
                "category": "Consumables",
                "mass_kg": 5.7,
                "tg_loss_pct": 2.2,
                "ega_loss_pct": 0.5,
                "water_l_per_kg": 0.06,
                "energy_kwh_per_kg": 0.55,
            },
            {
                "item": "Panel dieléctrico reciclable",
                "category": "Energy systems",
                "mass_kg": 4.6,
                "tg_loss_pct": 2.0,
                "ega_loss_pct": 0.3,
                "water_l_per_kg": 0.03,
                "energy_kwh_per_kg": 0.62,
            },
            {
                "item": "Resina fotopolimerizable",
                "category": "Additive manufacturing",
                "mass_kg": 2.9,
                "tg_loss_pct": 1.4,
                "ega_loss_pct": 0.25,
                "water_l_per_kg": 0.0,
                "energy_kwh_per_kg": 0.7,
            },
        ],
    },
]


def _synthesise_audio_sequence(
    sequence: Sequence[tuple[float, float, float]],
    *,
    sample_rate: int = 22050,
) -> bytes:
    """Return a WAV-encoded byte payload for the provided tone sequence."""

    frames = bytearray()
    for frequency, duration, amplitude in sequence:
        duration = max(float(duration), 0.0)
        amplitude = max(min(float(amplitude), 1.0), 0.0)
        samples = max(1, int(duration * sample_rate))
        for index in range(samples):
            if frequency <= 0:
                value = 0
            else:
                theta = 2.0 * math.pi * frequency * (index / sample_rate)
                value = int(amplitude * 32767.0 * math.sin(theta))
            frames.extend(int(value).to_bytes(2, "little", signed=True))

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(bytes(frames))
    return buffer.getvalue()


def _ensure_demo_audio_clip(clip_id: str) -> tuple[str | None, bytes | None]:
    """Create (if needed) and return the audio clip for ``clip_id``."""

    clip_id = clip_id.strip()
    if not clip_id:
        return None, None

    if clip_id in _DEMO_AUDIO_CACHE:
        data = _DEMO_AUDIO_CACHE[clip_id]
    else:
        sequence = _DEMO_AUDIO_CLIPS.get(clip_id)
        if not sequence:
            return None, None
        data = _synthesise_audio_sequence(sequence)
        _DEMO_AUDIO_CACHE[clip_id] = data

    try:
        _STATIC_AUDIO_ROOT.mkdir(parents=True, exist_ok=True)
    except OSError:
        # If we cannot write to disk we still return the bytes so Streamlit can play them.
        return None, data

    path = _STATIC_AUDIO_ROOT / f"{clip_id}.wav"
    if not path.exists():
        try:
            path.write_bytes(data)
        except OSError:
            return None, data

    return str(path), data


def _attach_demo_audio(event: "DemoEvent") -> "DemoEvent":
    """Ensure the demo event carries synthesised audio assets."""

    clip_id = event.audio_clip
    if not clip_id and event.audio_path:
        clip_id = Path(event.audio_path).stem
    if not clip_id:
        return event

    path, data = _ensure_demo_audio_clip(clip_id)
    if data is None and path is None:
        return event

    updates: dict[str, Any] = {"audio_clip": clip_id, "audio_bytes": data}
    if path:
        updates["audio_path"] = path
    return replace(event, **updates)


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


@dataclass(slots=True, frozen=True)
class DemoEvent:
    """Preset event used to drive the demo control room experience."""

    event_id: str
    category: str
    title: str
    message: str
    severity: str = "info"
    icon: str = "🛰️"
    audio_clip: str | None = None
    audio_path: str | None = None
    audio_bytes: bytes | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    emitted_at: datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "event_id": self.event_id,
            "category": self.category,
            "title": self.title,
            "message": self.message,
            "severity": self.severity,
            "icon": self.icon,
            "audio_clip": self.audio_clip,
            "audio_path": self.audio_path,
            "metadata": deepcopy(self.metadata),
        }
        if self.emitted_at is not None:
            payload["emitted_at"] = self.emitted_at.isoformat()
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

_JEZERO_GEODATA_CACHE: dict[str, Any] | None = None


def _default_jezero_geometry() -> dict[str, Any]:
    return {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"name": "Jezero", "kind": "boundary"},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [
                            [77.2, 18.2],
                            [77.6, 18.4],
                            [77.9, 18.6],
                            [77.6, 18.8],
                            [77.2, 18.6],
                            [77.2, 18.2],
                        ]
                    ],
                },
            }
        ],
    }


def load_jezero_geodata(*, refresh: bool = False) -> dict[str, Any]:
    """Return the operational geometry for the Jezero landing site."""

    global _JEZERO_GEODATA_CACHE
    if _JEZERO_GEODATA_CACHE is not None and not refresh:
        return deepcopy(_JEZERO_GEODATA_CACHE)

    geometry: dict[str, Any]
    if _JEZERO_GEOJSON_PATH.is_file():
        try:
            with _JEZERO_GEOJSON_PATH.open("r", encoding="utf-8") as handle:
                geometry = json.load(handle)
        except json.JSONDecodeError as exc:  # pragma: no cover - defensive path
            st.warning(f"GeoJSON inválido en { _JEZERO_GEOJSON_PATH.name }: {exc}")
            geometry = _default_jezero_geometry()
    else:
        geometry = _default_jezero_geometry()

    _JEZERO_GEODATA_CACHE = geometry
    return deepcopy(_JEZERO_GEODATA_CACHE)


def _jezero_bounds_from_geometry(geometry: Mapping[str, Any]) -> tuple[float, float, float, float] | None:
    if not isinstance(geometry, Mapping):
        return None

    coordinates: list[tuple[float, float]] = []

    def _collect_points(geom: Mapping[str, Any]) -> None:
        if not isinstance(geom, Mapping):
            return
        gtype = str(geom.get("type", ""))
        raw_coords = geom.get("coordinates")
        if gtype == "Point" and isinstance(raw_coords, Sequence) and len(raw_coords) >= 2:
            try:
                lon = float(raw_coords[0])
                lat = float(raw_coords[1])
            except (TypeError, ValueError):
                return
            coordinates.append((lon, lat))
            return
        if gtype in {"LineString", "MultiPoint"} and isinstance(raw_coords, Sequence):
            for item in raw_coords:
                if isinstance(item, Sequence) and len(item) >= 2:
                    try:
                        lon = float(item[0])
                        lat = float(item[1])
                    except (TypeError, ValueError):
                        continue
                    coordinates.append((lon, lat))
            return
        if gtype in {"Polygon", "MultiLineString"} and isinstance(raw_coords, Sequence):
            for ring in raw_coords:
                if isinstance(ring, Sequence):
                    for item in ring:
                        if isinstance(item, Sequence) and len(item) >= 2:
                            try:
                                lon = float(item[0])
                                lat = float(item[1])
                            except (TypeError, ValueError):
                                continue
                            coordinates.append((lon, lat))
            return
        if gtype == "GeometryCollection":
            for entry in geom.get("geometries", []):
                if isinstance(entry, Mapping):
                    _collect_points(entry)
            return
        if gtype == "MultiPolygon" and isinstance(raw_coords, Sequence):
            for polygon in raw_coords:
                if isinstance(polygon, Sequence):
                    for ring in polygon:
                        if isinstance(ring, Sequence):
                            for item in ring:
                                if isinstance(item, Sequence) and len(item) >= 2:
                                    try:
                                        lon = float(item[0])
                                        lat = float(item[1])
                                    except (TypeError, ValueError):
                                        continue
                                    coordinates.append((lon, lat))
            return

    features = geometry.get("features") if isinstance(geometry.get("features"), Sequence) else []
    for feature in features:
        if isinstance(feature, Mapping):
            geom = feature.get("geometry")
            if isinstance(geom, Mapping):
                _collect_points(geom)

    if not coordinates:
        return None

    longitudes, latitudes = zip(*coordinates)
    return (
        float(min(longitudes)),
        float(min(latitudes)),
        float(max(longitudes)),
        float(max(latitudes)),
    )


def _load_bitmap_dimensions(image_path: Path) -> tuple[int | None, int | None]:
    try:
        from PIL import Image  # type: ignore

        with Image.open(image_path) as image:
            return int(image.width), int(image.height)
    except Exception:  # pragma: no cover - optional dependency
        return None, None


def _load_bitmap_layer(
    image_path: Path,
    *,
    label: str,
    description: str,
    layer_type: str,
    attribution: str,
    source: str,
    default_opacity: float = 0.75,
    legend: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if not image_path.is_file():
        raise FileNotFoundError(
            "No se encontró la textura requerida para Jezero. "
            f"Asegurate de copiar '{image_path.name}' dentro de '{image_path.parent.name}/'."
        )

    asset_path, asset_url = _ensure_static_bitmap(image_path)

    try:
        payload = image_path.read_bytes()
    except OSError as exc:  # pragma: no cover - defensive path
        raise RuntimeError(f"No se pudo leer la textura de Jezero ({image_path.name}): {exc}") from exc

    suffix = image_path.suffix.lower()
    mime_type = "image/jpeg" if suffix in {".jpg", ".jpeg"} else "image/png"
    encoded = b64encode(payload).decode("ascii")
    image_uri = f"data:{mime_type};base64,{encoded}"

    geometry = load_jezero_geodata()
    bounds = _jezero_bounds_from_geometry(geometry) or _JEZERO_DEFAULT_BOUNDS

    width_px, height_px = _load_bitmap_dimensions(image_path)
    center = {
        "longitude": (bounds[0] + bounds[2]) / 2.0,
        "latitude": (bounds[1] + bounds[3]) / 2.0,
    }

    metadata: dict[str, Any] = {
        "path": str(image_path),
        "asset_path": str(asset_path),
        "asset_url": asset_url,
        "mime_type": mime_type,
        "width_px": width_px,
        "height_px": height_px,
        "attribution": attribution,
        "source": source,
        "license": attribution,
        "provenance": source,
        "label": label,
        "description": description,
        "layer_type": layer_type,
        "default_opacity": float(max(0.0, min(default_opacity, 1.0))),
    }
    if legend:
        metadata["legend"] = legend

    return {
        "image_uri": image_uri,
        "image": {"url": asset_url},
        "bounds": bounds,
        "center": center,
        "metadata": metadata,
    }


@st.cache_data(show_spinner=False)
def load_jezero_bitmap() -> dict[str, Any]:
    """Return cached bitmap metadata covering the Jezero operational area."""

    image_path = _JEZERO_BITMAP_PATH if _JEZERO_BITMAP_PATH.is_file() else _JEZERO_BITMAP_FALLBACK_PATH
    return _load_bitmap_layer(
        image_path,
        label="Textura base Jezero",
        description="Mosaico orbital del cráter Jezero utilizado como fondo base.",
        layer_type="base",
        attribution=(
            "Mars 2020 Jezero landing site mosaic · NASA/JPL-Caltech/University of Arizona/"
            "DLR/FU Berlin. Public domain unless otherwise noted."
        ),
        source="Processed Jezero crater mosaic supplied with the hackathon dataset.",
        default_opacity=0.92,
    )


@st.cache_data(show_spinner=False)
def load_jezero_slope_bitmap() -> dict[str, Any]:
    """Return slope-derived bitmap aligned to the Jezero operational footprint."""

    legend = {
        "description": "Pendiente del terreno (grados)",
        "units": "degrees",
        "range": [0, 35],
        "ticks": [
            {"value": 0, "label": "0°", "color": "#0f172a"},
            {"value": 5, "label": "5°", "color": "#0ea5e9"},
            {"value": 15, "label": "15°", "color": "#facc15"},
            {"value": 25, "label": "25°", "color": "#f97316"},
            {"value": 35, "label": "35°", "color": "#ef4444"},
        ],
        "gradient_css": "linear-gradient(90deg,#0f172a 0%,#0ea5e9 30%,#facc15 60%,#ef4444 100%)",
    }

    return _load_bitmap_layer(
        _JEZERO_SLOPE_BITMAP_PATH,
        label="Pendiente 20 m",
        description="Modelo de pendientes de Jezero a 20 m/px derivado del DEM CTX.",
        layer_type="slope",
        attribution=(
            "CTX DEM slope model · Caltech/JPL/ASU · procesado por el equipo de la misión Mars 2020."
        ),
        source="j03_045994_1986_j03_046060_1986_20m_slope_20m (hackathon dataset).",
        default_opacity=0.65,
        legend=legend,
    )


@st.cache_data(show_spinner=False)
def load_jezero_ortho_bitmap() -> dict[str, Any]:
    """Return orthoimagery aligned to the Jezero polygon."""

    return _load_bitmap_layer(
        _JEZERO_ORTHO_BITMAP_PATH,
        label="Ortofoto HiRISE 6 m",
        description="Ortoimagen HiRISE/CTX corregida geométricamente sobre el delta de Jezero.",
        layer_type="orthophoto",
        attribution=(
            "HiRISE/CTX orthorectified mosaic · NASA/JPL/University of Arizona · uso educativo."
        ),
        source="j03_045994_1986_xn_18n282w_6m_ortho (hackathon dataset).",
        default_opacity=0.75,
    )


def _static_base_url() -> str:
    try:
        base_url = st.get_option("server.baseUrlPath") or ""
    except Exception:  # pragma: no cover - Streamlit not initialised in tests
        base_url = ""
    if base_url and not base_url.endswith("/"):
        base_url = f"{base_url}/"
    return base_url


def _ensure_static_asset(
    source: Path, destination: Path, *, missing_message: str | None = None
) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not source.is_file():
        message = missing_message or f"No se encontró el asset requerido en {source}."
        raise FileNotFoundError(message)
    should_copy = True
    if destination.exists():
        try:
            should_copy = source.stat().st_mtime > destination.stat().st_mtime
        except OSError:
            should_copy = True
    if should_copy:
        shutil.copy2(source, destination)
    return destination


def _ensure_static_bitmap(image_path: Path) -> tuple[Path, str]:
    asset_path = _ensure_static_asset(
        image_path,
        _STATIC_MARS_ROOT / image_path.name,
        missing_message=(
            "No se encontró la textura requerida para Jezero. "
            f"Asegurate de copiar '{image_path.name}' dentro de '{image_path.parent.name}/'."
        ),
    )
    asset_url = f"{_static_base_url()}static/mars/{asset_path.name}"
    return asset_path, asset_url


@st.cache_data(show_spinner=False)
def load_mars_scenegraph() -> dict[str, Any]:
    """Expose the Mars orbital model as a static asset ready for the 3D scene."""

    source = _MARS_DATASETS_ROOT / _MARS_SCENEGRAPH_FILENAME
    destination = _STATIC_ROOT / "models" / _MARS_SCENEGRAPH_FILENAME
    asset_path = _ensure_static_asset(
        source,
        destination,
        missing_message=f"No se encontró el modelo 3D requerido en {source}.",
    )

    asset_url = f"{_static_base_url()}static/models/{asset_path.name}"
    return {
        "url": asset_url,
        "path": str(asset_path),
        "scale": (0.0075, 0.0075, 0.0075),
        "orientation": (0.0, 180.0, 0.0),
        "translation": (0.0, 0.0, 0.0),
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


def _material_group_key(record: Mapping[str, Any]) -> str:
    category = str(record.get("category") or "").lower()
    family = str(record.get("material_family") or "").lower()
    flags = str(record.get("flags") or "").lower()
    key_materials = str(record.get("key_materials") or "").lower()
    tokens = " ".join([category, family, flags, key_materials])

    if "foam" in tokens or "pvdf" in tokens:
        return "espumas"
    if any(term in tokens for term in ("textile", "fabric", "cotton", "nomex", "garment", "towel")):
        return "textiles"
    if any(term in tokens for term in ("alloy", "aluminium", "titanium", "steel", "structural", "strut")):
        return "metales"
    if any(term in tokens for term in ("poly", "plastic", "polymer", "eva")):
        return "polimeros"
    return "mixtos"


def _dominant_percentage(text: str) -> float | None:
    matches = _PERCENTAGE_PATTERN.findall(text)
    if not matches:
        return None
    try:
        values = [float(match) for match in matches]
    except ValueError:
        return None
    if not values:
        return None
    return max(values)


def _estimate_purity(record: Mapping[str, Any]) -> float:
    key_materials = str(record.get("key_materials") or "")
    flags = str(record.get("flags") or "").lower()
    dominant = _dominant_percentage(key_materials)

    if dominant is not None:
        purity = float(dominant)
    else:
        separators = re.split(r"[;\n\-/|]", key_materials)
        tokens = [token.strip() for token in separators if token.strip()]
        if not tokens:
            purity = 80.0
        elif len(tokens) == 1:
            purity = 92.0
        elif len(tokens) == 2:
            purity = 74.0
        else:
            purity = 58.0

    if "multilayer" in flags or "composite" in flags:
        purity -= 20.0
    if "foam" in flags:
        purity -= 10.0
    return float(max(20.0, min(purity, 100.0)))


def _estimate_contamination(record: Mapping[str, Any], *, purity: float) -> float:
    flags = str(record.get("flags") or "").lower()
    category = str(record.get("category") or "").lower()
    base = max(0.0, 100.0 - purity)
    if record.get("_problematic"):
        base += 12.0
    if any(term in flags for term in ("multilayer", "adhesive", "foam")):
        base += 10.0
    if any(term in category for term in ("food", "packaging")):
        base += 6.0
    return float(max(0.0, min(base, 100.0)))


def _allocate_destination_masses(
    group: str,
    *,
    mass_kg: float,
    purity: float,
    contamination: float,
) -> dict[str, float]:
    if not math.isfinite(mass_kg) or mass_kg <= 0:
        return {key: 0.0 for key in _DESTINATION_INFO}

    base_weights = _DESTINATION_WEIGHTS.get(group, _DESTINATION_WEIGHTS["mixtos"]).copy()
    purity_norm = max(0.0, min(purity / 100.0, 1.0))
    contamination_norm = max(0.0, min(contamination / 100.0, 1.0))

    adjusted = {
        "recycle": base_weights["recycle"] * (0.45 + 0.55 * contamination_norm),
        "reuse": base_weights["reuse"] * (0.4 + 0.6 * purity_norm),
        "stock": base_weights["stock"] * (0.35 + 0.65 * purity_norm),
    }
    total_weight = sum(adjusted.values()) or 1.0
    return {key: mass_kg * weight / total_weight for key, weight in adjusted.items()}


def aggregate_inventory_by_category(
    inventory: pd.DataFrame | None,
) -> dict[str, Any]:
    """Aggregate inventory metrics grouped by category and destination."""

    empty_payload = {
        "normalized": pd.DataFrame(),
        "categories": pd.DataFrame(
            columns=[
                "category",
                "material_group",
                "material_group_label",
                "total_mass_kg",
                "water_l",
                "energy_kwh",
                "purity_index",
                "cross_contamination_risk",
                "recycle_mass_kg",
                "reuse_mass_kg",
                "stock_mass_kg",
            ]
        ),
        "flows": pd.DataFrame(columns=["category", "destination_key", "destination_label", "mass_kg"]),
        "material_groups": _MATERIAL_GROUP_LABELS.copy(),
        "group_palette": _MATERIAL_GROUP_COLORS.copy(),
        "destinations": _DESTINATION_INFO.copy(),
    }

    if inventory is None or inventory.empty:
        return empty_payload

    working = inventory.copy(deep=True)
    working["category"] = working.get("category", "").fillna("").astype(str)
    working["material_family"] = working.get("material_family", "").fillna("").astype(str)
    working["flags"] = working.get("flags", "").fillna("").astype(str)
    working["key_materials"] = working.get("key_materials", "").fillna("").astype(str)
    if "_problematic" in working.columns:
        working["_problematic"] = working["_problematic"].astype(bool)
    else:
        working["_problematic"] = False

    mass = pd.to_numeric(working.get("mass_kg", working.get("kg")), errors="coerce").fillna(0.0)
    volume_l = pd.to_numeric(working.get("volume_l"), errors="coerce").fillna(0.0)
    moisture_ratio = pd.to_numeric(working.get("moisture_pct"), errors="coerce").fillna(0.0) / 100.0
    difficulty = pd.to_numeric(working.get("difficulty_factor"), errors="coerce").fillna(1.0)
    difficulty = difficulty.clip(lower=1.0, upper=3.0)

    working["mass_kg"] = mass
    working["volume_l"] = volume_l
    working["material_group"] = working.apply(_material_group_key, axis=1)
    working["material_group_label"] = working["material_group"].map(
        _MATERIAL_GROUP_LABELS
    )
    working["material_group_label"] = working["material_group_label"].fillna(
        _MATERIAL_GROUP_LABELS["mixtos"]
    )

    purity_series = working.apply(_estimate_purity, axis=1)
    working["purity_index"] = purity_series
    working["cross_contamination_risk"] = working.apply(
        lambda row: _estimate_contamination(row, purity=float(row["purity_index"])),
        axis=1,
    )

    base_energy = 0.12
    max_energy = 0.70
    energy_per_kg = base_energy + (difficulty - 1.0) / 2.0 * (max_energy - base_energy)
    working["water_l"] = mass * moisture_ratio
    working["energy_kwh"] = mass * energy_per_kg
    working["volume_m3"] = volume_l / 1000.0

    allocations = working.apply(
        lambda row: _allocate_destination_masses(
            str(row.get("material_group")),
            mass_kg=float(row.get("mass_kg", 0.0)),
            purity=float(row.get("purity_index", 0.0)),
            contamination=float(row.get("cross_contamination_risk", 0.0)),
        ),
        axis=1,
    )
    allocation_df = pd.DataFrame(list(allocations), index=working.index).fillna(0.0)
    for key in _DESTINATION_INFO:
        working[f"{key}_mass_kg"] = allocation_df.get(key, 0.0)

    working["purity_mass"] = working["purity_index"] * working["mass_kg"]
    working["contamination_mass"] = working["cross_contamination_risk"] * working["mass_kg"]

    def _mode(series: pd.Series) -> str:
        try:
            modes = series.mode(dropna=True)
            if not modes.empty:
                return str(modes.iloc[0])
        except Exception:
            pass
        return str(series.iloc[0]) if not series.empty else ""

    grouped = (
        working.groupby("category", as_index=False)
        .agg(
            material_group=("material_group", _mode),
            material_group_label=("material_group_label", _mode),
            total_mass_kg=("mass_kg", "sum"),
            water_l=("water_l", "sum"),
            energy_kwh=("energy_kwh", "sum"),
            purity_mass=("purity_mass", "sum"),
            contamination_mass=("contamination_mass", "sum"),
            recycle_mass_kg=("recycle_mass_kg", "sum"),
            reuse_mass_kg=("reuse_mass_kg", "sum"),
            stock_mass_kg=("stock_mass_kg", "sum"),
        )
        .reset_index(drop=True)
    )

    safe_mass = grouped["total_mass_kg"].where(grouped["total_mass_kg"] != 0, pd.NA)
    grouped["purity_index"] = (grouped["purity_mass"] / safe_mass).fillna(0.0)
    grouped["cross_contamination_risk"] = (
        grouped["contamination_mass"] / safe_mass
    ).fillna(0.0)

    grouped = grouped.drop(columns=["purity_mass", "contamination_mass"])
    grouped = grouped.sort_values("total_mass_kg", ascending=False).reset_index(drop=True)

    flow_records: list[dict[str, Any]] = []
    for _, row in grouped.iterrows():
        for key, info in _DESTINATION_INFO.items():
            mass_value = float(row.get(f"{key}_mass_kg", 0.0))
            if mass_value <= 0:
                continue
            flow_records.append(
                {
                    "category": row["category"],
                    "destination_key": key,
                    "destination_label": info["label"],
                    "destination_display": info["display"],
                    "mass_kg": mass_value,
                }
            )

    flows = pd.DataFrame(flow_records)

    payload = {
        "normalized": working,
        "categories": grouped,
        "flows": flows,
        "material_groups": _MATERIAL_GROUP_LABELS.copy(),
        "group_palette": _MATERIAL_GROUP_COLORS.copy(),
        "destinations": _DESTINATION_INFO.copy(),
    }
    return payload


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


def _ensure_demo_state(session: MutableMapping[str, Any]) -> MutableMapping[str, Any]:
    state = session.setdefault(_DEMO_STATE_KEY, {})
    if not isinstance(state.get(_DEMO_QUEUE_KEY), list) or not state.get(_DEMO_QUEUE_KEY):
        state[_DEMO_QUEUE_KEY] = [deepcopy(record) for record in _DEMO_EVENT_PLAYLIST]
        state[_DEMO_CURSOR_KEY] = 0
        state[_DEMO_HISTORY_KEY] = []
        state[_DEMO_LAST_TS_KEY] = None
    state.setdefault(_DEMO_CURSOR_KEY, 0)
    state.setdefault(_DEMO_HISTORY_KEY, [])
    state.setdefault(_DEMO_LAST_TS_KEY, None)
    return state


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


def _demo_event_from_record(record: Mapping[str, Any]) -> DemoEvent:
    metadata = record.get("metadata")
    if not isinstance(metadata, Mapping):
        metadata = {}
    emitted_at = None
    emitted_raw = record.get("emitted_at")
    if emitted_raw:
        try:
            emitted_at = datetime.fromisoformat(str(emitted_raw))
        except ValueError:
            emitted_at = None
    return DemoEvent(
        event_id=str(record.get("event_id", "")).strip() or "demo-event",
        category=str(record.get("category", "info")).strip(),
        title=str(record.get("title", "Evento demo")).strip(),
        message=str(record.get("message", "")).strip(),
        severity=str(record.get("severity", "info")).strip(),
        icon=str(record.get("icon", "🛰️")),
        audio_clip=(str(record.get("audio_clip", "")).strip() or None),
        audio_path=(str(record["audio_path"]).strip() if record.get("audio_path") else None),
        metadata=dict(metadata),
        emitted_at=emitted_at,
    )


def demo_event_script() -> list[DemoEvent]:
    """Return the static demo event playlist."""

    return [_attach_demo_audio(_demo_event_from_record(record)) for record in _DEMO_EVENT_PLAYLIST]


def generate_demo_event(
    interval_seconds: float,
    *,
    session: MutableMapping[str, Any] | None = None,
    force: bool = False,
    now: datetime | None = None,
) -> DemoEvent | None:
    """Emit the next demo event when the interval has elapsed."""

    session = session or st.session_state
    state = _ensure_demo_state(session)

    queue: list[dict[str, Any]] = state.get(_DEMO_QUEUE_KEY, [])
    if not queue:
        queue = [deepcopy(record) for record in _DEMO_EVENT_PLAYLIST]
        state[_DEMO_QUEUE_KEY] = queue
        state[_DEMO_CURSOR_KEY] = 0

    cursor = int(state.get(_DEMO_CURSOR_KEY, 0) or 0)
    if cursor >= len(queue):
        cursor = 0

    now_dt = now or datetime.now(timezone.utc)
    last_raw = state.get(_DEMO_LAST_TS_KEY)
    due = force
    if not due:
        if interval_seconds <= 0:
            due = True
        elif last_raw:
            try:
                last_dt = datetime.fromisoformat(str(last_raw))
            except ValueError:
                last_dt = None
            if last_dt is None or (now_dt - last_dt).total_seconds() >= interval_seconds:
                due = True
        else:
            due = True

    if not due or not queue:
        return None

    record = deepcopy(queue[cursor])
    state[_DEMO_CURSOR_KEY] = (cursor + 1) % len(queue)
    state[_DEMO_LAST_TS_KEY] = now_dt.isoformat()

    record["emitted_at"] = now_dt.isoformat()
    event = _attach_demo_audio(_demo_event_from_record(record))
    if event.audio_path:
        record["audio_path"] = event.audio_path
    if event.audio_clip:
        record["audio_clip"] = event.audio_clip

    history = state.setdefault(_DEMO_HISTORY_KEY, [])
    history.append(deepcopy(record))

    return event


def get_demo_event_history(
    limit: int | None = None,
    *,
    session: MutableMapping[str, Any] | None = None,
) -> list[DemoEvent]:
    """Return the emitted demo events in reverse chronological order."""

    session = session or st.session_state
    state = _ensure_demo_state(session)
    history = state.get(_DEMO_HISTORY_KEY, [])
    records = list(history)
    if limit is not None and limit >= 0:
        records = records[-limit:]
    return [_attach_demo_audio(_demo_event_from_record(record)) for record in reversed(records)]


def reset_demo_events(*, session: MutableMapping[str, Any] | None = None) -> None:
    """Reset the demo event generator state."""

    session = session or st.session_state
    session[_DEMO_STATE_KEY] = {
        _DEMO_QUEUE_KEY: [deepcopy(record) for record in _DEMO_EVENT_PLAYLIST],
        _DEMO_CURSOR_KEY: 0,
        _DEMO_LAST_TS_KEY: None,
        _DEMO_HISTORY_KEY: [],
    }


def demo_manifest_catalogue() -> list[dict[str, Any]]:
    """Expose the demo manifest presets for UI consumption."""

    return [deepcopy(entry) for entry in _DEMO_MANIFESTS]


def load_demo_manifest(key: str) -> pd.DataFrame:
    """Return a demo manifest dataframe matching ``key``."""

    for entry in _DEMO_MANIFESTS:
        if entry.get("key") == key:
            return pd.DataFrame(deepcopy(entry.get("rows", [])))
    raise KeyError(f"Unknown demo manifest key: {key}")


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
    "DemoEvent",
    "load_jezero_geodata",
    "load_jezero_bitmap",
    "load_jezero_slope_bitmap",
    "load_jezero_ortho_bitmap",
    "load_mars_scenegraph",
    "load_logistics_baseline",
    "load_live_inventory",
    "compute_mission_summary",
    "aggregate_inventory_by_category",
    "score_manifest_batch",
    "summarise_policy_actions",
    "demo_event_script",
    "generate_demo_event",
    "get_demo_event_history",
    "reset_demo_events",
    "demo_manifest_catalogue",
    "load_demo_manifest",
    "iterate_events",
    "apply_simulation_tick",
]
