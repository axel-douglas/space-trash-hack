from __future__ import annotations

from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Any
import json
import uuid

import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
LOGS_DIR = DATA_DIR / "logs"


@dataclass
class ImpactEntry:
    ts_iso: str
    scenario: str
    target_name: str
    materials: str
    weights: str
    process_id: str
    process_name: str
    mass_final_kg: float
    energy_kwh: float
    water_l: float
    crew_min: float
    score: float
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class FeedbackEntry:
    ts_iso: str
    astronaut: str
    scenario: str
    target_name: str
    option_idx: int
    rigidity_ok: bool
    ease_ok: bool
    issues: str
    notes: str
    extra: dict[str, Any] = field(default_factory=dict)


def _ensure_dirs() -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)


def _serialize_extra(extra: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    extras = extra or {}
    return json.dumps(extras, ensure_ascii=False, sort_keys=True), {
        f"extra_{key}": value for key, value in extras.items()
    }


def _prepare_payload(entry_dict: dict[str, Any]) -> dict[str, Any]:
    extra_payload = entry_dict.pop("extra", {}) or {}
    extra_serialized, extra_columns = _serialize_extra(extra_payload)
    payload = {**entry_dict, **extra_columns}
    payload["extra"] = extra_serialized
    payload["run_id"] = uuid.uuid4().hex
    return payload


def _entry_date(ts_iso: str) -> str:
    try:
        dt = datetime.fromisoformat(ts_iso.replace("Z", "+00:00"))
    except ValueError:
        dt = datetime.utcnow()
    return dt.strftime("%Y-%m-%d")


def _append_record(filename: str, payload: dict[str, Any]) -> None:
    _ensure_dirs()
    path = LOGS_DIR / filename
    row_df = pd.DataFrame([payload])
    if path.exists():
        existing = pd.read_parquet(path)
        df = pd.concat([existing, row_df], ignore_index=True, sort=False)
    else:
        df = row_df
    df.to_parquet(path, index=False)


def append_impact(entry: ImpactEntry) -> str:
    payload = _prepare_payload(asdict(entry))
    date_str = _entry_date(entry.ts_iso)
    _append_record(f"impact_{date_str}.parquet", payload)
    return payload["run_id"]


def append_feedback(entry: FeedbackEntry) -> str:
    payload = _prepare_payload(asdict(entry))
    date_str = _entry_date(entry.ts_iso)
    _append_record(f"feedback_{date_str}.parquet", payload)
    return payload["run_id"]


def _parse_extra_column(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value:
        try:
            parsed = json.loads(value)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass
    return {}


def _load_parquet(pattern: str) -> pd.DataFrame:
    _ensure_dirs()
    files = sorted(LOGS_DIR.glob(pattern))
    if not files:
        return pd.DataFrame()
    frames = [pd.read_parquet(path) for path in files]
    df = pd.concat(frames, ignore_index=True, sort=False)
    if "extra" in df.columns:
        df["extra"] = df["extra"].apply(_parse_extra_column)
    else:
        df["extra"] = [{} for _ in range(len(df))]
    return df


def load_impact_df() -> pd.DataFrame:
    return _load_parquet("impact_*.parquet")


def load_feedback_df() -> pd.DataFrame:
    return _load_parquet("feedback_*.parquet")


def summarize_impact(df: pd.DataFrame) -> dict:
    if df.empty:
        return {"runs":0, "kg":0.0, "kwh":0.0, "water_l":0.0, "crew_min":0.0}
    metrics = {}
    for column in ("mass_final_kg", "energy_kwh", "water_l", "crew_min"):
        if column in df.columns:
            metrics[column] = pd.to_numeric(df[column], errors="coerce")
        else:
            metrics[column] = pd.Series(dtype="float64")
    return {
        "runs": int(len(df)),
        "kg": float(metrics["mass_final_kg"].sum(skipna=True)),
        "kwh": float(metrics["energy_kwh"].sum(skipna=True)),
        "water_l": float(metrics["water_l"].sum(skipna=True)),
        "crew_min": float(metrics["crew_min"].sum(skipna=True))
    }
