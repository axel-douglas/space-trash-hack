from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass, asdict, field
from datetime import datetime
from typing import Any

import pandas as pd

try:  # Optional dependency used for streaming Parquet writes
    import pyarrow as pa
    import pyarrow.parquet as pq
except Exception:  # pragma: no cover - pyarrow is optional at runtime
    pa = None  # type: ignore[assignment]
    pq = None  # type: ignore[assignment]

from .paths import DATA_ROOT, LOGS_DIR

DATA_DIR = DATA_ROOT


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


def _align_table_to_schema(table: "pa.Table" | "pa.RecordBatch", schema: "pa.Schema") -> "pa.Table":
    """Return *table* aligned to *schema*, filling missing columns with nulls."""

    if isinstance(table, pa.RecordBatch):
        base_table = pa.Table.from_batches([table])
    else:
        base_table = table

    columns = []
    for field in schema:
        if field.name in base_table.column_names:
            column = base_table[field.name]
            if column.type != field.type:
                column = column.cast(field.type)
        else:
            column = pa.nulls(base_table.num_rows, type=field.type)
        columns.append(column)

    return pa.Table.from_arrays(columns, schema=schema)


def _updated_schema(
    existing_schema: "pa.Schema | None", row_schema: "pa.Schema"
) -> tuple["pa.Schema", bool]:
    """Return schema extended with any new fields from *row_schema*."""

    if existing_schema is None:
        return row_schema, False

    schema = existing_schema
    changed = False
    existing_names = set(existing_schema.names)
    for field in row_schema:
        if field.name not in existing_names:
            inferred = pa.field(field.name, field.type, nullable=True)
            schema = schema.append(inferred)
            changed = True
    return schema, changed


def _append_record(filename: str, payload: dict[str, Any]) -> None:
    _ensure_dirs()
    path = LOGS_DIR / filename

    if pa is None or pq is None:
        row_df = pd.DataFrame([payload])
        if path.exists():
            existing = pd.read_parquet(path)
            df = pd.concat([existing, row_df], ignore_index=True, sort=False)
        else:
            df = row_df
        df.to_parquet(path, index=False)
        return

    row_table = pa.Table.from_pydict({key: [value] for key, value in payload.items()})

    if not path.exists():
        pq.write_table(row_table, path)
        return

    try:
        parquet_file = pq.ParquetFile(path)
        existing_schema = parquet_file.schema_arrow
    except Exception:
        parquet_file = None
        existing_schema = None

    target_schema, schema_changed = _updated_schema(existing_schema, row_table.schema)
    aligned_row = _align_table_to_schema(row_table, target_schema)

    temp_path = path.with_suffix(path.suffix + ".tmp")
    with pq.ParquetWriter(temp_path, target_schema) as writer:
        if parquet_file is not None:
            for batch in parquet_file.iter_batches():
                if schema_changed:
                    aligned_batch = _align_table_to_schema(batch, target_schema)
                else:
                    aligned_batch = pa.Table.from_batches([batch])
                writer.write_table(aligned_batch)
        writer.write_table(aligned_row)

    if parquet_file is not None:
        del parquet_file

    os.replace(temp_path, path)


def append_impact(entry: ImpactEntry) -> str:
    payload = _prepare_payload(asdict(entry))
    date_str = _entry_date(entry.ts_iso)
    _append_record(f"impact_{date_str}.parquet", payload)
    return payload["run_id"]


def append_feedback(entry: FeedbackEntry) -> str:
    entry_dict = asdict(entry)
    if not entry_dict.get("scenario"):
        entry_dict["scenario"] = "-"
    payload = _prepare_payload(entry_dict)
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
