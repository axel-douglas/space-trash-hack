"""Data ingestion and feature engineering utilities for manufacturing runs.

The module loads raw inventories, process definitions and execution logs from the
``data`` folder, validates them with :mod:`pydantic` models and builds a tidy
``pandas.DataFrame`` that contains both numerical/categorical features and the
labels required for model training (``rigidez``, ``estanqueidad`` and
``consumo_real``).

It also provides persistence helpers that write the dataset to timestamped
Parquet or Delta Lake artefacts inside ``data/processed`` so that experiments
can be reproduced reliably.
"""

from __future__ import annotations

import csv
import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import pandas as pd
from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator, model_validator

LOGGER = logging.getLogger(__name__)

from .paths import LOGS_DIR

INGESTION_ERROR_LOG_PATH = LOGS_DIR / "ingestion.errors.jsonl"


class InventoryRecord(BaseModel):
    """Representation of a single inventory entry from the waste catalog."""

    model_config = ConfigDict(populate_by_name=True, frozen=True)

    inventory_id: str = Field(validation_alias=AliasChoices("inventory_id", "id"))
    category: str
    material_family: str
    mass_kg: float
    volume_l: float
    flags: List[str] = Field(default_factory=list)

    @field_validator("mass_kg", "volume_l")
    @classmethod
    def _ensure_positive(cls, value: float, info: Field) -> float:
        if value <= 0:
            msg = f"{info.field_name.replace('_', ' ')} must be greater than zero"
            raise ValueError(msg)
        return value

    @field_validator("flags", mode="before")
    @classmethod
    def _normalise_flags(cls, value: Any) -> List[str]:
        if value in (None, ""):
            return []
        if isinstance(value, str):
            return [flag.strip() for flag in value.split(",") if flag.strip()]
        if isinstance(value, Iterable):
            return [str(flag).strip() for flag in value if str(flag).strip()]
        raise TypeError("flags must be provided as a string or iterable of strings")

    @property
    def density_kg_m3(self) -> float:
        """Density derived from the mass and volume measurements."""

        # Convert volume from litres to cubic metres before dividing.
        return self.mass_kg / (self.volume_l / 1000.0)


class ProcessRecord(BaseModel):
    """Representation of an available transformation process."""

    model_config = ConfigDict(populate_by_name=True, frozen=True)

    process_id: str = Field(validation_alias=AliasChoices("process_id", "id"))
    name: str
    location: str
    energy_kwh_per_kg: float
    water_l_per_kg: float
    crew_min_per_batch: float
    notes: Optional[str] = None

    @field_validator("energy_kwh_per_kg", "water_l_per_kg", "crew_min_per_batch")
    @classmethod
    def _ensure_non_negative(cls, value: float, info: Field) -> float:
        if value < 0:
            msg = f"{info.field_name.replace('_', ' ')} cannot be negative"
            raise ValueError(msg)
        return value


class ProcessRunLog(BaseModel):
    """Validated execution log for a single manufacturing run."""

    model_config = ConfigDict(populate_by_name=True, extra="allow")

    run_id: str
    timestamp: datetime
    inventory_id: str
    process_id: str
    batch_mass_kg: float
    measured_energy_kwh: float
    measured_water_l: float
    crew_time_min: float
    rigidity: float = Field(
        validation_alias=AliasChoices("rigidez", "rigidity", "rigidity_score")
    )
    tightness: float = Field(
        validation_alias=AliasChoices("estanqueidad", "tightness", "tightness_score")
    )
    actual_energy_kwh: float = Field(
        validation_alias=AliasChoices(
            "consumo_real", "actual_energy_kwh", "real_energy_kwh", "real_consumption_kwh"
        )
    )
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _capture_extras(cls, data: Mapping[str, Any]) -> Mapping[str, Any]:
        if not isinstance(data, Mapping):
            return data
        # Collect extra keys so they are not lost when validating.
        recognised = set(cls.model_fields)
        extras = {key: value for key, value in data.items() if key not in recognised}
        metadata = dict(data.get("metadata", {}))
        if extras:
            metadata.update(extras)
        updated = dict(data)
        updated["metadata"] = metadata
        return updated

    @field_validator(
        "batch_mass_kg",
        "measured_energy_kwh",
        "measured_water_l",
        "crew_time_min",
        "actual_energy_kwh",
    )
    @classmethod
    def _ensure_non_negative(cls, value: float, info: Field) -> float:
        if info.field_name == "batch_mass_kg" and value <= 0:
            msg = "batch mass kg must be greater than zero"
            raise ValueError(msg)
        if value < 0:
            msg = f"{info.field_name.replace('_', ' ')} cannot be negative"
            raise ValueError(msg)
        return value

    @field_validator("rigidity", "tightness")
    @classmethod
    def _ensure_unit_interval(cls, value: float, info: Field) -> float:
        if not 0.0 <= value <= 1.0:
            msg = f"{info.field_name.replace('_', ' ')} must be between 0 and 1"
            raise ValueError(msg)
        return value

    @field_validator("timestamp", mode="before")
    @classmethod
    def _parse_timestamp(cls, value: Any) -> Any:
        if isinstance(value, datetime):
            return value
        if isinstance(value, (int, float)):
            return datetime.fromtimestamp(value, tz=UTC)
        if isinstance(value, str):
            try:
                return datetime.fromisoformat(value.replace("Z", "+00:00"))
            except ValueError as exc:  # pragma: no cover - defensive logging.
                LOGGER.warning("Failed to parse timestamp %s: %s", value, exc)
        return value


class GoldFeatureRow(BaseModel):
    """Schema for curated feature entries used by the training pipeline."""

    model_config = ConfigDict(extra="allow")

    recipe_id: str
    process_id: str
    regolith_pct: float = Field(ge=0.0, le=1.0)
    total_mass_kg: float = Field(gt=0.0)
    mass_input_kg: float = Field(gt=0.0)
    num_items: int = Field(ge=1)
    density_kg_m3: float = Field(gt=0.0)
    moisture_frac: float = Field(ge=0.0)
    difficulty_index: float = Field(ge=0.0)
    problematic_mass_frac: float = Field(ge=0.0)
    problematic_item_frac: float = Field(ge=0.0)
    aluminum_frac: float = Field(ge=0.0)
    foam_frac: float = Field(ge=0.0)
    eva_frac: float = Field(ge=0.0)
    textile_frac: float = Field(ge=0.0)
    multilayer_frac: float = Field(ge=0.0)
    glove_frac: float = Field(ge=0.0)
    polyethylene_frac: float = Field(ge=0.0)
    carbon_fiber_frac: float = Field(ge=0.0)
    hydrogen_rich_frac: float = Field(ge=0.0)
    packaging_frac: float = Field(ge=0.0)
    gas_recovery_index: float = Field(ge=0.0)
    logistics_reuse_index: float = Field(ge=0.0)
    oxide_sio2: float = Field(ge=0.0)
    oxide_feot: float = Field(ge=0.0)
    oxide_mgo: float = Field(ge=0.0)
    oxide_cao: float = Field(ge=0.0)
    oxide_so3: float = Field(ge=0.0)
    oxide_h2o: float = Field(ge=0.0)


class GoldLabelRow(BaseModel):
    """Schema for curated label entries used by the training pipeline."""

    model_config = ConfigDict(extra="allow")

    recipe_id: str
    process_id: str
    rigidez: float = Field(ge=0.0, le=1.0)
    estanqueidad: float = Field(ge=0.0, le=1.0)
    energy_kwh: float = Field(ge=0.0)
    water_l: float = Field(ge=0.0)
    crew_min: float = Field(ge=0.0)
    tightness_pass: int = Field(ge=0)
    rigidity_level: int = Field(ge=0)
    label_source: str = Field(min_length=1)
    label_weight: float = Field(gt=0.0)
    provenance: str = Field(min_length=1)
    conf_lo_rigidez: float = Field(ge=0.0)
    conf_hi_rigidez: float = Field(ge=0.0)
    conf_lo_estanqueidad: float = Field(ge=0.0)
    conf_hi_estanqueidad: float = Field(ge=0.0)
    conf_lo_energy_kwh: float = Field(ge=0.0)
    conf_hi_energy_kwh: float = Field(ge=0.0)
    conf_lo_water_l: float = Field(ge=0.0)
    conf_hi_water_l: float = Field(ge=0.0)
    conf_lo_crew_min: float = Field(ge=0.0)
    conf_hi_crew_min: float = Field(ge=0.0)

def _read_csv_records(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return [dict(row) for row in reader]


def _append_validation_error(
    source_file: Path,
    *,
    error: Exception,
    row_index: Optional[int] = None,
    raw_entry: Optional[Any] = None,
) -> None:
    """Persist validation errors to a JSON lines log file."""

    payload: Dict[str, Any] = {
        "source_file": str(source_file),
        "error": str(error),
    }
    if row_index is not None:
        payload["row_index"] = row_index
    if raw_entry is not None:
        payload["raw_entry"] = raw_entry

    try:
        INGESTION_ERROR_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with INGESTION_ERROR_LOG_PATH.open("a", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, default=str)
            handle.write("\n")
    except Exception as exc:  # pragma: no cover - defensive logging.
        LOGGER.warning("Failed to write ingestion error log: %s", exc)


def load_inventory(path: Path) -> List[InventoryRecord]:
    """Load and validate the inventory catalog from ``path``."""

    if not path.exists():
        raise FileNotFoundError(f"Inventory file not found: {path}")
    records: List[InventoryRecord] = []
    for index, row in enumerate(_read_csv_records(path), start=1):
        try:
            record = InventoryRecord.model_validate(row)
        except Exception as exc:
            LOGGER.warning(
                "Skipping invalid inventory record at row %s from %s: %s",
                index,
                path,
                exc,
            )
            _append_validation_error(path, row_index=index, raw_entry=row, error=exc)
            continue
        records.append(record)
    return records


def load_process_catalog(path: Path) -> List[ProcessRecord]:
    """Load and validate the process catalog from ``path``."""

    if not path.exists():
        raise FileNotFoundError(f"Process catalog file not found: {path}")
    records: List[ProcessRecord] = []
    for index, row in enumerate(_read_csv_records(path), start=1):
        try:
            record = ProcessRecord.model_validate(row)
        except Exception as exc:
            LOGGER.warning(
                "Skipping invalid process record at row %s from %s: %s",
                index,
                path,
                exc,
            )
            _append_validation_error(path, row_index=index, raw_entry=row, error=exc)
            continue
        records.append(record)
    return records


def _load_json_file(path: Path) -> List[Mapping[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, Mapping):
        # Accept single entry or nested list under common keys.
        if "runs" in payload and isinstance(payload["runs"], Sequence):
            return list(payload["runs"])
        if "data" in payload and isinstance(payload["data"], Sequence):
            return list(payload["data"])
        return [payload]
    if isinstance(payload, Sequence):
        return list(payload)
    msg = f"Unsupported JSON structure in {path}"
    raise ValueError(msg)


def load_process_logs(path: Path) -> List[ProcessRunLog]:
    """Load and validate execution logs from ``path``.

    The path can refer to a directory (``*.json``/``*.csv`` files) or to a
    single file. Records that fail validation are logged and skipped.
    """

    if not path.exists():
        LOGGER.info("No process logs found at %s", path)
        return []

    if path.is_file():
        files = [path]
    else:
        files = sorted(
            [
                candidate
                for candidate in path.glob("**/*")
                if candidate.suffix.lower() in {".json", ".csv"}
            ]
        )

    records: List[ProcessRunLog] = []
    for file_path in files:
        raw_entries: List[Mapping[str, Any]]
        try:
            if file_path.suffix.lower() == ".json":
                raw_entries = _load_json_file(file_path)
            else:
                raw_entries = _read_csv_records(file_path)
        except Exception as exc:  # pragma: no cover - defensive logging.
            LOGGER.warning("Failed to read log file %s: %s", file_path, exc)
            continue

        for index, entry in enumerate(raw_entries, start=1):
            try:
                record = ProcessRunLog.model_validate(entry)
            except Exception as exc:  # pragma: no cover - defensive logging.
                LOGGER.warning("Skipping invalid log record from %s: %s", file_path, exc)
                _append_validation_error(
                    file_path,
                    row_index=index,
                    raw_entry=entry if isinstance(entry, Mapping) else {"raw": entry},
                    error=exc,
                )
                continue
            records.append(record)
    return records


def build_feature_dataframe(
    inventory: Sequence[InventoryRecord],
    processes: Sequence[ProcessRecord],
    runs: Sequence[ProcessRunLog],
) -> pd.DataFrame:
    """Combine sources into a feature-rich :class:`~pandas.DataFrame`.

    Parameters
    ----------
    inventory, processes, runs:
        Validated datasets returned by the ``load_*`` helpers.
    """

    if not runs:
        LOGGER.info("No runs available to build a dataset")
        return pd.DataFrame(
            columns=[
                "run_id",
                "timestamp",
                "category",
                "material_family",
                "process_id",
                "process_name",
                "mass_kg",
                "volume_l",
                "density_kg_m3",
                "batch_mass_kg",
                "specific_energy_kwh_kg",
                "water_intensity_l_kg",
                "crew_intensity_min_kg",
                "flag_count",
                "rigidez",
                "estanqueidad",
                "consumo_real",
            ]
        )

    if inventory:
        inventory_rows: List[Dict[str, Any]] = []
        for record in inventory:
            payload = record.model_dump()
            payload["density_kg_m3"] = record.density_kg_m3
            payload["flag_count"] = len(record.flags)
            inventory_rows.append(payload)
        inventory_df = pd.DataFrame(inventory_rows)
    else:
        inventory_df = pd.DataFrame(columns=["inventory_id", "density_kg_m3", "flag_count"])

    process_df = pd.DataFrame([record.model_dump() for record in processes])
    if not process_df.empty:
        process_df = process_df.rename(columns={"process_id": "process_id"})

    runs_df = pd.DataFrame([
        record.model_dump(mode="python", round_trip=True) for record in runs
    ])

    # Normalise timestamp to pandas datetime and ensure label names in Spanish.
    runs_df["timestamp"] = pd.to_datetime(runs_df["timestamp"], utc=True)
    runs_df = runs_df.rename(
        columns={
            "rigidity": "rigidez",
            "tightness": "estanqueidad",
            "actual_energy_kwh": "consumo_real",
        }
    )

    dataset = runs_df.merge(inventory_df, how="left", on="inventory_id", suffixes=("", "_inv"))
    dataset = dataset.merge(process_df, how="left", on="process_id", suffixes=("", "_proc"))

    # Compute engineered numerical features.
    if "density_kg_m3" in dataset:
        dataset["density_kg_m3"] = dataset["density_kg_m3"].astype(float)
    if "flag_count" in dataset:
        dataset["flag_count"] = dataset["flag_count"].fillna(0).astype(int)
    else:
        dataset["flag_count"] = 0
    dataset["specific_energy_kwh_kg"] = dataset["measured_energy_kwh"] / dataset["batch_mass_kg"]
    dataset["water_intensity_l_kg"] = dataset["measured_water_l"] / dataset["batch_mass_kg"]
    dataset["crew_intensity_min_kg"] = dataset["crew_time_min"] / dataset["batch_mass_kg"]
    dataset["energy_gap_kwh"] = dataset["measured_energy_kwh"] - dataset["consumo_real"]

    # Ensure categorical columns carry the proper dtype for downstream encoders.
    for column in ["category", "material_family", "location", "name"]:
        if column in dataset:
            dataset[column] = dataset[column].astype("category")

    if "name" in dataset:
        dataset = dataset.rename(columns={"name": "process_name"})

    # Arrange final column order for readability.
    ordered_columns = [
        "run_id",
        "timestamp",
        "inventory_id",
        "category",
        "material_family",
        "flags",
        "flag_count",
        "process_id",
        "process_name",
        "location",
        "mass_kg",
        "volume_l",
        "density_kg_m3",
        "batch_mass_kg",
        "specific_energy_kwh_kg",
        "water_intensity_l_kg",
        "crew_intensity_min_kg",
        "energy_gap_kwh",
        "rigidez",
        "estanqueidad",
        "consumo_real",
        "metadata",
    ]
    # Preserve additional engineered columns that may not be in the default list.
    ordered_columns.extend(column for column in dataset.columns if column not in ordered_columns)
    return dataset[ordered_columns]


def persist_dataset(
    dataset: pd.DataFrame,
    output_dir: Path,
    format_: str = "parquet",
    run_id: Optional[str] = None,
) -> Path:
    """Persist the dataset with a timestamped artefact name.

    Parameters
    ----------
    dataset:
        The dataframe returned by :func:`build_feature_dataframe`.
    output_dir:
        Target folder inside ``data/processed``.
    format_:
        ``"parquet"`` (default) or ``"delta"``. The latter requires the
        :mod:`deltalake` package to be installed.
    run_id:
        Optional identifier. When omitted a UTC timestamp is used.
    """

    if dataset.empty:
        raise ValueError("Cannot persist an empty dataset")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_label = run_id or datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")

    format_lower = format_.lower()
    if format_lower == "parquet":
        file_path = output_dir / f"dataset_{run_label}.parquet"
        dataset.to_parquet(file_path, index=False)
        return file_path
    if format_lower == "delta":
        try:
            from deltalake import write_deltalake
        except ModuleNotFoundError as exc:  # pragma: no cover - dependency guard.
            raise RuntimeError("Delta Lake support requires the 'deltalake' package") from exc

        delta_path = output_dir / f"dataset_{run_label}.delta"
        write_deltalake(str(delta_path), dataset, mode="overwrite")
        return delta_path
    raise ValueError(f"Unsupported persistence format: {format_}")


class DataPipeline:
    """High level interface for building and persisting training datasets."""

    def __init__(self, data_dir: Path | str = Path("data")) -> None:
        self.data_dir = Path(data_dir)

    @property
    def inventory_path(self) -> Path:
        return self.data_dir / "waste_inventory_sample.csv"

    @property
    def process_path(self) -> Path:
        return self.data_dir / "process_catalog.csv"

    @property
    def logs_path(self) -> Path:
        return self.data_dir / "logs"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    def load_sources(self) -> tuple[List[InventoryRecord], List[ProcessRecord], List[ProcessRunLog]]:
        inventory = load_inventory(self.inventory_path)
        processes = load_process_catalog(self.process_path)
        runs = load_process_logs(self.logs_path)
        return inventory, processes, runs

    def build_dataset(self) -> pd.DataFrame:
        inventory, processes, runs = self.load_sources()
        return build_feature_dataframe(inventory, processes, runs)

    def ingest_runs(self, raw_runs: Iterable[Mapping[str, Any]]) -> List[ProcessRunLog]:
        """Validate arbitrary log entries and return structured objects."""

        validated: List[ProcessRunLog] = []
        for entry in raw_runs:
            record = ProcessRunLog.model_validate(entry)
            validated.append(record)
        return validated

    def persist(self, dataset: pd.DataFrame, format_: str = "parquet", run_id: Optional[str] = None) -> Path:
        """Persist the dataset to ``data/processed`` using ``format_``."""

        target_dir = self.processed_dir / format_.lower()
        return persist_dataset(dataset, target_dir, format_=format_, run_id=run_id)

