"""Construction of curated gold datasets from NASA source material.

This module consolidates a handful of public NASA data tables describing
non-metabolic waste streams, logistics reuse scenarios and waste management
metrics (trash-to-gas) together with the MGS-1 simulant characterization. The
goal is to produce deterministic feature/label tables that can be consumed by
the model training pipeline as high quality "gold" references.

The builder keeps the transformation logic in Python so it can be unit tested
and re-executed whenever new raw data drops. The resulting parquet artefacts
are written to ``data/gold`` and validated against the pydantic models
defined in :mod:`app.modules.data_pipeline`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np
import pandas as pd

from app.modules import generator
from app.modules.dataset_validation import validate_waste_inventory
from app.modules.data_sources import REGOLITH_CHARACTERIZATION
from app.modules.data_pipeline import GoldFeatureRow, GoldLabelRow
from app.modules.model_training import FEATURE_COLUMNS
from app.modules.label_mapper import derive_recipe_id
from .paths import DATA_ROOT, GOLD_DIR

DATASETS_ROOT = DATA_ROOT.parent / "datasets"
RAW_DIR = DATASETS_ROOT / "raw"

MOXIE_PEAK_O2_G_PER_HOUR = 12.0
MOXIE_POWER_W = 300.0
MOXIE_ENERGY_KWH_PER_KG = (MOXIE_POWER_W / 1000.0) / (MOXIE_PEAK_O2_G_PER_HOUR / 1000.0)
CREW_O2_KG_PER_DAY = 0.84
SABATIER_RECOVERY_FRACTION = 0.54
WATER_RECOVERY_RATE = 0.98
CREW_WATER_L_PER_DAY = 3.8
WATER_EXTRACTION_KWH_PER_KG = 0.45
MAINTENANCE_CREW_HOURS = 24.0

NASA_RIGIDITY_MPA = {
    "P02": 50.0,  # PLA + basalto impreso en 3D
    "P03": 206.0,  # Regolito sinterizado basáltico
}

NASA_TIGHTNESS_LEAK_LPH = {
    "P02": 0.05,  # Sellado hermético en hábitats 3D (prueba hidrostática)
    "P03": 0.2,
}

RIGIDITY_MIN_MPA = 20.0  # Concreto Portland convencional
RIGIDITY_MAX_MPA = 220.0  # Margen superior de experimentos con regolito
MAX_LEAK_LPH = 10.0

AUTONOMOUS_PROCESSES = {"P03"}

MISSION_SCENARIO_MAP = {
    "Gateway Phase I": "Short Sortie",
    "Gateway Phase II": "Outpost 12.1",
    "Mars Transit": "Outpost 12.1 (max)",
}

CATEGORY_PROCESS_MAP = {
    "Aggregate": "P03",
    "Trash-to-Supply": "P02",
}

MASS_KEYS = ("packaging", "structural", "eva", "textile")


@dataclass(slots=True)
class GoldRecord:
    """Container holding the artefacts generated for a mission scenario."""

    mission: str
    category: str
    scenario: str
    process: pd.Series
    regolith_pct: float
    picks: pd.DataFrame
    weights: np.ndarray
    features: dict[str, float]
    labels: dict[str, float]


def _load_inventory() -> pd.DataFrame:
    inventory = pd.read_csv(RAW_DIR / "nasa_waste_inventory.csv")
    validate_waste_inventory(inventory, dataset_label="el archivo nasa_waste_inventory.csv")
    prepared = generator.prepare_waste_frame(inventory)
    return prepared


def _load_trash_to_gas() -> pd.DataFrame:
    return pd.read_csv(RAW_DIR / "nasa_trash_to_gas.csv")


def _load_logistics() -> pd.DataFrame:
    return pd.read_csv(RAW_DIR / "logistics_to_living.csv")


def _load_regolith_properties() -> pd.Series:
    properties = pd.read_csv(RAW_DIR / "mgs1_properties.csv")
    series = (
        properties.assign(property=lambda df: df["property"].str.lower())
        .set_index("property")["value"]
        .astype(float)
    )
    amorphous = float(series.get("amorphous_fraction", 0.3))
    water_release = float(series.get("water_release", 3.0))
    regolith_pct = float(np.clip(0.18 + 0.4 * amorphous + 0.01 * water_release, 0.15, 0.45))
    return pd.Series({
        "baseline_pct": regolith_pct,
        "water_release": water_release,
        "amorphous_fraction": amorphous,
    })


def _subset_by_category(tokens: pd.Series, keywords: Iterable[str]) -> pd.Series:
    pattern = "|".join(keywords)
    return tokens.str.contains(pattern, case=False, regex=True, na=False)


def _select_inventory(inventory: pd.DataFrame, key: str, mass_kg: float) -> pd.DataFrame:
    if mass_kg <= 0:
        return pd.DataFrame(columns=inventory.columns)

    if key == "packaging":
        mask = _subset_by_category(
            inventory["tokens"],
            ("packaging", "bubble", "pouch", "foam", "overwrap"),
        )
    elif key == "structural":
        mask = _subset_by_category(
            inventory["tokens"],
            ("structural", "strut", "composite", "partition"),
        )
    elif key == "eva":
        mask = _subset_by_category(
            inventory["tokens"],
            ("eva", "ctb", "glove", "nomex"),
        )
    else:  # textiles and the default branch fall back to fabrics/cloth entries.
        mask = _subset_by_category(
            inventory["tokens"],
            ("fabric", "cloth", "towel", "garment"),
        )

    subset = inventory.loc[mask].copy()
    if subset.empty:
        subset = inventory.copy()

    subset = subset.sort_values("kg", ascending=False).head(5).copy()
    weights = subset["kg"].astype(float)
    if weights.sum() <= 0:
        weights = pd.Series(np.ones(len(subset)), index=subset.index)
    fractions = weights / weights.sum()
    subset["kg"] = fractions * mass_kg
    density = subset["density_kg_m3"].replace(0, np.nan).fillna(250.0)
    subset["volume_l"] = subset["kg"] / density * 1000.0
    return subset


def _compose_material_mix(
    inventory: pd.DataFrame,
    breakdown: dict[str, float],
) -> pd.DataFrame:
    picks = [
        _select_inventory(inventory, key, breakdown.get(key, 0.0))
        for key in MASS_KEYS
        if breakdown.get(key, 0.0) > 0
    ]
    picks = [frame for frame in picks if not frame.empty]
    if not picks:
        return pd.DataFrame(columns=inventory.columns)
    merged = pd.concat(picks, ignore_index=True)
    total_mass = merged["kg"].sum()
    if total_mass <= 0:
        return pd.DataFrame(columns=inventory.columns)
    merged["pct_mass"] = merged["kg"] / total_mass * 100.0
    volume_sum = merged["volume_l"].sum()
    if volume_sum <= 0:
        merged["pct_volume"] = 0.0
    else:
        merged["pct_volume"] = merged["volume_l"] / volume_sum * 100.0
    return merged


def _build_breakdown(scenario_row: pd.Series) -> dict[str, float]:
    goods = float(scenario_row.get("goods_kg", 0.0))
    packaging = float(scenario_row.get("packaging_kg", 0.0))
    residual = float(scenario_row.get("residual_waste_kg", 0.0))
    outfitting = float(scenario_row.get("outfitting_replaced_kg", 0.0))
    eva_mass = float(scenario_row.get("ctb_count", 0.0)) * 1.2
    eva_mass = min(eva_mass, goods * 0.25 if goods > 0 else eva_mass)
    textile_mass = max(goods - eva_mass, 0.0)
    structural = residual + outfitting + max(goods * 0.2, 0.0)
    breakdown = {
        "packaging": packaging,
        "structural": structural,
        "eva": eva_mass,
        "textile": textile_mass,
    }
    return breakdown


def _compute_confidence_bounds(value: float, rel_width: float = 0.08) -> tuple[float, float]:
    span = abs(value) * rel_width
    return max(0.0, value - span), value + span


def _normalise_rigidity(mpa: float) -> float:
    span = max(RIGIDITY_MAX_MPA - RIGIDITY_MIN_MPA, 1.0)
    score = (float(mpa) - RIGIDITY_MIN_MPA) / span
    return float(np.clip(score, 0.0, 1.0))


def _normalise_tightness(leak_lph: float) -> float:
    score = 1.0 - float(leak_lph) / max(MAX_LEAK_LPH, 1.0)
    return float(np.clip(score, 0.0, 1.0))


def _build_gold_records() -> Iterator[GoldRecord]:
    inventory = _load_inventory()
    trash_to_gas = _load_trash_to_gas()
    logistics = _load_logistics().set_index("scenario")
    regolith = _load_regolith_properties()
    process_catalog = pd.read_csv(DATA_ROOT / "process_catalog.csv")

    for _, row in trash_to_gas.iterrows():
        mission = str(row["mission"])
        category = str(row["category"])
        scenario_name = MISSION_SCENARIO_MAP.get(mission)
        if scenario_name is None or scenario_name not in logistics.index:
            continue

        process_id = CATEGORY_PROCESS_MAP.get(category, "P02")
        process = process_catalog.loc[process_catalog["process_id"] == process_id].iloc[0]
        scenario = logistics.loc[scenario_name]

        breakdown = _build_breakdown(scenario)
        picks = _compose_material_mix(inventory, breakdown)
        if picks.empty:
            continue

        total_mass = picks["kg"].sum()
        weights = (picks["kg"] / total_mass).to_numpy(dtype=float)

        regolith_pct = 0.0
        if process_id == "P03":
            regolith_pct = float(np.clip(regolith["baseline_pct"], 0.1, 0.45))

        features = generator.compute_feature_vector(picks, weights, process, regolith_pct)
        features["total_mass_kg"] = float(total_mass)
        features["mass_input_kg"] = float(total_mass)
        features["num_items"] = int(len(picks))
        features["mission"] = mission
        features["scenario"] = scenario_name

        for feature_name, baseline in REGOLITH_CHARACTERIZATION.feature_items:
            features.setdefault(feature_name, float(baseline) * float(regolith_pct))

        recipe_id = derive_recipe_id(picks, process["process_id"], features)
        features["recipe_id"] = recipe_id

        heuristics = generator.heuristic_props(picks, process, weights, regolith_pct)
        packaging_frac = float(features.get("packaging_frac", 0.0))
        eva_frac = float(features.get("eva_frac", 0.0))

        rigidity_mpa = NASA_RIGIDITY_MPA.get(process_id, 60.0)
        rigidity_mpa += 20.0 * float(np.clip(heuristics.rigidity - 0.5, -0.5, 0.5))
        if process_id == "P03":
            rigidity_mpa += 15.0 * regolith_pct
        rigidity = _normalise_rigidity(rigidity_mpa)

        base_leak = NASA_TIGHTNESS_LEAK_LPH.get(process_id, 1.0)
        leak_modifier = 1.0 - 0.4 * packaging_frac - 0.1 * eva_frac
        leak_modifier += 0.25 * regolith_pct
        leak_rate = max(base_leak * float(np.clip(leak_modifier, 0.3, 1.5)), 0.01)
        tightness = _normalise_tightness(leak_rate)

        crew_days = float(scenario.get("crew_days", 0.0))
        crew_count = float(scenario.get("crew_count", 1.0))
        water_demand = crew_count * crew_days * CREW_WATER_L_PER_DAY
        recovered_water = water_demand * WATER_RECOVERY_RATE
        water_l = water_demand - recovered_water

        o2_need = crew_count * crew_days * CREW_O2_KG_PER_DAY
        closed_loop = o2_need * SABATIER_RECOVERY_FRACTION
        o2_external = max(o2_need - closed_loop, 0.0)
        moxie_energy = o2_external * MOXIE_ENERGY_KWH_PER_KG

        mission_mass = max(total_mass, scenario.get("packaging_kg", 0.0) + scenario.get("residual_waste_kg", 0.0))
        process_energy = float(process.get("energy_kwh_per_kg", 0.0)) * mission_mass
        water_energy = recovered_water * WATER_EXTRACTION_KWH_PER_KG
        energy_kwh = process_energy + water_energy + moxie_energy

        crew_hours = MAINTENANCE_CREW_HOURS
        if process_id in AUTONOMOUS_PROCESSES:
            crew_hours = 0.0
        else:
            crew_hours += 0.1 * crew_days
        crew_min = crew_hours * 60.0

        label_weight = float(1.0 + 0.002 * mission_mass + 0.15 * np.log1p(crew_count))
        provenance = f"nasa/{mission.lower().replace(' ', '_')}"

        tight_pass = int(tightness >= 0.6)
        rigidity_level = 1 if rigidity < 0.5 else 2 if rigidity < 0.8 else 3

        labels: dict[str, float] = {
            "recipe_id": recipe_id,
            "process_id": str(process_id),
            "rigidez": rigidity,
            "estanqueidad": tightness,
            "energy_kwh": energy_kwh,
            "water_l": water_l,
            "crew_min": float(crew_min),
            "tightness_pass": tight_pass,
            "rigidity_level": rigidity_level,
            "label_source": "mission",
            "label_weight": label_weight,
            "provenance": provenance,
            "mission": mission,
            "scenario": scenario_name,
        }

        for metric, value in (
            ("rigidez", rigidity),
            ("estanqueidad", tightness),
            ("energy_kwh", energy_kwh),
            ("water_l", water_l),
            ("crew_min", float(crew_min)),
        ):
            lo, hi = _compute_confidence_bounds(value)
            labels[f"conf_lo_{metric}"] = lo
            labels[f"conf_hi_{metric}"] = hi

        features["process_id"] = str(process_id)

        yield GoldRecord(
            mission=mission,
            category=category,
            scenario=scenario_name,
            process=process,
            regolith_pct=regolith_pct,
            picks=picks,
            weights=weights,
            features=features,
            labels=labels,
        )


def build_gold_dataset(
    output_dir: Path | str | None = None,
    *,
    return_frames: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    """Build and persist the gold feature/label tables.

    Parameters
    ----------
    output_dir:
        Destination folder for the parquet artefacts. When omitted the default
        ``data/gold`` directory is used.
    return_frames:
        When ``True`` the in-memory dataframes are returned alongside the
        persistence step. This is particularly handy for tests.
    """

    records = list(_build_gold_records())
    if not records:
        raise RuntimeError("No se pudieron generar registros gold a partir de los datos NASA")

    feature_rows = [rec.features for rec in records]
    label_rows = [rec.labels for rec in records]

    features_df = pd.DataFrame(feature_rows)
    labels_df = pd.DataFrame(label_rows)

    features_df = features_df.drop(columns=["mission", "scenario"], errors="ignore")
    labels_df = labels_df.drop(columns=["mission", "scenario"], errors="ignore")

    ordered_columns = ["recipe_id", "process_id", *FEATURE_COLUMNS[1:]]
    missing = [column for column in ordered_columns if column not in features_df.columns]
    if missing:
        raise ValueError(f"Faltan columnas de características esperadas: {', '.join(missing)}")
    features_df = features_df[ordered_columns]

    features_df = features_df.sort_values(["process_id", "recipe_id"]).reset_index(drop=True)
    labels_df = labels_df.sort_values(["process_id", "recipe_id"]).reset_index(drop=True)

    validated_features = [GoldFeatureRow.model_validate(row).model_dump() for row in features_df.to_dict("records")]
    validated_labels = [GoldLabelRow.model_validate(row).model_dump() for row in labels_df.to_dict("records")]

    features_df = pd.DataFrame(validated_features)
    labels_df = pd.DataFrame(validated_labels)

    target_dir = Path(output_dir) if output_dir is not None else GOLD_DIR
    target_dir.mkdir(parents=True, exist_ok=True)
    features_path = target_dir / "features.parquet"
    labels_path = target_dir / "labels.parquet"
    features_df.to_parquet(features_path, index=False)
    labels_df.to_parquet(labels_path, index=False)

    if return_frames:
        return features_df, labels_df
    return None


def ensure_gold_dataset(output_dir: Path | str | None = None) -> tuple[Path, Path]:
    """Ensure the curated gold artefacts exist on disk.

    Parameters
    ----------
    output_dir:
        Optional base directory where the artefacts should live. When omitted
        the default :data:`GOLD_DIR` location is used.

    Returns
    -------
    tuple[Path, Path]
        Paths to the features and labels Parquet files respectively.
    """

    target_dir = Path(output_dir) if output_dir is not None else GOLD_DIR
    features_path = target_dir / "features.parquet"
    labels_path = target_dir / "labels.parquet"

    if features_path.exists() and labels_path.exists():
        return features_path, labels_path

    build_gold_dataset(target_dir)
    return features_path, labels_path


def generate_gold_records() -> list[GoldRecord]:
    """Return the list of in-memory gold records without persisting artefacts."""

    return list(_build_gold_records())


__all__ = [
    "GoldRecord",
    "build_gold_dataset",
    "ensure_gold_dataset",
    "generate_gold_records",
]

