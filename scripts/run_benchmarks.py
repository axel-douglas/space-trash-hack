"""Utility to benchmark Rex-AI predictions against heuristic baselines.

The command evaluates three fixed scenarios that are representative of the
hackathon narrative:

* Multicapa + laminación térmica (P02)
* Espuma técnica + regolito MGS-1 + sinterizado (P03)
* Reconfiguración de kits CTB y estructuras (P04)

Para cada escenario el script carga el RandomForest multisalida ya entrenado
desde ``data/models`` mediante :class:`app.modules.ml_models.ModelRegistry`,
calcula las predicciones y las compara frente al baseline determinístico
``heuristic_props``. Las tablas resultantes incluyen el error absoluto medio
(MAE), el RMSE y los intervalos de confianza al 95% reportados por el modelo.

Results are persisted under ``data/benchmarks/`` so they can be versioned or
shared alongside the repository. Optionally, the command can perform ablation
studies where relevant groups of engineered features (composición MGS-1,
banderas de materiales NASA y los índices logísticos) se desactivan para medir
su impacto en MAE/RMSE/CI95.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd

from app.modules.generator import compute_feature_vector, heuristic_props, prepare_waste_frame
from app.modules.ml_models import ModelRegistry
DATA_DIR = REPO_ROOT / "data"
WASTE_SAMPLE = DATA_DIR / "waste_inventory_sample.csv"
PROCESS_CATALOG = DATA_DIR / "process_catalog.csv"
BENCHMARK_DIR = DATA_DIR / "benchmarks"
TARGET_COLUMNS: List[str] = ["rigidez", "estanqueidad", "energy_kwh", "water_l", "crew_min"]


def _zeroed_features(features: Mapping[str, object], keys: Iterable[str]) -> Dict[str, object]:
    """Return a copy of *features* where *keys* were set to 0.0 when present."""

    mutated = dict(features)
    for key in keys:
        if key in mutated:
            mutated[key] = 0.0
    return mutated


def _ablation_variants(features: Mapping[str, object]) -> Dict[str, Dict[str, object]]:
    """Produce feature ablations grouped by domain-meaningful subsets."""

    oxide_keys = [key for key in features.keys() if key.startswith("oxide_")]
    oxide_keys.append("regolith_pct")

    nasa_flag_keys = [
        "packaging_frac",
        "aluminum_frac",
        "foam_frac",
        "eva_frac",
        "textile_frac",
        "multilayer_frac",
        "glove_frac",
        "polyethylene_frac",
        "carbon_fiber_frac",
        "hydrogen_rich_frac",
    ]

    logistics_keys = [
        "gas_recovery_index",
        "logistics_reuse_index",
        "problematic_mass_frac",
        "problematic_item_frac",
        "difficulty_index",
    ]

    return {
        "mgs1_composition": _zeroed_features(features, oxide_keys),
        "nasa_flags": _zeroed_features(features, nasa_flag_keys),
        "logistics_indices": _zeroed_features(features, logistics_keys),
    }


@dataclass(frozen=True)
class Scenario:
    """Definition of a fixed benchmark scenario."""

    name: str
    title: str
    description: str
    process_id: str
    waste_ids: Sequence[str]
    regolith_pct: float = 0.0


def _load_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load and normalise waste & process reference tables."""

    if not WASTE_SAMPLE.exists():
        raise FileNotFoundError(
            f"No se encontró el inventario de residuos en {WASTE_SAMPLE}"
        )
    if not PROCESS_CATALOG.exists():
        raise FileNotFoundError(
            f"No se encontró el catálogo de procesos en {PROCESS_CATALOG}"
        )

    waste_df = pd.read_csv(WASTE_SAMPLE)
    waste_df = prepare_waste_frame(waste_df)

    process_df = pd.read_csv(PROCESS_CATALOG)
    process_df["process_id"] = process_df["process_id"].astype(str)
    process_df = process_df.set_index("process_id", drop=False)

    return waste_df, process_df


def _scenario_catalog() -> List[Scenario]:
    return [
        Scenario(
            name="multicapa_laminar",
            title="Multicapa + Laminar",
            description="Films y pouches multicapa prensados en P02",
            process_id="P02",
            waste_ids=["W006", "W007", "W008"],
            regolith_pct=0.0,
        ),
        Scenario(
            name="espuma_mgs1_sinter",
            title="Espuma + MGS-1 + Sinter",
            description="Espumas técnicas con 20% de regolito en P03",
            process_id="P03",
            waste_ids=["W001", "W011", "W015"],
            regolith_pct=0.2,
        ),
        Scenario(
            name="ctb_reconfig",
            title="CTB Reconfig",
            description="Reutilización de CTB + estructuras en P04",
            process_id="P04",
            waste_ids=["W002", "W009", "W010"],
            regolith_pct=0.0,
        ),
    ]


def _normalised_weights(masses: Iterable[float]) -> List[float]:
    masses_arr = np.asarray(list(masses), dtype=float)
    total = float(masses_arr.sum())
    if total <= 0:
        return [0.0 for _ in masses_arr]
    weights = masses_arr / total
    return [float(round(w, 6)) for w in weights]


def _extract_ci(prediction: Mapping[str, object], target: str) -> tuple[float, float]:
    ci_payload = prediction.get("confidence_interval", {})
    if isinstance(ci_payload, Mapping):
        ci_value = ci_payload.get(target)
        if isinstance(ci_value, (list, tuple)) and len(ci_value) == 2:
            return float(ci_value[0]), float(ci_value[1])
    return math.nan, math.nan


def _mae(values: Sequence[float]) -> float:
    arr = np.asarray(list(values), dtype=float)
    return float(np.mean(np.abs(arr))) if arr.size else math.nan


def _rmse(values: Sequence[float]) -> float:
    arr = np.asarray(list(values), dtype=float)
    return float(np.sqrt(np.mean(np.square(arr)))) if arr.size else math.nan


def _nanmean(values: Sequence[float]) -> float:
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return math.nan
    mask = ~np.isnan(arr)
    if not mask.any():
        return math.nan
    return float(np.mean(arr[mask]))


def _build_metrics(predictions_df: pd.DataFrame, group_cols: Sequence[str] | None = None) -> pd.DataFrame:
    group_cols = list(group_cols or [])

    metrics_by_scenario = (
        predictions_df.groupby(group_cols + ["scenario", "target"], as_index=False)
        .agg(
            mae=("absolute_error", _mae),
            rmse=("signed_error", _rmse),
            ci95_low_mean=("ci95_low", _nanmean),
            ci95_high_mean=("ci95_high", _nanmean),
            ci95_width_mean=("ci95_width", _nanmean),
        )
        .assign(level="scenario")
    )

    scenario_overall = (
        predictions_df.groupby(group_cols + ["scenario"], as_index=False)
        .agg(
            mae=("absolute_error", _mae),
            rmse=("signed_error", _rmse),
            ci95_low_mean=("ci95_low", _nanmean),
            ci95_high_mean=("ci95_high", _nanmean),
            ci95_width_mean=("ci95_width", _nanmean),
        )
        .assign(target="overall", level="scenario")
    )

    metrics_by_target = (
        predictions_df.groupby(group_cols + ["target"], as_index=False)
        .agg(
            mae=("absolute_error", _mae),
            rmse=("signed_error", _rmse),
            ci95_low_mean=("ci95_low", _nanmean),
            ci95_high_mean=("ci95_high", _nanmean),
            ci95_width_mean=("ci95_width", _nanmean),
        )
        .assign(scenario="overall", level="target")
    )

    if group_cols:
        overall = (
            predictions_df.groupby(group_cols, as_index=False)
            .agg(
                mae=("absolute_error", _mae),
                rmse=("signed_error", _rmse),
                ci95_low_mean=("ci95_low", _nanmean),
                ci95_high_mean=("ci95_high", _nanmean),
                ci95_width_mean=("ci95_width", _nanmean),
            )
            .assign(scenario="overall", target="overall", level="global")
        )
    else:
        overall = pd.DataFrame(
            [
                {
                    "scenario": "overall",
                    "target": "overall",
                    "mae": _mae(predictions_df["absolute_error"]),
                    "rmse": _rmse(predictions_df["signed_error"]),
                    "ci95_low_mean": _nanmean(predictions_df["ci95_low"]),
                    "ci95_high_mean": _nanmean(predictions_df["ci95_high"]),
                    "ci95_width_mean": _nanmean(predictions_df["ci95_width"]),
                    "level": "global",
                }
            ]
        )

    metrics_df = pd.concat(
        [metrics_by_scenario, scenario_overall, metrics_by_target, overall],
        ignore_index=True,
        sort=False,
    )
    return metrics_df


def run_benchmarks(
    output_dir: Path, output_format: str = "csv", with_ablation: bool = False
) -> dict:
    waste_df, process_df = _load_inputs()

    registry = ModelRegistry()
    if not registry.ready:
        raise RuntimeError(
            "ModelRegistry.ready es False. Descargá `data/models/rexai_regressor.joblib` "
            "o ejecutá el pipeline de entrenamiento antes de correr benchmarks."
        )

    scenarios = _scenario_catalog()
    predictions_records: List[Dict[str, object]] = []

    for scenario in scenarios:
        picks = waste_df[waste_df["id"].isin(scenario.waste_ids)]
        if picks.empty:
            raise RuntimeError(
                f"El escenario {scenario.name} no encontró residuos con IDs {scenario.waste_ids}"
            )

        try:
            process = process_df.loc[scenario.process_id]
        except KeyError as exc:  # pragma: no cover - data corruption
            raise RuntimeError(
                f"Proceso {scenario.process_id} inexistente en process_catalog"
            ) from exc

        weights = _normalised_weights(picks["kg"])
        heuristics = heuristic_props(picks, process, weights, scenario.regolith_pct)
        features = compute_feature_vector(picks, weights, process, scenario.regolith_pct)
        prediction = registry.predict(features)
        if not prediction:
            raise RuntimeError(
                "La inferencia devolvió un diccionario vacío; revisar el modelo entrenado"
            )

        heur_targets = heuristics.to_targets()

        for target in TARGET_COLUMNS:
            model_value = float(prediction[target])
            heuristic_value = float(heur_targets[target])
            diff = model_value - heuristic_value
            ci_low, ci_high = _extract_ci(prediction, target)
            ci_width = ci_high - ci_low if not math.isnan(ci_low) and not math.isnan(ci_high) else math.nan
            predictions_records.append(
                {
                    "scenario": scenario.name,
                    "scenario_title": scenario.title,
                    "target": target,
                    "model_prediction": model_value,
                    "ci95_low": ci_low,
                    "ci95_high": ci_high,
                    "ci95_width": ci_width,
                    "heuristic_prediction": heuristic_value,
                    "signed_error": diff,
                    "absolute_error": abs(diff),
                    "regolith_pct": scenario.regolith_pct,
                    "process_id": scenario.process_id,
                    "waste_ids": ",".join(scenario.waste_ids),
                }
            )

    predictions_df = pd.DataFrame(predictions_records)

    metrics_df = _build_metrics(predictions_df)

    output_dir.mkdir(parents=True, exist_ok=True)

    predictions_path = output_dir / "scenario_predictions"
    metrics_path = output_dir / "scenario_metrics"

    ablation_records: List[Dict[str, object]] = []

    if with_ablation:
        for scenario in scenarios:
            picks = waste_df[waste_df["id"].isin(scenario.waste_ids)]
            weights = _normalised_weights(picks["kg"])
            heuristics = heuristic_props(picks, process_df.loc[scenario.process_id], weights, scenario.regolith_pct)
            base_features = compute_feature_vector(
                picks,
                weights,
                process_df.loc[scenario.process_id],
                scenario.regolith_pct,
            )

            for group, ablated_features in _ablation_variants(base_features).items():
                prediction = registry.predict(ablated_features)
                if not prediction:
                    continue

                heur_targets = heuristics.to_targets()
                for target in TARGET_COLUMNS:
                    model_value = float(prediction[target])
                    heuristic_value = float(heur_targets[target])
                    diff = model_value - heuristic_value
                    ci_low, ci_high = _extract_ci(prediction, target)
                    ci_width = (
                        ci_high - ci_low
                        if not math.isnan(ci_low) and not math.isnan(ci_high)
                        else math.nan
                    )
                    ablation_records.append(
                        {
                            "scenario": scenario.name,
                            "scenario_title": scenario.title,
                            "target": target,
                            "model_prediction": model_value,
                            "ci95_low": ci_low,
                            "ci95_high": ci_high,
                            "ci95_width": ci_width,
                            "heuristic_prediction": heuristic_value,
                            "signed_error": diff,
                            "absolute_error": abs(diff),
                            "regolith_pct": scenario.regolith_pct,
                            "process_id": scenario.process_id,
                            "waste_ids": ",".join(scenario.waste_ids),
                            "ablation_group": group,
                        }
                    )

    if output_format in {"csv", "both"}:
        predictions_df.to_csv(predictions_path.with_suffix(".csv"), index=False)
        metrics_df.to_csv(metrics_path.with_suffix(".csv"), index=False)

    if output_format in {"parquet", "both"}:
        predictions_df.to_parquet(predictions_path.with_suffix(".parquet"), index=False)
        metrics_df.to_parquet(metrics_path.with_suffix(".parquet"), index=False)

    ablation_summary = None
    if with_ablation and ablation_records:
        ablation_df = pd.DataFrame(ablation_records)
        ablation_metrics_df = _build_metrics(ablation_df, ["ablation_group"])

        ablation_predictions_path = output_dir / "ablation_predictions"
        ablation_metrics_path = output_dir / "ablation_metrics"

        if output_format in {"csv", "both"}:
            ablation_df.to_csv(ablation_predictions_path.with_suffix(".csv"), index=False)
            ablation_metrics_df.to_csv(ablation_metrics_path.with_suffix(".csv"), index=False)

        if output_format in {"parquet", "both"}:
            ablation_df.to_parquet(ablation_predictions_path.with_suffix(".parquet"), index=False)
            ablation_metrics_df.to_parquet(ablation_metrics_path.with_suffix(".parquet"), index=False)

        ablation_summary = {
            "predictions_path": str(
                ablation_predictions_path.with_suffix(
                    f".{output_format if output_format != 'both' else 'csv'}"
                )
            ),
            "metrics_path": str(
                ablation_metrics_path.with_suffix(
                    f".{output_format if output_format != 'both' else 'csv'}"
                )
            ),
            "groups": sorted({row["ablation_group"] for row in ablation_records}),
        }

    summary = {
        "model": {
            "ready": registry.ready,
            "trained_at": registry.metadata.get("trained_at"),
            "trained_on": registry.metadata.get("trained_on"),
            "source": registry.metadata.get("model_name", "rexai-rf-ensemble"),
        },
        "predictions_path": str(predictions_path.with_suffix(f".{output_format if output_format != 'both' else 'csv'}")),
        "metrics_path": str(metrics_path.with_suffix(f".{output_format if output_format != 'both' else 'csv'}")),
    }

    if ablation_summary:
        summary["ablation"] = ablation_summary

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark Rex-AI predictions versus heuristic baselines",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=BENCHMARK_DIR,
        help="Directorio donde se guardarán las tablas (por defecto data/benchmarks)",
    )
    parser.add_argument(
        "--format",
        choices=["csv", "parquet", "both"],
        default="csv",
        help="Formato de salida para las tablas",
    )
    parser.add_argument(
        "--with-ablation",
        action="store_true",
        help="Incluye barridos de ablation desactivando grupos de features",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Sequence[str] | None = None) -> dict:
    args = parse_args(argv)
    return run_benchmarks(args.output_dir, args.format, with_ablation=args.with_ablation)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
