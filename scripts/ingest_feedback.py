"""CLI utility to merge astronaut feedback into the gold dataset."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

import pandas as pd

from app.modules import model_training
from app.modules.label_mapper import CLASS_TARGET_COLUMNS, TARGET_COLUMNS

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FEEDBACK = ROOT / "feedback" / "recipes.parquet"
DEFAULT_GOLD_DIR = ROOT / "datasets" / "gold"

FEATURE_COLUMNS = list(model_training.FEATURE_COLUMNS)
TARGET_LIST = list(TARGET_COLUMNS)
CLASS_TARGET_LIST = list(CLASS_TARGET_COLUMNS)
KEY_COLUMNS = ["recipe_id", "process_id"]


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_existing(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def _merge_tables(existing: pd.DataFrame, incoming: pd.DataFrame) -> pd.DataFrame:
    if existing.empty:
        return incoming.reset_index(drop=True)

    combined = pd.concat([existing, incoming], ignore_index=True, sort=False)
    combined = combined.drop_duplicates(subset=KEY_COLUMNS, keep="last")
    combined = combined.reset_index(drop=True)
    return combined


def _prepare_feedback(paths: Sequence[str]) -> pd.DataFrame:
    df = model_training.load_feedback_logs(paths)
    if df.empty:
        raise ValueError(
            "No se encontraron archivos de feedback válidos. "
            "Revisa feedback/README.md para generar un recipes.parquet local."
        )

    prepared = model_training.prepare_feedback_dataframe(df)

    missing_keys = [col for col in KEY_COLUMNS if col not in prepared.columns]
    if missing_keys:
        raise ValueError(
            "Los archivos de feedback deben incluir las columnas clave: "
            + ", ".join(missing_keys)
        )

    missing_features = [
        col for col in FEATURE_COLUMNS if col not in prepared.columns
    ]
    if missing_features:
        raise ValueError(
            "Faltan columnas de features requeridas en el feedback: "
            + ", ".join(missing_features)
        )

    for column in TARGET_LIST + CLASS_TARGET_LIST:
        if column not in prepared.columns:
            raise ValueError(
                "Falta la columna objetivo en el feedback: " + column
            )

    prepared = prepared.copy()
    prepared["ingested_at"] = _timestamp()
    return prepared


def _split_feature_label(prepared: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    feature_cols = KEY_COLUMNS + FEATURE_COLUMNS
    feature_cols = list(dict.fromkeys(feature_cols))  # dedupe preserving order
    feature_df = prepared[feature_cols + ["ingested_at"]].copy()

    label_cols = KEY_COLUMNS + ["label_source", "label_weight"] + TARGET_LIST + CLASS_TARGET_LIST
    confidence_cols = [
        col for col in prepared.columns if col.startswith("conf_lo_") or col.startswith("conf_hi_")
    ]
    extra_cols = [
        col
        for col in prepared.columns
        if col not in feature_cols
        and col not in label_cols
        and col not in {"ingested_at"}
    ]
    label_df = prepared[label_cols + confidence_cols].copy()
    for column in extra_cols:
        label_df[column] = prepared[column]

    label_df["ingested_at"] = prepared["ingested_at"]
    return feature_df, label_df


def ingest_feedback(
    feedback_paths: Sequence[str] | None,
    *,
    gold_dir: Path = DEFAULT_GOLD_DIR,
    dry_run: bool = False,
) -> Mapping[str, object]:
    paths = list(feedback_paths or [str(DEFAULT_FEEDBACK)])

    prepared = _prepare_feedback(paths)
    feature_df, label_df = _split_feature_label(prepared)

    gold_dir = Path(gold_dir)
    gold_dir.mkdir(parents=True, exist_ok=True)

    features_path = gold_dir / "features.parquet"
    labels_path = gold_dir / "labels.parquet"

    existing_features = _load_existing(features_path)
    existing_labels = _load_existing(labels_path)

    merged_features = _merge_tables(existing_features, feature_df)
    merged_labels = _merge_tables(existing_labels, label_df)

    summary = {
        "features_path": str(features_path),
        "labels_path": str(labels_path),
        "rows_ingested": len(feature_df),
        "total_features": len(merged_features),
        "total_labels": len(merged_labels),
    }

    if not dry_run:
        merged_features.to_parquet(features_path, index=False)
        merged_labels.to_parquet(labels_path, index=False)

    return summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Incorpora feedback humano al dataset gold")
    parser.add_argument(
        "--feedback",
        nargs="*",
        default=None,
        help="Archivos o globs Parquet con feedback (por defecto feedback/recipes.parquet)",
    )
    parser.add_argument(
        "--gold-dir",
        type=Path,
        default=DEFAULT_GOLD_DIR,
        help="Directorio donde viven features.parquet y labels.parquet",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Procesa el feedback sin escribir los Parquet resultantes",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    summary = ingest_feedback(args.feedback, gold_dir=args.gold_dir, dry_run=args.dry_run)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()

