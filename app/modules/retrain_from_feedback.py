"""Utility CLI to retrain Rex-AI models using captured astronaut feedback.

The command scans ``data/logs/feedback_*.parquet`` by default, converts the
recorded corrections into supervised targets (RandomForest regression +
classification) and reuses the main training pipeline with ``--append-logs``.

Usage
-----
``python -m app.modules.retrain_from_feedback``

Optional arguments expose the same knobs as ``app.modules.model_training`` to
control synthetic sampling, gold datasets and feedback glob patterns.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Sequence

from app.modules import model_training

from .paths import LOGS_DIR

DEFAULT_PATTERN = LOGS_DIR / "feedback_*.parquet"


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Reentrena Rex-AI combinando datasets base con feedback humano. "
            "Carga data/logs/feedback_*.parquet por defecto y reutiliza el "
            "pipeline principal (train_and_save)."
        )
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=1600,
        help=(
            "Número de muestras sintéticas a generar si faltan etiquetas doradas. "
            "Se pasa directamente a train_and_save()."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=21,
        help="Semilla global para generación y entrenamiento.",
    )
    parser.add_argument(
        "--gold",
        type=Path,
        default=None,
        help="Directorio con features.parquet/labels.parquet dorados.",
    )
    parser.add_argument(
        "--features",
        type=Path,
        default=None,
        help="Ruta alternativa a features.parquet si difiere del --gold.",
    )
    parser.add_argument(
        "--logs",
        nargs="*",
        default=None,
        help=(
            "Globs o archivos Parquet con feedback humano. Si no se pasan se usa "
            "data/logs/feedback_*.parquet."
        ),
    )
    return parser


def _resolve_gold_paths(args: argparse.Namespace) -> tuple[Path | None, Path | None]:
    gold_features: Path | None = None
    gold_labels: Path | None = None

    if args.gold is not None:
        gold_dir = Path(args.gold)
        gold_features = gold_dir / "features.parquet"
        gold_labels = gold_dir / "labels.parquet"

    if args.features is not None:
        features_path = Path(args.features)
        if features_path.is_dir():
            features_path = features_path / "features.parquet"
        gold_features = features_path

    return gold_features, gold_labels


def _resolve_patterns(log_args: Iterable[str] | None) -> list[str]:
    patterns: list[str] = []
    if log_args:
        patterns.extend(str(Path(p)) for p in log_args)
    if not patterns:
        patterns.append(str(DEFAULT_PATTERN))
    return patterns


def cli(argv: Sequence[str] | None = None) -> dict:
    args = _build_arg_parser().parse_args(list(argv) if argv is not None else None)
    gold_features_path, gold_labels_path = _resolve_gold_paths(args)
    patterns = _resolve_patterns(args.logs)

    feedback_df = model_training.load_feedback_logs(patterns)
    metadata = model_training.train_and_save(
        n_samples=args.samples,
        seed=args.seed,
        gold_features_path=gold_features_path,
        gold_labels_path=gold_labels_path,
        feedback_logs=feedback_df,
    )

    print(json.dumps(metadata, indent=2, ensure_ascii=False))
    return metadata


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    cli()
