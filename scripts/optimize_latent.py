"""Explore candidate recipes in the Rex-AI latent space."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable

import pandas as pd

from app.modules.latent_optimizer import LatentSpaceExplorer, make_objective


def _load_candidates(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"Input file not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix in {".csv", ".tsv"}:
        sep = "," if suffix == ".csv" else "\t"
        return pd.read_csv(path, sep=sep)
    if suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        return pd.DataFrame(data)

    raise SystemExit("Unsupported input format. Use Parquet, CSV/TSV or JSON.")


def _parse_weights(raw: str) -> Dict[str, float]:
    weights: Dict[str, float] = {}
    if not raw:
        return weights

    for fragment in raw.split(","):
        if not fragment.strip():
            continue
        if ":" in fragment:
            key, value = fragment.split(":", 1)
        else:
            key, value = fragment, "1.0"
        try:
            weights[key.strip()] = float(value)
        except ValueError:
            raise SystemExit(f"Invalid weight specification: {fragment}")
    return weights


def _collect_seeds(frame: pd.DataFrame, limit: int) -> Iterable[Dict[str, object]]:
    records = frame.to_dict(orient="records")
    for row in records[:limit]:
        yield row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidates", type=Path, help="Path to the candidate feature table (Parquet/CSV/JSON)")
    parser.add_argument("--objective", type=str, default="rigidez:1.0,crew_min:-0.1,energy_kwh:-0.05")
    parser.add_argument("--samples", type=int, default=96, help="Latent samples per seed recipe")
    parser.add_argument("--radius", type=float, default=0.45, help="Gaussian radius applied in latent space")
    parser.add_argument("--seeds", type=int, default=5, help="Number of seed recipes to expand")
    parser.add_argument("--top", type=int, default=10, help="Number of suggestions to print")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument(
        "--maximise",
        action="store_true",
        help="Treat weights as positive contributions (default). Use --minimise to invert.",
    )
    parser.add_argument("--minimise", action="store_true", help="Invert the objective (multiply by -1)")
    parser.add_argument(
        "--duplicates-threshold",
        type=float,
        default=0.1,
        help="Optional latent distance threshold to report near-duplicates",
    )
    args = parser.parse_args()

    explorer = LatentSpaceExplorer()
    if not explorer.available():
        raise SystemExit("Autoencoder not available. Train optional artefacts before using this script.")

    frame = _load_candidates(args.candidates)
    if frame.empty:
        raise SystemExit("Candidate table is empty")

    weights = _parse_weights(args.objective)
    maximise = not args.minimise
    if args.maximise:
        maximise = True

    objective = make_objective(weights, maximise=maximise)

    seeds = list(_collect_seeds(frame, args.seeds))
    if not seeds:
        raise SystemExit("No seed candidates found in the input dataset")

    proposals = []
    for seed in seeds:
        proposals.extend(
            explorer.propose_candidates(
                seed,
                objective,
                radius=args.radius,
                samples=args.samples,
                top_k=args.top,
                random_state=args.random_state,
            )
        )

    proposals.sort(key=lambda item: item.score, reverse=True)
    formatted = [candidate.as_dict() for candidate in proposals[: args.top]]

    payload: Dict[str, object] = {"suggestions": formatted}

    if args.duplicates_threshold > 0:
        dupes = explorer.detect_duplicates(frame, threshold=args.duplicates_threshold)
        payload["near_duplicates"] = dupes

    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    main()
