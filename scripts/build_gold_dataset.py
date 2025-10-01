"""CLI helper to construct the gold feature/label datasets."""

from __future__ import annotations

import argparse
from pathlib import Path

from app.modules.data_build import build_gold_dataset
from app.modules.data_pipeline import GoldFeatureRow, GoldLabelRow


def main() -> None:
    """Generate the curated gold dataset artefacts."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Directory where features.parquet and labels.parquet will be written. "
            "Defaults to data/gold when omitted."
        ),
    )
    args = parser.parse_args()
    build_gold_dataset(args.output_dir, return_frames=False)


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    main()


__all__ = ["build_gold_dataset", "GoldFeatureRow", "GoldLabelRow", "main"]

