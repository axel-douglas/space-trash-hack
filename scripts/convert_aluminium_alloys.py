"""Normalise the public aluminium alloy dataset shipped with the challenge."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_FILE = REPO_ROOT / "datasets" / "raw" / "external_aluminium" / "al_data.csv"
CSV_OUT = REPO_ROOT / "datasets" / "aluminium_alloys.csv"
PARQUET_OUT = CSV_OUT.with_suffix(".parquet")


COLUMN_RENAME = {
    "Processing": "processing_route",
    "class": "class_id",
    "Elongation (%)": "elongation_pct",
    "Tensile Strength (MPa)": "tensile_strength_mpa",
    "Yield Strength (MPa)": "yield_strength_mpa",
}


def slugify_element(column: str) -> str:
    return f"element_{column.lower()}_mass_fraction"


def convert_alloys() -> None:
    df = pd.read_csv(RAW_FILE)
    df = df.drop(columns=[col for col in df.columns if str(col).startswith("Unnamed")])
    excluded = set(COLUMN_RENAME.keys()) | {
        "Processing",
        "class",
        "Elongation (%)",
        "Tensile Strength (MPa)",
        "Yield Strength (MPa)",
    }
    chemical_columns = [col for col in df.columns if col not in excluded]
    rename_map = {col: slugify_element(col) for col in chemical_columns}
    df = df.rename(columns=rename_map | COLUMN_RENAME)
    df["class_id"] = pd.to_numeric(df["class_id"], errors="coerce").astype("Int64")
    df["composition_basis"] = "mass_fraction"
    df["test_type"] = "Room-temperature tensile test"
    df["property_source"] = (
        "Aggregate of MatWeb/ALCOA datasheets (Kaggle Aluminium Alloys dataset)"
    )
    df["elongation_unit"] = "percent"
    df["tensile_strength_unit"] = "megapascal"
    df["yield_strength_unit"] = "megapascal"
    df["source_path"] = str(RAW_FILE.relative_to(REPO_ROOT))
    df.to_csv(CSV_OUT, index=False)
    df.to_parquet(PARQUET_OUT, index=False)
    print(f"Exported {CSV_OUT.relative_to(REPO_ROOT)} ({len(df)} rows)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    convert_alloys()


if __name__ == "__main__":
    main()
