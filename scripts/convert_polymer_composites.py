"""Utilities for exporting proprietary polymer composite characterization
projects into open CSV/Parquet tables.

The composite supplier delivered several OriginLab projects (.opj) and a
supporting Excel workbook with ignition testing results.  This module automates
(or documents) the conversion so jurors can reproduce the processed datasets
that ship with the hackathon prototype.

Usage
-----
The .opj files require OriginLab automation on Windows.  When running on a
workstation with Origin installed you can execute the full pipeline:

    python -m scripts.convert_polymer_composites --origin-exe "C:\\Program Files\\OriginLab\\Origin2023b\\Origin96.exe"

If Origin automation is not available (e.g. on CI or Linux) you may export each
workbook to CSV manually from Origin and drop the files next to the raw project
using the same stem (``PC-Analisis Termico Composito.csv`` etc.).  The script
will pick up those fallbacks and still materialise the open datasets.

The ignition Excel workbook is parsed directly with pandas/openpyxl and does not
require Origin.
"""

from __future__ import annotations

import argparse
import importlib
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = REPO_ROOT / "datasets" / "raw" / "external_polymer_composites"
OUT_DIR = REPO_ROOT / "datasets"


class OriginAutomationUnavailable(RuntimeError):
    """Raised when Origin automation is required but cannot be initialised."""


@dataclass
class OriginSheetExport:
    """Configuration for exporting a sheet from an Origin project."""

    name: str
    project: Path
    workbook: str | None
    sheet: str | int | None
    output_csv: Path
    output_parquet: Path
    metadata: dict[str, str] = field(default_factory=dict)
    column_units: dict[str, str] = field(default_factory=dict)
    rename_overrides: dict[str, str] = field(default_factory=dict)

    @property
    def fallback_csv(self) -> Path:
        return self.project.with_suffix(".csv")


POLYMER_EXPORTS: list[OriginSheetExport] = [
    OriginSheetExport(
        name="thermal",
        project=RAW_DIR / "PC-Analisis Termico Composito.opj",
        workbook="Book1",
        sheet=0,
        output_csv=OUT_DIR / "polymer_composite_thermal.csv",
        output_parquet=OUT_DIR / "polymer_composite_thermal.parquet",
        metadata={
            "dataset": "polymer_composite_thermal",
            "test_type": "Differential scanning calorimetry",
            "environment": "Nitrogen purge, 10 °C/min heating ramp",
            "source_file": "PC-Analisis Termico Composito.opj",
        },
        column_units={
            "temperature_c": "celsius",
            "heat_flow_w_per_g": "watt_per_gram",
            "heat_capacity_j_per_g_k": "joule_per_gram_kelvin",
        },
    ),
    OriginSheetExport(
        name="density",
        project=RAW_DIR / "PC-Densidad.opj",
        workbook="Book1",
        sheet=0,
        output_csv=OUT_DIR / "polymer_composite_density.csv",
        output_parquet=OUT_DIR / "polymer_composite_density.parquet",
        metadata={
            "dataset": "polymer_composite_density",
            "test_type": "Density measurement (ASTM D792 pycnometer)",
            "environment": "23 °C, deionised water",
            "source_file": "PC-Densidad.opj",
        },
        column_units={
            "density_g_per_cm3": "gram_per_cubic_centimetre",
        },
    ),
    OriginSheetExport(
        name="mechanics",
        project=RAW_DIR / "PC-Propiedades Mecanicas Composito Pastico NMF.opj",
        workbook="Book1",
        sheet=0,
        output_csv=OUT_DIR / "polymer_composite_mechanics.csv",
        output_parquet=OUT_DIR / "polymer_composite_mechanics.parquet",
        metadata={
            "dataset": "polymer_composite_mechanics",
            "test_type": "Tensile and flexural mechanical testing",
            "environment": "ASTM D638 Type I specimens at 23 °C",
            "source_file": "PC-Propiedades Mecanicas Composito Pastico NMF.opj",
        },
        column_units={
            "stress_mpa": "megapascal",
            "strain_pct": "percent",
            "modulus_gpa": "gigapascal",
        },
    ),
]

IGNITION_OUTPUT_CSV = OUT_DIR / "polymer_composite_ignition.csv"
IGNITION_OUTPUT_PARQUET = OUT_DIR / "polymer_composite_ignition.parquet"
IGNITION_METADATA = {
    "dataset": "polymer_composite_ignition",
    "test_type": "UL-94 style ignition and burning rate",
    "environment": "Horizontal burn, ambient pressure",
    "source_file": "PC-Ignicion.xlsx",
}


def slugify(value: str) -> str:
    text = value.strip().lower()
    text = text.replace("%", "pct").replace("°", "deg").replace("/", "_per_")
    text = re.sub(r"[^0-9a-z]+", "_", text)
    text = re.sub(r"_+", "_", text)
    return text.strip("_")


def ensure_origin_session(executable: str | None):
    """Create an Origin automation session via win32com."""

    win32com = importlib.import_module("win32com.client")
    origin = win32com.Dispatch("Origin.ApplicationSI")
    # Configure visibility/headless mode.
    origin.Visible = 0
    if executable:
        # When multiple Origin versions are installed you can force the
        # executable path so the LT commands resolve correctly.
        origin.Execute(f'system.path.program$ = "{executable}";')
    return origin


def run_labtalk(origin, commands: Iterable[str]) -> None:
    for command in commands:
        result = origin.LTExecute(command)
        if result != 0:
            raise RuntimeError(f"Origin command failed: {command}\nReturn code: {result}")


def export_origin_sheet(spec: OriginSheetExport, origin, tmp_dir: Path) -> Path:
    """Export a worksheet to CSV using Origin automation."""

    destination = tmp_dir / f"{spec.project.stem}.csv"
    dest_parent = destination.parent.as_posix()
    dest_name = destination.name
    lt_commands = [
        f'doc -o "{spec.project}";',
    ]
    if spec.workbook:
        lt_commands.append(f'win -a "{spec.workbook}";')
    if spec.sheet is not None:
        lt_commands.append(f'page.active$ = "{spec.workbook}!{spec.sheet}";')
    lt_commands.extend(
        [
            "worksheet -s 0 0 -1 -1;",  # select all columns/rows
            (
                "expASC type:=csv path:=\"{parent}\" fname:=\"{name}\""
                " options.names:=1 options.units:=1 options.comments:=1;"
            ).format(parent=dest_parent, name=dest_name),
            "doc -t;",  # close the project to avoid modified prompts
        ]
    )
    run_labtalk(origin, lt_commands)
    return destination


def load_origin_dataframe(spec: OriginSheetExport, origin) -> pd.DataFrame:
    with tempfile.TemporaryDirectory() as tmp_root:
        tmp_dir = Path(tmp_root)
        if origin is None:
            if spec.fallback_csv.exists():
                intermediate = spec.fallback_csv
            else:
                raise OriginAutomationUnavailable(
                    "Origin automation is required for "
                    f"{spec.project.name}; alternatively provide {spec.fallback_csv.name}."
                )
        else:
            intermediate = export_origin_sheet(spec, origin, tmp_dir)
        df = pd.read_csv(intermediate)
    return df


def tidy_origin_dataframe(spec: OriginSheetExport, df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.dropna(how="all")
    df.columns = [slugify(str(col)) for col in df.columns]
    if spec.rename_overrides:
        df = df.rename(columns=spec.rename_overrides)
    # Attach units and metadata as explicit columns so they survive CSV export.
    for column, unit in spec.column_units.items():
        if column in df.columns:
            df[f"{column}_unit"] = unit
    for key, value in spec.metadata.items():
        df[key] = value
    df["source_path"] = str(spec.project.relative_to(REPO_ROOT))
    return df


def convert_origin_projects(origin_executable: str | None, *, skip_origin: bool = False) -> None:
    origin = None
    if not skip_origin:
        try:
            origin = ensure_origin_session(origin_executable)
        except ImportError as exc:
            raise OriginAutomationUnavailable(
                "win32com is required to drive Origin; install pywin32 on Windows"
            ) from exc
    for spec in POLYMER_EXPORTS:
        try:
            df = load_origin_dataframe(spec, origin)
        except OriginAutomationUnavailable as exc:
            if skip_origin:
                print(f"[skip] {exc}")
                continue
            raise
        tidy = tidy_origin_dataframe(spec, df)
        tidy.to_csv(spec.output_csv, index=False)
        tidy.to_parquet(spec.output_parquet, index=False)
        print(f"Exported {spec.output_csv.relative_to(REPO_ROOT)} ({len(tidy)} rows)")
    if origin is not None:
        try:
            origin.Exit()
        except AttributeError:
            pass


def parse_burn_value(value: object) -> tuple[float | None, float | None]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None, None
    text = str(value).strip().replace(",", ".")
    match_temp = re.search(r"(\d+(?:\.\d+)?)\s*°c", text, flags=re.IGNORECASE)
    temperature = float(match_temp.group(1)) if match_temp else None
    time_text = text.split("(")[0].strip()
    segments = time_text.split(".")
    if len(segments) == 3:
        minutes = float(segments[0])
        seconds = float(segments[1])
        hundredths = float(segments[2])
        time_minutes = minutes + (seconds + hundredths / 100.0) / 60.0
    else:
        try:
            time_minutes = float(time_text) if time_text else None
        except ValueError:
            time_minutes = None
    return time_minutes, temperature


def convert_ignition_workbook() -> None:
    df = pd.read_excel(RAW_DIR / "PC-Ignicion.xlsx", skiprows=5)
    df = df.drop(columns=[col for col in df.columns if col.startswith("Unnamed")])
    replicas = []
    suffixes = ["", ".1", ".2", ".3"]
    for idx, suffix in enumerate(suffixes, start=1):
        required = [
            f"Muestra{suffix}",
            f"Espesor (mm){suffix}",
            f"Ancho (mm){suffix}",
            f"Tiempo (min){suffix}",
            f"L (mm){suffix}",
        ]
        existing = [col for col in required if col in df.columns]
        if len(existing) != len(required):
            continue
        subset = df[required].rename(
            columns={
                f"Muestra{suffix}": "sample_label",
                f"Espesor (mm){suffix}": "thickness_mm",
                f"Ancho (mm){suffix}": "width_mm",
                f"Tiempo (min){suffix}": "burn_time_raw",
                f"L (mm){suffix}": "burn_length_mm",
            }
        )
        subset["replicate"] = idx
        replicas.append(subset)
    if not replicas:
        raise RuntimeError("Ignition workbook structure not recognised")
    tidy = pd.concat(replicas, ignore_index=True)
    tidy = tidy.dropna(how="all")
    tidy = tidy.dropna(subset=["sample_label"], how="all")
    tidy["sample_label"] = tidy["sample_label"].astype("string").str.strip()
    for numeric in ("thickness_mm", "width_mm", "burn_length_mm"):
        tidy[numeric] = pd.to_numeric(tidy[numeric], errors="coerce")
    tidy[["burn_time_min", "ignition_temperature_c"]] = tidy.apply(
        lambda row: pd.Series(parse_burn_value(row["burn_time_raw"])), axis=1
    )
    tidy = tidy.drop(columns=["burn_time_raw"])
    tidy["burn_time_unit"] = "minute"
    tidy["burn_length_unit"] = "millimetre"
    tidy["thickness_unit"] = "millimetre"
    tidy["width_unit"] = "millimetre"
    for key, value in IGNITION_METADATA.items():
        tidy[key] = value
    tidy["source_path"] = str((RAW_DIR / "PC-Ignicion.xlsx").relative_to(REPO_ROOT))
    tidy.to_csv(IGNITION_OUTPUT_CSV, index=False)
    tidy.to_parquet(IGNITION_OUTPUT_PARQUET, index=False)
    print(f"Exported {IGNITION_OUTPUT_CSV.relative_to(REPO_ROOT)} ({len(tidy)} rows)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--origin-exe",
        help="Optional path to a specific Origin executable (useful when multiple versions are installed)",
    )
    parser.add_argument(
        "--skip-origin",
        action="store_true",
        help="Skip Origin automation and rely solely on manually exported CSV fallbacks.",
    )
    args = parser.parse_args()

    convert_origin_projects(args.origin_exe, skip_origin=args.skip_origin)
    convert_ignition_workbook()


if __name__ == "__main__":
    main()
