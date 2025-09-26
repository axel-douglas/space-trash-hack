# NASA reference datasets

This directory now contains processed NASA reference tables that can be merged
into `rexai_nasa_waste_features.csv` via the helper
`app.modules.generator._merge_reference_dataset`. The original spreadsheet style
exports that used text ranges have been archived under `datasets/raw/` with a
`*_raw.csv` suffix so the preprocessing can be reproduced.

## Source material

| Processed file | NASA reference | Notes |
| -------------- | -------------- | ----- |
| `nasa_waste_summary.csv` | Logistics Reduction & Repurposing (LRR) – Logistics-to-Living Phase I/II non-metabolic waste inventory | Ranges such as `15-58 kg` are converted to numeric midpoints and mapped onto the canonical waste categories used by the demo (e.g. Foam Packaging → `Zotek F30`). |
| `nasa_waste_processing_products.csv` | LRR Trash-to-Gas architecture study | Provides propellant and water yields for the Trash-to-Gas (TtG) and Trash-to-Supply-Gas (TtSG) flows. |
| `nasa_leo_mass_savings.csv` | LRR delivery trade study for TtG/TtSG logistics | Captures chemical vs. solar-electric propulsion savings and their gear ratios. |
| `nasa_propellant_benefits.csv` | LRR mission delta-V budget summary | Lists vehicle concepts and the propellant required to exploit trash-derived propellants. |
| `l2l_parameters.csv` | NASA/TM-2021002072 "Logistics to Living" appendix A1b | Bag geometries, volumetric ratios, and other Logistics-to-Living constants; the `feature` column mirrors the terminology used in the report. |

The raw CSVs under `datasets/raw/` contain the literal values transcribed from
the NASA PDFs and slide decks so the preprocessing logic below remains
reproducible.

## Regeneration steps

Use the following Python snippet to rebuild the processed CSVs from the raw
inputs. It converts textual ranges (e.g. `15-58`) to numeric midpoints, sums
multi-mission totals, and exposes the NASA identifiers as the `category`/
`subitem` join keys expected by `_merge_reference_dataset`.

```python
from pathlib import Path
import pandas as pd
import polars as pl
import re

RAW_DIR = Path("datasets/raw")
OUT_DIR = Path("datasets")


def parse_numeric(value: object) -> float | None:
    if value is None or value != value:  # handles None and NaN
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().replace(",", "")
    if not text:
        return None
    numbers = [abs(float(match)) for match in re.findall(r"[-+]?[0-9]*\.?[0-9]+", text)]
    if "-" in text and len(numbers) >= 2:
        return sum(numbers[:2]) / 2.0
    if "+" in text and len(numbers) >= 2:
        return sum(numbers)
    return numbers[0] if numbers else None

# Waste summary: map known waste types to canonical categories/subitems and
# compute mission totals.
summary_map = {
    "Clothing": ("Fabrics", "Clothing"),
    "Wipes/Tissues": ("Fabrics", "Cleaning Wipes"),
    "Towels and Hygiene": ("Fabrics", "Towels/Wash Cloths"),
    "Foam Packaging for Launch": ("Foam Packaging", "Zotek F30 (PVDF foam)"),
    "Other Crew Supplies": ("Other Packaging/Gloves (B)", "Nitrile gloves"),
    "Food & Packaging": ("Food Packaging", "Overwrap"),
    "EVA Supplies": ("EVA Waste", "Cargo Transfer Bags (CTB)"),
}
summary_rows = []
for row in pd.read_csv(RAW_DIR / "nasa_waste_summary_raw.csv").to_dict(orient="records"):
    category, subitem = summary_map.get(row["waste_type"], (row["waste_type"], None))
    gateway_i = parse_numeric(row["gateway_phase_I_total_kg"])
    gateway_ii = parse_numeric(row["gateway_phase_II_total_kg"])
    mars = parse_numeric(row["mars_transit_total_kg"])
    summary_rows.append({
        "category": category,
        "subitem": subitem,
        "kg_per_cm_day": parse_numeric(row["kg_per_cm_day"]),
        "gateway_phase_i_mass_kg": gateway_i,
        "gateway_phase_ii_mass_kg": gateway_ii,
        "mars_transit_mass_kg": mars,
        "total_mass_kg": sum(v for v in (gateway_i, gateway_ii, mars) if v is not None),
    })
pl.DataFrame(summary_rows).write_csv(OUT_DIR / "nasa_waste_summary.csv")

# Trash processing products: keep the NASA "approach" label and expose
# propellant/water figures.
processing_rows = []
for row in pd.read_csv(RAW_DIR / "nasa_waste_processing_products_raw.csv").to_dict(orient="records"):
    prop_day = parse_numeric(row["kg_propellant_per_cm_day"])
    gateway_i = parse_numeric(row["gateway_phase_I_total_kg"])
    gateway_ii = parse_numeric(row["gateway_phase_II_total_kg"])
    mars = parse_numeric(row["mars_outbound_total_kg"])
    processing_rows.append({
        "category": "Trash Total",
        "subitem": row["approach"],
        "approach": row["approach"],
        "propellant_per_cm_day_kg": prop_day,
        "gateway_phase_i_propellant_kg": gateway_i,
        "gateway_phase_ii_propellant_kg": gateway_ii,
        "mars_outbound_propellant_kg": mars,
        "total_propellant_kg": sum(v for v in (gateway_i, gateway_ii, mars) if v is not None),
        "makeup_water_per_cm_day_kg": parse_numeric(row["makeup_water_kg_cm_day"]),
        "o2_ch4_yield_kg": prop_day if ("O2" in row["products"] or "CH4" in row["products"]) else 0.0,
    })
pl.DataFrame(processing_rows).write_csv(OUT_DIR / "nasa_waste_processing_products.csv")

# LEO mass savings: expose propulsion mode, ISP, gear ratio and convert the
# savings range to min/max/mean values.
leo_rows = []
for row in pd.read_csv(RAW_DIR / "nasa_leo_mass_savings_raw.csv").to_dict(orient="records"):
    numbers = [abs(float(match)) for match in re.findall(r"[-+]?[0-9]*\.?[0-9]+", str(row["leo_mass_savings_kg"]))]
    if "-" in str(row["leo_mass_savings_kg"]) and len(numbers) >= 2:
        savings_min, savings_max = sorted(numbers[:2])
        savings = (savings_min + savings_max) / 2.0
    elif numbers:
        savings_min = savings_max = savings = numbers[0]
    else:
        savings_min = savings_max = savings = None
    leo_rows.append({
        "category": "Trash Total",
        "subitem": row["delivery_method"],
        "propulsion": row["delivery_method"],
        "isp_seconds": parse_numeric(row["isp_sec"]),
        "gear_ratio": parse_numeric(row["gear_ratio"]),
        "max_phase_i_propellant_kg": parse_numeric(row["max_phase_I_propellant_kg"]),
        "max_phase_ii_propellant_kg": parse_numeric(row["max_phase_II_propellant_kg"]),
        "mass_savings_min_kg": savings_min,
        "mass_savings_max_kg": savings_max,
        "mass_savings_kg": savings,
    })
pl.DataFrame(leo_rows).write_csv(OUT_DIR / "nasa_leo_mass_savings.csv")

# Propellant benefits: retain the mission/vehicle labels and convert the
# delta-V requirements to floats.
propellant_rows = []
for row in pd.read_csv(RAW_DIR / "nasa_propellant_benefits_raw.csv").to_dict(orient="records"):
    propellant_rows.append({
        "category": "Trash Total",
        "subitem": row["mission"],
        "mission": row["mission"],
        "vehicle": row.get("spacecraft"),
        "propellant_mass_kg": parse_numeric(row["mass_kg"]),
        "delta_v_ttg_m_s": parse_numeric(row["deltaV_ttg_m_s"]),
        "delta_v_ttsg_m_s": parse_numeric(row["deltaV_ttsg_m_s"]),
        "delta_v_requirement_m_s": parse_numeric(row["deltaV_requirement_m_s"]),
    })
pl.DataFrame(propellant_rows).write_csv(OUT_DIR / "nasa_propellant_benefits.csv")
```

Running the script will refresh the processed tables in place, ensuring the
merged feature columns such as `summary_total_mass_kg`,
`processing_o2_ch4_yield_kg`, `leo_mass_savings_kg`, and
`propellant_propellant_mass_kg` stay in sync with NASA's published values.

Because the Logistics-to-Living loader expects descriptor names, the processed
`l2l_parameters.csv` also renames the raw `variable` header to `feature`. This
ensures constants such as `l2l_geometry_ctb_small_volume_value` are materialized
as distinct columns during feature engineering.
