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

## Mars surface textures

| File | Source | Licence | Notes |
| ---- | ------ | ------- | ----- |
| `mars/Katie_1_-_DLR_Jezero_hi_v2.jpg` | HiRISE orthomosaic for the Mars 2020 Jezero landing ellipse · NASA/JPL-Caltech/University of Arizona with processing by DLR/FU Berlin | NASA imagery is released into the public domain; DLR requires the credit line above when reusing the mosaic. | High-resolution texture used for the Jezero bitmap layer. Derived from the Mars 2020 mission press kit curated for the hackathon. |
| `mars/8k_mars.jpg` | Mars Global Color Map (MGS/MOLA/VO) assembled by NASA/JPL-Caltech/USGS | Public domain (NASA) | Global fallback texture when the Jezero-specific mosaic is not available. Downsampled to 8K resolution for browser rendering. |

These assets are mirrored into the Streamlit static directory and exposed under
`/static/mars/<archivo>` so the dashboard layers can reference a CDN-friendly
URL while keeping the original provenance metadata bundled with the dataset.

## Polymer composite supplier data

| Processed file | Source asset | Notes |
| -------------- | ------------ | ----- |
| `polymer_composite_thermal.csv` | `raw/external_polymer_composites/PC-Analisis Termico Composito.opj` | Differential scanning calorimetry traces exported from Origin. Metadata columns preserve the heating ramp, purge atmosphere and units (°C, W/g). |
| `polymer_composite_density.csv` | `raw/external_polymer_composites/PC-Densidad.opj` | Specific gravity measurements (ASTM D792) with explicit density units and ambient conditions. |
| `polymer_composite_mechanics.csv` | `raw/external_polymer_composites/PC-Propiedades Mecanicas Composito Pastico NMF.opj` | Tensile/flexural property table with stress/strain units and specimen descriptors pulled from the Origin workbook. |
| `polymer_composite_ignition.csv` | `raw/external_polymer_composites/PC-Ignicion.xlsx` | Parsed UL-94 style ignition test replicates. The script normalises the mixed time/temperature annotations and expands the replicate columns into tidy rows. |

The OriginLab projects must be opened on Windows. To regenerate the open
derivatives launch Origin (2023 or later), ensure `pywin32` and `originpro`
are installed in the same environment and run:

```bash
python -m scripts.convert_polymer_composites --origin-exe "C:\\Program Files\\OriginLab\\Origin2023b\\Origin96.exe"
```

When the automation layer is not available (e.g. on CI) export each worksheet to
CSV manually from Origin using the same stem name and re-run the command with
`--skip-origin`. The module will still enrich the tables with metadata and write
the CSV/Parquet outputs described above. Parsing the ignition workbook only
requires `pandas` + `openpyxl`.

> Licensing note: these composites were supplied by the challenge organisers for
> audit purposes. Redistribution should remain limited to the hackathon jury –
> the raw files therefore live under `datasets/raw/external_polymer_composites/`
> and the processed CSV/Parquet exports can be regenerated locally when needed.

## Zenodo material reference bundle

The `datasets/zenodo/` bundle consolidates public reference sheets that feed
the material lookup used by the generator and the policy engine. Each artefact
is mirrored from an open repository and normalised into tidy CSV/Parquet tables
so the attribution metadata survives downstream merges.

| Local file | Upstream record | DOI / Licence | Normalisation steps |
| ---------- | --------------- | ------------- | ------------------- |
| `MNL1 Mecha.xlsx` | *PE/EVOH Multinanolayer films – mechanical characterisation* (Zenodo) | DOI: 10.5281/zenodo.7998429 · CC BY 4.0 | Extract the `Composition`/`Mecha MNL1` sheets, coerce wt.% to fractions, infer laminate geometry (`layers`, `lme_position`) and reshape mechanical metrics into canonical `series` mixing rules. |
| `PS_c4_50.csv` | *ATR-FTIR spectral library of microplastics* (Zenodo, Primpke et al.) | DOI: 10.5281/zenodo.1195724 · CC BY 4.0 | Strip metadata headers, parse the wavenumber/transmittance pairs and store them as float columns while preserving instrument notes inside `bundle.metadata["polystyrene_transmittance"]`. |
| `pvdf_ftir_phases_1um_160C.csv` | *Crystallisation behaviour of PVDF thin films* (Mendeley Data) | DOI: 10.17632/sfbpt6sjb3.1 · CC BY 4.0 | Import the FTIR absorbance curves, tag the phase/temperature in metadata and expose them via `bundle.spectral_curves["pvdf_alpha_160c"]`. |
| `rexai_materials_ref_polyolefins_evoh_nbr.csv` | Combination of the Zenodo laminate workbook above + manufacturer data sheets | DOI as above · Mixed licences (CC BY 4.0 / vendor terms) | Harmonise densities and tensile metrics to SI units, surface laminate variants, and aggregate the composition evidence that seeds the compatibility matrix. |

All tables retain the original `source`/`license` strings to keep downstream
reports auditable. The loader in `app.modules.data_sources.load_material_reference_bundle`
builds a `MaterialReferenceBundle` that now exposes the following artefacts:

* `mixing_rules` – composite formulations derived from `MNL1 Mecha.xlsx` with
  explicit `series` vs. `parallel` assumptions and fraction evidence.
* `compatibility_matrix` – the same workbook plus regolith heuristics yields a
  canonical compatibility graph stored under `bundle.compatibility_matrix` and
  exported as `compatibility_matrix.parquet` during passport generation.
* `spectral_curves` – FTIR references for PVDF and polystyrene are exposed as
  pandas dataframes (`bundle.spectral_curves[material]`) so the Streamlit views
  can plot them without re-reading raw files.

When refreshing the bundle, run the ingestion helpers below to keep the
normalised files reproducible:

```bash
python -m scripts.convert_polymer_composites --skip-origin
python -m scripts.convert_aluminium_alloys
```

These commands regenerate the vendor CSVs, after which executing
`app.modules.data_sources.load_material_reference_bundle.cache_clear()` inside a
Python shell will rebuild the cached Polars table, spectral curves and
compatibility graph from the mirrored Zenodo/Mendeley sources.

## Aluminium alloy reference

| Processed file | Source asset | Notes |
| -------------- | ------------ | ----- |
| `aluminium_alloys.csv` | `raw/external_aluminium/al_data.csv` | Composition/strength dataset originally shared on Kaggle (“Aluminium Alloys”) and compiled from MatWeb/ALCOA datasheets. The script renames elements to `element_<symbol>_mass_fraction` and tags the mechanical units. |

Regenerate the alloy table with:

```bash
python -m scripts.convert_aluminium_alloys
```

`convert_aluminium_alloys.py` depends on `pandas` (already bundled) and writes
both CSV and Parquet to keep the processing pipeline portable.

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

## MGS-1 Regolith

The `datasets` directory also bundles public artifacts from the MGS-1 Martian
regolith simulant release (Cannon et al. 2019, DOI: 10.17632/8vhmgcczwr.1). The
deposit provides machine-readable exports for the key characterization figures:

* **Figure 3 – Grain-size distribution** → `fig3_psizeData.csv` tabulates the
  sieve-retained mass fraction for each particle diameter channel, enabling the
  coarse/fine split used when sampling bulk regolith into waste streams.
* **Figure 4 – Visible/near-IR spectra** → `fig4_spectralData.csv` holds the
  bidirectional reflectance curves for the MMS-1, MMS-2, JSC Mars-1 and MGS-1
  prototype blends, which we reference for optical heuristics.
* **Figure 5A – Thermogravimetric (TG) profile** → `fig5_tgData.csv` records the
  cumulative mass loss vs. furnace temperature, constraining devolatilisation
  assumptions during pyrolysis modeling.
* **Figure 5B – Evolved Gas Analysis (EGA)** → `fig5_egaData.csv` captures the
  quadrupole mass spectrometry channels (m/z 18, 32, 44, 64) that inform our
  expected water, oxygen, carbon dioxide and sulfur dioxide release rates.

### Regeneration steps

The original CSVs from the DOI live under `datasets/raw/` with the `mgs1_*.csv`
prefix. Run the snippet below to rebuild the derived recipe file consumed by
`app.modules.data_sources` and to materialise the oxide vector used by
`REGOLITH_VECTOR`.

```python
from pathlib import Path
import pandas as pd

RAW = Path("datasets/raw")
OUT = Path("datasets")

# Mineral recipe grouped by crystalline vs. amorphous fractions.
composition = pd.read_csv(RAW / "mgs1_composition.csv")
phase_map = {
    "plagioclasa": ("Plagioclase", "Crystalline"),
    "piroxeno": ("Pyroxene", "Crystalline"),
    "olivino": ("Olivine", "Crystalline"),
    "magnetita": ("Magnetite", "Crystalline"),
    "hematita": ("Hematite", "Crystalline"),
    "anhidrita": ("Anhydrite", "Crystalline"),
    "basalto vítreo": ("Basaltic Glass", "Amorphous"),
    "sílice hidratada": ("Hydrated Silica (Opal)", "Amorphous"),
    "mg-sulfato": ("Mg-sulfate", "Amorphous"),
    "ferrihidrita": ("Ferrihydrite", "Amorphous"),
    "fe-carbonato": ("Fe-carbonate", "Amorphous"),
}

recipe = (
    composition.assign(key=composition["mineral"].str.lower())
    .assign(
        phase=lambda df: df["key"].map(lambda name: phase_map.get(name, (name, "Crystalline"))[0]),
        type=lambda df: df["key"].map(lambda name: phase_map.get(name, (name, "Crystalline"))[1]),
    )
    [["phase", "type", "wt_percent"]]
    .rename(columns={"wt_percent": "wt_pct"})
)

padding = pd.DataFrame(
    [
        {"phase": "Quartz", "type": "Crystalline", "wt_pct": 0.0},
        {"phase": "Ilmenite", "type": "Crystalline", "wt_pct": 0.0},
    ]
)

recipe = pd.concat([recipe, padding], ignore_index=True)
recipe.to_csv(OUT / "MGS-1_Martian_Regolith_Simulant_Recipe.csv", index=False)

# Oxide vector normalised to unit mass for REGOLITH_VECTOR.
oxides = pd.read_csv(RAW / "mgs1_oxides.csv")
cleaned = (
    oxides.assign(oxide=lambda df: df["oxide"].str.lower().str.replace(r"[^0-9a-z]+", "_", regex=True))
    .groupby("oxide", as_index=False)["wt_percent"].sum()
)

oxide_vector = cleaned.assign(fraction=lambda df: df["wt_percent"] / df["wt_percent"].sum())
oxide_vector.to_csv(OUT / "mgs1_oxide_vector.csv", index=False)

summary = {
    row["oxide"]: row["fraction"] for _, row in oxide_vector.iterrows()
}
```

The `summary` mapping mirrors the dictionary returned by
`app.modules.data_sources._load_regolith_vector()` and therefore the dynamic
feature columns (`oxide_<oxide_name>`) written by the generator.

### Feature traceability

* `MGS-1_Martian_Regolith_Simulant_Recipe.csv` → columns `phase`, `type`,
  `wt_pct` feed the regolith composition lookup. When present, the keys become
  the feature suffixes appended after `oxide_`.
* `datasets/raw/mgs1_oxides.csv` (and its normalised export above) → the
  `oxide`/`wt_percent` columns produce `REGOLITH_VECTOR` entries such as
  `sio2`, `feot`, `mgo`, `cao`, `so3`, `h2o`, `co2`.
* `datasets/raw/mgs1_properties.csv` → the `amorphous_fraction` and
  `water_release` fields set the default `regolith_pct` used throughout
  candidate generation, while the remaining property columns (e.g.
  `density_bulk`) stay available for future feature engineering.
