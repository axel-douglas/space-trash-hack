"""Centralised data ingestion helpers for Rex-AI reference datasets.

The :mod:`app.modules.generator` module depends on a fairly eclectic mix of
CSV/Delta inputs curated by NASA.  Historically these helpers were sprinkled
throughout ``generator.py`` which made the core candidate-building logic hard
to audit.  This module collects the read/parse utilities so that both the
runtime and training pipelines have a consistent contract for obtaining
reference data.

Responsibilities handled here:

* resolving dataset locations inside :mod:`datasets`
* loading CSV artifacts into :class:`polars` or :class:`pandas` structures
* preparing cached bundles with official NASA feature metadata
* exposing normalisation helpers shared across generator routines
"""

from __future__ import annotations

import itertools
import math
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, NamedTuple

import numpy as np
import pandas as pd
import polars as pl

from .paths import DATA_ROOT

DATASETS_ROOT = DATA_ROOT.parent / "datasets"

__all__ = [
    "DATASETS_ROOT",
    "to_lazy_frame",
    "from_lazy_frame",
    "resolve_dataset_path",
    "slugify",
    "normalize_text",
    "normalize_category",
    "extend_category_synonyms",
    "normalize_item",
    "token_set",
    "merge_reference_dataset",
    "extract_grouped_metrics",
    "extract_reference_metrics",
    "load_regolith_particle_size",
    "load_regolith_spectra",
    "load_regolith_thermogravimetry",
    "RegolithCharacterization",
    "load_regolith_characterization",
    "L2LParameters",
    "load_l2l_parameters",
    "OfficialFeaturesBundle",
    "official_features_bundle",
    "lookup_official_feature_values",
    "REGOLITH_VECTOR",
    "REGOLITH_CHARACTERIZATION",
    "GAS_MEAN_YIELD",
    "MEAN_REUSE",
    "RegolithThermalBundle",
    "load_regolith_granulometry",
    "load_regolith_spectral_curves",
    "load_regolith_thermal_profiles",
    "regolith_observation_lines",
    "MaterialReferenceBundle",
    "load_material_reference_bundle",
]


def to_lazy_frame(
    frame: pd.DataFrame | pl.DataFrame | pl.LazyFrame,
) -> tuple[pl.LazyFrame, str]:
    """Return a :class:`polars.LazyFrame` along with the original frame type."""

    if isinstance(frame, pl.LazyFrame):
        return frame, "lazy"
    if isinstance(frame, pl.DataFrame):
        return frame.lazy(), "polars"
    if isinstance(frame, pd.DataFrame):
        return pl.from_pandas(frame).lazy(), "pandas"
    raise TypeError(f"Unsupported frame type: {type(frame)!r}")


def from_lazy_frame(lazy: pl.LazyFrame, frame_kind: str) -> pd.DataFrame | pl.DataFrame | pl.LazyFrame:
    """Convert *lazy* back to the representation described by *frame_kind*."""

    if frame_kind == "lazy":
        return lazy

    collected = lazy.collect()
    if frame_kind == "polars":
        return collected
    if frame_kind == "pandas":
        return collected.to_pandas()
    raise ValueError(f"Unsupported frame kind: {frame_kind}")


def resolve_dataset_path(name: str) -> Path | None:
    """Return the first dataset path that exists for *name*."""

    candidates = (
        DATASETS_ROOT / name,
        DATASETS_ROOT / "raw" / name,
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def slugify(value: str) -> str:
    """Convert *value* into a snake_case identifier safe for feature names."""

    text = re.sub(r"[^0-9a-zA-Z]+", "_", str(value).strip().lower())
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "value"


def _feature_name_from_parts(*parts: str) -> str:
    return "_".join(part for part in (slugify(part) for part in parts if part) if part)


def normalize_text(value: Any) -> str:
    text = str(value or "").lower()
    text = text.replace("—", " ").replace("/", " ")
    text = re.sub(r"\(.*?\)", " ", text)
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    tokens = []
    for token in text.split():
        if len(token) > 3 and token.endswith("s"):
            token = token[:-1]
        tokens.append(token)
    return " ".join(tokens).strip()


_CATEGORY_SYNONYMS = {
    "foam": "foam packaging",
    "foam packaging": "foam packaging",
    "foam packaging for launch": "foam packaging",
    "packaging": "packaging",
    "other packaging": "other packaging",
    "other packaging glove": "other packaging",
    "glove": "gloves",
    "gloves": "gloves",
    "food packaging": "food packaging",
    "structural elements": "structural elements",
    "structural element": "structural elements",
    "eva": "eva waste",
    "eva waste": "eva waste",
}


def normalize_category(value: Any) -> str:
    normalized = normalize_text(value)
    return _CATEGORY_SYNONYMS.get(normalized, normalized)


def extend_category_synonyms(
    synonyms: Mapping[str, str] | Iterable[tuple[str, str]],
) -> None:
    """Extend the category synonym map with *synonyms*.

    The provided mapping is normalised using :func:`normalize_text` to ensure
    consistent lookups regardless of capitalisation or punctuation.  Existing
    entries with the same normalised key are overwritten so callers can update
    canonical targets when ingesting new inventories.
    """

    if isinstance(synonyms, Mapping):
        items = synonyms.items()
    else:
        items = synonyms

    for source, target in items:
        normalized_source = normalize_text(source)
        if not normalized_source:
            continue
        normalized_target = normalize_text(target)
        if not normalized_target:
            normalized_target = normalized_source
        _CATEGORY_SYNONYMS[normalized_source] = normalized_target


def normalize_item(value: Any) -> str:
    return normalize_text(value)


def token_set(value: Any) -> frozenset[str]:
    normalized = normalize_item(value)
    if not normalized:
        return frozenset()
    return frozenset(normalized.split())


def merge_reference_dataset(
    base: pd.DataFrame | pl.DataFrame | pl.LazyFrame, filename: str, prefix: str
) -> pd.DataFrame | pl.DataFrame | pl.LazyFrame:
    path = resolve_dataset_path(filename)
    if path is None:
        return base

    base_lazy, base_kind = to_lazy_frame(base)
    base_columns = list(base_lazy.collect_schema().names())
    base_column_set = set(base_columns)

    extra_lazy = pl.scan_csv(path)
    extra_columns = list(extra_lazy.collect_schema().names())
    extra_column_set = set(extra_columns)

    join_cols = [
        col for col in ("category", "subitem") if col in base_column_set and col in extra_column_set
    ]
    if not join_cols:
        return base

    existing = set(base_columns)
    rename_map: Dict[str, str] = {}
    drop_cols: list[str] = []
    for column in extra_columns:
        if column in join_cols:
            continue
        if column in existing:
            drop_cols.append(column)
            continue
        rename_map[column] = f"{prefix}_{slugify(column)}"

    if drop_cols:
        extra_lazy = extra_lazy.drop(drop_cols)
    if rename_map:
        extra_lazy = extra_lazy.rename(rename_map)

    added_columns = [
        rename_map.get(col, col)
        for col in extra_columns
        if col not in join_cols and col not in drop_cols
    ]

    merged_lazy = base_lazy.join(extra_lazy, on=join_cols, how="left")
    if added_columns:
        projection = base_columns + [col for col in added_columns if col not in base_columns]
        merged_lazy = merged_lazy.select([pl.col(name) for name in projection])

    result = from_lazy_frame(merged_lazy, base_kind)
    if isinstance(result, pd.DataFrame):
        return result.loc[:, ~result.columns.duplicated()]
    if isinstance(result, pl.DataFrame):
        unique_cols = []
        seen: set[str] = set()
        for name in result.columns:
            if name in seen:
                continue
            seen.add(name)
            unique_cols.append(name)
        return result.select(unique_cols)
    return result


class MaterialReferenceBundle(NamedTuple):
    """Normalized material properties sourced from Zenodo technical sheets.

    The bundle exposes reference values that can be merged with the NASA waste
    inventories and reused throughout the generator, data build and training
    pipelines.  Keys are normalised with :func:`slugify` so callers can lookup
    properties by any canonical ``material`` / ``material_family`` string seen
    in the waste tables.

    Attributes
    ----------
    table:
        A :class:`polars.DataFrame` where each row represents a canonical
        material and the columns correspond to harmonised properties (density,
        modulus, strength, etc.).  The frame is ready for dataframe joins.
    properties:
        Mapping from ``material_key`` to the same values contained in
        :attr:`table` for quick dictionary based lookups.
    density_map:
        Convenience map returning ``material_density_kg_m3`` entries for each
        canonical key.  Used by heuristics to override legacy density defaults.
    alias_map:
        Dictionary translating slugified aliases (``normalize_item`` +
        :func:`slugify`) to the canonical ``material_key``.  The aliases cover
        common NASA spellings so lookups remain robust.
    property_columns:
        Tuple enumerating the numeric property columns guaranteed to be present
        in :attr:`table`.  This allows feature builders to iterate deterministically
        without hard-coding the column names in multiple modules.
    spectral_curves:
        Dictionary with raw spectral curves (e.g. FTIR measurements) stored as
        :class:`pandas.DataFrame` instances.  The curves retain their original
        resolution so downstream visualisations can resample as needed.
    metadata:
        Additional descriptors extracted from the Zenodo artefacts (licence,
        source file names, etc.).  These are serialised together with the model
        metadata to preserve attribution in downstream reports.
    mixing_rules:
        Dictionary describing known composite formulations and the mixing rule
        (``series`` or ``parallel``) inferred from the Zenodo workbooks.  Each
        entry documents the component fractions and the source workbook so the
        generator can surface traceability information alongside the features.
    compatibility_matrix:
        Nested dictionary capturing polymer↔polymer/regolith compatibility. The
        outer keys refer to canonical material identifiers (``slugify`` of the
        reference key) and the inner mapping lists the supported partners with
        their associated mixing rule and evidences extracted from the source
        datasets.
    """

    table: pl.DataFrame
    properties: Dict[str, Dict[str, float]]
    density_map: Dict[str, float]
    alias_map: Dict[str, str]
    property_columns: tuple[str, ...]
    spectral_curves: Dict[str, pd.DataFrame]
    metadata: Dict[str, Dict[str, Any]]
    mixing_rules: Dict[str, Dict[str, Any]]
    compatibility_matrix: Dict[str, Dict[str, Any]]


@lru_cache(maxsize=1)
def load_material_reference_bundle() -> MaterialReferenceBundle:
    """Return the consolidated Zenodo material reference bundle.

    The loader is cached because it performs several CSV reads and normalisation
    steps.  Callers are expected to treat the returned bundle as immutable.
    """

    bundle_dir = DATASETS_ROOT / "zenodo"
    property_columns = (
        "material_density_kg_m3",
        "material_modulus_gpa",
        "material_tensile_strength_mpa",
        "material_elongation_pct",
        "material_oxygen_index_pct",
        "material_water_absorption_pct",
        "material_thermal_conductivity_w_mk",
        "material_glass_transition_c",
        "material_melting_temperature_c",
        "material_service_temperature_short_c",
        "material_service_temperature_long_c",
        "material_service_temperature_min_c",
        "material_coefficient_thermal_expansion_per_k_min",
        "material_coefficient_thermal_expansion_per_k_max",
        "material_ball_indentation_hardness_mpa",
        "material_shore_d_hardness",
        "material_rockwell_m_hardness",
        "material_surface_resistivity_ohm",
        "material_volume_resistivity_ohm_cm",
        "material_dielectric_strength_kv_mm",
        "material_relative_permittivity_low_freq",
        "material_relative_permittivity_high_freq",
        "material_dielectric_loss_tan_delta_low_freq",
        "material_dielectric_loss_tan_delta_high_freq",
        "material_comparative_tracking_index_cti",
    )

    default = MaterialReferenceBundle(
        pl.DataFrame(),
        {},
        {},
        {},
        property_columns,
        {},
        {},
        {},
        {},
    )

    if not bundle_dir.exists():
        return default

    records_map: Dict[str, dict[str, Any]] = {}
    properties: Dict[str, Dict[str, float]] = {}
    density_map: Dict[str, float] = {}
    alias_map: Dict[str, str] = {}
    spectral_curves: Dict[str, pd.DataFrame] = {}
    metadata: Dict[str, Dict[str, Any]] = {}
    mixing_rules: Dict[str, Dict[str, Any]] = {}
    compatibility_matrix: Dict[str, Dict[str, Any]] = {}

    def _safe_float(value: Any) -> float | None:
        if value is None:
            return None
        if isinstance(value, (int, float, np.floating)):
            numeric = float(value)
            if math.isnan(numeric):
                return None
            return numeric
        try:
            numeric = float(str(value))
        except (TypeError, ValueError):
            return None
        if math.isnan(numeric):
            return None
        return numeric

    def _register_alias(canonical: str, candidate: str | None) -> None:
        if not candidate:
            return
        normalized = normalize_item(candidate)
        if not normalized:
            return
        slug = slugify(normalized)
        if slug and slug not in alias_map:
            alias_map[slug] = canonical

    def _coefficient(value: Any, scale: float) -> float | None:
        numeric = _safe_float(value)
        if numeric is None:
            return None
        return float(numeric) * scale

    def _aggregate_range(values: Iterable[float | None]) -> tuple[float | None, float | None]:
        numeric_values = [float(v) for v in values if v is not None and math.isfinite(v)]
        if not numeric_values:
            return None, None
        return min(numeric_values), max(numeric_values)

    def _add_record(
        name: str,
        payload: Mapping[str, Any],
        *,
        aliases: Iterable[str] = (),
        meta: Mapping[str, Any] | None = None,
    ) -> None:
        canonical_name = normalize_item(name)
        slug = slugify(canonical_name)
        if not slug:
            return

        row: dict[str, Any] = records_map.get(slug, {"material_key": slug, "material_name": name})
        if "material_name" not in row or not row["material_name"]:
            row["material_name"] = name

        clean_props: Dict[str, float] = {}
        for column in property_columns:
            raw_value = payload.get(column)
            numeric = _safe_float(raw_value)
            if numeric is None:
                row[column] = float("nan")
            else:
                row[column] = float(numeric)
                clean_props[column] = float(numeric)
                if column == "material_density_kg_m3":
                    if slug in density_map and not math.isnan(density_map[slug]):
                        density_map[slug] = float(
                            np.nanmean([density_map[slug], float(numeric)])
                        )
                    else:
                        density_map[slug] = float(numeric)

        if slug in properties:
            merged = properties[slug]
            for column, value in clean_props.items():
                if column in merged:
                    existing = merged[column]
                    if math.isnan(existing):
                        merged[column] = value
                    elif not math.isnan(value):
                        merged[column] = float(np.nanmean([existing, value]))
                else:
                    merged[column] = value
        else:
            properties[slug] = clean_props

        existing_row = records_map.get(slug)
        if existing_row is None:
            records_map[slug] = row
        else:
            for column, value in row.items():
                if column in {"material_key", "material_name"}:
                    continue
                numeric = _safe_float(value)
                if numeric is None:
                    continue
                prev = _safe_float(existing_row.get(column))
                if prev is None:
                    existing_row[column] = float(numeric)
                else:
                    existing_row[column] = float(np.nanmean([prev, numeric]))

        _register_alias(slug, name)
        for alias in aliases:
            _register_alias(slug, alias)
        _register_alias(slug, canonical_name)
        # Provide token aliases for key words (nylon -> nylon66, etc.).
        for token in canonical_name.split():
            if len(token) >= 3:
                _register_alias(slug, token)

        if meta:
            metadata[slug] = {str(k): v for k, v in meta.items()}

    def _merge_compatibility(
        base: Dict[str, Dict[str, Any]],
        extra: Mapping[str, Mapping[str, Any]],
    ) -> None:
        for material, partners in extra.items():
            target = base.setdefault(material, {})
            for partner, payload in partners.items():
                existing = target.get(partner, {})
                merged: Dict[str, Any] = dict(existing)
                for key, value in payload.items():
                    if key in {"sources", "evidence"}:
                        current: list[Any] = list(merged.get(key, []))
                        for item in value:
                            if item not in current:
                                current.append(item)
                        merged[key] = current
                    elif key == "rule":
                        merged[key] = value or merged.get(key)
                    else:
                        merged[key] = value
                if "sources" not in merged:
                    merged["sources"] = []
                if "evidence" not in merged:
                    merged["evidence"] = []
                target[partner] = merged

    def _parse_mnl1_mecha_workbook(
        workbook: Path,
        alias_snapshot: Mapping[str, str],
    ) -> tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
        if not workbook.exists():
            return {}, {}

        try:
            composition_df = pd.read_excel(workbook, sheet_name="Composition", header=None)
            mechanical_df = pd.read_excel(workbook, sheet_name="Mecha MNL1", header=None)
        except Exception:
            return {}, {}

        if composition_df.empty:
            return {}, {}

        # Identify component descriptors (PE, EVOH) from the composition sheet.
        component_descriptors: list[dict[str, Any]] = []
        if len(composition_df.index) > 2:
            header_row = composition_df.iloc[2]
            alias_override = {
                "pe": "polyethylene",
                "evoh": "ethylene_vinyl_alcohol",
            }
            for column_idx, raw in header_row.items():
                if not isinstance(raw, str):
                    continue
                tokens = [token.strip() for token in raw.splitlines() if token.strip()]
                if not tokens:
                    continue
                base_label = tokens[-1]
                slug = slugify(normalize_item(base_label))
                canonical_slug = slugify(alias_override.get(slug, slug))
                canonical_key = alias_snapshot.get(canonical_slug)
                descriptor = {
                    "column": column_idx,
                    "label": base_label,
                    "slug": slug,
                    "canonical_slug": canonical_slug,
                    "canonical_key": canonical_key,
                }
                component_descriptors.append(descriptor)

        if not component_descriptors:
            return {}, {}

        component_lookup: Dict[str, dict[str, Any]] = {}
        for descriptor in component_descriptors:
            for key in {
                descriptor["slug"],
                descriptor["canonical_slug"],
                descriptor.get("canonical_key") or "",
            }:
                if key:
                    component_lookup[key] = descriptor

        composite_slug = slugify("pe_evoh_multilayer_film")
        composite_key = alias_snapshot.get(composite_slug, composite_slug)

        def _add_pair(
            matrix: Dict[str, Dict[str, Any]],
            a: str | None,
            b: str | None,
            *,
            rule: str,
            source: str,
            evidence: Mapping[str, Any] | None = None,
        ) -> None:
            if not a or not b:
                return
            partner_map = matrix.setdefault(a, {})
            payload = partner_map.setdefault(
                b,
                {"rule": rule, "sources": [], "evidence": []},
            )
            if rule and rule not in {payload.get("rule"), None}:
                payload["rule"] = rule
            sources: list[str] = list(payload.get("sources", []))
            if source and source not in sources:
                sources.append(source)
            payload["sources"] = sources
            if evidence:
                evidence_list = list(payload.get("evidence", []))
                evidence_list.append(dict(evidence))
                payload["evidence"] = evidence_list

        variants: list[dict[str, Any]] = []
        if not mechanical_df.empty:
            for _, row in mechanical_df.iterrows():
                label = str(row.iloc[0]) if len(row) > 0 else ""
                if not label or "PE/EVOH" not in label:
                    continue
                label = label.strip()
                setup = str(row.iloc[1]) if len(row) > 1 else ""
                lme_match = re.search(r"(\d+)\s*LME", setup)
                layers_match = re.search(r"(\d+)\s*layer", setup, flags=re.IGNORECASE)
                lme_position = int(lme_match.group(1)) if lme_match else None
                layers = int(layers_match.group(1)) if layers_match else None

                tokens = label.split()
                materials_part = tokens[0] if tokens else ""
                ratio_part = tokens[1] if len(tokens) > 1 else ""
                material_tokens = [tok.strip() for tok in materials_part.split("/") if tok.strip()]
                ratio_tokens = [tok.strip() for tok in ratio_part.split("/") if tok.strip()]
                ratios: list[float] = []
                for tok in ratio_tokens:
                    try:
                        ratios.append(float(tok))
                    except (TypeError, ValueError):
                        ratios.append(float("nan"))
                if len(material_tokens) != len(ratios) or not material_tokens:
                    continue
                total = sum(value for value in ratios if math.isfinite(value))
                if not total:
                    continue

                composition: Dict[str, float] = {}
                component_aliases: Dict[str, str] = {}
                for token, ratio in zip(material_tokens, ratios):
                    slug = slugify(normalize_item(token))
                    descriptor = component_lookup.get(slug)
                    if descriptor is None:
                        descriptor = {
                            "slug": slug,
                            "canonical_slug": slug,
                            "canonical_key": alias_snapshot.get(slug),
                            "label": token,
                        }
                    key = descriptor.get("canonical_key") or descriptor.get("canonical_slug") or descriptor.get("slug")
                    if not key:
                        continue
                    fraction = float(ratio) / float(total)
                    composition[key] = fraction
                    component_aliases[key] = descriptor.get("label", token)

                if not composition:
                    continue

                metric_mapping = [
                    (2, "machine_direction", "tensile_modulus_mpa"),
                    (3, "machine_direction", "stress_at_yield_mpa"),
                    (4, "machine_direction", "elongation_at_yield_pct"),
                    (5, "machine_direction", "stress_at_break_mpa"),
                    (6, "machine_direction", "elongation_at_break_pct"),
                    (7, "transversal_direction", "tensile_modulus_mpa"),
                    (8, "transversal_direction", "stress_at_yield_mpa"),
                    (9, "transversal_direction", "elongation_at_yield_pct"),
                    (10, "transversal_direction", "stress_at_break_mpa"),
                    (11, "transversal_direction", "elongation_at_break_pct"),
                ]

                mechanical: Dict[str, Dict[str, float | None]] = {}
                for idx, section, metric in metric_mapping:
                    if idx >= len(row):
                        continue
                    value = row.iloc[idx]
                    numeric = None
                    if isinstance(value, str):
                        match = re.search(r"([-+]?[0-9]+(?:\.[0-9]+)?)", value.replace(",", "."))
                        if match:
                            numeric = float(match.group(1))
                    elif isinstance(value, (int, float, np.floating)) and not math.isnan(value):
                        numeric = float(value)
                    if section not in mechanical:
                        mechanical[section] = {}
                    mechanical[section][metric] = numeric

                variants.append(
                    {
                        "label": label,
                        "composition": composition,
                        "component_labels": component_aliases,
                        "lme_position": lme_position,
                        "layers": layers,
                        "mechanical": mechanical,
                    }
                )

        if not variants:
            return {}, {}

        source = f"datasets/zenodo/{workbook.name}"
        components_payload: Dict[str, Dict[str, Any]] = {}
        for descriptor in component_descriptors:
            key = descriptor.get("canonical_key") or descriptor.get("canonical_slug") or descriptor.get("slug")
            if not key:
                continue
            components_payload[key] = {
                "label": descriptor.get("label"),
                "canonical_slug": descriptor.get("canonical_slug"),
                "canonical_key": descriptor.get("canonical_key"),
            }

        mixing_payload = {
            composite_key: {
                "rule": "series",
                "source": source,
                "components": components_payload,
                "variants": variants,
            }
        }

        compatibility_payload: Dict[str, Dict[str, Any]] = {}
        for variant in variants:
            composition = variant.get("composition", {})
            for key_a, frac_a in composition.items():
                for key_b, frac_b in composition.items():
                    if key_a == key_b:
                        continue
                    evidence = {
                        "variant": variant.get("label"),
                        "component_fraction": frac_a,
                        "partner_fraction": frac_b,
                        "layers": variant.get("layers"),
                        "lme_position": variant.get("lme_position"),
                    }
                    _add_pair(
                        compatibility_payload,
                        key_a,
                        key_b,
                        rule="series",
                        source=source,
                        evidence=evidence,
                    )
                for component_key, fraction in composition.items():
                    evidence = {
                        "variant": variant.get("label"),
                        "component_fraction": fraction,
                        "layers": variant.get("layers"),
                        "lme_position": variant.get("lme_position"),
                    }
                    _add_pair(
                        compatibility_payload,
                        composite_key,
                        component_key,
                        rule="series",
                        source=source,
                        evidence=evidence,
                    )
                    _add_pair(
                        compatibility_payload,
                        component_key,
                        composite_key,
                        rule="series",
                        source=source,
                        evidence=evidence,
                    )

        return mixing_payload, compatibility_payload

    def _build_regolith_compatibility(alias_snapshot: Mapping[str, str]) -> Dict[str, Dict[str, Any]]:
        regolith_path = DATASETS_ROOT / "raw" / "mgs1_properties.csv"
        if not regolith_path.exists():
            return {}
        try:
            reg_df = pd.read_csv(regolith_path)
        except Exception:
            return {}

        metrics: Dict[str, float | None] = {}
        for _, row in reg_df.iterrows():
            prop = str(row.get("property", "")).strip().lower()
            value = _safe_float(row.get("value"))
            if prop:
                metrics[prop] = value

        source = f"datasets/raw/{regolith_path.name}"
        regolith_key = "mgs_1_regolith"
        compatibility_payload: Dict[str, Dict[str, Any]] = {}

        def _add_regolith_pair(material_key: str) -> None:
            evidence = {
                "density_bulk_g_cm3": metrics.get("density_bulk"),
                "median_grain_size_um": metrics.get("median_grain_size"),
                "assumption": "Regolith fillers disperse in parallel with polymer matrices.",
            }
            _merge_compatibility(
                compatibility_payload,
                {
                    material_key: {
                        regolith_key: {
                            "rule": "parallel",
                            "sources": [source],
                            "evidence": [evidence],
                        }
                    },
                    regolith_key: {
                        material_key: {
                            "rule": "parallel",
                            "sources": [source],
                            "evidence": [evidence],
                        }
                    },
                },
            )

        for candidate in (
            "polyethylene",
            "polypropylene",
            "nitrile rubber",
            "pe_evoh_multilayer_film",
        ):
            slug = slugify(normalize_item(candidate))
            key = alias_snapshot.get(slug)
            if key:
                _add_regolith_pair(key)

        return compatibility_payload

    # ------------------------------------------------------------------
    # HDPE / Polyethylene
    # ------------------------------------------------------------------
    hdpe_path = bundle_dir / "hdpe_properties.csv"
    if hdpe_path.exists():
        hdpe_df = pd.read_csv(hdpe_path)
        for _, row in hdpe_df.iterrows():
            density = _safe_float(row.get("density_g_cm3"))
            service_short = _safe_float(row.get("service_temp_short_C"))
            service_long = _safe_float(row.get("service_temp_long_C"))
            cte_min = _coefficient(row.get("CTE_1e-5_perK_min"), 1e-5)
            cte_max = _coefficient(row.get("CTE_1e-5_perK_max"), 1e-5)
            modulus = _safe_float(row.get("E_modulus_tension_GPa"))
            strength = _safe_float(row.get("tensile_strength_yield_MPa"))
            elongation = _safe_float(row.get("elongation_yield_pct"))
            water = _safe_float(row.get("water_absorption_saturation_23C_pct_lt"))
            conductivity = _safe_float(row.get("k_W_mK_min")) or _safe_float(
                row.get("k_W_mK_max")
            )
            payload = {
                "material_density_kg_m3": density * 1000.0 if density is not None else None,
                "material_modulus_gpa": modulus,
                "material_tensile_strength_mpa": strength,
                "material_elongation_pct": elongation,
                "material_water_absorption_pct": water,
                "material_thermal_conductivity_w_mk": conductivity,
                "material_glass_transition_c": _safe_float(row.get("Tg_C")),
                "material_melting_temperature_c": _safe_float(row.get("Tm_C")),
                "material_service_temperature_short_c": service_short,
                "material_service_temperature_long_c": service_long,
                "material_coefficient_thermal_expansion_per_k_min": cte_min,
                "material_coefficient_thermal_expansion_per_k_max": cte_max,
                "material_ball_indentation_hardness_mpa": _safe_float(
                    row.get("ball_indentation_hardness_MPa")
                ),
                "material_shore_d_hardness": _safe_float(row.get("shore_D")),
                "material_surface_resistivity_ohm": _safe_float(
                    row.get("surface_resistivity_ohm")
                ),
                "material_dielectric_strength_kv_mm": _safe_float(
                    row.get("dielectric_strength_kV_mm")
                ),
            }
            aliases = [
                str(row.get("material", "")),
                str(row.get("color", "")),
                "polyethylene",
                "hdpe",
            ]
            _add_record(str(row.get("material", "HDPE")), payload, aliases=aliases)

    # ------------------------------------------------------------------
    # Nomex 410 (aramid paper)
    # ------------------------------------------------------------------
    nomex_path = bundle_dir / "nomex410_properties.csv"
    if nomex_path.exists():
        nomex_df = pd.read_csv(nomex_path)
        for _, row in nomex_df.iterrows():
            thickness_mm = _safe_float(row.get("thickness_mm")) or 0.0
            thickness_m = thickness_mm / 1000.0
            tensile_n_cm = _safe_float(row.get("tensile_MD_N_cm"))
            elongation_pct = _safe_float(row.get("elongation_MD_pct"))
            tensile_strength = None
            modulus = None
            if tensile_n_cm and thickness_m > 0:
                area_m2 = thickness_m * 0.01  # assume 1 cm width sample
                stress_pa = tensile_n_cm / max(area_m2, 1e-9)
                tensile_strength = stress_pa / 1e6
                if elongation_pct and elongation_pct > 0:
                    modulus = tensile_strength / (elongation_pct / 100.0)
            density = _safe_float(row.get("density_g_cm3"))
            oxygen_min = _safe_float(row.get("LOI_room_temp_pct_min"))
            oxygen_max = _safe_float(row.get("LOI_room_temp_pct_max"))
            oxygen_index = None
            if oxygen_min and oxygen_max:
                oxygen_index = 0.5 * (oxygen_min + oxygen_max)
            elif oxygen_min:
                oxygen_index = oxygen_min
            elif oxygen_max:
                oxygen_index = oxygen_max
            dissipation = _safe_float(row.get("dissipation_factor_60Hz_x1e_3"))
            if dissipation is not None:
                dissipation = dissipation * 1e-3
            payload = {
                "material_density_kg_m3": density * 1000.0 if density is not None else None,
                "material_modulus_gpa": modulus,
                "material_tensile_strength_mpa": tensile_strength,
                "material_elongation_pct": elongation_pct,
                "material_oxygen_index_pct": oxygen_index,
                "material_thermal_conductivity_w_mk": (
                    (_safe_float(row.get("thermal_conductivity_mW_mK_150C")) or 0.0) / 1000.0
                ),
                "material_dielectric_strength_kv_mm": _safe_float(
                    row.get("dielectric_strength_AC_kV_per_mm")
                ),
                "material_relative_permittivity_low_freq": _safe_float(
                    row.get("dielectric_constant_60Hz")
                ),
                "material_dielectric_loss_tan_delta_low_freq": dissipation,
            }
            meta = {
                "source": str(row.get("source", "")),
                "family": str(row.get("family", "")),
                "reference_pdf": str(row.get("reference_pdf", "")),
            }
            _add_record("Nomex 410", payload, aliases=["nomex", "aramid"], meta=meta)

    # ------------------------------------------------------------------
    # Nylon 6/6 technical sheet
    # ------------------------------------------------------------------
    nylon_path = bundle_dir / "rexai_material_reference_nylon66.csv"
    if nylon_path.exists():
        nylon_df = pd.read_csv(nylon_path)
        for _, row in nylon_df.iterrows():
            density = _safe_float(row.get("density_g_cm3"))
            cte_values = [
                _coefficient(row.get("cte_23_60_e6_per_k"), 1e-6),
                _coefficient(row.get("cte_23_100_e6_per_k"), 1e-6),
            ]
            cte_min, cte_max = _aggregate_range(cte_values)
            service_short = _safe_float(row.get("service_temp_short_c"))
            service_candidates = [
                _safe_float(row.get("service_temp_long_c")),
                _safe_float(row.get("service_temp_5000h_c")),
                _safe_float(row.get("service_temp_20000h_c")),
            ]
            service_candidates = [value for value in service_candidates if value is not None]
            service_long = max(service_candidates) if service_candidates else None
            payload = {
                "material_density_kg_m3": density * 1000.0 if density is not None else None,
                "material_modulus_gpa": _safe_float(row.get("tensile_modulus_gpa")),
                "material_tensile_strength_mpa": _safe_float(row.get("tensile_strength_mpa")),
                "material_elongation_pct": _safe_float(row.get("tensile_strain_break_pct")),
                "material_oxygen_index_pct": _safe_float(row.get("oxygen_index_pct")),
                "material_water_absorption_pct": _safe_float(row.get("water_absorption_96h_pct")),
                "material_thermal_conductivity_w_mk": _safe_float(
                    row.get("thermal_conductivity_w_mk")
                ),
                "material_melting_temperature_c": _safe_float(row.get("melting_temperature_c")),
                "material_service_temperature_short_c": service_short,
                "material_service_temperature_long_c": service_long,
                "material_service_temperature_min_c": _safe_float(
                    row.get("min_service_temp_c")
                ),
                "material_coefficient_thermal_expansion_per_k_min": cte_min,
                "material_coefficient_thermal_expansion_per_k_max": cte_max,
                "material_ball_indentation_hardness_mpa": _safe_float(
                    row.get("ball_indentation_hardness_n_mm2")
                ),
                "material_rockwell_m_hardness": _safe_float(row.get("rockwell_hardness_M")),
                "material_surface_resistivity_ohm": _safe_float(
                    row.get("surface_resistivity_ohm")
                ),
                "material_volume_resistivity_ohm_cm": _safe_float(
                    row.get("volume_resistivity_ohm_cm")
                ),
                "material_dielectric_strength_kv_mm": _safe_float(
                    row.get("dielectric_strength_kv_mm")
                ),
                "material_relative_permittivity_low_freq": _safe_float(
                    row.get("relative_permittivity_100hz")
                ),
                "material_relative_permittivity_high_freq": _safe_float(
                    row.get("relative_permittivity_1mhz")
                ),
                "material_dielectric_loss_tan_delta_low_freq": _safe_float(
                    row.get("dielectric_loss_tan_delta_100hz")
                ),
                "material_dielectric_loss_tan_delta_high_freq": _safe_float(
                    row.get("dielectric_loss_tan_delta_1mhz")
                ),
                "material_comparative_tracking_index_cti": _safe_float(
                    row.get("comparative_tracking_index_cti")
                ),
            }
            meta = {
                "condition": str(row.get("condition", "")),
                "source": str(row.get("source_brand", "")),
                "source_file": str(row.get("source_file", "")),
            }
            aliases = [
                str(row.get("material_name", "")),
                str(row.get("material_key", "")),
                "nylon",
                "polyamide",
            ]
            _add_record(str(row.get("material_name", "Nylon 6/6")), payload, aliases=aliases, meta=meta)

    # ------------------------------------------------------------------
    # Polyolefins / EVOH / NBR composites
    # ------------------------------------------------------------------
    poly_path = bundle_dir / "rexai_materials_ref_polyolefins_evoh_nbr.csv"
    if poly_path.exists():
        poly_df = pd.read_csv(poly_path)
        for _, row in poly_df.iterrows():
            density = _safe_float(row.get("density_g_cm3"))
            payload = {
                "material_density_kg_m3": density * 1000.0 if density is not None else None,
                "material_modulus_gpa": _safe_float(row.get("tensile_modulus_GPa")),
                "material_tensile_strength_mpa": _safe_float(row.get("tensile_strength_MPa")),
                "material_elongation_pct": _safe_float(row.get("elongation_percent")),
            }
            material_name = str(row.get("material", "")) or "polymer"
            family = str(row.get("family", ""))
            aliases = [material_name, family, str(row.get("form", "")), "polypropylene", "evoh", "nbr"]
            _add_record(material_name, payload, aliases=aliases)

    # ------------------------------------------------------------------
    # Textile fibres (cotton, flax, etc.)
    # ------------------------------------------------------------------
    textile_path = bundle_dir / "textile_fibers_reference.csv"
    if textile_path.exists():
        textile_df = pd.read_csv(textile_path)
        for _, row in textile_df.iterrows():
            density_min = _safe_float(row.get("density_g_cm3_min"))
            density_max = _safe_float(row.get("density_g_cm3_max"))
            density = None
            if density_min and density_max:
                density = 0.5 * (density_min + density_max)
            elif density_min:
                density = density_min
            elif density_max:
                density = density_max
            strength_min = _safe_float(row.get("tensile_strength_MPa_min"))
            strength_max = _safe_float(row.get("tensile_strength_MPa_max"))
            strength = None
            if strength_min and strength_max:
                strength = 0.5 * (strength_min + strength_max)
            elif strength_min:
                strength = strength_min
            elif strength_max:
                strength = strength_max
            modulus_min = _safe_float(row.get("youngs_modulus_GPa_min"))
            modulus_max = _safe_float(row.get("youngs_modulus_GPa_max"))
            modulus = None
            if modulus_min and modulus_max:
                modulus = 0.5 * (modulus_min + modulus_max)
            elif modulus_min:
                modulus = modulus_min
            elif modulus_max:
                modulus = modulus_max
            elong_min = _safe_float(row.get("elongation_pct_min"))
            elong_max = _safe_float(row.get("elongation_pct_max"))
            elong = None
            if elong_min and elong_max:
                elong = 0.5 * (elong_min + elong_max)
            elif elong_min:
                elong = elong_min
            elif elong_max:
                elong = elong_max
            payload = {
                "material_density_kg_m3": density * 1000.0 if density is not None else None,
                "material_modulus_gpa": modulus,
                "material_tensile_strength_mpa": strength,
                "material_elongation_pct": elong,
            }
            aliases = [
                str(row.get("material", "")),
                str(row.get("category", "")),
                "textile",
                "fabric",
                "cotton",
                "flax",
            ]
            _add_record(str(row.get("material", "textile")), payload, aliases=aliases)

    # ------------------------------------------------------------------
    # PET-P (Ertalyte) mechanical sheet (wide range of properties)
    # ------------------------------------------------------------------
    petp_path = bundle_dir / "ertalyte_petp_properties.csv"
    if petp_path.exists():
        petp_df = pd.read_csv(petp_path)
        if not petp_df.empty:
            pivot = (
                petp_df.assign(property=lambda df: df["property"].str.lower())
                .pivot_table(index="material", columns="property", values="value", aggfunc="mean")
            )
            for material, row in pivot.iterrows():
                payload = {
                    "material_modulus_gpa": _safe_float(row.get("tensile_modulus"))
                    if row.get("tensile_modulus") is not None
                    else None,
                    "material_tensile_strength_mpa": _safe_float(row.get("tensile_strength")),
                    "material_elongation_pct": _safe_float(row.get("elongation_at_break")),
                    "material_thermal_conductivity_w_mk": _safe_float(row.get("thermal_conductivity")),
                }
                _add_record(str(material), payload, aliases=["pet", "polyester", "petp", "ertalyte"])

    # ------------------------------------------------------------------
    # PVDF FTIR spectra
    # ------------------------------------------------------------------
    pvdf_path = bundle_dir / "pvdf_ftir_phases_1um_160C.csv"
    if pvdf_path.exists():
        pvdf_df = pd.read_csv(pvdf_path)
        if not pvdf_df.empty:
            spectral_curves["pvdf_alpha_160c"] = pvdf_df
            metadata["pvdf_alpha_160c"] = {
                "phase": "alpha",
                "temperature_c": int(pvdf_df.get("temperature_C", pd.Series([160])).iloc[0]),
                "material": "PVDF",
                "source": str(pvdf_df.get("source", pd.Series(["Zenodo"])).iloc[0]),
                "license": str(pvdf_df.get("license", pd.Series(["CC BY 4.0"])).iloc[0]),
            }
            _add_record(
                "PVDF",
                {
                    "material_density_kg_m3": 1780.0,
                    "material_modulus_gpa": None,
                    "material_tensile_strength_mpa": None,
                },
                aliases=["polyvinylidene fluoride", "pvdf"],
            )

    # ------------------------------------------------------------------
    # Polystyrene spectral reference (transmittance)
    # ------------------------------------------------------------------
    ps_path = bundle_dir / "PS_c4_50.csv"
    if ps_path.exists():
        data_rows: list[dict[str, float]] = []
        meta: Dict[str, str] = {}
        with ps_path.open("r", encoding="utf-8") as handle:
            for raw_line in handle:
                parts = [part.strip() for part in raw_line.strip().split(",") if part.strip()]
                if not parts:
                    continue
                try:
                    wavenumber = float(parts[0])
                    transmittance = float(parts[1])
                except (ValueError, IndexError):
                    if ":" in parts[0]:
                        key, *rest = parts[0].split(":", 1)
                        if rest:
                            meta_key = slugify(normalize_item(key))
                            meta_value = rest[0].strip()
                        elif len(parts) > 1:
                            meta_key = slugify(normalize_item(parts[0]))
                            meta_value = parts[1]
                        else:
                            meta_key = slugify(normalize_item(parts[0]))
                            meta_value = ""
                        if meta_key:
                            meta[meta_key] = meta_value
                    continue
                data_rows.append(
                    {
                        "wavenumber_cm_1": wavenumber,
                        "transmittance_pct": transmittance,
                    }
                )
        if data_rows:
            spectral_curves["polystyrene_transmittance"] = pd.DataFrame(data_rows)
            metadata["polystyrene_transmittance"] = meta or {
                "material": "polystyrene",
                "instrument": "FTIR",
            }
            _add_record(
                "Polystyrene",
                {
                    "material_density_kg_m3": 1050.0,
                    "material_modulus_gpa": None,
                    "material_tensile_strength_mpa": None,
                },
                aliases=["ps", "polystyrene"],
            )

    workbook_path = bundle_dir / "MNL1 Mecha.xlsx"
    workbook_rules, workbook_compat = _parse_mnl1_mecha_workbook(workbook_path, alias_map)
    if workbook_rules:
        mixing_rules.update(workbook_rules)
    if workbook_compat:
        _merge_compatibility(compatibility_matrix, workbook_compat)

    regolith_compat = _build_regolith_compatibility(alias_map)
    if regolith_compat:
        _merge_compatibility(compatibility_matrix, regolith_compat)

    if not records_map:
        return default

    records = list(records_map.values())
    table = pl.from_dicts(records)
    table = table.unique(subset=["material_key"], keep="first")
    return MaterialReferenceBundle(
        table,
        properties,
        density_map,
        alias_map,
        property_columns,
        spectral_curves,
        metadata,
        mixing_rules,
        compatibility_matrix,
    )


class _WasteSummary(NamedTuple):
    mass_by_key: Dict[str, Dict[str, float]]
    mission_totals: Dict[str, float]


def _mission_slug(column: str) -> str:
    cleaned = column.lower()
    cleaned = cleaned.replace("summary_", "")
    cleaned = cleaned.replace("mass", "")
    cleaned = cleaned.replace("kg", "")
    cleaned = cleaned.replace("total", "")
    cleaned = cleaned.replace("__", "_")
    return slugify(cleaned)


def _load_waste_summary_data() -> _WasteSummary:
    path = resolve_dataset_path("nasa_waste_summary.csv")
    if path is None:
        return _WasteSummary({}, {})

    table = pl.scan_csv(path)
    column_names = list(table.collect_schema().names())
    column_set = set(column_names)

    if "category" not in column_set:
        return _WasteSummary({}, {})

    mass_columns = [
        column
        for column in column_names
        if column.lower().endswith("mass_kg") and not column.lower().startswith("subitem_")
    ]
    if not mass_columns:
        return _WasteSummary({}, {})

    has_subitem = "subitem" in column_set
    subitem_expr = (
        pl.when(pl.col("subitem").is_not_null())
        .then(pl.col("subitem").map_elements(normalize_item, return_dtype=pl.String))
        .otherwise(pl.lit(""))
        .alias("subitem_norm")
        if has_subitem
        else pl.lit("").alias("subitem_norm")
    )

    melted = (
        table.with_columns(
            pl.col("category")
            .map_elements(normalize_category, return_dtype=pl.String)
            .alias("category_norm"),
            subitem_expr,
        )
        .with_columns(
            pl.when(pl.col("subitem_norm").str.len_bytes() > 0)
            .then(pl.col("category_norm") + pl.lit("|") + pl.col("subitem_norm"))
            .otherwise(pl.col("category_norm"))
            .alias("item_key"),
            pl.col("category_norm").alias("category_key"),
        )
        .melt(
            id_vars=["category_key", "item_key"],
            value_vars=mass_columns,
        )
        .with_columns(
            pl.col("variable")
            .map_elements(_mission_slug, return_dtype=pl.String)
            .alias("mission"),
            pl.col("value").cast(pl.Float64, strict=False).alias("mass"),
        )
        .drop_nulls("mission")
        .drop_nulls("mass")
    )

    mass_by_key: Dict[str, Dict[str, float]] = {}
    mission_totals: Dict[str, float] = {}
    for row in melted.collect().to_dicts():
        key = str(row["item_key"])
        mission = str(row["mission"])
        mass = float(row["mass"])
        if not mission:
            continue
        mission_totals[mission] = mission_totals.get(mission, 0.0) + mass
        entry = mass_by_key.setdefault(key, {})
        entry[mission] = entry.get(mission, 0.0) + mass
        category_key = str(row["category_key"])
        if key != category_key:
            category_entry = mass_by_key.setdefault(category_key, {})
            category_entry[mission] = category_entry.get(mission, 0.0) + mass

    return _WasteSummary(mass_by_key, mission_totals)


def extract_grouped_metrics(filename: str, prefix: str) -> Dict[str, Dict[str, float]]:
    path = resolve_dataset_path(filename)
    if path is None:
        return {}

    table = pl.scan_csv(path)

    row_count = table.select(pl.len().alias("rows")).collect().row(0)[0]
    if row_count == 0:
        return {}

    schema = table.collect_schema()
    numeric_cols = [
        name
        for name, dtype in schema.items()
        if dtype.is_numeric()
    ]
    if not numeric_cols:
        return {}

    group_candidates = {
        "mission",
        "scenario",
        "approach",
        "vehicle",
        "propulsion",
        "architecture",
    }
    column_names = list(schema.names())
    group_columns = [col for col in column_names if col.lower() in group_candidates]

    aggregated: Dict[str, Dict[str, float]] = {}

    if not group_columns:
        summary = (
            table.select([pl.col(col).cast(pl.Float64, strict=False).mean().alias(col) for col in numeric_cols])
            .collect()
            .to_dicts()
        )
        metrics = {}
        if summary:
            metrics = {}
            for column, value in summary[0].items():
                if value is None:
                    continue
                if isinstance(value, float) and math.isnan(value):
                    continue
                metrics[f"{prefix}_{slugify(column)}"] = float(value)
        if metrics:
            aggregated[prefix] = metrics
        return aggregated

    combinations: list[tuple[str, ...]] = []
    for length in range(1, len(group_columns) + 1):
        combinations.extend(itertools.combinations(group_columns, length))

    for combo in combinations:
        grouped = (
            table.group_by(list(combo))
            .agg([pl.col(col).cast(pl.Float64, strict=False).mean().alias(col) for col in numeric_cols])
            .collect()
            .to_dicts()
        )
        for row in grouped:
            slug_parts: list[str] = []
            for column in combo:
                value = row.get(column)
                if isinstance(value, str):
                    slug = slugify(value)
                elif value is not None:
                    slug = slugify(str(value))
                else:
                    slug = ""
                if slug:
                    slug_parts.append(slug)
            slug = "_".join(part for part in slug_parts if part)
            if not slug:
                continue

            metrics: Dict[str, float] = {}
            for column in numeric_cols:
                value = row.get(column)
                if value is None or (isinstance(value, float) and math.isnan(value)):
                    continue
                metrics[f"{prefix}_{slugify(column)}"] = float(value)

            if metrics:
                aggregated[slug] = metrics

    return aggregated


def _load_regolith_vector() -> Dict[str, float]:
    path = resolve_dataset_path("MGS-1_Martian_Regolith_Simulant_Recipe.csv")
    if path is None:
        path = DATASETS_ROOT / "raw" / "mgs1_oxides.csv"

    if path and path.exists():
        table = pd.read_csv(path)
        key_cols = [
            col
            for col in table.columns
            if col.lower() in {"oxide", "component", "phase", "mineral"}
        ]
        value_cols = [
            col
            for col in table.columns
            if any(token in col.lower() for token in ("wt", "weight", "percent"))
        ]

        key_col = key_cols[0] if key_cols else None
        value_col = value_cols[0] if value_cols else None

        if key_col and value_col:
            working = table[[key_col, value_col]].dropna()

            def _clean_label(value: Any) -> str:
                text = str(value or "").lower()
                text = re.sub(r"[^0-9a-z]+", "_", text)
                text = re.sub(r"_+", "_", text).strip("_")
                return text

            working[key_col] = working[key_col].map(_clean_label)
            weights = pd.to_numeric(working[value_col], errors="coerce")
            total = float(weights.sum())
            if total > 0:
                normalised = weights.div(total)
                return {
                    str(key): float(normalised.iloc[idx])
                    for idx, key in enumerate(working[key_col])
                    if pd.notna(normalised.iloc[idx])
                }

    return {"sio2": 0.48, "feot": 0.18, "mgo": 0.13, "cao": 0.055, "so3": 0.07, "h2o": 0.032}


def _load_gas_mean_yield() -> float:
    path = DATASETS_ROOT / "raw" / "nasa_trash_to_gas.csv"
    if path.exists():
        table = pd.read_csv(path)
        ratio = table["o2_ch4_yield_kg"] / table["water_makeup_kg"].clip(lower=1e-6)
        return float(ratio.mean())
    return 6.0


def _load_mean_reuse() -> float:
    path = DATASETS_ROOT / "raw" / "logistics_to_living.csv"
    if path.exists():
        table = pd.read_csv(path)
        efficiency = (
            (table["outfitting_replaced_kg"] - table["residual_waste_kg"]) / table["packaging_kg"].clip(lower=1e-6)
        ).clip(lower=0)
        return float(efficiency.mean())
    return 0.6


def _load_regolith_properties_table() -> pd.DataFrame:
    path = resolve_dataset_path("mgs1_properties.csv")
    if path is None or not path.exists():
        return pd.DataFrame(
            {
                "property": [
                    "median_grain_size",
                    "spectral_slope_1um",
                    "mass_loss_400c",
                    "h2o_peak_temp",
                ],
                "value": [122.0, 0.18, 0.03, 360.0],
                "units": ["µm", "%/100nm", "fraction", "c"],
            }
        )
    return pd.read_csv(path)


def _parse_regolith_properties(table: pd.DataFrame) -> dict[str, float]:
    working = (
        table.assign(property=lambda df: df["property"].astype(str).str.strip().str.lower())
        .dropna(subset=["property", "value"])
        .copy()
    )

    values = pd.to_numeric(working.set_index("property")["value"], errors="coerce")
    units = (
        working.set_index("property")["units"].astype(str).str.strip().str.lower()
        if "units" in working
        else pd.Series(dtype="string")
    )

    def _extract(name: str, default: float, *, percent: bool = False) -> float:
        value = values.get(name)
        if value is None or not np.isfinite(float(value)):
            return float(default)
        numeric = float(value)
        unit = str(units.get(name, "") or "").lower()
        if percent or (not percent and unit.startswith("wt%") and name == "water_release"):
            numeric /= 100.0
        return float(numeric)

    d50_um = float(np.clip(_extract("median_grain_size", 120.0), 1.0, 5_000.0))
    slope = float(_extract("spectral_slope_1um", 0.18))
    mass_loss = float(
        np.clip(
            _extract(
                "mass_loss_400c",
                _extract("water_release", 0.03, percent=True),
                percent=True,
            ),
            0.0,
            1.0,
        )
    )
    h2o_peak = float(np.clip(_extract("h2o_peak_temp", 360.0), 0.0, 2_000.0))

    return {
        "d50_um": d50_um,
        "spectral_slope_1um": slope,
        "mass_loss_400c": mass_loss,
        "h2o_peak_c": h2o_peak,
    }


def _load_regolith_baselines() -> dict[str, float]:
    table = _load_regolith_properties_table()
    return _parse_regolith_properties(table)


def _log_interp_percentile(diameters: np.ndarray, cdf: np.ndarray, target: float) -> float:
    """Return the diameter at *target* cumulative percent finer using log interpolation."""

    if diameters.size == 0 or cdf.size == 0:
        return float("nan")

    mask = np.isfinite(diameters) & np.isfinite(cdf)
    if not np.any(mask):
        return float("nan")

    diameters = diameters[mask]
    cdf = cdf[mask]

    if not np.all(np.diff(cdf) >= 0):
        order = np.argsort(cdf)
        cdf = cdf[order]
        diameters = diameters[order]

    unique_cdf, unique_idx = np.unique(cdf, return_index=True)
    diameters = diameters[unique_idx]

    if target <= unique_cdf[0]:
        return float(diameters[0])
    if target >= unique_cdf[-1]:
        return float(diameters[-1])

    log_diam = np.log(diameters)
    interpolated = np.interp(target, unique_cdf, log_diam)
    return float(np.exp(interpolated))


def _log_size_slope(diameters: np.ndarray, cdf: np.ndarray) -> float:
    """Return the slope of log10(size) vs. log10(percent finer) for the central distribution."""

    mask = (
        np.isfinite(diameters)
        & np.isfinite(cdf)
        & (diameters > 0)
        & (cdf > 0)
        & (cdf < 100)
    )
    if not np.any(mask):
        return float("nan")

    diameters = diameters[mask]
    cdf = cdf[mask] / 100.0

    central = (cdf >= 0.1) & (cdf <= 0.9)
    if np.count_nonzero(central) >= 2:
        diameters = diameters[central]
        cdf = cdf[central]

    if diameters.size < 2:
        return float("nan")

    x = np.log10(diameters)
    y = np.log10(np.clip(cdf, 1e-6, 1.0))
    slope, _intercept = np.polyfit(x, y, 1)
    return float(slope)


def _mass_loss_between(
    temperatures: np.ndarray, mass: np.ndarray, start: float, stop: float
) -> float:
    """Return the mass loss percentage between *start* and *stop* temperatures."""

    if temperatures.size == 0 or mass.size == 0:
        return float("nan")

    ordered = np.argsort(temperatures)
    temperatures = temperatures[ordered]
    mass = mass[ordered]

    lower = float(np.interp(start, temperatures, mass))
    upper = float(np.interp(stop, temperatures, mass))
    return max(0.0, lower - upper)


@lru_cache(maxsize=1)
def load_regolith_particle_size() -> tuple[pl.DataFrame, Dict[str, float]]:
    """Return the MGS-1 particle size distribution and derived metrics."""

    path = resolve_dataset_path("fig3_psizeData.csv")
    if path is None or not path.exists():
        empty = pl.DataFrame(
            {
                "diameter_microns": pl.Series(dtype=pl.Float64),
                "percent_retained": pl.Series(dtype=pl.Float64),
                "percent_channel": pl.Series(dtype=pl.Float64),
            }
        )
        return empty, {}

    frame = pl.read_csv(path).rename(
        {
            "Diameter (microns)": "diameter_microns",
            "% Retained": "percent_retained",
            "% Channel": "percent_channel",
        }
    )

    frame = frame.select(
        [
            pl.col("diameter_microns").cast(pl.Float64),
            pl.col("percent_retained").cast(pl.Float64),
            pl.col("percent_channel").cast(pl.Float64),
        ]
    ).sort("diameter_microns", descending=True)

    frame = frame.with_columns(
        [
            (pl.col("percent_channel") / 100.0).alias("fraction_channel"),
            pl.col("percent_channel").cum_sum().alias("cumulative_percent_finer"),
            pl.col("percent_retained").alias("cumulative_percent_retained"),
            (100.0 - pl.col("percent_retained")).alias("percent_finer_than"),
        ]
    )

    metric_frame = frame.filter(pl.col("percent_channel") > 0).select(
        "diameter_microns", "cumulative_percent_finer"
    )

    metrics: Dict[str, float] = {}
    if metric_frame.height > 0:
        diameters = metric_frame.get_column("diameter_microns").to_numpy()
        cdf = metric_frame.get_column("cumulative_percent_finer").to_numpy()
        metrics.update(
            {
                "d10_microns": _log_interp_percentile(diameters, cdf, 90.0),
                "d50_microns": _log_interp_percentile(diameters, cdf, 50.0),
                "d90_microns": _log_interp_percentile(diameters, cdf, 10.0),
                "log_slope_fraction_finer": _log_size_slope(diameters, cdf),
            }
        )

    return frame, metrics


@lru_cache(maxsize=1)
def load_regolith_spectra() -> tuple[pl.DataFrame, Dict[str, float]]:
    """Return reflectance spectra for the regolith simulants with summary metrics."""

    path = resolve_dataset_path("fig4_spectralData.csv")
    if path is None or not path.exists():
        return pl.DataFrame(), {}

    frame = pl.read_csv(path).rename(
        {
            "Wavelength (nm)": "wavelength_nm",
            "MMS1": "reflectance_mms1",
            "MMS2": "reflectance_mms2",
            "JSC Mars-1": "reflectance_jsc_mars_1",
            "MGS-1 Prototype": "reflectance_mgs_1",
        }
    )

    frame = frame.select(
        [
            pl.col("wavelength_nm").cast(pl.Float64),
            pl.col("reflectance_mms1").cast(pl.Float64),
            pl.col("reflectance_mms2").cast(pl.Float64),
            pl.col("reflectance_jsc_mars_1").cast(pl.Float64),
            pl.col("reflectance_mgs_1").cast(pl.Float64),
        ]
    ).sort("wavelength_nm")

    metrics: Dict[str, float] = {}
    for column in frame.columns:
        if column == "wavelength_nm":
            continue
        metrics[f"mean_{column}"] = float(frame.get_column(column).mean())

    window = frame.filter(
        (pl.col("wavelength_nm") >= 700.0) & (pl.col("wavelength_nm") <= 1000.0)
    )
    if window.height >= 2:
        wavelengths = window.get_column("wavelength_nm").to_numpy()
        for column in window.columns:
            if column == "wavelength_nm":
                continue
            values = window.get_column(column).to_numpy()
            slope = float(np.polyfit(wavelengths, values, 1)[0])
            metrics[f"slope_{column}_700_1000"] = slope

    return frame, metrics


@lru_cache(maxsize=1)
def load_regolith_thermogravimetry() -> tuple[
    pl.DataFrame,
    pl.DataFrame,
    Dict[str, float],
    Dict[str, float],
]:
    """Return thermogravimetric and evolved gas analysis data with summaries."""

    tg_path = resolve_dataset_path("fig5_tgData.csv")
    ega_path = resolve_dataset_path("fig5_egaData.csv")

    if tg_path is None or not tg_path.exists():
        return pl.DataFrame(), pl.DataFrame(), {}, {}

    tg_frame = (
        pl.read_csv(tg_path, encoding="latin1")
        .rename({"Temperature (¡C)": "temperature_c", "Mass (%)": "mass_percent"})
        .select(
            [
                pl.col("temperature_c").cast(pl.Float64),
                pl.col("mass_percent").cast(pl.Float64),
            ]
        )
        .sort("temperature_c")
    )

    ega_metrics: Dict[str, float] = {}
    ega_frame = pl.DataFrame()
    if ega_path is not None and ega_path.exists():
        ega_frame = (
            pl.read_csv(ega_path, encoding="latin1")
            .rename({"Temperature (¡C)": "temperature_c"})
            .select([pl.all().cast(pl.Float64)])
            .sort("temperature_c")
        )

        if ega_frame.height > 0:
            temperatures = ega_frame.get_column("temperature_c").to_numpy()
            for column in ega_frame.columns:
                if column == "temperature_c":
                    continue
                series = ega_frame.get_column(column).to_numpy()
                if series.size == 0:
                    continue
                peak_idx = int(np.argmax(series))
                ega_metrics[f"peak_temperature_{slugify(column)}"] = float(
                    temperatures[peak_idx]
                )

    temperatures = tg_frame.get_column("temperature_c").to_numpy()
    mass = tg_frame.get_column("mass_percent").to_numpy()

    thermal_metrics: Dict[str, float] = {}
    if temperatures.size > 0 and mass.size > 0:
        initial_mass = float(mass[0])
        final_mass = float(mass[-1])
        thermal_metrics["mass_loss_total_percent"] = max(0.0, initial_mass - final_mass)
        thermal_metrics["residual_mass_percent"] = final_mass

        ranges = (
            (30.0, 200.0),
            (200.0, 400.0),
            (400.0, 600.0),
            (600.0, min(800.0, float(temperatures[-1]))),
        )
        for start, stop in ranges:
            if stop <= start:
                continue
            loss = _mass_loss_between(temperatures, mass, start, stop)
            key = f"mass_loss_{int(start)}_{int(stop)}_c"
            thermal_metrics[key] = loss

    return tg_frame, ega_frame, thermal_metrics, ega_metrics


@dataclass(frozen=True)
class RegolithCharacterization:
    particle_size: pl.DataFrame
    particle_metrics: Mapping[str, float]
    spectra: pl.DataFrame
    spectral_metrics: Mapping[str, float]
    thermogravimetry: pl.DataFrame
    evolved_gas: pl.DataFrame
    thermal_metrics: Mapping[str, float]
    gas_release_peaks: Mapping[str, float]
    d50_um: float
    spectral_slope_1um: float
    mass_loss_400c: float
    h2o_peak_c: float

    @property
    def feature_items(self) -> tuple[tuple[str, float], ...]:
        return (
            ("regolith_d50_um", float(self.d50_um)),
            ("regolith_spectral_slope_1um", float(self.spectral_slope_1um)),
            ("regolith_mass_loss_400c", float(self.mass_loss_400c)),
            ("regolith_h2o_peak_c", float(self.h2o_peak_c)),
        )


@lru_cache(maxsize=1)
def load_regolith_characterization() -> RegolithCharacterization:
    """Return a cached bundle with regolith particle, spectral and thermal summaries."""

    particle_size, particle_metrics = load_regolith_particle_size()
    spectra, spectral_metrics = load_regolith_spectra()
    tg_frame, ega_frame, thermal_metrics, ega_metrics = load_regolith_thermogravimetry()
    baselines = _load_regolith_baselines()

    return RegolithCharacterization(
        particle_size=particle_size,
        particle_metrics=particle_metrics,
        spectra=spectra,
        spectral_metrics=spectral_metrics,
        thermogravimetry=tg_frame,
        evolved_gas=ega_frame,
        thermal_metrics=thermal_metrics,
        gas_release_peaks=ega_metrics,
        d50_um=float(baselines["d50_um"]),
        spectral_slope_1um=float(baselines["spectral_slope_1um"]),
        mass_loss_400c=float(baselines["mass_loss_400c"]),
        h2o_peak_c=float(baselines["h2o_peak_c"]),
    )


@dataclass
class L2LParameters:
    constants: Dict[str, float]
    category_features: Dict[str, Dict[str, float]]
    item_features: Dict[str, Dict[str, float]]
    hints: Dict[str, str]


def _parse_l2l_numeric(value: Any) -> Dict[str, float]:
    """Return numeric representations for Logistics-to-Living values."""

    result: Dict[str, float] = {}
    if value is None:
        return result

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if np.isfinite(value):
            result["value"] = float(value)
        return result

    text = str(value).strip()
    if not text:
        return result

    cleaned = text.replace(",", "").replace("—", "-").replace("–", "-").replace("−", "-")
    numbers = [
        float(match)
        for match in re.findall(r"-?\d+(?:\.\d+)?", cleaned)
        if match not in {"-"}
    ]
    if not numbers:
        return result

    lowered = cleaned.lower()
    if any(token in lowered for token in (" per ", ":", "/")) and len(numbers) >= 2:
        denominator = numbers[1]
        if denominator:
            result["value"] = numbers[0] / denominator
            result["numerator"] = numbers[0]
            result["denominator"] = denominator
        else:
            result["value"] = numbers[0]
        return result

    if "-" in cleaned and len(numbers) >= 2:
        result["min"] = numbers[0]
        result["max"] = numbers[1]
        result["value"] = float(np.mean(numbers[:2]))
        return result

    result["value"] = numbers[0]
    if len(numbers) > 1:
        result["extra"] = numbers[1]
    return result


def load_l2l_parameters() -> L2LParameters:
    path = resolve_dataset_path("l2l_parameters.csv")
    if path is None or not path.exists():
        return L2LParameters({}, {}, {}, {})

    table = pd.read_csv(path)
    if table.empty:
        return L2LParameters({}, {}, {}, {})

    normalized_cols = {col.lower(): col for col in table.columns}
    category_col = normalized_cols.get("category")
    subitem_col = normalized_cols.get("subitem")

    descriptor_cols = [
        normalized_cols[name]
        for name in ("parameter", "metric", "key", "feature", "name", "field")
        if name in normalized_cols
    ]
    value_candidates = [
        column
        for column in table.columns
        if column not in {category_col, subitem_col}
        and column not in descriptor_cols
        and column.lower() not in {"page_hint", "units", "unit", "notes"}
    ]

    constants: Dict[str, float] = {}
    category_features: Dict[str, Dict[str, float]] = {}
    item_features: Dict[str, Dict[str, float]] = {}
    hints: Dict[str, str] = {}

    global_categories = {
        "geometry",
        "logistics",
        "scenario",
        "scenarios",
        "testbed",
        "ops",
        "operations",
        "materials",
        "material",
        "global",
        "constants",
    }

    for _, row in table.iterrows():
        category_value = row.get(category_col, "") if category_col else ""
        category_norm = normalize_category(category_value)
        subitem_value = row.get(subitem_col, "") if subitem_col else ""
        subitem_norm = normalize_item(subitem_value) if subitem_value else ""

        descriptor = ""
        for candidate in descriptor_cols:
            value = str(row.get(candidate, "")).strip()
            if value:
                descriptor = value
                break

        hint = str(row.get(normalized_cols.get("page_hint", "page_hint"), "")).strip()

        target_map: Dict[str, Dict[str, float]] | None
        key: str | None

        if category_norm in global_categories or not category_norm:
            target_map = None
            key = None
        elif subitem_norm:
            key = f"{category_norm}|{subitem_norm}"
            target_map = item_features
        else:
            key = category_norm
            target_map = category_features

        base_parts = ["l2l", category_norm]
        if subitem_norm:
            base_parts.append(subitem_norm)
        if descriptor:
            base_parts.append(descriptor)

        for column in value_candidates:
            payload = _parse_l2l_numeric(row.get(column))
            if not payload:
                continue

            for suffix, numeric_value in payload.items():
                if not np.isfinite(numeric_value):
                    continue
                name_parts = list(base_parts)
                if column:
                    name_parts.append(column)
                if suffix not in {"value"}:
                    name_parts.append(suffix)
                feature_name = _feature_name_from_parts(*name_parts)
                if not feature_name:
                    continue

                if category_norm in global_categories or not category_norm:
                    constants[feature_name] = float(numeric_value)
                elif target_map is not None and key is not None:
                    entry = target_map.setdefault(key, {})
                    entry[feature_name] = float(numeric_value)
                else:
                    constants[feature_name] = float(numeric_value)

                if hint:
                    hints[feature_name] = hint

    return L2LParameters(constants, category_features, item_features, hints)


@dataclass(frozen=True)
class ReferenceMetricSpec:
    filename: str
    prefix: str
    group_columns: tuple[str, ...]
    value_columns: tuple[str, ...]


def extract_reference_metrics(spec: ReferenceMetricSpec) -> Dict[str, Dict[str, float]]:
    path = resolve_dataset_path(spec.filename)
    if path is None:
        return {}

    table = pl.scan_csv(path)
    column_names = list(table.collect_schema().names())
    column_set = set(column_names)

    missing_groups = [column for column in spec.group_columns if column not in column_set]
    if missing_groups:
        return {}

    numeric_columns = [column for column in spec.value_columns if column in column_set]
    if not numeric_columns:
        return {}

    aggregations = [
        pl.col(column).cast(pl.Float64, strict=False).mean().alias(column)
        for column in numeric_columns
    ]

    grouped = (
        table.group_by(list(spec.group_columns)).agg(aggregations).collect().to_dicts()
    )

    metrics: Dict[str, Dict[str, float]] = {}
    for row in grouped:
        slug_parts = [spec.prefix]
        for column in spec.group_columns:
            value = row.get(column)
            if value is None:
                continue
            slug_parts.append(slugify(value))
        slug = _feature_name_from_parts(*slug_parts)
        if not slug:
            continue

        payload: Dict[str, float] = {}
        for column in numeric_columns:
            value = row.get(column)
            if value is None:
                continue
            if isinstance(value, float) and math.isnan(value):
                continue
            try:
                payload[f"{spec.prefix}_{slugify(column)}"] = float(value)
            except (TypeError, ValueError):
                continue

        if payload:
            metrics[slug] = payload

    return metrics


_REFERENCE_METRIC_SPECS: tuple[ReferenceMetricSpec, ...] = (
    ReferenceMetricSpec(
        filename="polymer_composite_density.csv",
        prefix="pc_density",
        group_columns=("sample_label",),
        value_columns=("density_g_per_cm3",),
    ),
    ReferenceMetricSpec(
        filename="polymer_composite_mechanics.csv",
        prefix="pc_mechanics",
        group_columns=("sample_label",),
        value_columns=("stress_mpa", "modulus_gpa", "strain_pct", "tensile_strength_mpa", "yield_strength_mpa"),
    ),
    ReferenceMetricSpec(
        filename="polymer_composite_thermal.csv",
        prefix="pc_thermal",
        group_columns=("sample_label",),
        value_columns=("glass_transition_c", "onset_temperature_c", "heat_capacity_j_per_g_k", "heat_flow_w_per_g"),
    ),
    ReferenceMetricSpec(
        filename="polymer_composite_ignition.csv",
        prefix="pc_ignition",
        group_columns=("sample_label",),
        value_columns=("ignition_temperature_c", "burn_time_min"),
    ),
    ReferenceMetricSpec(
        filename="aluminium_alloys.csv",
        prefix="aluminium",
        group_columns=("processing_route",),
        value_columns=("tensile_strength_mpa", "yield_strength_mpa", "elongation_pct"),
    ),
)


def _coerce_metric_value(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        numeric = float(value)
        if not math.isfinite(numeric):
            return None
        return numeric
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric):
        return None
    return numeric


def _translate_reference_metrics(
    payload: Mapping[str, Any],
    bundle: OfficialFeaturesBundle,
    match_key: str,
) -> Dict[str, float]:
    translated: Dict[str, float] = {}

    def _apply(column: str, raw_value: Any) -> None:
        numeric = _coerce_metric_value(raw_value)
        if numeric is None:
            return
        for source_column, target_name, transform in _MATERIAL_METRIC_COLUMNS:
            if column != source_column or target_name in translated:
                continue
            try:
                translated[target_name] = float(transform(numeric))
            except (TypeError, ValueError):
                continue
            return

    for column, value in payload.items():
        _apply(column, value)

    reference_metrics = getattr(bundle, "reference_metrics", {}) or {}
    if not reference_metrics:
        return translated

    candidates: list[str] = []
    if match_key:
        candidates.append(str(match_key))

    for prefix, columns in _REFERENCE_METRIC_LOOKUPS.items():
        values = []
        for column in columns:
            candidate = payload.get(column)
            if candidate is None or candidate == "":
                continue
            if isinstance(candidate, float) and not math.isfinite(candidate):
                continue
            values.append(candidate)
        if not values:
            continue
        slug = _feature_name_from_parts(prefix, *values)
        if slug:
            candidates.append(slug)

    for slug in candidates:
        metrics = reference_metrics.get(slug)
        if not metrics:
            continue
        for column, value in metrics.items():
            _apply(column, value)

    return translated

_MATERIAL_METRIC_COLUMNS: tuple[tuple[str, str, Any], ...] = (
    ("pc_density_density_g_per_cm3", "official_density_kg_m3", lambda value: float(value) * 1000.0),
    ("pc_density_density_kg_m3", "official_density_kg_m3", float),
    ("pc_mechanics_tensile_strength_mpa", "official_tensile_strength_mpa", float),
    ("pc_mechanics_stress_mpa", "official_tensile_strength_mpa", float),
    ("pc_mechanics_yield_strength_mpa", "official_yield_strength_mpa", float),
    ("pc_mechanics_modulus_gpa", "official_modulus_gpa", float),
    ("pc_mechanics_strain_pct", "official_strain_pct", float),
    ("pc_thermal_glass_transition_c", "official_glass_transition_c", float),
    ("pc_thermal_onset_temperature_c", "official_decomposition_onset_c", float),
    ("pc_ignition_ignition_temperature_c", "official_ignition_temperature_c", float),
    ("pc_ignition_burn_time_min", "official_burn_time_min", float),
    ("aluminium_tensile_strength_mpa", "official_tensile_strength_mpa", float),
    ("aluminium_yield_strength_mpa", "official_yield_strength_mpa", float),
    ("aluminium_elongation_pct", "official_elongation_pct", float),
)

_REFERENCE_METRIC_LOOKUPS: Dict[str, tuple[str, ...]] = {
    "pc_density": ("pc_density_sample_label",),
    "pc_mechanics": ("pc_mechanics_sample_label",),
    "pc_thermal": ("pc_thermal_sample_label",),
    "pc_ignition": ("pc_ignition_sample_label",),
    "aluminium": ("aluminium_processing_route", "aluminium_class_id"),
}


class OfficialFeaturesBundle(NamedTuple):
    value_columns: tuple[str, ...]
    composition_columns: tuple[str, ...]
    direct_map: Dict[str, Dict[str, float]]
    category_tokens: Dict[str, list[tuple[frozenset[str], Dict[str, float], str]]]
    table: pl.DataFrame
    mission_mass: Dict[str, Dict[str, float]]
    mission_totals: Dict[str, float]
    processing_metrics: Dict[str, Dict[str, float]]
    leo_mass_savings: Dict[str, Dict[str, float]]
    propellant_benefits: Dict[str, Dict[str, float]]
    reference_metrics: Dict[str, Dict[str, float]]
    l2l_constants: Dict[str, float]
    l2l_category_features: Dict[str, Dict[str, float]]
    l2l_item_features: Dict[str, Dict[str, float]]
    l2l_hints: Dict[str, str]


@dataclass(frozen=True)
class RegolithThermalBundle:
    """Container for MGS-1 thermogravimetric / EGA reference curves."""

    tg_curve: pd.DataFrame
    ega_curve: pd.DataFrame
    ega_long: pd.DataFrame
    gas_peaks: pd.DataFrame
    mass_events: pd.DataFrame


_L2L_PARAMETERS = load_l2l_parameters()
_OFFICIAL_FEATURES_PATH = DATASETS_ROOT / "rexai_nasa_waste_features.csv"


@lru_cache(maxsize=1)
def official_features_bundle() -> OfficialFeaturesBundle:
    l2l = _L2L_PARAMETERS
    default = OfficialFeaturesBundle(
        (),
        (),
        {},
        {},
        pl.DataFrame(),
        {},
        {},
        {},
        {},
        {},
        {},
        l2l.constants,
        l2l.category_features,
        l2l.item_features,
        l2l.hints,
    )

    if not _OFFICIAL_FEATURES_PATH.exists():
        return default

    table_lazy = pl.scan_csv(_OFFICIAL_FEATURES_PATH)
    initial_columns = list(table_lazy.collect_schema().names())
    duplicate_suffixes = [column for column in initial_columns if column.endswith(".1")]
    if duplicate_suffixes:
        table_lazy = table_lazy.drop(duplicate_suffixes)

    table_lazy = merge_reference_dataset(table_lazy, "nasa_waste_summary.csv", "summary")
    table_lazy = merge_reference_dataset(table_lazy, "nasa_waste_processing_products.csv", "processing")
    table_lazy = merge_reference_dataset(table_lazy, "nasa_leo_mass_savings.csv", "leo")
    table_lazy = merge_reference_dataset(table_lazy, "nasa_propellant_benefits.csv", "propellant")
    table_lazy = merge_reference_dataset(table_lazy, "polymer_composite_density.csv", "pc_density")
    table_lazy = merge_reference_dataset(table_lazy, "polymer_composite_mechanics.csv", "pc_mechanics")
    table_lazy = merge_reference_dataset(table_lazy, "polymer_composite_thermal.csv", "pc_thermal")
    table_lazy = merge_reference_dataset(table_lazy, "polymer_composite_ignition.csv", "pc_ignition")
    table_lazy = merge_reference_dataset(table_lazy, "aluminium_alloys.csv", "aluminium")

    table_lazy = table_lazy.with_columns(
        [
            pl.col("category")
            .map_elements(normalize_category)
            .alias("category_norm"),
            pl.col("subitem")
            .map_elements(normalize_item)
            .alias("subitem_norm"),
        ]
    )

    table_lazy = table_lazy.with_columns(
        pl.when(pl.col("subitem_norm").str.len_bytes() > 0)
        .then(pl.col("category_norm") + pl.lit("|") + pl.col("subitem_norm"))
        .otherwise(pl.col("category_norm"))
        .alias("key")
    )

    if isinstance(table_lazy, pd.DataFrame):  # pragma: no cover - defensive
        table_df = pl.from_pandas(table_lazy)
    elif isinstance(table_lazy, pl.DataFrame):
        table_df = table_lazy
    else:
        table_df = table_lazy.collect()

    if table_df.height == 0:
        return default

    columns = table_df.columns
    excluded = {"category", "subitem", "category_norm", "subitem_norm", "token_set", "key"}
    value_columns = tuple(col for col in columns if col not in excluded)
    composition_columns = tuple(
        col for col in value_columns if col.endswith("_pct") and not col.startswith("subitem_")
    )

    direct_map: Dict[str, Dict[str, float]] = {}
    category_tokens: Dict[str, list[tuple[frozenset[str], Dict[str, float], str]]] = {}

    for row in table_df.to_dicts():
        category_raw = row.get("category")
        subitem_raw = row.get("subitem")

        if category_raw is None:
            continue

        key = build_match_key(category_raw, subitem_raw)
        category_norm = normalize_category(category_raw)
        tokens = token_set(subitem_raw)

        payload: Dict[str, float] = {}
        for column in value_columns:
            value = row.get(column)
            if value is None:
                payload[column] = float("nan")
                continue
            if isinstance(value, (int, float)):
                payload[column] = float(value)
                continue
            try:
                payload[column] = float(value)
            except (TypeError, ValueError):
                payload[column] = float("nan")

        direct_map[key] = payload
        category_tokens.setdefault(category_norm, []).append((tokens, payload, key))

    waste_summary = _load_waste_summary_data()
    processing_metrics = extract_grouped_metrics("nasa_waste_processing_products.csv", "processing")
    leo_savings = extract_grouped_metrics("nasa_leo_mass_savings.csv", "leo")
    propellant_metrics = extract_grouped_metrics("nasa_propellant_benefits.csv", "propellant")

    reference_metrics: Dict[str, Dict[str, float]] = {}
    for spec in _REFERENCE_METRIC_SPECS:
        metrics = extract_reference_metrics(spec)
        if metrics:
            reference_metrics.update(metrics)

    table_join = table_df.select(
        ["category_norm", "subitem_norm", *value_columns]
    ).unique(subset=["category_norm", "subitem_norm"], maintain_order=True)

    return OfficialFeaturesBundle(
        value_columns,
        composition_columns,
        direct_map,
        category_tokens,
        table_join,
        waste_summary.mass_by_key,
        waste_summary.mission_totals,
        processing_metrics,
        leo_savings,
        propellant_metrics,
        reference_metrics,
        l2l.constants,
        l2l.category_features,
        l2l.item_features,
        l2l.hints,
    )


def build_match_key(category: Any, subitem: Any | None = None) -> str:
    """Return the canonical key used to match NASA reference tables."""

    if subitem:
        return f"{normalize_category(category)}|{normalize_item(subitem)}"
    return normalize_category(category)


def lookup_official_feature_values(row: pd.Series) -> tuple[Dict[str, float], str]:
    bundle = official_features_bundle()
    if not bundle.value_columns:
        return {}, ""

    category = normalize_category(row.get("category", ""))
    if not category:
        return {}, ""

    candidates = (
        row.get("material"),
        row.get("material_family"),
        row.get("key_materials"),
    )

    for candidate in candidates:
        normalized = normalize_item(candidate)
        if not normalized:
            continue
        key = f"{category}|{normalized}"
        payload = bundle.direct_map.get(key)
        if payload:
            metrics = _translate_reference_metrics(payload, bundle, key)
            if metrics:
                payload = dict(payload)
                payload.update(metrics)
            return payload, key

    token_candidates = [value for value in candidates if value]
    if not token_candidates:
        return {}, ""

    matches = bundle.category_tokens.get(category)
    if not matches:
        return {}, ""

    for candidate in token_candidates:
        tokens = token_set(candidate)
        if not tokens:
            continue
        for reference_tokens, payload, match_key in matches:
            if tokens.issubset(reference_tokens):
                metrics = _translate_reference_metrics(payload, bundle, str(match_key))
                if metrics:
                    payload = dict(payload)
                    payload.update(metrics)
                return payload, match_key

    return {}, ""


__all__.extend([
    "build_match_key",
])


def _empty_dataframe(columns: Iterable[str] | None = None) -> pd.DataFrame:
    if not columns:
        return pd.DataFrame()
    return pd.DataFrame({col: [] for col in columns})


@lru_cache(maxsize=1)
def load_regolith_granulometry() -> pd.DataFrame:
    """Return particle size distribution for the MGS-1 simulant."""

    path = resolve_dataset_path("fig3_psizeData.csv")
    if path is None:
        return _empty_dataframe(["diameter_microns", "pct_retained", "pct_channel", "cumulative_retained", "pct_passing"])

    data = pd.read_csv(path, encoding="latin-1")
    rename_map = {
        "Diameter (microns)": "diameter_microns",
        "% Retained": "pct_retained",
        "% Channel": "pct_channel",
    }
    data = data.rename(columns=rename_map)

    for column in ("diameter_microns", "pct_retained", "pct_channel"):
        data[column] = pd.to_numeric(data[column], errors="coerce")

    data = data.dropna(subset=["diameter_microns"]).sort_values("diameter_microns", ascending=False).reset_index(drop=True)
    data["pct_retained"] = data["pct_retained"].fillna(0.0)
    data["pct_channel"] = data["pct_channel"].fillna(0.0)
    data["cumulative_retained"] = data["pct_retained"].cumsum().clip(upper=100.0)
    data["pct_passing"] = (100.0 - data["cumulative_retained"]).clip(lower=0.0, upper=100.0)
    return data


@lru_cache(maxsize=1)
def load_regolith_spectral_curves() -> pd.DataFrame:
    """Return VNIR reflectance curves for Martian soil simulants."""

    path = resolve_dataset_path("fig4_spectralData.csv")
    if path is None:
        return _empty_dataframe(["wavelength_nm", "sample", "reflectance", "reflectance_pct", "sample_slug"])

    table = pd.read_csv(path, encoding="latin-1")
    table = table.rename(columns={"Wavelength (nm)": "wavelength_nm"})
    table["wavelength_nm"] = pd.to_numeric(table["wavelength_nm"], errors="coerce")
    table = table.dropna(subset=["wavelength_nm"])

    samples = [column for column in table.columns if column != "wavelength_nm"]
    for column in samples:
        table[column] = pd.to_numeric(table[column], errors="coerce")

    melted = table.melt(id_vars=["wavelength_nm"], var_name="sample", value_name="reflectance").dropna(subset=["reflectance"])
    melted["sample"] = melted["sample"].astype(str).str.strip()
    melted["sample_slug"] = melted["sample"].map(slugify)
    melted["reflectance_pct"] = melted["reflectance"] * 100.0
    melted = melted.sort_values(["sample", "wavelength_nm"]).reset_index(drop=True)
    return melted


@lru_cache(maxsize=1)
def load_regolith_thermal_profiles() -> RegolithThermalBundle:
    """Return thermogravimetric (TG) and EGA curves for MGS-1."""

    tg_path = resolve_dataset_path("fig5_tgData.csv")
    ega_path = resolve_dataset_path("fig5_egaData.csv")

    empty = RegolithThermalBundle(
        tg_curve=_empty_dataframe(["temperature_c", "mass_pct", "mass_loss_pct"]),
        ega_curve=_empty_dataframe(["temperature_c", "mz_18_h2o", "mz_32_o2", "mz_44_co2", "mz_64_so2"]),
        ega_long=_empty_dataframe(["temperature_c", "species", "signal", "signal_ppb", "species_label"]),
        gas_peaks=_empty_dataframe(["species", "species_label", "temperature_c", "signal", "signal_ppb"]),
        mass_events=_empty_dataframe(["event", "temperature_c", "mass_pct"]),
    )

    if tg_path is None or ega_path is None:
        return empty

    tg_raw = pd.read_csv(tg_path, encoding="latin-1")
    tg_raw = tg_raw.rename(columns={"Temperature (¡C)": "temperature_c", "Mass (%)": "mass_pct"})
    tg_raw["temperature_c"] = pd.to_numeric(tg_raw["temperature_c"], errors="coerce")
    tg_raw["mass_pct"] = pd.to_numeric(tg_raw["mass_pct"], errors="coerce")
    tg_raw = tg_raw.dropna(subset=["temperature_c", "mass_pct"]).sort_values("temperature_c").reset_index(drop=True)
    tg_raw["mass_loss_pct"] = (100.0 - tg_raw["mass_pct"]).clip(lower=0.0)

    if len(tg_raw) > 1200:
        step = max(1, len(tg_raw) // 1200)
        tg_curve = tg_raw.iloc[::step].reset_index(drop=True)
    else:
        tg_curve = tg_raw.copy()

    ega_raw = pd.read_csv(ega_path, encoding="latin-1")
    ega_raw = ega_raw.rename(
        columns={
            "Temperature (¡C)": "temperature_c",
            "m/z 18 (H2O)": "mz_18_h2o",
            "m/z 32 (O2)": "mz_32_o2",
            "m/z 44 (CO2)": "mz_44_co2",
            "m/z 64 (SO2)": "mz_64_so2",
        }
    )
    ega_raw["temperature_c"] = pd.to_numeric(ega_raw["temperature_c"], errors="coerce")
    gas_columns = [col for col in ega_raw.columns if col != "temperature_c"]
    for column in gas_columns:
        ega_raw[column] = pd.to_numeric(ega_raw[column], errors="coerce")
    ega_raw = ega_raw.dropna(subset=["temperature_c"]).sort_values("temperature_c").reset_index(drop=True)

    ega_long = ega_raw.melt(id_vars=["temperature_c"], var_name="species", value_name="signal").dropna(subset=["signal"])
    species_labels = {
        "mz_18_h2o": "H₂O (m/z 18)",
        "mz_32_o2": "O₂ (m/z 32)",
        "mz_44_co2": "CO₂ (m/z 44)",
        "mz_64_so2": "SO₂ (m/z 64)",
    }
    ega_long["species_label"] = ega_long["species"].map(species_labels).fillna(ega_long["species"])
    ega_long["signal_ppb"] = ega_long["signal"] * 1e9

    gas_peaks: list[dict[str, float | str]] = []
    for column in gas_columns:
        series = ega_raw[column]
        if series.isnull().all():
            continue
        idx = series.idxmax()
        temperature = float(ega_raw.loc[idx, "temperature_c"])
        signal = float(series.loc[idx])
        gas_peaks.append(
            {
                "species": column,
                "species_label": species_labels.get(column, column),
                "temperature_c": temperature,
                "signal": signal,
                "signal_ppb": signal * 1e9,
            }
        )

    peaks_df = pd.DataFrame(gas_peaks).sort_values("temperature_c").reset_index(drop=True)

    mass_events: list[dict[str, float | str]] = []
    thresholds = [99.5, 99.0, 98.0, 97.0]
    for threshold in thresholds:
        mask = tg_raw["mass_pct"] <= threshold
        if mask.any():
            temp = float(tg_raw.loc[mask, "temperature_c"].iloc[0])
            mass_events.append(
                {
                    "event": f"mass_{threshold}",
                    "temperature_c": temp,
                    "mass_pct": float(threshold),
                }
            )

    if "mass_pct" in tg_raw.columns and tg_raw.shape[0] > 2:
        diff = tg_raw[["temperature_c", "mass_pct"]].copy()
        diff["mass_pct_next"] = diff["mass_pct"].shift(-1)
        diff["temperature_next"] = diff["temperature_c"].shift(-1)
        diff["mass_loss_rate"] = (diff["mass_pct_next"] - diff["mass_pct"]) / (
            diff["temperature_next"] - diff["temperature_c"]
        )
        diff["mass_loss_rate"] = diff["mass_loss_rate"].abs()
        diff = diff.dropna(subset=["mass_loss_rate"])
        if not diff.empty:
            idx = diff["mass_loss_rate"].idxmax()
            mass_events.append(
                {
                    "event": "max_mass_loss_rate",
                    "temperature_c": float(diff.loc[idx, "temperature_c"]),
                    "mass_pct": float(diff.loc[idx, "mass_pct"]),
                }
            )

    events_df = pd.DataFrame(mass_events).sort_values("temperature_c").reset_index(drop=True)

    return RegolithThermalBundle(
        tg_curve=tg_curve,
        ega_curve=ega_raw,
        ega_long=ega_long,
        gas_peaks=peaks_df,
        mass_events=events_df,
    )


def regolith_observation_lines(
    regolith_pct: float, thermo: Mapping[str, Any] | RegolithThermalBundle | None
) -> list[str]:
    """Return human-readable TG/EGA notes for a regolito blend."""

    if regolith_pct <= 0 or not thermo:
        return []

    lines = [
        (
            f"{regolith_pct * 100:.0f}% de MGS-1: monitorear densificación, ventilación y "
            "sellos al liberar volátiles."
        )
    ]

    if isinstance(thermo, RegolithThermalBundle):
        peaks_data: Iterable[Any]
        events_data: Iterable[Any]
        if isinstance(thermo.gas_peaks, pd.DataFrame):
            peaks_data = thermo.gas_peaks.to_dict("records")
        else:
            peaks_data = thermo.gas_peaks or []
        if isinstance(thermo.mass_events, pd.DataFrame):
            events_data = thermo.mass_events.to_dict("records")
        else:
            events_data = thermo.mass_events or []
        peaks = list(peaks_data)
        events = list(events_data)
    else:
        peaks = list(thermo.get("peaks", [])) if isinstance(thermo, Mapping) else []
        events = list(thermo.get("events", [])) if isinstance(thermo, Mapping) else []

    for peak in peaks[:2]:
        temperature = peak.get("temperature_c") if isinstance(peak, Mapping) else None
        species = (
            peak.get("species_label")
            if isinstance(peak, Mapping)
            else None
        ) or (
            peak.get("species") if isinstance(peak, Mapping) else None
        ) or "Volátiles"
        signal = peak.get("signal_ppb") if isinstance(peak, Mapping) else None
        temp_txt = (
            f"{temperature:.0f} °C"
            if isinstance(temperature, (int, float))
            else "pico térmico"
        )
        signal_txt = (
            f" (~{signal:.2f} ppb eq.)"
            if isinstance(signal, (int, float))
            else ""
        )
        lines.append(f"TG/EGA: {species} con liberación cerca de {temp_txt}{signal_txt}.")

    for event in events[:2]:
        if isinstance(event, Mapping):
            label = (event.get("event") or "").replace("_", " ").strip().capitalize()
            mass_pct = event.get("mass_pct")
            temperature = event.get("temperature_c")
        else:
            label = "Evento"
            mass_pct = None
            temperature = None
        mass_txt = (
            f"{mass_pct:.1f}%" if isinstance(mass_pct, (int, float)) else "variación"
        )
        temp_txt = (
            f"{temperature:.0f} °C"
            if isinstance(temperature, (int, float))
            else "el perfil térmico"
        )
        lines.append(f"TG: {label or 'Evento'} → {mass_txt} alrededor de {temp_txt}.")

    return lines


REGOLITH_VECTOR = _load_regolith_vector()
GAS_MEAN_YIELD = _load_gas_mean_yield()
MEAN_REUSE = _load_mean_reuse()
REGOLITH_CHARACTERIZATION = load_regolith_characterization()
