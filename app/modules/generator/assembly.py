"""Domain helpers for assembling candidate feature rows."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Mapping

import numpy as np
import pandas as pd

from app.modules import data_sources as ds

from .normalization import normalize_category, normalize_item


_COMPOSITION_DENSITY_MAP: Mapping[str, float] = {
    "Aluminum_pct": 2700.0,
    "Carbon_Fiber_pct": 1700.0,
    "Polyethylene_pct": 950.0,
    "PVDF_pct": 1780.0,
    "Nomex_pct": 1350.0,
    "Nylon_pct": 1140.0,
    "Polyester_pct": 1380.0,
    "Cotton_Cellulose_pct": 1550.0,
    "EVOH_pct": 1250.0,
    "PET_pct": 1370.0,
    "Nitrile_pct": 1030.0,
    "approx_moisture_pct": 1000.0,
    "Other_pct": 500.0,
    "Plastic_Resin_pct": 950.0,
}

_CATEGORY_DENSITY_DEFAULTS: Mapping[str, float] = {
    "foam packaging": 100.0,
    "food packaging": 650.0,
    "structural elements": 1800.0,
    "structural element": 1800.0,
    "packaging": 420.0,
    "other packaging": 420.0,
    "gloves": 420.0,
    "eva waste": 240.0,
    "fabric": 350.0,
}


@dataclass(slots=True)
class CandidateAssembler:
    """Utility responsible for candidate level feature derivations."""

    material_reference: ds.MaterialReferenceBundle = field(
        default_factory=ds.load_material_reference_bundle
    )
    composition_density_map: Mapping[str, float] = field(
        default_factory=lambda: dict(_COMPOSITION_DENSITY_MAP)
    )
    category_density_defaults: Mapping[str, float] = field(
        default_factory=lambda: dict(_CATEGORY_DENSITY_DEFAULTS)
    )
    _alias_map: Dict[str, str] = field(init=False, repr=False, default_factory=dict)
    _properties: Dict[str, Dict[str, float]] = field(init=False, repr=False, default_factory=dict)
    _property_columns: tuple[str, ...] = field(init=False, repr=False, default_factory=tuple)
    _density_map: Dict[str, float] = field(init=False, repr=False, default_factory=dict)
    _mixing_rules: Dict[str, Dict[str, Any]] = field(init=False, repr=False, default_factory=dict)
    _compatibility: Dict[str, Dict[str, Any]] = field(
        init=False, repr=False, default_factory=dict
    )
    _metadata: Dict[str, Dict[str, Any]] = field(init=False, repr=False, default_factory=dict)

    def __post_init__(self) -> None:
        reference = self.material_reference
        self._alias_map = dict(getattr(reference, "alias_map", {}))
        self._properties = dict(getattr(reference, "properties", {}))
        self._property_columns = tuple(getattr(reference, "property_columns", ()))
        self._density_map = dict(getattr(reference, "density_map", {}))
        self._mixing_rules = dict(getattr(reference, "mixing_rules", {}))
        self._compatibility = {
            key: dict(value) for key, value in getattr(reference, "compatibility_matrix", {}).items()
        }
        self._metadata = dict(getattr(reference, "metadata", {}))

        overrides = {
            "Polyethylene_pct": "polyethylene",
            "PVDF_pct": "pvdf",
            "Nomex_pct": "nomex",
            "Nylon_pct": "nylon",
            "Polyester_pct": "pet_p_ertalyte_tx",
            "Cotton_Cellulose_pct": "cotton_gossypium_hirsutum_barbadense",
            "Other_pct": "polypropylene",
        }
        for column, alias in overrides.items():
            density = None
            alias_slug = ds.slugify(alias)
            if alias_slug:
                density = self._density_map.get(alias_slug)
            if density is not None:
                self.composition_density_map = dict(self.composition_density_map)
                self.composition_density_map[column] = float(density)

    def build_mixing_profile(
        self,
        picks: pd.DataFrame,
        regolith_pct: float | None = None,
    ) -> Dict[str, Any]:
        if picks.empty and (not regolith_pct or regolith_pct <= 0):
            return {}

        canonical_keys: list[str] = []
        row_indices: Dict[str, list[int]] = {}
        for idx, row in picks.iterrows():
            key = self._resolve_material_key(row)
            if not key:
                continue
            canonical_keys.append(key)
            row_indices.setdefault(key, []).append(int(idx))

        if regolith_pct and regolith_pct > 0:
            canonical_keys.append("mgs_1_regolith")

        if not canonical_keys:
            return {}

        composites: list[Dict[str, Any]] = []
        for key in canonical_keys:
            rule = self._mixing_rules.get(key)
            if not rule:
                continue
            entry = {
                "material_key": key,
                "rule": rule.get("rule"),
                "source": rule.get("source"),
                "components": rule.get("components", {}),
                "variants": rule.get("variants", []),
                "rows": row_indices.get(key, []),
            }
            meta = self._metadata.get(key)
            if meta:
                entry["metadata"] = meta
            composites.append(entry)

        compatibility_pairs: list[Dict[str, Any]] = []
        seen_pairs: set[tuple[str, str]] = set()
        for key_a in canonical_keys:
            partners = self._compatibility.get(key_a)
            if not partners:
                continue
            for key_b, payload in partners.items():
                if key_b not in canonical_keys:
                    continue
                pair = tuple(sorted((key_a, key_b)))
                if pair in seen_pairs:
                    continue
                detail = dict(payload)
                evidence = [dict(item) for item in detail.get("evidence", [])]
                detail["evidence"] = evidence
                detail["sources"] = list(detail.get("sources", []))
                meta_a = self._metadata.get(key_a)
                meta_b = self._metadata.get(key_b)
                if meta_a or meta_b:
                    detail = dict(detail)
                    detail["metadata"] = {}
                    if meta_a:
                        detail["metadata"]["material_a"] = meta_a
                    if meta_b:
                        detail["metadata"]["material_b"] = meta_b
                compatibility_pairs.append(
                    {
                        "materials": [key_a, key_b],
                        "details": detail,
                    }
                )
                seen_pairs.add(pair)

        profile: Dict[str, Any] = {}
        if composites:
            profile["composites"] = composites
        if compatibility_pairs:
            profile["compatibility_pairs"] = compatibility_pairs
        return profile

    def _resolve_material_key(self, row: Mapping[str, Any]) -> str | None:
        for candidate in (
            row.get("_material_reference_key"),
            row.get("material"),
            row.get("material_family"),
            row.get("key_materials"),
            row.get("flags"),
        ):
            if not candidate:
                continue
            normalized = normalize_item(candidate)
            slug = ds.slugify(normalized)
            if not slug:
                continue
            key = self._alias_map.get(slug)
            if key:
                return key
        return None

    def lookup_properties(self, row: Mapping[str, Any]) -> tuple[dict[str, float], str | None]:
        key = self._resolve_material_key(row)
        if key is None:
            return {}, None
        props = self._properties.get(key)
        if not props:
            return {}, key
        return dict(props), key

    def aggregate_material_properties(
        self, picks: pd.DataFrame, weights: Iterable[float]
    ) -> Dict[str, float]:
        if picks.empty or not self._property_columns:
            return {}

        weight_array = np.asarray(list(weights), dtype=float)
        if weight_array.size == 0:
            weight_array = picks["kg"].to_numpy(dtype=float)
        if weight_array.size != len(picks):
            weight_array = np.resize(weight_array, len(picks))
        mass = picks.get("kg")
        if isinstance(mass, pd.Series):
            mass_weights = mass.to_numpy(dtype=float)
        else:
            mass_weights = np.ones(len(picks), dtype=float)
        weights_combined = weight_array
        if np.all(weights_combined == 0):
            weights_combined = mass_weights
        total = np.sum(weights_combined)
        if not np.isfinite(total) or total <= 0:
            total = float(len(picks))
        normalized = weights_combined / total

        property_arrays: Dict[str, np.ndarray] = {}
        for column in self._property_columns:
            if column in picks.columns:
                values = pd.to_numeric(picks[column], errors="coerce").to_numpy(dtype=float)
            else:
                values = np.full(len(picks), float("nan"), dtype=float)
            property_arrays[column] = values

        mixing_cache: Dict[str, Dict[str, float]] = {}
        for position, (_, row) in enumerate(picks.iterrows()):
            key = self._resolve_material_key(row)
            if not key:
                continue

            reference = self._properties.get(key)
            if reference:
                for column, values in property_arrays.items():
                    value = reference.get(column)
                    if value is None:
                        continue
                    numeric = self._to_float(value)
                    if numeric is None:
                        continue
                    values[position] = numeric

            rule = self._mixing_rules.get(key)
            if not rule:
                continue
            composite = mixing_cache.get(key)
            if composite is None:
                composite = self._compute_mixed_property_values(rule)
                mixing_cache[key] = composite
            if not composite:
                continue
            for column, values in property_arrays.items():
                if column not in composite:
                    continue
                values[position] = composite[column]

        aggregates: Dict[str, float] = {}
        for column, values in property_arrays.items():
            mask = np.isfinite(values)
            if not mask.any():
                continue
            denom = normalized[mask].sum()
            if denom <= 0:
                continue
            aggregates[column] = float(np.dot(values[mask], normalized[mask]) / denom)
        return aggregates

    def _compute_mixed_property_values(self, rule: Mapping[str, Any]) -> Dict[str, float]:
        fractions = self._derive_component_fractions(rule)
        if not fractions:
            return {}

        components_meta = rule.get("components") or {}
        rule_type = str(rule.get("rule") or "parallel").lower()
        mixed: Dict[str, float] = {}
        for column in self._property_columns:
            weighted: list[tuple[float, float]] = []
            total_fraction = 0.0
            for component_key, fraction in fractions.items():
                if not fraction or fraction <= 0:
                    continue
                value = self._lookup_component_property(
                    component_key,
                    column,
                    components_meta,
                )
                if value is None:
                    continue
                weighted.append((float(fraction), value))
                total_fraction += float(fraction)

            if not weighted or total_fraction <= 0:
                continue

            normalized = [(frac / total_fraction, val) for frac, val in weighted]
            if rule_type == "series":
                denom = 0.0
                for frac, val in normalized:
                    if val == 0:
                        continue
                    denom += frac / val
                if denom > 0:
                    mixed[column] = float(1.0 / denom)
            else:
                mixed[column] = float(sum(frac * val for frac, val in normalized))

        return mixed

    def _derive_component_fractions(self, rule: Mapping[str, Any]) -> Dict[str, float]:
        variants = rule.get("variants") or []
        accum: Dict[str, float] = {}
        variant_count = 0
        for variant in variants:
            composition = variant.get("composition") or {}
            normalized: Dict[str, float] = {}
            total = 0.0
            for component_key, raw_fraction in composition.items():
                numeric = self._to_float(raw_fraction)
                if numeric is None or numeric <= 0:
                    continue
                normalized[str(component_key)] = numeric
                total += numeric
            if total <= 0 or not normalized:
                continue
            variant_count += 1
            for component_key, numeric in normalized.items():
                accum[component_key] = accum.get(component_key, 0.0) + numeric / total

        if variant_count == 0:
            components = list((rule.get("components") or {}).keys())
            if not components:
                return {}
            equal = 1.0 / float(len(components))
            return {str(key): equal for key in components}

        scale = 1.0 / float(variant_count)
        for component_key in list(accum):
            accum[component_key] *= scale

        total_fraction = sum(accum.values())
        if total_fraction <= 0:
            return {}
        for component_key in list(accum):
            accum[component_key] /= total_fraction
        return accum

    def _lookup_component_property(
        self,
        component_key: str,
        column: str,
        components_meta: Mapping[str, Any],
    ) -> float | None:
        meta = components_meta.get(component_key) if isinstance(components_meta, Mapping) else {}

        candidates: list[str] = []
        seen: set[str] = set()

        def _add_candidate(value: Any) -> None:
            if not value:
                return
            if not isinstance(value, str):
                value = str(value)
            if not value or value in seen:
                return
            seen.add(value)
            candidates.append(value)

        _add_candidate(component_key)
        _add_candidate(ds.slugify(component_key))
        if isinstance(meta, Mapping):
            _add_candidate(meta.get("canonical_key"))
            canonical_slug = meta.get("canonical_slug")
            _add_candidate(canonical_slug)
            if canonical_slug:
                _add_candidate(self._alias_map.get(canonical_slug))
        slug = ds.slugify(component_key)
        if slug:
            _add_candidate(self._alias_map.get(slug))
        if isinstance(meta, Mapping):
            canonical_key = meta.get("canonical_key")
            if canonical_key:
                _add_candidate(self._alias_map.get(ds.slugify(canonical_key)))
        for candidate in list(candidates):
            slug_candidate = ds.slugify(candidate)
            if slug_candidate:
                _add_candidate(self._alias_map.get(slug_candidate))

        for candidate in candidates:
            props = self._properties.get(candidate)
            if not props:
                continue
            value = props.get(column)
            numeric = self._to_float(value)
            if numeric is None:
                continue
            return numeric
        return None

    @staticmethod
    def _to_float(value: Any) -> float | None:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return None
        if not np.isfinite(numeric):
            return None
        return float(numeric)

    def estimate_density_from_row(self, row: pd.Series) -> float | None:
        """Estimate a material density with packaging-aware fallbacks."""

        category = normalize_category(row.get("category", ""))

        properties, _ = self.lookup_properties(row)
        density_from_reference = properties.get("material_density_kg_m3")
        if density_from_reference and np.isfinite(density_from_reference):
            return float(np.clip(density_from_reference, 20.0, 4000.0))

        try:
            cat_mass = float(row.get("category_total_mass_kg"))
            cat_volume = float(row.get("category_total_volume_m3"))
        except (TypeError, ValueError):
            cat_mass = cat_volume = float("nan")

        if pd.notna(cat_mass) and pd.notna(cat_volume) and cat_volume > 0:
            return float(np.clip(cat_mass / cat_volume, 20.0, 4000.0))

        composition_weights: list[tuple[float, float]] = []
        total = 0.0
        for column, density in self.composition_density_map.items():
            try:
                pct = float(row.get(column, 0.0))
            except (TypeError, ValueError):
                pct = 0.0
            if pct and not np.isnan(pct):
                frac = pct / 100.0
                if frac > 0:
                    composition_weights.append((frac, density))
                    total += frac

        if total > 0 and composition_weights:
            weighted = sum(frac * density for frac, density in composition_weights) / total
            if category == "foam packaging":
                return float(min(weighted, self.category_density_defaults.get(category, weighted)))
            return float(np.clip(weighted, 20.0, 4000.0))

        if category in self.category_density_defaults:
            return float(self.category_density_defaults[category])

        return None


COMPOSITION_DENSITY_MAP: Mapping[str, float] = dict(_COMPOSITION_DENSITY_MAP)
CATEGORY_DENSITY_DEFAULTS: Mapping[str, float] = dict(_CATEGORY_DENSITY_DEFAULTS)


__all__ = [
    "CandidateAssembler",
    "CATEGORY_DENSITY_DEFAULTS",
    "COMPOSITION_DENSITY_MAP",
]
