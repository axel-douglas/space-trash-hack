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

    def __post_init__(self) -> None:
        reference = self.material_reference
        self._alias_map = dict(getattr(reference, "alias_map", {}))
        self._properties = dict(getattr(reference, "properties", {}))
        self._property_columns = tuple(getattr(reference, "property_columns", ()))
        self._density_map = dict(getattr(reference, "density_map", {}))

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

        aggregates: Dict[str, float] = {}
        for column in self._property_columns:
            if column not in picks.columns:
                continue
            values = pd.to_numeric(picks[column], errors="coerce").to_numpy(dtype=float)
            mask = np.isfinite(values)
            if not mask.any():
                continue
            denom = normalized[mask].sum()
            if denom <= 0:
                continue
            aggregates[column] = float(np.dot(values[mask], normalized[mask]) / denom)
        return aggregates

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
