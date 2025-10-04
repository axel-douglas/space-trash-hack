"""Domain helpers for assembling candidate feature rows."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import numpy as np
import pandas as pd

from .normalization import normalize_category


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

    composition_density_map: Mapping[str, float] = field(
        default_factory=lambda: dict(_COMPOSITION_DENSITY_MAP)
    )
    category_density_defaults: Mapping[str, float] = field(
        default_factory=lambda: dict(_CATEGORY_DENSITY_DEFAULTS)
    )

    def estimate_density_from_row(self, row: pd.Series) -> float | None:
        """Estimate a material density with packaging-aware fallbacks."""

        category = normalize_category(row.get("category", ""))

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
