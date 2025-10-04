"""Mission planning helpers combining material, process and logistics data."""

from __future__ import annotations

import random
from dataclasses import dataclass
from functools import lru_cache
from types import SimpleNamespace
from typing import Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from app.modules import mars_control, policy_engine, process_planner

PROCESS_PRODUCT_MAP: dict[str, str] = {
    "P01": "Gránulos reforzados",
    "P02": "Paneles laminados",
    "P03": "Composites sinterizados",
    "P04": "Estructuras modulares",
}


@lru_cache(maxsize=1)
def load_process_catalog(path: str = "data/process_catalog.csv") -> pd.DataFrame:
    """Return the curated process catalog as a DataFrame."""

    catalog = pd.read_csv(path)
    catalog["energy_kwh_per_kg"] = pd.to_numeric(
        catalog.get("energy_kwh_per_kg"), errors="coerce"
    )
    catalog["water_l_per_kg"] = pd.to_numeric(
        catalog.get("water_l_per_kg"), errors="coerce"
    )
    catalog["crew_min_per_batch"] = pd.to_numeric(
        catalog.get("crew_min_per_batch"), errors="coerce"
    )
    return catalog


@lru_cache(maxsize=1)
def load_inventory(path: str = "data/waste_inventory_sample.csv") -> pd.DataFrame:
    """Return the reference waste inventory used for mission planning."""

    inventory = pd.read_csv(path)
    numeric_cols = ("mass_kg", "volume_l", "moisture_pct", "pct_mass", "pct_volume")
    for column in numeric_cols:
        if column in inventory.columns:
            inventory[column] = pd.to_numeric(inventory[column], errors="coerce")
    inventory["flags"] = inventory.get("flags", "").fillna("")
    return inventory


@lru_cache(maxsize=1)
def load_logistics() -> mars_control.MarsLogisticsData:
    """Return the Mars logistics baseline dataset."""

    return mars_control.load_logistics_baseline()


@dataclass(slots=True)
class Assignment:
    """Recommended process assignment for a material."""

    material_id: str
    material: str
    category: str
    process_id: str
    process_name: str
    match_score: float
    match_reason: str
    energy_kwh_per_kg: float
    water_l_per_kg: float
    crew_min_per_batch: float
    mass_kg: float

    @property
    def total_energy(self) -> float:
        return float(self.mass_kg * self.energy_kwh_per_kg)

    @property
    def total_water(self) -> float:
        return float(self.mass_kg * self.water_l_per_kg)

    @property
    def product_label(self) -> str:
        return PROCESS_PRODUCT_MAP.get(self.process_id, "Producto avanzado")


def recommend_processes(
    materials: pd.DataFrame,
    *,
    scenario: str | None = None,
    crew_time_low: bool = False,
    max_energy_kwh_per_kg: float | None = None,
    max_crew_min_per_batch: float | None = None,
    top_n: int = 3,
) -> list[Assignment]:
    """Compute process recommendations for the provided *materials*."""

    if materials.empty:
        return []

    catalog = load_process_catalog()
    energy_limit = float(max_energy_kwh_per_kg) if max_energy_kwh_per_kg else None
    crew_limit = float(max_crew_min_per_batch) if max_crew_min_per_batch else None

    assignments: list[Assignment] = []
    for _, row in materials.iterrows():
        descriptor = " ".join(
            part
            for part in (
                str(row.get("material", "")),
                str(row.get("flags", "")),
                str(row.get("notes", "")),
            )
            if part and part != "nan"
        )
        ranked = process_planner.choose_process(
            descriptor,
            catalog,
            scenario=scenario,
            crew_time_low=crew_time_low,
        )
        if energy_limit is not None:
            ranked = ranked[
                pd.to_numeric(ranked.get("energy_kwh_per_kg"), errors="coerce").fillna(np.inf)
                <= energy_limit
            ]
        if crew_limit is not None:
            ranked = ranked[
                pd.to_numeric(ranked.get("crew_min_per_batch"), errors="coerce").fillna(np.inf)
                <= crew_limit
            ]
        if ranked.empty:
            continue
        for _, proc in ranked.head(top_n).iterrows():
            assignments.append(
                Assignment(
                    material_id=str(row.get("id", row.get("material", "unknown"))),
                    material=str(row.get("material", "Material")),
                    category=str(row.get("category", "")),
                    process_id=str(proc.get("process_id")),
                    process_name=str(proc.get("name", proc.get("process_id"))),
                    match_score=float(proc.get("match_score", 0.0) or 0.0),
                    match_reason=str(proc.get("match_reason", "")),
                    energy_kwh_per_kg=float(proc.get("energy_kwh_per_kg", 0.0) or 0.0),
                    water_l_per_kg=float(proc.get("water_l_per_kg", 0.0) or 0.0),
                    crew_min_per_batch=float(proc.get("crew_min_per_batch", 0.0) or 0.0),
                    mass_kg=float(row.get("mass_kg", 0.0) or 0.0),
                )
            )
    return assignments


def assignments_to_dataframe(assignments: Sequence[Assignment]) -> pd.DataFrame:
    """Serialise *assignments* into a tabular structure."""

    if not assignments:
        return pd.DataFrame(
            columns=
            [
                "material_id",
                "material",
                "category",
                "process_id",
                "process_name",
                "match_score",
                "match_reason",
                "energy_kwh_per_kg",
                "total_energy_kwh",
                "water_l_per_kg",
                "total_water_l",
                "crew_min_per_batch",
                "product",
            ]
        )

    rows = []
    for entry in assignments:
        rows.append(
            {
                "material_id": entry.material_id,
                "material": entry.material,
                "category": entry.category,
                "process_id": entry.process_id,
                "process_name": entry.process_name,
                "match_score": entry.match_score,
                "match_reason": entry.match_reason,
                "energy_kwh_per_kg": entry.energy_kwh_per_kg,
                "total_energy_kwh": entry.total_energy,
                "water_l_per_kg": entry.water_l_per_kg,
                "total_water_l": entry.total_water,
                "crew_min_per_batch": entry.crew_min_per_batch,
                "product": entry.product_label,
            }
        )
    return pd.DataFrame(rows)


def build_manifest(materials: pd.DataFrame) -> pd.DataFrame:
    """Prepare a manifest-like table for policy evaluation."""

    if materials.empty:
        return pd.DataFrame(
            columns=["inventory_id", "item", "description", "mass_kg", "flags", "category"]
        )

    manifest = materials.copy()
    manifest["inventory_id"] = manifest.get("id").fillna(manifest.get("material"))
    manifest["item"] = manifest.get("material")
    manifest["description"] = manifest.get("notes", "")
    manifest["mass_kg"] = pd.to_numeric(manifest.get("mass_kg"), errors="coerce").fillna(0.0)
    manifest["flags"] = manifest.get("flags", "").fillna("")
    manifest["category"] = manifest.get("category", "").fillna("")
    return manifest[
        ["inventory_id", "item", "description", "mass_kg", "flags", "category"]
    ]


def evaluate_policy_signals(manifest: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Return manifest scores and textual policy alerts."""

    if manifest.empty:
        return manifest, []

    manifest_payload = manifest.copy()
    manifest_payload["material"] = manifest_payload.get("item")
    manifest_payload["item_name"] = manifest_payload.get("item")
    manifest_payload["notes"] = manifest_payload.get("description")
    mapped = policy_engine.map_manifest_to_bundle(manifest_payload)
    scored = policy_engine.compute_material_utility_scores(mapped)
    recommendations = policy_engine.propose_policy_actions(scored)

    alerts: list[str] = []

    low_confidence = scored[pd.to_numeric(scored.get("match_confidence"), errors="coerce") < 0.5]
    for _, row in low_confidence.iterrows():
        alerts.append(
            f"⚠️ Coincidencia débil para {row.get('item', row.get('material', 'ítem'))}: revisar equivalentes."
        )

    low_score = scored[pd.to_numeric(scored.get("material_utility_score"), errors="coerce") < 0.45]
    for _, row in low_score.iterrows():
        alerts.append(
            f"🔍 Puntaje bajo ({row.get('material_utility_score', 0):.2f}) en {row.get('item', 'ítem')}: considerar sustitución."
        )

    if not recommendations.empty:
        for _, row in recommendations.iterrows():
            alerts.append(
                "🔄 Sugerencia política: reemplazar "
                f"{row.get('item_name', 'ítem')} por {row.get('recommended_material_key', 'alternativa')}"
                f" (ganancia estimada {row.get('recommended_score', 0):.2f})."
            )

    mass_by_category = scored.groupby("category")["mass_kg"].sum().sort_values(ascending=False)
    overstock = mass_by_category[mass_by_category > mass_by_category.mean() * 1.8]
    for category, mass in overstock.items():
        alerts.append(
            f"📦 Overstock detectado en {category or 'categoría general'}: {mass:.1f} kg disponibles, priorizar su uso."
        )

    return scored, alerts


def build_sankey(assignments: Sequence[Assignment]) -> go.Figure | None:
    """Render a Sankey diagram describing waste → process → product → ruta."""

    if not assignments:
        return None

    logistics = load_logistics()
    primary_route = None
    if logistics.flights:
        flight = logistics.flights[0]
        primary_route = f"Ruta {flight.origin} → {flight.destination}"
    else:
        primary_route = "Ruta logística"

    nodes: list[str] = []
    links_source: list[int] = []
    links_target: list[int] = []
    links_value: list[float] = []
    links_label: list[str] = []

    def _idx(label: str) -> int:
        if label not in nodes:
            nodes.append(label)
        return nodes.index(label)

    grouped: dict[str, Assignment] = {}
    for assignment in assignments:
        grouped.setdefault(assignment.material_id, assignment)

    for assignment in grouped.values():
        waste_node = f"Residuo · {assignment.material}"
        process_node = f"Proceso {assignment.process_id}"
        product_node = assignment.product_label

        waste_idx = _idx(waste_node)
        process_idx = _idx(process_node)
        product_idx = _idx(product_node)
        route_idx = _idx(primary_route)

        links_source.extend([waste_idx, process_idx, product_idx])
        links_target.extend([process_idx, product_idx, route_idx])
        links_value.extend([assignment.mass_kg, assignment.mass_kg, assignment.mass_kg])
        links_label.extend(
            [
                f"{assignment.material} → {assignment.process_name}",
                f"{assignment.process_name} produce {assignment.product_label}",
                f"{assignment.product_label} listo para {primary_route}",
            ]
        )

    figure = go.Figure(
        data=[
            go.Sankey(
                node=dict(label=nodes, pad=18, thickness=18),
                link=dict(
                    source=links_source,
                    target=links_target,
                    value=links_value,
                    label=links_label,
                ),
            )
        ]
    )
    figure.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10))
    return figure


def _build_candidate(
    subset: Sequence[Assignment],
    scores: pd.DataFrame,
    objective: str,
) -> dict:
    if not subset:
        return {}

    total_energy = sum(assignment.total_energy for assignment in subset)
    total_water = sum(assignment.total_water for assignment in subset)
    total_crew = sum(assignment.crew_min_per_batch for assignment in subset)
    ids = [assignment.material_id for assignment in subset]
    score_rows = scores[scores["inventory_id"].isin(ids)]
    mech_mean = float(score_rows.get("mechanical_score", pd.Series([0.0])).mean()) if not score_rows.empty else 0.0
    util_mean = float(score_rows.get("material_utility_score", pd.Series([0.0])).mean()) if not score_rows.empty else 0.0

    if objective == "max_rigidity":
        base = mech_mean
        bonus = 0.2 * util_mean
        score = float(np.clip(base + bonus, 0.0, 1.5))
    else:
        denom = max(total_energy, 1e-3)
        base = 1.0 / (1.0 + denom / 50.0)
        bonus = 0.15 * util_mean
        score = float(np.clip(base + bonus, 0.0, 1.5))

    processes = sorted({assignment.process_id for assignment in subset})
    products = sorted({assignment.product_label for assignment in subset})

    candidate = {
        "score": score,
        "materials": [assignment.material for assignment in subset],
        "processes": processes,
        "products": products,
        "props": SimpleNamespace(
            energy_kwh=float(total_energy),
            water_l=float(total_water),
            crew_min=float(total_crew),
        ),
        "features": {
            "mechanical_score": mech_mean,
            "material_utility_score": util_mean,
        },
    }
    return candidate


def optimize_assignments(
    assignments: Sequence[Assignment],
    scores: pd.DataFrame,
    *,
    lot_size: int,
    objective: str,
    target_limits: dict[str, float],
    n_evals: int = 24,
) -> tuple[pd.DataFrame, list[dict]]:
    """Run the optimisation loop over *assignments* and return Pareto candidates."""

    from app.modules import optimizer

    unique_map: dict[str, Assignment] = {}
    for assignment in assignments:
        unique_map.setdefault(assignment.material_id, assignment)

    unique_assignments = list(unique_map.values())
    if not unique_assignments:
        return pd.DataFrame(), []

    lot_size = int(max(1, min(lot_size, len(unique_assignments))))

    def _sample(_override: dict | None) -> dict:
        subset = random.sample(unique_assignments, lot_size)
        return _build_candidate(subset, scores, objective)

    deterministic: list[Assignment] = sorted(
        unique_assignments,
        key=lambda a: (a.energy_kwh_per_kg, -a.match_score),
    )
    seed_low_energy = _build_candidate(deterministic[:lot_size], scores, objective)

    mechanical_lookup: dict[str, float] = {}
    if (
        not scores.empty
        and "inventory_id" in scores.columns
        and "mechanical_score" in scores.columns
    ):
        mechanical_series = scores.set_index("inventory_id")["mechanical_score"].dropna()
        mechanical_lookup = mechanical_series.to_dict()
    deterministic_mech = sorted(
        unique_assignments,
        key=lambda a: mechanical_lookup.get(a.material_id, 0.0),
        reverse=True,
    )
    seed_high_mech = _build_candidate(deterministic_mech[:lot_size], scores, objective)

    seeds = [cand for cand in (seed_low_energy, seed_high_mech) if cand]
    if not seeds:
        seeds.append(_sample(None))

    pareto, _history = optimizer.optimize_candidates(
        seeds,
        _sample,
        target=target_limits,
        n_evals=n_evals,
    )

    if not pareto:
        return pd.DataFrame(), []

    rows = []
    for candidate in pareto:
        props = candidate.get("props", SimpleNamespace(energy_kwh=0.0, water_l=0.0, crew_min=0.0))
        rows.append(
            {
                "score": float(candidate.get("score", 0.0)),
                "energy_kwh": float(getattr(props, "energy_kwh", 0.0)),
                "water_l": float(getattr(props, "water_l", 0.0)),
                "crew_min": float(getattr(props, "crew_min", 0.0)),
                "materials": ", ".join(candidate.get("materials", [])),
                "products": ", ".join(candidate.get("products", [])),
            }
        )

    return pd.DataFrame(rows).sort_values("score", ascending=False), pareto


__all__ = [
    "Assignment",
    "PROCESS_PRODUCT_MAP",
    "assignments_to_dataframe",
    "build_manifest",
    "build_sankey",
    "evaluate_policy_signals",
    "load_inventory",
    "load_logistics",
    "load_process_catalog",
    "optimize_assignments",
    "recommend_processes",
]

