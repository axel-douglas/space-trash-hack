"""Policy evaluation helpers for manifest compliance workflows.

This module provides lightweight heuristics to translate customs manifests
into the canonical material bundle used throughout the Rex-AI stack.  The
core responsibilities are:

* map manifest rows to the Zenodo material reference bundle
* derive a Material Utility Score that balances mechanical performance,
  spectral compatibility and operational penalties
* propose substitutions or quota adjustments along with traceable evidence
  extracted from the material bundle

The implementation intentionally favours deterministic, explainable rules so
that Streamlit views can surface the reasoning to domain experts without
requiring ML inference.
"""
from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from app.modules import data_sources as ds

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class PolicyArtifacts:
    """Container grouping the artefacts generated during analysis."""

    mapped_manifest: pd.DataFrame
    scored_manifest: pd.DataFrame
    policy_recommendations: pd.DataFrame
    compatibility_records: pd.DataFrame
    material_passport: dict[str, object]


_MANIFEST_TOKEN_COLUMNS: tuple[str, ...] = (
    "material_key",
    "material",
    "item",
    "item_name",
    "description",
    "category",
    "subcategory",
)

_MECHANICAL_COLUMNS: tuple[str, ...] = (
    "material_density_kg_m3",
    "material_modulus_gpa",
    "material_tensile_strength_mpa",
    "material_elongation_pct",
)

_COMPATIBILITY_PRIORITIES: tuple[str, ...] = (
    "mgs_1_regolith",
    "regolith",
    "simulant",
)

_PENALTY_SPECS: tuple[tuple[str, float, float], ...] = (
    ("thermogravimetric", 0.25, 30.0),
    ("ega", 0.2, 15.0),
    ("water", 0.2, 10.0),
    ("energy", 0.2, 6.0),
)

_PENALTY_COLUMNS: dict[str, tuple[str, ...]] = {
    "thermogravimetric": (
        "tg_loss_pct",
        "tga_loss_pct",
        "thermograv_loss_pct",
        "thermogravimetric_loss_pct",
        "mass_loss_pct",
    ),
    "ega": (
        "ega_loss_pct",
        "gas_analysis_loss_pct",
    ),
    "water": (
        "water_l_per_kg",
        "water_consumption_l_per_kg",
        "water_l",
    ),
    "energy": (
        "energy_kwh_per_kg",
        "energy_kwh",
        "energy_consumption_kwh",
    ),
}


def _coerce_manifest_df(manifest: pd.DataFrame | Mapping[str, Sequence[object]] | Sequence[Mapping[str, object]]) -> pd.DataFrame:
    if isinstance(manifest, pd.DataFrame):
        return manifest.copy()
    if isinstance(manifest, Mapping):
        return pd.DataFrame(manifest)
    return pd.DataFrame(list(manifest))


def _build_alias_index(bundle: ds.MaterialReferenceBundle) -> tuple[dict[str, str], dict[str, frozenset[str]], dict[str, set[str]]]:
    alias_map = dict(bundle.alias_map)
    for key in bundle.properties:
        alias_map.setdefault(key, key)
        slug = ds.slugify(ds.normalize_item(key))
        if slug:
            alias_map.setdefault(slug, key)
    token_index: dict[str, frozenset[str]] = {}
    inverted: dict[str, set[str]] = {}
    for alias, target in alias_map.items():
        tokens = ds.token_set(alias)
        if not tokens:
            continue
        token_index[alias] = tokens
        for token in tokens:
            inverted.setdefault(token, set()).add(alias)
    return alias_map, token_index, inverted


def _resolve_numeric(row: Mapping[str, object], columns: Sequence[str]) -> float | None:
    for column in columns:
        value = row.get(column)
        if value is None:
            continue
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if math.isnan(numeric):
            continue
        return numeric
    return None


def map_manifest_to_bundle(
    manifest: pd.DataFrame | Mapping[str, Sequence[object]] | Sequence[Mapping[str, object]],
    *,
    bundle: ds.MaterialReferenceBundle | None = None,
) -> pd.DataFrame:
    """Return manifest rows annotated with ``material_key`` matches."""

    manifest_df = _coerce_manifest_df(manifest)
    if manifest_df.empty:
        manifest_df["material_key"] = None
        manifest_df["match_confidence"] = 0.0
        manifest_df["match_method"] = "unmatched"
        return manifest_df

    if bundle is None:
        bundle = ds.load_material_reference_bundle()

    alias_map, token_index, inverted = _build_alias_index(bundle)

    results: list[str | None] = []
    confidences: list[float] = []
    methods: list[str] = []

    for _, row in manifest_df.iterrows():
        resolved_key: str | None = None
        confidence = 0.0
        method = "unmatched"

        # First try direct slug matches
        for column in _MANIFEST_TOKEN_COLUMNS:
            value = row.get(column)
            if not isinstance(value, str):
                continue
            slug = ds.slugify(ds.normalize_item(value))
            if not slug:
                continue
            target = alias_map.get(slug)
            if target:
                resolved_key = target
                confidence = 1.0 if slug in bundle.alias_map else 0.92
                method = "alias" if slug in bundle.alias_map else "canonical"
                break
        if resolved_key is not None:
            results.append(resolved_key)
            confidences.append(confidence)
            methods.append(method)
            continue

        tokens: set[str] = set()
        for column in _MANIFEST_TOKEN_COLUMNS:
            value = row.get(column)
            if isinstance(value, str):
                tokens.update(ds.token_set(value))
        if not tokens:
            results.append(None)
            confidences.append(0.0)
            methods.append(method)
            continue

        candidate_aliases: set[str] = set()
        for token in tokens:
            candidate_aliases.update(inverted.get(token, set()))

        best_score = 0.0
        best_key: str | None = None
        for alias in candidate_aliases:
            alias_tokens = token_index.get(alias)
            if not alias_tokens:
                continue
            intersection = len(tokens & alias_tokens)
            if intersection == 0:
                continue
            union = len(tokens | alias_tokens)
            score = intersection / union
            if score > best_score:
                best_score = score
                best_key = alias_map.get(alias)
        if best_key:
            results.append(best_key)
            confidences.append(round(best_score * 0.8 + 0.1, 4))
            methods.append("jaccard")
        else:
            results.append(None)
            confidences.append(0.0)
            methods.append(method)

    manifest_df = manifest_df.copy()
    manifest_df["material_key"] = results
    manifest_df["match_confidence"] = confidences
    manifest_df["match_method"] = methods
    return manifest_df


def _bundle_property_stats(bundle: ds.MaterialReferenceBundle) -> dict[str, tuple[float, float, float]]:
    stats: dict[str, tuple[float, float, float]] = {}
    if bundle.table.is_empty():
        return stats
    table = bundle.table.to_pandas()
    for column in bundle.property_columns:
        if column not in table.columns:
            continue
        series = pd.to_numeric(table[column], errors="coerce").dropna()
        if series.empty:
            continue
        stats[column] = (float(series.min()), float(series.max()), float(series.mean()))
    return stats


def _mechanical_score(material_key: str, stats: Mapping[str, tuple[float, float, float]], bundle: ds.MaterialReferenceBundle) -> float:
    properties = bundle.properties.get(material_key)
    if not properties:
        return 0.0
    values: list[float] = []
    for column in _MECHANICAL_COLUMNS:
        if column not in stats:
            continue
        bounds = stats[column]
        val = properties.get(column)
        if val is None:
            continue
        try:
            numeric = float(val)
        except (TypeError, ValueError):
            continue
        low, high, _ = bounds
        if not math.isfinite(numeric) or not math.isfinite(low) or not math.isfinite(high):
            continue
        if high <= low:
            normalized = 0.5
        else:
            normalized = (numeric - low) / (high - low)
        values.append(float(np.clip(normalized, 0.0, 1.0)))
    if not values:
        return 0.0
    return float(np.mean(values))


def _spectral_score(material_key: str, bundle: ds.MaterialReferenceBundle) -> float:
    compat = bundle.compatibility_matrix.get(material_key)
    if not compat:
        return 0.35
    score = 0.4 + 0.1 * min(len(compat), 4)
    has_regolith = False
    has_parallel = False
    for partner, meta in compat.items():
        lower_partner = partner.lower()
        if any(priority in lower_partner for priority in _COMPATIBILITY_PRIORITIES):
            has_regolith = True
        if str(meta.get("rule", "")).lower().strip() == "parallel":
            has_parallel = True
    if has_regolith:
        score += 0.3
    if has_parallel:
        score += 0.1
    return float(np.clip(score, 0.0, 1.0))


def _mass_factor(row: Mapping[str, object]) -> float:
    mass = _resolve_numeric(row, ("mass_kg", "mass", "kg", "available_mass_kg"))
    if mass is None or mass <= 0:
        return 0.45
    return float(1.0 - math.exp(-float(mass) / 25.0))


def _penalty_factor(row: Mapping[str, object]) -> tuple[float, dict[str, dict[str, float]]]:
    base = 1.0
    breakdown: dict[str, dict[str, float]] = {}
    for label, weight, scale in _PENALTY_SPECS:
        columns = _PENALTY_COLUMNS[label]
        value = _resolve_numeric(row, columns)
        if value is None:
            continue
        normalized = float(np.clip(value / scale, 0.0, 1.0))
        deduction = weight * normalized
        base -= deduction
        breakdown[label] = {
            "value": float(value),
            "deduction": float(deduction),
            "weight": weight,
            "scale": scale,
        }
    base = float(np.clip(base, 0.2, 1.0))
    return base, breakdown


def compute_material_utility_scores(
    manifest: pd.DataFrame,
    *,
    bundle: ds.MaterialReferenceBundle | None = None,
) -> pd.DataFrame:
    """Add Material Utility Score components to *manifest*."""

    if manifest.empty:
        manifest = manifest.copy()
        manifest["spectral_score"] = 0.0
        manifest["mechanical_score"] = 0.0
        manifest["mass_factor"] = 0.0
        manifest["penalty_factor"] = 1.0
        manifest["penalty_breakdown"] = [{} for _ in range(len(manifest))]
        manifest["penalty_breakdown_json"] = "{}"
        manifest["material_utility_score"] = 0.0
        return manifest

    if bundle is None:
        bundle = ds.load_material_reference_bundle()
    stats = _bundle_property_stats(bundle)

    spectral_scores: list[float] = []
    mechanical_scores: list[float] = []
    mass_factors: list[float] = []
    penalties: list[float] = []
    breakdowns: list[dict[str, dict[str, float]]] = []

    for _, row in manifest.iterrows():
        material_key = row.get("material_key")
        if not isinstance(material_key, str):
            spectral = 0.2
            mechanical = 0.1
        else:
            spectral = _spectral_score(material_key, bundle)
            mechanical = _mechanical_score(material_key, stats, bundle)
        mass_factor = _mass_factor(row)
        penalty, breakdown = _penalty_factor(row)

        spectral_scores.append(spectral)
        mechanical_scores.append(mechanical)
        mass_factors.append(mass_factor)
        penalties.append(penalty)
        breakdowns.append(breakdown)

    manifest = manifest.copy()
    manifest["spectral_score"] = spectral_scores
    manifest["mechanical_score"] = mechanical_scores
    manifest["mass_factor"] = mass_factors
    manifest["penalty_factor"] = penalties
    manifest["penalty_breakdown"] = breakdowns
    manifest["penalty_breakdown_json"] = [
        json.dumps(breakdown, ensure_ascii=False) if breakdown else "{}"
        for breakdown in breakdowns
    ]

    utility = []
    for spectral, mechanical, mass_factor, penalty in zip(
        spectral_scores, mechanical_scores, mass_factors, penalties, strict=False
    ):
        base = 0.45 * spectral + 0.4 * mechanical + 0.15 * mass_factor
        utility.append(float(np.clip(base * penalty, 0.0, 1.0)))
    manifest["material_utility_score"] = utility
    return manifest


def _estimate_material_score(
    material_key: str,
    *,
    bundle: ds.MaterialReferenceBundle,
    stats: Mapping[str, tuple[float, float, float]],
    mass_factor: float,
) -> float:
    spectral = _spectral_score(material_key, bundle)
    mechanical = _mechanical_score(material_key, stats, bundle)
    base = 0.45 * spectral + 0.4 * mechanical + 0.15 * mass_factor
    return float(np.clip(base, 0.0, 1.0))


def propose_policy_actions(
    manifest: pd.DataFrame,
    *,
    bundle: ds.MaterialReferenceBundle | None = None,
    top_n: int = 3,
) -> pd.DataFrame:
    """Suggest substitutions or quota adjustments for low scoring items."""

    if manifest.empty:
        return pd.DataFrame(
            columns=[
                "item_index",
                "item_name",
                "current_material_key",
                "current_score",
                "recommended_material_key",
                "recommended_score",
                "recommended_quota",
                "action",
                "justification",
                "evidence_json",
            ]
        )

    if bundle is None:
        bundle = ds.load_material_reference_bundle()
    stats = _bundle_property_stats(bundle)

    recommendations: list[dict[str, object]] = []

    for idx, row in manifest.iterrows():
        score = float(row.get("material_utility_score", 0.0) or 0.0)
        material_key = row.get("material_key")
        if not isinstance(material_key, str):
            continue
        if score >= 0.7:
            continue
        compat = bundle.compatibility_matrix.get(material_key, {})
        if not compat:
            continue
        mass_factor = float(row.get("mass_factor", _mass_factor(row)))
        candidates: list[tuple[float, str, Mapping[str, object]]] = []
        for partner, meta in compat.items():
            partner_key = bundle.alias_map.get(partner, partner)
            partner_key = bundle.alias_map.get(ds.slugify(ds.normalize_item(partner_key)), partner_key)
            if partner_key not in bundle.properties:
                continue
            candidate_score = _estimate_material_score(
                partner_key,
                bundle=bundle,
                stats=stats,
                mass_factor=mass_factor,
            )
            candidates.append((candidate_score, partner_key, meta))
        if not candidates:
            continue
        candidates.sort(key=lambda entry: entry[0], reverse=True)
        best_candidates = candidates[:top_n]
        for candidate_score, partner_key, meta in best_candidates:
            if candidate_score <= score + 0.05:
                continue
            sources = meta.get("sources", []) if isinstance(meta, Mapping) else []
            evidence = meta.get("evidence", []) if isinstance(meta, Mapping) else []
            quota = float(np.clip(1.0 - score, 0.1, 1.0))
            action = "substitute"
            justification = (
                f"Compatibilidad documentada con {partner_key} (regla {meta.get('rule', 'N/A')})"
                if isinstance(meta, Mapping)
                else f"Compatibilidad documentada con {partner_key}"
            )
            recommendations.append(
                {
                    "item_index": idx,
                    "item_name": row.get("item") or row.get("material") or row.get("description"),
                    "current_material_key": material_key,
                    "current_score": round(score, 4),
                    "recommended_material_key": partner_key,
                    "recommended_score": round(candidate_score, 4),
                    "recommended_quota": round(quota, 3),
                    "action": action,
                    "justification": justification,
                    "evidence_json": json.dumps(
                        {
                            "sources": sources,
                            "evidence": evidence,
                        },
                        ensure_ascii=False,
                    ),
                }
            )
            break

    if not recommendations:
        return pd.DataFrame(
            columns=[
                "item_index",
                "item_name",
                "current_material_key",
                "current_score",
                "recommended_material_key",
                "recommended_score",
                "recommended_quota",
                "action",
                "justification",
                "evidence_json",
            ]
        )

    return pd.DataFrame(recommendations)


def build_manifest_compatibility(
    manifest: pd.DataFrame,
    *,
    bundle: ds.MaterialReferenceBundle | None = None,
) -> pd.DataFrame:
    """Flatten compatibility entries referenced by *manifest*."""

    if bundle is None:
        bundle = ds.load_material_reference_bundle()

    unique_keys = [
        key for key in manifest.get("material_key", []) if isinstance(key, str)
    ]
    if not unique_keys:
        return pd.DataFrame(
            columns=[
                "material_key",
                "partner_key",
                "rule",
                "sources_json",
                "evidence_json",
            ]
        )

    records: list[dict[str, object]] = []
    seen_pairs: set[tuple[str, str]] = set()
    for key in dict.fromkeys(unique_keys):
        compat = bundle.compatibility_matrix.get(key, {})
        for partner, meta in compat.items():
            pair = (key, partner)
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            sources = []
            evidence = []
            if isinstance(meta, Mapping):
                sources = list(meta.get("sources", [])) if isinstance(meta.get("sources"), Sequence) else []
                evidence = meta.get("evidence", []) or []
            records.append(
                {
                    "material_key": key,
                    "partner_key": partner,
                    "rule": (meta.get("rule") if isinstance(meta, Mapping) else None),
                    "sources_json": json.dumps(list(sources), ensure_ascii=False),
                    "evidence_json": json.dumps(evidence, ensure_ascii=False),
                }
            )
    return pd.DataFrame(records)


def build_material_passport(
    manifest: pd.DataFrame,
    recommendations: pd.DataFrame,
    compatibility: pd.DataFrame,
    *,
    original_manifest: pd.DataFrame | None = None,
) -> dict[str, object]:
    """Assemble a serialisable material passport payload."""

    timestamp = datetime.now(UTC).isoformat()
    total_mass = pd.to_numeric(manifest.get("mass_kg"), errors="coerce").fillna(0.0).sum()
    mean_score = float(pd.to_numeric(manifest.get("material_utility_score"), errors="coerce").fillna(0.0).mean())

    passport = {
        "generated_at": timestamp,
        "total_items": int(len(manifest)),
        "total_mass_kg": float(total_mass),
        "mean_material_utility_score": mean_score,
        "source_manifest_columns": list(original_manifest.columns) if isinstance(original_manifest, pd.DataFrame) else [],
        "items": [],
        "recommendations": [],
        "compatibility_sources": [],
    }

    item_records = manifest.copy()
    item_records["penalty_breakdown"] = [
        breakdown if isinstance(breakdown, Mapping) else {}
        for breakdown in item_records.get("penalty_breakdown", [])
    ]
    passport["items"] = json.loads(item_records.to_json(orient="records"))
    if not recommendations.empty:
        passport["recommendations"] = json.loads(recommendations.to_json(orient="records"))
    if not compatibility.empty:
        entries = json.loads(compatibility.to_json(orient="records"))
        sources_index: list[dict[str, object]] = []
        for entry in entries:
            sources = entry.get("sources_json")
            if isinstance(sources, str) and sources:
                try:
                    parsed = json.loads(sources)
                except json.JSONDecodeError:
                    parsed = []
            else:
                parsed = []
            for source in parsed:
                sources_index.append(
                    {
                        "material_key": entry.get("material_key"),
                        "partner_key": entry.get("partner_key"),
                        "source": source,
                    }
                )
        passport["compatibility_sources"] = sources_index
    return passport


def export_material_passport_pdf(passport: Mapping[str, object], path: Path) -> bool:
    """Render *passport* into a PDF file.  Returns ``True`` on success."""

    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.lib.units import inch
        from reportlab.pdfgen import canvas
    except Exception as exc:  # pragma: no cover - optional dependency
        logger.warning("reportlab no disponible para generar PDF: %s", exc)
        return False

    path.parent.mkdir(parents=True, exist_ok=True)
    canvas_obj = canvas.Canvas(str(path), pagesize=letter)
    width, height = letter
    text = canvas_obj.beginText(0.75 * inch, height - 1 * inch)
    text.setFont("Helvetica", 10)

    def _emit(line: str) -> None:
        text.textLine(line[:120])

    _emit("Material Passport")
    _emit("")
    _emit(f"Generado: {passport.get('generated_at', 'N/A')}")
    _emit(f"Total ítems: {passport.get('total_items', 0)}")
    _emit(f"Masa total (kg): {passport.get('total_mass_kg', 0):.2f}")
    _emit(f"Puntaje medio: {passport.get('mean_material_utility_score', 0):.3f}")
    _emit("")
    _emit("Top recomendaciones:")
    recommendations = passport.get("recommendations") or []
    for item in recommendations[:6]:
        if not isinstance(item, Mapping):
            continue
        _emit(
            f"- {item.get('item_name', 'N/A')} → {item.get('recommended_material_key', 'N/A')} (cuota {item.get('recommended_quota', 0)})"
        )
    _emit("")
    _emit("Resumen de ítems (primeros 8):")
    items = passport.get("items") or []
    for item in items[:8]:
        if not isinstance(item, Mapping):
            continue
        _emit(
            f"• {item.get('item', item.get('material', 'item'))}: puntaje {item.get('material_utility_score', 0):.3f}"
        )
    canvas_obj.drawText(text)
    canvas_obj.showPage()
    canvas_obj.save()
    return True


__all__ = [
    "PolicyArtifacts",
    "map_manifest_to_bundle",
    "compute_material_utility_scores",
    "propose_policy_actions",
    "build_manifest_compatibility",
    "build_material_passport",
    "export_material_passport_pdf",
]
