"""Spectral mixing utilities.

This module exposes a convenience wrapper on top of the material reference
bundle so downstream services can propose FTIR mixes that approximate a target
signature under logistic constraints.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import optimize

from app.modules import data_sources as ds


_NUMERIC_PRIORITY = (
    "absorbance_norm_1um",
    "absorbance",
    "reflectance_pct",
    "transmittance_pct",
)


@dataclass(slots=True)
class SpectralMixResult:
    """Container returned by :func:`solve_spectral_recipe`."""

    basis: list[str]
    coefficients: np.ndarray
    target_curve: pd.DataFrame
    synthetic_curve: pd.DataFrame
    error_metrics: dict[str, float]

    def as_dict(self) -> dict[str, object]:
        """Return a serialisable payload used by Streamlit views."""

        return {
            "coefficients": [
                {"material": key, "fraction": float(value)}
                for key, value in zip(self.basis, self.coefficients, strict=False)
            ],
            "synthetic_curve": self.synthetic_curve,
            "target_curve": self.target_curve,
            "error": dict(self.error_metrics),
        }


def _load_target_curve(target_curve: pd.DataFrame | Sequence[Mapping[str, float]] | str | Path) -> pd.DataFrame:
    if isinstance(target_curve, pd.DataFrame):
        df = target_curve.copy()
    elif isinstance(target_curve, (str, Path)):
        df = pd.read_csv(target_curve)
    else:
        df = pd.DataFrame(list(target_curve))

    if df.empty:
        raise ValueError("target_curve must contain at least one row")

    df = df.copy()
    wave_column = _resolve_wavenumber_column(df)
    value_column = _resolve_value_column(df)
    df = df[[wave_column, value_column]].dropna()
    df = df.rename(columns={wave_column: "wavenumber_cm_1", value_column: "intensity"})
    df = df.sort_values("wavenumber_cm_1").reset_index(drop=True)
    return df


def _resolve_wavenumber_column(df: pd.DataFrame) -> str:
    for candidate in df.columns:
        normalized = candidate.lower()
        if "wavenumber" in normalized or normalized.endswith("_cm_1"):
            return candidate
    raise ValueError("target_curve must include a wavenumber column")


def _resolve_value_column(df: pd.DataFrame) -> str:
    numeric_columns = [col for col in df.columns if np.issubdtype(df[col].dtype, np.number)]
    for preferred in _NUMERIC_PRIORITY:
        if preferred in df.columns and preferred in numeric_columns:
            return preferred
    if numeric_columns:
        return numeric_columns[0]
    raise ValueError("target_curve must include a numeric spectral column")


def _resolve_available_basis(
    availability: Mapping[str, object] | pd.DataFrame | Sequence[Mapping[str, object]] | Sequence[str] | None,
    bundle: ds.MaterialReferenceBundle,
) -> list[str]:
    if availability is None:
        return list(bundle.spectral_curves.keys())

    keys: set[str] = set()
    rows: Iterable[Mapping[str, object]]
    if isinstance(availability, pd.DataFrame):
        rows = availability.to_dict("records")
    elif isinstance(availability, Mapping):
        rows = [availability]
    elif isinstance(availability, Sequence):
        if availability and isinstance(availability[0], Mapping):
            rows = availability  # type: ignore[assignment]
        else:
            values = [str(value) for value in availability]
            rows = [{"value": value} for value in values]
    else:
        return list(bundle.spectral_curves.keys())

    alias_index: dict[str, str] = {}
    for spectral_key, meta in bundle.metadata.items():
        tokens = {spectral_key, ds.slugify(spectral_key)}
        for value in meta.values():
            if isinstance(value, str):
                slug = ds.slugify(ds.normalize_item(value))
                if slug:
                    tokens.add(slug)
        for token in tokens:
            if token:
                alias_index[token] = spectral_key

    for row in rows:
        for candidate in ("spectral_key", "material_key", "material", "value"):
            value = row.get(candidate) if isinstance(row, Mapping) else None
            if not value:
                continue
            text = str(value).strip()
            if not text:
                continue
            if text in bundle.spectral_curves:
                keys.add(text)
                continue
            slug = ds.slugify(ds.normalize_item(text))
            if not slug:
                continue
            resolved = alias_index.get(slug)
            if resolved:
                keys.add(resolved)
    if not keys:
        return list(bundle.spectral_curves.keys())
    return sorted(keys)


def _build_design_matrix(
    basis: list[str],
    bundle: ds.MaterialReferenceBundle,
    target_grid: np.ndarray,
) -> np.ndarray:
    matrix = np.zeros((target_grid.size, len(basis)), dtype=float)
    for idx, key in enumerate(basis):
        curve = bundle.spectral_curves.get(key)
        if curve is None or curve.empty:
            continue
        curve = curve.copy()
        if "wavenumber_cm_1" not in curve.columns:
            raise ValueError(f"spectral curve '{key}' is missing 'wavenumber_cm_1'")
        value_column = _resolve_value_column(curve)
        sorted_curve = curve.sort_values("wavenumber_cm_1")
        x = sorted_curve["wavenumber_cm_1"].to_numpy(dtype=float)
        y = sorted_curve[value_column].to_numpy(dtype=float)
        matrix[:, idx] = np.interp(target_grid, x, y, left=y[0], right=y[-1])
    return matrix


def _apply_component_cap(coefficients: np.ndarray, max_components: int | None) -> np.ndarray:
    if not max_components or max_components <= 0:
        return coefficients
    if max_components >= coefficients.size:
        return coefficients
    mask = np.argsort(coefficients)[::-1]
    keep = mask[:max_components]
    filtered = np.zeros_like(coefficients)
    filtered[keep] = coefficients[keep]
    return filtered


def _run_constrained_least_squares(
    design_matrix: np.ndarray,
    target: np.ndarray,
    upper_bounds: np.ndarray,
    total_cap: float,
) -> np.ndarray:
    initial, _ = optimize.nnls(design_matrix, target)
    bounds = optimize.Bounds(lb=np.zeros_like(initial), ub=upper_bounds)
    linear_constraint = optimize.LinearConstraint(np.ones_like(initial), lb=0.0, ub=total_cap)

    def objective(x: np.ndarray) -> float:
        residual = design_matrix @ x - target
        return 0.5 * float(residual @ residual)

    def gradient(x: np.ndarray) -> np.ndarray:
        residual = design_matrix @ x - target
        return design_matrix.T @ residual

    result = optimize.minimize(
        objective,
        x0=initial,
        jac=gradient,
        bounds=bounds,
        constraints=[linear_constraint],
        method="SLSQP",
        options={"maxiter": 200, "ftol": 1e-9},
    )
    if result.success and isinstance(result.x, np.ndarray):
        return result.x

    # Fallback to projecting the NNLS solution if optimisation fails.
    solution = initial
    total = float(solution.sum())
    if total <= total_cap + 1e-9:
        return solution
    scale = total_cap / total if total > 0 else 0.0
    return np.clip(solution * scale, 0.0, upper_bounds)


def solve_spectral_recipe(
    target_curve: pd.DataFrame | Sequence[Mapping[str, float]] | str | Path,
    availability: Mapping[str, object] | pd.DataFrame | Sequence[Mapping[str, object]] | Sequence[str] | None,
    constraints: Mapping[str, object] | None = None,
) -> SpectralMixResult:
    """Solve a constrained least squares mixing problem for FTIR curves."""

    constraints = dict(constraints or {})
    bundle = ds.load_material_reference_bundle()
    target_df = _load_target_curve(target_curve)
    target_grid = target_df["wavenumber_cm_1"].to_numpy(dtype=float)
    target_vector = target_df["intensity"].to_numpy(dtype=float)

    basis = _resolve_available_basis(availability, bundle)
    if not basis:
        raise ValueError("No spectral curves available to build a mix")

    design_matrix = _build_design_matrix(basis, bundle, target_grid)

    per_material_max = {
        key: float(value)
        for key, value in dict(constraints.get("per_material_max", {})).items()
        if key in basis
    }
    default_cap = float(constraints.get("max_fraction_per_component", 1.0))
    upper_bounds = np.full(len(basis), default_cap, dtype=float)
    for idx, key in enumerate(basis):
        cap = per_material_max.get(key)
        if cap is not None:
            upper_bounds[idx] = min(default_cap, float(cap))
    total_cap = float(constraints.get("max_fraction", 1.0))

    coefficients = _run_constrained_least_squares(design_matrix, target_vector, upper_bounds, total_cap)
    coefficients = _apply_component_cap(coefficients, constraints.get("max_components"))

    total = coefficients.sum()
    if total > total_cap + 1e-9:
        coefficients *= total_cap / total

    synthetic = design_matrix @ coefficients
    residual = synthetic - target_vector
    mae = float(np.mean(np.abs(residual)))
    rmse = float(np.sqrt(np.mean(residual**2)))
    max_abs = float(np.max(np.abs(residual)))

    synthetic_df = target_df.copy()
    synthetic_df["synthetic_intensity"] = synthetic

    return SpectralMixResult(
        basis=basis,
        coefficients=coefficients,
        target_curve=target_df,
        synthetic_curve=synthetic_df,
        error_metrics={"mae": mae, "rmse": rmse, "max_abs": max_abs},
    )


__all__ = ["solve_spectral_recipe", "SpectralMixResult"]
