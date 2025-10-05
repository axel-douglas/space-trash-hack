"""Utility helpers shared across Streamlit pages."""

from __future__ import annotations

import math
from typing import Any, Mapping

__all__ = [
    "safe_int",
    "safe_float",
    "format_number",
    "format_resource_text",
    "format_label_summary",
    "uses_physical_dataset",
    "physical_dataset_tooltip",
]


def safe_int(value: Any, default: int | None = 0) -> int | None:
    """Return ``value`` converted to ``int`` when possible.

    The helper mirrors :func:`int` but guards against ``None`` inputs,
    malformed strings and ``NaN`` floats. When conversion fails the
    provided ``default`` is returned instead of raising an exception.
    """

    try:
        if isinstance(value, str):
            candidate = value.strip()
            if not candidate:
                raise ValueError("empty string")
            number = float(candidate)
        else:
            number = float(value)
    except (TypeError, ValueError):
        return default

    if math.isnan(number):
        return default

    try:
        return int(number)
    except (OverflowError, ValueError, TypeError):
        return default


def safe_float(value: Any, default: float | None = None) -> float | None:
    """Convert ``value`` to ``float`` guardando contra valores inválidos."""

    try:
        number = float(value)
    except (TypeError, ValueError):
        return default

    if math.isnan(number):
        return default

    return number


def format_number(
    value: Any,
    *,
    precision: int = 2,
    placeholder: str = "—",
) -> str:
    """Renderiza un número con precisión configurable o un placeholder."""

    number = safe_float(value)
    if number is None:
        return placeholder
    return f"{number:.{precision}f}"


def format_resource_text(
    value: Any,
    limit: Any,
    *,
    precision: int = 2,
    placeholder: str = "—",
) -> str:
    """Devuelve ``valor / límite`` formateado o ``placeholder`` cuando aplique."""

    value_text = format_number(value, precision=precision, placeholder=placeholder)
    limit_number = safe_float(limit)
    if limit_number is None:
        return value_text
    limit_text = f"{limit_number:.{precision}f}"
    return f"{value_text} / {limit_text}"


def format_label_summary(summary: Mapping[str, Mapping[str, float]] | None) -> str:
    """Compacta estadísticas de labels en un texto amigable para la UI."""

    if not summary:
        return ""

    parts: list[str] = []
    for source, stats in summary.items():
        if not isinstance(stats, Mapping):
            parts.append(str(source))
            continue

        label = str(source)
        fragment = label

        count = stats.get("count")
        try:
            if count is not None:
                fragment = f"{label}×{int(count)}"
        except (TypeError, ValueError):
            fragment = label

        mean_weight = stats.get("mean_weight")
        try:
            if mean_weight is not None:
                fragment = f"{fragment} (w≈{float(mean_weight):.2f})"
        except (TypeError, ValueError):
            pass

        parts.append(fragment)

    return " · ".join(parts)


def uses_physical_dataset(source: Any) -> bool:
    """Return ``True`` when the prediction source maps to a Rex-AI physical model."""

    if isinstance(source, str):
        return source.lower().startswith("rexai")
    return False


def physical_dataset_tooltip(*, summary: str | None = None, trained_at: Any | None = None) -> str:
    """Compose a concise tooltip describing NASA datasets backing Rex-AI predictions."""

    base = (
        "Respaldado por datasets físicos NASA ISRU: granulometría MGS-1, espectros (Fig.4) y "
        "perfiles térmicos (Fig.5)."
    )
    parts = [base]

    if summary:
        parts.append(f"Cobertura labels: {summary}.")

    if trained_at:
        trained_label = str(trained_at).strip()
        if trained_label and trained_label not in {"?", "—"}:
            parts.append(f"Entrenado {trained_label}.")

    return " ".join(parts)
