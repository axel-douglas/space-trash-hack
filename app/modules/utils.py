"""Utility helpers shared across Streamlit pages."""

from __future__ import annotations

import math
from typing import Any

__all__ = ["safe_int"]


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
