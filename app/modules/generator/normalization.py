"""Helpers that proxy the shared data normalisation primitives."""
from __future__ import annotations

from typing import Any

from app.modules import data_sources as ds

__all__ = [
    "normalize_category",
    "normalize_item",
    "token_set",
    "build_match_key",
]


normalize_category = ds.normalize_category
normalize_item = ds.normalize_item
token_set = ds.token_set


def build_match_key(category: Any, subitem: Any | None = None) -> str:
    """Return the canonical key used to match NASA reference tables."""

    if subitem:
        return f"{normalize_category(category)}|{normalize_item(subitem)}"
    return normalize_category(category)
