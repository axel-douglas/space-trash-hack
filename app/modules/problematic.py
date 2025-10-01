"""Utilities for classifying problematic waste inventory rows."""

from __future__ import annotations

import pandas as pd


def _lower_text_column(frame: pd.DataFrame, column: str) -> pd.Series:
    """Return a lowercased string Series for ``column`` within ``frame``."""

    if column in frame.columns:
        series = frame[column]
    else:
        series = pd.Series("", index=frame.index)

    return series.fillna("").astype(str).str.lower()


def problematic_mask(frame: pd.DataFrame) -> pd.Series:
    """Vectorized rules to flag problematic inventory rows.

    Parameters
    ----------
    frame:
        DataFrame with the NASA inventory schema (or compatible).

    Returns
    -------
    pandas.Series
        Boolean mask indicating rows considered problematic.
    """

    if frame.empty:
        return pd.Series(False, index=frame.index, dtype=bool)

    category = _lower_text_column(frame, "category")
    material_family = _lower_text_column(frame, "material_family")
    flags = _lower_text_column(frame, "flags")

    contains = dict(
        category=lambda needle: category.str.contains(needle, regex=False),
        material=lambda needle: material_family.str.contains(needle, regex=False),
        flags=lambda needle: flags.str.contains(needle, regex=False),
    )

    return (
        contains["category"]("pouches")
        | contains["flags"]("multilayer")
        | contains["material"]("pe-pet-al")
        | contains["category"]("foam")
        | contains["material"]("zotek")
        | contains["flags"]("closed_cell")
        | contains["category"]("eva")
        | contains["flags"]("ctb")
        | contains["material"]("nomex")
        | contains["material"]("nylon")
        | contains["material"]("polyester")
        | contains["category"]("glove")
        | contains["material"]("nitrile")
        | contains["flags"]("wipe")
        | contains["category"]("textile")
    )

