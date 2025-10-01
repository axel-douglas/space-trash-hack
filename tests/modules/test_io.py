"""Tests for the io module helpers."""

from __future__ import annotations

import pandas as pd

from app.modules.io import load_waste_df
from app.modules.problematic import problematic_mask


def test_problematic_column_matches_helper() -> None:
    df = load_waste_df()

    assert "_problematic" in df.columns, "load_waste_df should populate the _problematic column"

    helper_mask = problematic_mask(df)

    pd.testing.assert_series_equal(
        df["_problematic"],
        helper_mask,
        check_names=False,
    )
