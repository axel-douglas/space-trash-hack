from __future__ import annotations

import itertools

import pytest

from app.modules import data_build
from app.modules.data_sources import REGOLITH_CHARACTERIZATION


def test_gold_record_contains_regolith_characterization() -> None:
    records = list(itertools.islice(data_build._build_gold_records(), 5))
    assert records, "the builder should yield at least one record"

    for record in records:
        assert record.features
        for name, baseline in REGOLITH_CHARACTERIZATION.feature_items:
            assert name in record.features
            expected = float(baseline) * float(record.regolith_pct)
            if record.regolith_pct > 0:
                assert record.features[name] == pytest.approx(expected, rel=1e-6)
            else:
                assert record.features[name] == pytest.approx(0.0, abs=1e-8)
