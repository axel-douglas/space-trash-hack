from datetime import datetime, timezone

import pandas as pd

from app.modules import data_sources, policy_engine


class _FrozenDateTime(datetime):
    @classmethod
    def now(cls, tz=None):
        tzinfo = tz or timezone.utc
        return datetime(2024, 1, 1, 0, 0, 0, tzinfo=tzinfo)


def test_material_passport_deterministic(monkeypatch):
    bundle = data_sources.load_material_reference_bundle()
    manifest = pd.DataFrame(
        [
            {
                "material_key": "pe_evoh_multilayer_film",
                "material": "PE/EVOH multilayer film",
                "material_utility_score": 0.82,
                "mass_kg": 4.0,
                "penalty_breakdown": {},
            }
        ]
    )

    compatibility = policy_engine.build_manifest_compatibility(manifest, bundle=bundle)
    assert not compatibility.empty

    recommendations = pd.DataFrame(
        [
            {
                "item_name": "Sample",
                "recommended_material_key": "pe_evoh_multilayer_film",
                "recommended_quota": 1,
            }
        ]
    )

    monkeypatch.setattr(policy_engine, "datetime", _FrozenDateTime)

    passport_a = policy_engine.build_material_passport(
        manifest,
        recommendations,
        compatibility,
        original_manifest=manifest,
    )
    passport_b = policy_engine.build_material_passport(
        manifest,
        recommendations,
        compatibility,
        original_manifest=manifest,
    )

    assert passport_a == passport_b
    assert passport_a["total_items"] == 1
    assert passport_a["compatibility_sources"]
