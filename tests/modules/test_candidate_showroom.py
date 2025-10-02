import pytest

from app.modules.candidate_showroom import (
    _collect_badges,
    _normalize_success,
    _prepare_rows,
)


def _base_candidate(**overrides):
    candidate = {
        "score": 0.8,
        "props": {
            "rigidity": 0.5,
            "water_l": 1.0,
            "energy_kwh": 2.0,
            "crew_min": 4,
        },
        "materials": [],
        "auxiliary": {},
        "timeline_badges": [],
        "process_id": "A1",
        "process_name": "Proceso Seguro",
    }
    candidate.update(overrides)
    return candidate


def test_prepare_rows_filters_by_score_and_safety():
    candidates = [
        _base_candidate(),
        _base_candidate(
            score=0.6,
            process_id="B1",
            process_name="Proceso Riesgo",
            materials=["PTFE"],
        ),
    ]

    rows = _prepare_rows(
        candidates,
        score_threshold=0.7,
        only_safe=True,
        threshold_active=True,
        resource_limits={"energy": 3.0, "water": 2.0, "crew": 6.0},
    )

    assert len(rows) == 1
    assert rows[0]["candidate"]["process_id"] == "A1"
    assert rows[0]["is_safe"] is True
    assert "🎯 Score ≥ 0.70" in rows[0]["badges"]


def test_prepare_rows_applies_resource_limits():
    candidates = [
        _base_candidate(score=0.9, props={"rigidity": 1.0, "water_l": 1.0, "energy_kwh": 4.0, "crew_min": 3}),
        _base_candidate(score=0.85, props={"rigidity": 0.8, "water_l": 1.2, "energy_kwh": 2.5, "crew_min": 3}),
    ]

    rows = _prepare_rows(
        candidates,
        score_threshold=0.5,
        only_safe=False,
        threshold_active=False,
        resource_limits={"energy": 3.0},
    )

    assert len(rows) == 1
    assert rows[0]["candidate"]["score"] == pytest.approx(0.85)


def test_normalize_success_variants():
    assert _normalize_success({"message": "ok", "candidate_idx": 2}) == {
        "message": "ok",
        "candidate_key": "2",
    }
    assert _normalize_success(" listo ") == {"message": " listo ", "candidate_key": None}
    assert _normalize_success(5) == {"message": "", "candidate_key": None}


def test_collect_badges_sources_and_auxiliary():
    cand = {
        "regolith_pct": 10,
        "source_categories": ["multilayer"],
    }
    aux = {"passes_seal": True}
    badges = _collect_badges(cand, aux)

    assert "⛰️ ISRU MGS-1" in badges
    assert "♻️ Valorización problemáticos" in badges
    assert "🛡️ Seal ready" in badges
