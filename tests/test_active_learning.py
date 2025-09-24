import math

from app.modules.active_learning import suggest_next_candidates


def _build_candidate(recipe: str, score: float, rig_ci: tuple[float, float], energy_ci: tuple[float, float]):
    return {
        "features": {"recipe_id": recipe, "process_id": "P02"},
        "score": score,
        "confidence_interval": {
            "rigidez": rig_ci,
            "energy_kwh": energy_ci,
        },
    }


def test_suggest_next_candidates_by_uncertainty():
    cand_a = _build_candidate("R-A", 0.62, (0.4, 0.8), (2.0, 4.5))
    cand_b = _build_candidate("R-B", 0.58, (0.2, 0.95), (1.5, 6.0))

    ranked = suggest_next_candidates([cand_a, cand_b], strategy="uncertainty", top_n=1)

    assert ranked, "Expected at least one suggestion"
    assert ranked[0].recipe_id == "R-B"
    assert ranked[0].metrics["uncertainty"] > ranked[0].priority * 0.9


def test_expected_improvement_combines_score_and_uncertainty():
    cand_a = {
        "features": {"recipe_id": "R-1", "process_id": "P03"},
        "score": 0.75,
        "confidence_interval": {"rigidez": (0.6, 0.82)},
    }
    cand_b = {
        "features": {"recipe_id": "R-2", "process_id": "P03"},
        "score": 0.7,
        "confidence_interval": {"rigidez": (0.65, 0.7)},
    }

    observed = [
        {"recipe_id": "OBS-1", "process_id": "P03", "score": 0.6},
    ]

    ranked = suggest_next_candidates(
        [cand_a, cand_b], strategy="expected_improvement", observations=observed, exploration_bonus=0.05
    )

    assert ranked[0].recipe_id == "R-1"
    metrics = ranked[0].metrics
    assert math.isclose(metrics["baseline_score"], 0.6)
    assert metrics["improvement"] > 0
    assert metrics["uncertainty"] > 0
