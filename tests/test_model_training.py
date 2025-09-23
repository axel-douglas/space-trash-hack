"""Tests for :mod:`app.modules.model_training`."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from app.modules import generator, label_mapper, model_training


@pytest.fixture(autouse=True)
def _reset_gold_caches() -> None:
    """Ensure cached gold datasets do not leak across tests."""

    original_features_cache = model_training._GOLD_FEATURES_CACHE
    original_targets_cache = model_training._GOLD_TARGETS_CACHE
    original_label_cache = label_mapper._LABELS_CACHE
    model_training._GOLD_FEATURES_CACHE = None
    model_training._GOLD_TARGETS_CACHE = None
    label_mapper._LABELS_CACHE = None
    try:
        yield
    finally:
        model_training._GOLD_FEATURES_CACHE = original_features_cache
        model_training._GOLD_TARGETS_CACHE = original_targets_cache
        label_mapper._LABELS_CACHE = original_label_cache


def _write_parquet(data: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data.to_parquet(path, index=False)


def _sample_inputs():
    picks = pd.DataFrame(
        {
            "_source_id": ["A1", "B2"],
            "kg": [3.0, 2.0],
            "material": ["Aluminum scrap", "Foam packaging"],
            "category": ["structural", "packaging"],
            "flags": ["", ""],
            "_source_category": ["structural", "packaging"],
            "_source_flags": ["", ""],
            "pct_mass": [60.0, 40.0],
            "pct_volume": [55.0, 45.0],
            "moisture_pct": [5.0, 8.0],
            "difficulty_factor": [2, 1],
            "density_kg_m3": [2.7, 0.2],
        }
    )
    process = pd.Series(
        {
            "process_id": "P01",
            "energy_kwh_per_kg": 1.2,
            "water_l_per_kg": 0.5,
            "crew_min_per_batch": 6.0,
            "name": "Test process",
        }
    )
    total_mass = max(0.001, float(picks["kg"].sum()))
    weights = (picks["kg"] / total_mass).tolist()
    return picks, process, weights


def test_build_training_dataframe_prefers_gold_labels(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """When gold labels exist they are merged and override heuristic targets."""

    features_path = tmp_path / "features.parquet"
    labels_path = tmp_path / "labels.parquet"

    feature_row = {
        "recipe_id": "REC-42",
        "process_id": "P01",
    }
    for column in model_training.FEATURE_COLUMNS:
        if column not in feature_row:
            feature_row[column] = 0.0

    _write_parquet(pd.DataFrame([feature_row]), features_path)

    labels_row = {
        "recipe_id": "rec-42",
        "process_id": "p01",
        "rigidez": 0.97,
        "estanqueidad": 0.42,
        "energy_kwh": 125.5,
        "water_l": 8.5,
        "crew_min": 73.0,
        "tightness_pass": 1,
        "rigidity_level": 3,
        "label_weight": 3.5,
        "provenance": "mission",
        "conf_lo_rigidez": 0.9,
        "conf_hi_rigidez": 1.05,
        "conf_lo_energy_kwh": 120.0,
        "conf_hi_energy_kwh": 130.0,
    }

    _write_parquet(pd.DataFrame([labels_row]), labels_path)

    monkeypatch.setattr(model_training, "GOLD_FEATURES_PATH", features_path)
    monkeypatch.setattr(model_training, "GOLD_LABELS_PATH", labels_path)
    monkeypatch.setattr(label_mapper, "GOLD_LABELS_PATH", labels_path)

    def _raise_generate(*_: object, **__: object) -> None:
        raise AssertionError("_generate_samples should not be invoked when gold labels exist")

    monkeypatch.setattr(model_training, "_generate_samples", _raise_generate)

    df = model_training.build_training_dataframe()

    assert len(df) == 1
    row = df.iloc[0]
    assert row["rigidez"] == pytest.approx(0.97)
    assert row["estanqueidad"] == pytest.approx(0.42)
    assert row["energy_kwh"] == pytest.approx(125.5)
    assert row["water_l"] == pytest.approx(8.5)
    assert row["crew_min"] == pytest.approx(73.0)
    assert row["tightness_pass"] == 1
    assert row["rigidity_level"] == 3
    assert row["label_weight"] == pytest.approx(3.5)
    assert row["label_source"] == "mission"
    assert row["provenance"] == "mission"
    assert row["conf_lo_rigidez"] == pytest.approx(0.9)
    assert row["conf_hi_rigidez"] == pytest.approx(1.05)
    assert row["conf_lo_energy_kwh"] == pytest.approx(120.0)
    assert row["conf_hi_energy_kwh"] == pytest.approx(130.0)


def test_build_training_dataframe_falls_back_when_labels_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Synthetic samples are generated when consolidated labels are unavailable."""

    features_path = tmp_path / "features.parquet"
    feature_row = {
        "recipe_id": "REC-99",
        "process_id": "P02",
    }
    for column in model_training.FEATURE_COLUMNS:
        if column not in feature_row:
            feature_row[column] = 0.0

    _write_parquet(pd.DataFrame([feature_row]), features_path)

    monkeypatch.setattr(model_training, "GOLD_FEATURES_PATH", features_path)
    monkeypatch.setattr(model_training, "GOLD_LABELS_PATH", tmp_path / "missing.parquet")
    monkeypatch.setattr(label_mapper, "GOLD_LABELS_PATH", tmp_path / "missing.parquet")

    samples_called: list[tuple[int, int | None]] = []

    def _fake_samples(n_samples: int, seed: int | None) -> list[model_training.SampledCombination]:
        samples_called.append((n_samples, seed))
        return [
            model_training.SampledCombination(
                features={"process_id": "P02", "recipe_id": "REC-99"},
                targets={
                    "rigidez": 0.5,
                    "estanqueidad": 0.4,
                    "energy_kwh": 10.0,
                    "water_l": 2.0,
                    "crew_min": 6.0,
                    "label_source": "simulated",
                    "label_weight": 0.7,
                },
            )
        ]

    monkeypatch.setattr(model_training, "_generate_samples", _fake_samples)

    df = model_training.build_training_dataframe(n_samples=2, seed=7)

    assert samples_called == [(2, 7)]
    assert len(df) == 1
    row = df.iloc[0]
    assert row["rigidez"] == pytest.approx(0.5)
    assert row["label_source"] == "simulated"
    assert row["label_weight"] == pytest.approx(0.7)


@pytest.mark.parametrize(
    "label_sources, expected",
    [
        ([], "synthetic_v0"),
        (["simulated"], "synthetic_v0"),
        (["mission"], "gold_v1"),
        (["measured", "mission"], "gold_v1"),
        (["simulated", "mission"], "hybrid_v1"),
    ],
)
def test_infer_trained_on_label(label_sources: list[str], expected: str) -> None:
    """Training provenance is derived from the label_source distribution."""

    df = pd.DataFrame({"label_source": label_sources})
    result = model_training._infer_trained_on_label(df)
    assert result == expected
def test_lookup_labels_returns_measured_metadata(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    picks, process, weights = _sample_inputs()
    recipe_id = label_mapper.derive_recipe_id(picks, process)

    labels_path = tmp_path / "labels.parquet"
    labels_row = {
        "recipe_id": recipe_id,
        "process_id": "P01",
        "rigidez": 0.88,
        "estanqueidad": 0.74,
        "energy_kwh": 92.0,
        "water_l": 4.5,
        "crew_min": 48.0,
        "tightness_pass": 1,
        "rigidity_level": 3,
        "label_weight": 2.0,
        "provenance": "measured",
        "conf_lo_rigidez": 0.8,
        "conf_hi_rigidez": 0.92,
    }
    _write_parquet(pd.DataFrame([labels_row]), labels_path)

    monkeypatch.setattr(label_mapper, "GOLD_LABELS_PATH", labels_path)
    monkeypatch.setattr(model_training, "GOLD_LABELS_PATH", labels_path)

    targets, metadata = label_mapper.lookup_labels(picks, "P01", {"recipe_id": recipe_id})
    assert targets["rigidez"] == pytest.approx(0.88)
    assert metadata["provenance"] == "measured"
    ci_rigidez = metadata["confidence_intervals"]["rigidez"]
    assert ci_rigidez[0] == pytest.approx(0.8)
    assert ci_rigidez[1] == pytest.approx(0.92)

    features = generator.compute_feature_vector(picks, weights, process, 0.0)
    features["recipe_id"] = recipe_id
    result = model_training._compute_targets(picks, process, features)
    assert result["rigidez"] == pytest.approx(0.88)
    assert result["estanqueidad"] == pytest.approx(0.74)
    assert result["label_source"] == "measured"


def test_compute_targets_fallback_without_curated_data(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    picks, process, weights = _sample_inputs()
    missing_path = tmp_path / "missing.parquet"
    monkeypatch.setattr(label_mapper, "GOLD_LABELS_PATH", missing_path)
    monkeypatch.setattr(model_training, "GOLD_LABELS_PATH", missing_path)

    features = generator.compute_feature_vector(picks, weights, process, 0.0)
    result = model_training._compute_targets(picks, process, features)

    assert result["label_source"] == "simulated"
    tightness = model_training.TIGHTNESS_SCORE_MAP[result["tightness_pass"]]
    assert result["estanqueidad"] == pytest.approx(tightness)
