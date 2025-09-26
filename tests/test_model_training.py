"""Tests for :mod:`app.modules.model_training`."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.modules import generator, label_mapper, model_training


@pytest.fixture(autouse=True)
def _reset_gold_caches() -> None:
    """Ensure cached gold datasets do not leak across tests."""

    original_features_cache = model_training._GOLD_FEATURES_CACHE
    original_features_path = model_training._GOLD_FEATURES_CACHE_PATH
    original_targets_cache = model_training._GOLD_TARGETS_CACHE
    original_targets_path = model_training._GOLD_TARGETS_CACHE_PATH
    original_label_cache = label_mapper._LABELS_CACHE
    original_label_path = label_mapper._LABELS_CACHE_PATH
    model_training._GOLD_FEATURES_CACHE = None
    model_training._GOLD_FEATURES_CACHE_PATH = None
    model_training._GOLD_TARGETS_CACHE = None
    model_training._GOLD_TARGETS_CACHE_PATH = None
    label_mapper._LABELS_CACHE = None
    label_mapper._LABELS_CACHE_PATH = None
    try:
        yield
    finally:
        model_training._GOLD_FEATURES_CACHE = original_features_cache
        model_training._GOLD_FEATURES_CACHE_PATH = original_features_path
        model_training._GOLD_TARGETS_CACHE = original_targets_cache
        model_training._GOLD_TARGETS_CACHE_PATH = original_targets_path
        label_mapper._LABELS_CACHE = original_label_cache
        label_mapper._LABELS_CACHE_PATH = original_label_path


def _write_parquet(data: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data.to_parquet(path, index=False)


def _make_gold_frames(count: int = 3) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return synthetic gold features/labels matching the expected schema."""

    feature_rows: list[dict[str, float | str]] = []
    label_rows: list[dict[str, float | str | int]] = []
    for idx in range(count):
        process_id = f"P{idx + 1:02d}"
        recipe_id = f"REC-{idx + 1:02d}"
        feature_row: dict[str, float | str] = {}
        for column in model_training.FEATURE_COLUMNS:
            if column == "process_id":
                feature_row[column] = process_id
            else:
                feature_row[column] = float(idx + 1)
        feature_row["recipe_id"] = recipe_id
        feature_rows.append(feature_row)

        label_row: dict[str, float | str | int] = {
            "recipe_id": recipe_id,
            "process_id": process_id,
            "label_source": "mission",
            "label_weight": 2.5,
            "provenance": "mission",
            "tightness_pass": 1,
            "rigidity_level": 3,
        }
        for offset, target in enumerate(model_training.TARGET_COLUMNS):
            label_row[target] = float(idx + offset + 1)
        label_rows.append(label_row)

    return pd.DataFrame(feature_rows), pd.DataFrame(label_rows)


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


def test_build_training_dataframe_uses_nasa_gold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The curated NASA dataset should be merged verbatim into the training frame."""

    features_df, labels_df = _make_gold_frames()
    monkeypatch.setattr(
        model_training,
        "_load_gold_features",
        lambda path=None: features_df.copy(),
    )
    monkeypatch.setattr(
        model_training,
        "_load_gold_targets",
        lambda path=None: labels_df.copy(),
    )

    def _fail_generate(*_: object, **__: object) -> None:  # pragma: no cover - guard
        raise AssertionError("_generate_samples should not be called when gold artefacts exist")

    monkeypatch.setattr(model_training, "_generate_samples", _fail_generate)

    df = model_training.build_training_dataframe()

    produced = df.set_index(["recipe_id", "process_id"]).sort_index()
    expected = labels_df.set_index(["recipe_id", "process_id"]).sort_index()

    assert len(produced) == len(expected)
    for column in model_training.TARGET_COLUMNS + ["tightness_pass", "rigidity_level"]:
        np.testing.assert_allclose(
            produced[column].to_numpy(dtype=float),
            expected[column].to_numpy(dtype=float),
            rtol=1e-6,
            atol=1e-6,
        )
    assert set(produced["label_source"]) == {"mission"}


def test_build_training_dataframe_uses_default_gold_dataset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the default gold artefacts exist sampling must be skipped."""

    features_df, labels_df = _make_gold_frames()
    monkeypatch.setattr(
        model_training,
        "_load_gold_features",
        lambda path=None: features_df.copy(),
    )
    monkeypatch.setattr(
        model_training,
        "_load_gold_targets",
        lambda path=None: labels_df.copy(),
    )

    def _fail_generate(*_: object, **__: object) -> None:  # pragma: no cover - guard
        raise AssertionError("_generate_samples should not run when gold artefacts are present")

    monkeypatch.setattr(model_training, "_generate_samples", _fail_generate)

    df = model_training.build_training_dataframe()

    assert not df.empty
    assert set(df["label_source"].str.lower()) == {"mission"}


def test_build_training_dataframe_falls_back_when_labels_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Synthetic samples are generated when consolidated labels are unavailable."""

    features_df, _ = _make_gold_frames(count=1)
    monkeypatch.setattr(
        model_training,
        "_load_gold_features",
        lambda path=None: features_df.copy(),
    )
    monkeypatch.setattr(
        model_training,
        "_load_gold_targets",
        lambda path=None: pd.DataFrame(),
    )

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


def test_compute_targets_prefers_curated_labels(monkeypatch: pytest.MonkeyPatch) -> None:
    """When a curated recipe exists the mission-provided targets must be used."""

    curated_targets = {
        "rigidez": 0.91,
        "estanqueidad": 0.47,
        "energy_kwh": 128.0,
        "water_l": 7.5,
        "crew_min": 42.0,
        "tightness_pass": 1,
        "rigidity_level": 3,
    }
    curated_meta = {"label_source": "mission", "provenance": "mission", "label_weight": 2.5}

    monkeypatch.setattr(
        model_training,
        "lookup_labels",
        lambda *args, **kwargs: (curated_targets, curated_meta),
    )

    picks, process, _ = _sample_inputs()
    features = {column: 0.0 for column in model_training.FEATURE_COLUMNS}
    features["process_id"] = process["process_id"]
    features["recipe_id"] = label_mapper.derive_recipe_id(picks, process)

    targets = model_training._compute_targets(picks, process, features)

    for key, value in curated_targets.items():
        assert targets[key] == pytest.approx(value)
    assert targets["label_source"] == "mission"


def test_compute_targets_uses_default_gold_dataset(monkeypatch: pytest.MonkeyPatch) -> None:
    """La tabla gold por defecto debe dominar los cálculos de etiquetas."""

    curated_targets = {
        "rigidez": 0.6,
        "estanqueidad": 0.55,
        "energy_kwh": 85.0,
        "water_l": 12.0,
        "crew_min": 33.0,
        "tightness_pass": 1,
        "rigidity_level": 2,
    }
    curated_meta = {"label_source": "mission", "provenance": "mission", "label_weight": 1.8}

    monkeypatch.setattr(
        model_training,
        "lookup_labels",
        lambda *args, **kwargs: (curated_targets, curated_meta),
    )

    picks, process, _ = _sample_inputs()
    features = {column: 0.1 for column in model_training.FEATURE_COLUMNS}
    features["process_id"] = process["process_id"]
    features["recipe_id"] = label_mapper.derive_recipe_id(picks, process)

    targets = model_training._compute_targets(picks, process, features)

    for column in model_training.TARGET_COLUMNS:
        assert targets[column] == pytest.approx(curated_targets[column])

    assert targets["label_source"] == "mission"


def test_infer_trained_on_label_with_gold_dataset() -> None:
    """The curated mission dataset implies gold provenance for training."""

    _, labels_df = _make_gold_frames()
    table = labels_df.set_index(["recipe_id", "process_id"])

    result = model_training._infer_trained_on_label(table)
    assert result == "gold_v1"

    mixed = table.copy()
    if not mixed.empty:
        mixed.loc[mixed.index[:1], "label_source"] = "simulated"
    assert model_training._infer_trained_on_label(mixed) == "hybrid_v1"


@pytest.mark.parametrize(
    "label_sources, expected",
    [
        ([], "synthetic_v0"),
        (["simulated"], "synthetic_v0"),
        (["weak"], "synthetic_v0"),
        (["weakly_supervised"], "synthetic_v0"),
        (["mission"], "gold_v1"),
        (["measured", "mission"], "gold_v1"),
        (["simulated", "mission"], "hybrid_v1"),
        (["feedback"], "hil_v1"),
        (["feedback", "mission"], "hil_v1"),
        (["feedback", "simulated"], "hybrid_v2"),
        (["feedback", "simulated", "mission"], "hybrid_v2"),
    ],
)
def test_infer_trained_on_label(label_sources: list[str], expected: str) -> None:
    """Training provenance is derived from the label_source distribution."""

    df = pd.DataFrame({"label_source": label_sources})
    result = model_training._infer_trained_on_label(df)
    assert result == expected


def test_cli_respects_custom_gold_paths(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """La interfaz CLI propaga las rutas indicadas hacia el pipeline de entrenamiento."""

    gold_dir = tmp_path / "gold"
    features_dir = tmp_path / "alt_features"
    features_path = features_dir / "features.parquet"
    labels_path = gold_dir / "labels.parquet"

    _write_parquet(pd.DataFrame([{column: 0.0 for column in model_training.FEATURE_COLUMNS}]), features_path)
    _write_parquet(pd.DataFrame([{column: 0.0 for column in model_training.TARGET_COLUMNS}]), labels_path)

    captured: dict[str, object] = {}
    expected_logs = pd.DataFrame({"dummy": [1]})

    def _fake_load_feedback_logs(patterns: object) -> pd.DataFrame:
        captured["append_patterns"] = patterns
        return expected_logs

    def _fake_train_and_save(**kwargs: object) -> dict[str, str]:
        captured.update(kwargs)
        return {"status": "ok"}

    monkeypatch.setattr(model_training, "load_feedback_logs", _fake_load_feedback_logs)
    monkeypatch.setattr(model_training, "train_and_save", _fake_train_and_save)

    result = model_training.cli(
        [
            "--gold",
            str(gold_dir),
            "--features",
            str(features_dir),
            "--samples",
            "3",
            "--seed",
            "17",
            "--append-logs",
            "logs/*.parquet",
        ]
    )

    assert result == {"status": "ok"}
    assert captured["n_samples"] == 3
    assert captured["seed"] == 17
    assert captured["gold_features_path"] == features_path
    assert captured["gold_labels_path"] == labels_path
    assert captured["append_patterns"] == ["logs/*.parquet"]
    assert captured["feedback_logs"] is expected_logs


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


@pytest.mark.skipif(
    not (
        model_training.HAS_LIGHTGBM
        and model_training.HAS_SKL2ONNX
        and model_training.HAS_ONNX
    ),
    reason="LightGBM/ONNX optional dependencies not available",
)
def test_train_lightgbm_gpu_exports_onnx(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rng = np.random.default_rng(0)
    rows: list[dict[str, float | str | int]] = []

    for idx in range(48):
        row: dict[str, float | str | int] = {"process_id": f"P{idx % 4:02d}"}
        for column in model_training.FEATURE_COLUMNS:
            if column == "process_id":
                continue
            row[column] = float(rng.uniform(0.0, 1.0))

        row["total_mass_kg"] = float(rng.uniform(5.0, 25.0))
        row["mass_input_kg"] = float(max(row["total_mass_kg"] - rng.uniform(0.0, 4.0), 1.0))
        row["num_items"] = float(rng.integers(3, 12))
        row["difficulty_index"] = float(rng.uniform(0.1, 0.9))
        row["moisture_frac"] = float(rng.uniform(0.0, 0.6))
        row["gas_recovery_index"] = float(rng.uniform(0.0, 1.0))

        rigidez = float(np.clip(0.35 + 0.45 * row["difficulty_index"] - 0.25 * row["moisture_frac"], 0.0, 1.0))
        estanqueidad = float(
            np.clip(0.4 + 0.4 * row["gas_recovery_index"] - 0.2 * row["moisture_frac"], 0.0, 1.0)
        )
        energy = float(25.0 + 55.0 * row["total_mass_kg"] + 12.0 * row["difficulty_index"])
        water = float(6.0 + 48.0 * row["moisture_frac"] + 15.0 * row["hydrogen_rich_frac"])
        crew = float(9.0 + 32.0 * row["difficulty_index"] + 4.0 * row["num_items"])

        row["rigidez"] = rigidez
        row["estanqueidad"] = estanqueidad
        row["energy_kwh"] = energy
        row["water_l"] = water
        row["crew_min"] = crew
        row["tightness_pass"] = 1 if estanqueidad > 0.6 else 0
        level = int(np.clip(round(rigidez * 2) + 1, 1, 3))
        row["rigidity_level"] = level
        row["label_source"] = "simulated"
        row["label_weight"] = 1.0
        rows.append(row)

    df = pd.DataFrame(rows)
    pipeline, *_ = model_training._train_random_forest(df, seed=0)

    onnx_path = tmp_path / "models" / "rexai_lightgbm.onnx"
    monkeypatch.setattr(model_training, "LIGHTGBM_ONNX_PATH", onnx_path)

    payload = model_training._train_lightgbm_gpu(pipeline, df, seed=0)

    assert payload.get("backend") in {"gpu", "cpu"}
    overall = payload.get("metrics", {}).get("overall", {})
    assert overall, "Expected overall metrics from LightGBM training"
    assert overall.get("mae", 1.0) < 5.0
    assert onnx_path.exists(), "ONNX artefact was not written to disk"
    path_label = payload.get("path")
    assert path_label is not None
    assert Path(path_label).name == onnx_path.name
    assert payload.get("format") == "onnx"


def test_compute_targets_relabels_weak_sources(monkeypatch: pytest.MonkeyPatch) -> None:
    picks, process, weights = _sample_inputs()
    features = generator.compute_feature_vector(picks, weights, process, 0.0)

    def _fake_lookup_labels(*_: object, **__: object) -> tuple[dict[str, float], dict[str, str]]:
        return (
            {"rigidez": 0.61},
            {"provenance": "weak", "label_source": "weak"},
        )

    monkeypatch.setattr(model_training, "lookup_labels", _fake_lookup_labels)

    result = model_training._compute_targets(picks, process, features)

    assert result["label_source"] == "simulated"
    assert result.get("provenance") == "weak"


def test_cli_appends_feedback_logs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    start_time = datetime.now(model_training.UTC)

    gold_dir = tmp_path / "gold"
    features_path = gold_dir / "features.parquet"
    labels_path = gold_dir / "labels.parquet"
    features_df, labels_df = _make_gold_frames()
    _write_parquet(features_df, features_path)
    _write_parquet(labels_df, labels_path)

    feedback_path = tmp_path / "feedback" / "human_feedback.parquet"
    feedback_row = {column: 1.0 for column in model_training.FEATURE_COLUMNS}
    feedback_row["process_id"] = "PFBK"
    feedback_row.update(
        {
            "rigidity_ok": False,
            "tightness_ok": True,
            "energy_kwh": 10.0,
            "energy_penalty": 2.5,
            "water_l": 5.0,
            "water_penalty": 1.25,
            "crew_min": 3.0,
            "crew_penalty": 0.75,
            "label_weight": 1.8,
        }
    )
    _write_parquet(pd.DataFrame([feedback_row]), feedback_path)

    models_dir = tmp_path / "models"
    processed_dir = tmp_path / "processed"
    processed_ml_dir = tmp_path / "processed_ml"
    monkeypatch.setattr(model_training, "MODEL_DIR", models_dir)
    monkeypatch.setattr(model_training, "PROCESSED_DIR", processed_dir)
    monkeypatch.setattr(model_training, "PROCESSED_ML", processed_ml_dir)
    monkeypatch.setattr(model_training, "GOLD_FEATURES_PATH", features_path)
    monkeypatch.setattr(model_training, "GOLD_LABELS_PATH", labels_path)
    monkeypatch.setattr(label_mapper, "GOLD_LABELS_PATH", labels_path)
    monkeypatch.setattr(
        model_training,
        "DATASET_PATH",
        processed_dir / "rexai_training_dataset.parquet",
    )
    monkeypatch.setattr(
        model_training,
        "DATASET_ML_PATH",
        processed_ml_dir / "synthetic_runs.parquet",
    )
    monkeypatch.setattr(
        model_training, "PIPELINE_PATH", models_dir / "rexai_regressor.joblib"
    )
    monkeypatch.setattr(
        model_training, "AUTOENCODER_PATH", models_dir / "rexai_autoencoder.pt"
    )
    monkeypatch.setattr(
        model_training, "XGBOOST_PATH", models_dir / "rexai_xgboost.joblib"
    )
    monkeypatch.setattr(
        model_training, "TABTRANSFORMER_PATH", models_dir / "rexai_tabtransformer.pt"
    )
    monkeypatch.setattr(
        model_training, "TIGHTNESS_MODEL_PATH", models_dir / "rexai_class_tightness.joblib"
    )
    monkeypatch.setattr(
        model_training, "RIGIDITY_MODEL_PATH", models_dir / "rexai_class_rigidity.joblib"
    )
    monkeypatch.setattr(
        model_training, "METADATA_PATH", models_dir / "metadata_gold.json"
    )
    monkeypatch.setattr(
        model_training, "LEGACY_METADATA_PATH", models_dir / "metadata.json"
    )
    monkeypatch.setattr(model_training, "_train_classifiers", lambda *_, **__: {})

    result = model_training.cli(
        [
            "--samples",
            "8",
            "--seed",
            "3",
            "--append-logs",
            str(feedback_path),
        ]
    )

    dataset = pd.read_parquet(model_training.DATASET_PATH)
    appended = dataset[dataset["process_id"] == "PFBK"]
    assert len(appended) == 1
    appended_row = appended.iloc[0]
    assert appended_row["rigidez"] == pytest.approx(model_training.RIGIDITY_SCORE_MAP[1])
    assert appended_row["rigidity_level"] == pytest.approx(1)
    assert appended_row["estanqueidad"] == pytest.approx(
        model_training.TIGHTNESS_SCORE_MAP[1]
    )
    assert appended_row["tightness_pass"] == pytest.approx(1)
    assert appended_row["energy_kwh"] == pytest.approx(12.5)
    assert appended_row["water_l"] == pytest.approx(6.25)
    assert appended_row["crew_min"] == pytest.approx(3.75)
    assert appended_row["label_source"] == "feedback"
    assert str(appended_row.get("provenance", "")).startswith("feedback:")

    metadata = json.loads(model_training.METADATA_PATH.read_text(encoding="utf-8"))
    trained_at = datetime.fromisoformat(metadata["trained_at"])
    assert trained_at >= start_time
    assert metadata["trained_on"] == "hil_v1"
    assert result["trained_on"] == "hil_v1"
