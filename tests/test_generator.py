from __future__ import annotations

import json
import numbers
import random
import shutil
import time
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

from deltalake import DeltaTable
from typing import Any, Dict

import pandas as pd
import pytest
import numpy as np
import pyarrow.parquet as pq

from app.modules import data_sources, generator, logging_utils

pl = generator.pl


@pytest.fixture
def reference_dataset_tables(monkeypatch):
    datasets_root = generator.DATASETS_ROOT
    dataset_map = {
        "summary": datasets_root / "nasa_waste_summary.csv",
        "processing": datasets_root / "nasa_waste_processing_products.csv",
        "leo": datasets_root / "nasa_leo_mass_savings.csv",
        "propellant": datasets_root / "nasa_propellant_benefits.csv",
    }

    tables = {prefix: pl.read_csv(path) for prefix, path in dataset_map.items()}

    l2l_params = data_sources.load_l2l_parameters()
    namespace = SimpleNamespace(
        constants=getattr(l2l_params, "constants", {}),
        category_features=getattr(l2l_params, "category_features", {}),
        item_features=getattr(l2l_params, "item_features", {}),
        hints=getattr(l2l_params, "hints", {}),
    )

    generator._official_features_bundle.cache_clear()
    monkeypatch.setattr(generator, "_L2L_PARAMETERS", namespace)

    try:
        yield tables
    finally:
        generator._official_features_bundle.cache_clear()


def test_append_inference_log_reuses_daily_writer(monkeypatch, tmp_path):
    generator._INFERENCE_LOG_MANAGER.close()
    monkeypatch.setattr(generator, "LOGS_ROOT", tmp_path)
    manager = generator._InferenceLogWriterManager()
    monkeypatch.setattr(generator, "_INFERENCE_LOG_MANAGER", manager)

    created_paths: list[Path] = []
    write_calls: list[tuple[Path, int]] = []

    class WriterSpy:
        def __init__(self, path: str, schema: Any) -> None:  # pragma: no cover - test helper
            self.path = Path(path)
            self.schema = schema
            self.write_count = 0

        def write_table(self, table: Any) -> None:  # pragma: no cover - test helper
            self.write_count += 1
            write_calls.append((self.path, self.write_count))

        def close(self) -> None:  # pragma: no cover - test helper
            pass

    def fake_writer(path: str, schema: Any) -> WriterSpy:
        writer = WriterSpy(path, schema)
        created_paths.append(writer.path)
        return writer

    monkeypatch.setattr(generator.pq, "ParquetWriter", fake_writer)

    day = datetime(2024, 5, 4, 12, 0, tzinfo=UTC)
    events = [
        (
            day,
            {
                "timestamp": day.isoformat(),
                "input_features": "{}",
                "prediction": "{}",
                "uncertainty": "{}",
                "model_hash": "abc",
            },
        ),
        (
            day.replace(hour=18),
            {
                "timestamp": day.replace(hour=18).isoformat(),
                "input_features": "{\"foo\": 1}",
                "prediction": "{\"bar\": 2}",
                "uncertainty": "{}",
                "model_hash": "def",
            },
        ),
    ]

    def fake_prepare(*args: Any, **kwargs: Any) -> tuple[datetime, Dict[str, str | None]]:
        ts, payload = events.pop(0)
        return ts, payload

    monkeypatch.setattr(generator, "_prepare_inference_event", fake_prepare)

    generator._append_inference_log({}, {}, {}, None)
    generator._append_inference_log({}, {}, {}, None)

    assert len(created_paths) == 1
    expected_path = (
        tmp_path
        / "inference"
        / "20240504"
        / "inference_20240504.parquet"
    )
    assert created_paths[0] == expected_path
    assert [count for _, count in write_calls] == [1, 2]

    manager.close()


def test_append_inference_log_rotates_daily(monkeypatch, tmp_path):
    generator._INFERENCE_LOG_MANAGER.close()
    monkeypatch.setattr(generator, "LOGS_ROOT", tmp_path)
    manager = generator._InferenceLogWriterManager()
    monkeypatch.setattr(generator, "_INFERENCE_LOG_MANAGER", manager)

    created_paths: list[Path] = []
    closed_paths: list[Path] = []

    class WriterSpy:
        def __init__(self, path: str, schema: Any) -> None:  # pragma: no cover - test helper
            self.path = Path(path)

        def write_table(self, table: Any) -> None:  # pragma: no cover - test helper
            pass

        def close(self) -> None:  # pragma: no cover - test helper
            closed_paths.append(self.path)

    def fake_writer(path: str, schema: Any) -> WriterSpy:
        writer = WriterSpy(path, schema)
        created_paths.append(writer.path)
        return writer

    monkeypatch.setattr(generator.pq, "ParquetWriter", fake_writer)

    day_one = datetime(2024, 5, 4, 23, 0, tzinfo=UTC)
    day_two = datetime(2024, 5, 5, 0, 5, tzinfo=UTC)
    events = [
        (
            day_one,
            {
                "timestamp": day_one.isoformat(),
                "input_features": "{}",
                "prediction": "{}",
                "uncertainty": "{}",
                "model_hash": None,
            },
        ),
        (
            day_two,
            {
                "timestamp": day_two.isoformat(),
                "input_features": "{}",
                "prediction": "{}",
                "uncertainty": "{}",
                "model_hash": None,
            },
        ),
    ]

    def fake_prepare(*args: Any, **kwargs: Any) -> tuple[datetime, Dict[str, str | None]]:
        ts, payload = events.pop(0)
        return ts, payload

    monkeypatch.setattr(generator, "_prepare_inference_event", fake_prepare)

    generator._append_inference_log({}, {}, {}, None)
    assert len(created_paths) == 1
    assert closed_paths == []

    generator._append_inference_log({}, {}, {}, None)
    assert len(created_paths) == 2
    assert closed_paths == [
        tmp_path / "inference" / "20240504" / "inference_20240504.parquet"
    ]

    manager.close()
    assert closed_paths[-1] == (
        tmp_path / "inference" / "20240505" / "inference_20240505.parquet"
    )
def _batched_feature_vectors(
    picks: pd.DataFrame,
    weights: list[float],
    process: pd.Series,
    regolith_pct: float,
    repeat: int = 2,
) -> list[dict]:
    picks_list = [picks.copy(deep=True) for _ in range(repeat)]
    weights_list = [list(weights) for _ in range(repeat)]
    process_list = [process.copy(deep=True) for _ in range(repeat)]
    regolith_list = [regolith_pct for _ in range(repeat)]
    tensor_batch = generator.build_feature_tensor_batch(
        picks_list,
        weights_list,
        process_list,
        regolith_list,
    )
    batched = generator.compute_feature_vector(tensor_batch)
    assert isinstance(batched, list)
    return batched


def test_load_waste_summary_data_polars(tmp_path, monkeypatch):
    summary_path = tmp_path / "nasa_waste_summary.csv"
    summary_path.write_text(
        "\n".join(
            [
                "category,subitem,Artemis_mass_kg,Gateway_mass_kg",
                "Packaging,Foam,10,5",
                "Packaging,,3,2",
            ]
        )
        + "\n"
    )

    monkeypatch.setattr(
        data_sources,
        "resolve_dataset_path",
        lambda name: summary_path if name == "nasa_waste_summary.csv" else None,
    )

    summary = data_sources._load_waste_summary_data()

    assert summary.mission_totals["artemis"] == pytest.approx(13.0)
    assert summary.mission_totals["gateway"] == pytest.approx(7.0)
    assert summary.mass_by_key["packaging"]["artemis"] == pytest.approx(13.0)
    assert summary.mass_by_key["packaging|foam"]["gateway"] == pytest.approx(5.0)


def test_extract_grouped_metrics_polars(tmp_path, monkeypatch):
    metrics_path = tmp_path / "nasa_waste_processing_products.csv"
    metrics_path.write_text(
        "\n".join(
            [
                "mission,scenario,value_kg",
                "Lunar,Nominal,10",
                "Lunar,Contingency,14",
            ]
        )
        + "\n"
    )

    monkeypatch.setattr(
        data_sources,
        "resolve_dataset_path",
        lambda name: metrics_path if name == "nasa_waste_processing_products.csv" else None,
    )

    aggregated = data_sources.extract_grouped_metrics(
        "nasa_waste_processing_products.csv", "processing"
    )

    assert aggregated["lunar"]["processing_value_kg"] == pytest.approx(12.0)
    assert aggregated["nominal"]["processing_value_kg"] == pytest.approx(10.0)
    assert aggregated["lunar_contingency"]["processing_value_kg"] == pytest.approx(14.0)


def test_official_features_bundle_polars_pipeline(tmp_path, monkeypatch):
    official_path = tmp_path / "official.csv"
    official_path.write_text(
        "\n".join(
            [
                "category,subitem,value_kg",
                "Packaging,Foam,2.5",
                "Packaging,,1.0",
                "Other Packaging/Gloves (B),Nitrile Gloves,4.5",
            ]
        )
        + "\n"
    )

    summary_path = tmp_path / "nasa_waste_summary.csv"
    summary_path.write_text(
        "\n".join(
            [
                "category,subitem,Artemis_mass_kg,Gateway_mass_kg",
                "Packaging,Foam,10,5",
                "Packaging,,3,2",
            ]
        )
        + "\n"
    )

    processing_path = tmp_path / "nasa_waste_processing_products.csv"
    processing_path.write_text(
        "\n".join(
            [
                "category,subitem,output_kg",
                "Packaging,Foam,2",
                "Packaging,Foam,4",
            ]
        )
        + "\n"
    )

    leo_path = tmp_path / "nasa_leo_mass_savings.csv"
    leo_path.write_text(
        "\n".join(
            [
                "category,subitem,savings_pct",
                "Packaging,Foam,12",
            ]
        )
        + "\n"
    )

    propellant_path = tmp_path / "nasa_propellant_benefits.csv"
    propellant_path.write_text(
        "\n".join(
            [
                "category,subitem,benefit",
                "Packaging,Foam,5",
            ]
        )
        + "\n"
    )

    file_map = {
        "nasa_waste_summary.csv": summary_path,
        "nasa_waste_processing_products.csv": processing_path,
        "nasa_leo_mass_savings.csv": leo_path,
        "nasa_propellant_benefits.csv": propellant_path,
    }

    monkeypatch.setattr(data_sources, "_OFFICIAL_FEATURES_PATH", official_path)
    monkeypatch.setattr(data_sources, "resolve_dataset_path", lambda name: file_map.get(name))
    data_sources.official_features_bundle.cache_clear()

    bundle = data_sources.official_features_bundle()

    assert "value_kg" in bundle.value_columns
    assert "processing_output_kg" in bundle.value_columns
    assert "leo_savings_pct" in bundle.value_columns
    assert "propellant_benefit" in bundle.value_columns
    idx = bundle.direct_map["packaging|foam"]
    assert "other packaging|nitrile glove" in bundle.direct_map

    generator._official_features_bundle.cache_clear()
    monkeypatch.setattr(generator, "_OFFICIAL_FEATURES_PATH", official_path)
    monkeypatch.setattr(generator, "_resolve_dataset_path", lambda name: file_map.get(name))
    vector_bundle = generator._official_features_bundle()
    vector_idx = vector_bundle.direct_map["packaging|foam"]
    glove_idx = vector_bundle.direct_map["gloves|nitrile glove"]
    payload = generator._build_payload_from_row(
        vector_bundle.value_matrix[vector_idx], vector_bundle.value_columns
    )
    assert payload["value_kg"] == pytest.approx(2.5)
    assert "category_norm" in bundle.table.columns
    assert "subitem_norm" in bundle.table.columns
    assert bundle.mission_totals["artemis"] == pytest.approx(13.0)
    assert bundle.processing_metrics["processing"]["processing_output_kg"] == pytest.approx(3.0)
    tokens, match_keys, indices = vector_bundle.category_tokens["packaging"]
    match_keys_array = match_keys.tolist() if hasattr(match_keys, "tolist") else list(match_keys)
    assert "packaging|foam" in match_keys_array
    indices_array = indices.tolist() if hasattr(indices, "tolist") else list(indices)
    assert vector_idx in indices_array

    glove_tokens, glove_match_keys, glove_indices = vector_bundle.category_tokens["gloves"]
    glove_match_array = (
        glove_match_keys.tolist() if hasattr(glove_match_keys, "tolist") else list(glove_match_keys)
    )
    assert "gloves|nitrile glove" in glove_match_array
    glove_indices_array = (
        glove_indices.tolist() if hasattr(glove_indices, "tolist") else list(glove_indices)
    )
    assert glove_idx in glove_indices_array

    data_sources.official_features_bundle.cache_clear()
    generator._official_features_bundle.cache_clear()


def test_vectorized_feature_map_benchmark():
    rows = 4000
    categories = ["Packaging" if idx % 2 == 0 else "Food Packaging" for idx in range(rows)]
    subitems = [f"Item {idx}" for idx in range(rows)]
    values = [float(idx % 17 + 1) for idx in range(rows)]

    table_df = pl.DataFrame(
        {
            "category": categories,
            "subitem": subitems,
            "value_kg": values,
        }
    ).with_columns(
        pl.col("category")
        .map_elements(generator._normalize_category, return_dtype=pl.String)
        .alias("category_norm"),
        pl.col("subitem")
        .map_elements(generator._normalize_item, return_dtype=pl.String)
        .alias("subitem_norm"),
    ).with_columns(
        pl.when(pl.col("subitem_norm").str.len_bytes() > 0)
        .then(pl.col("category_norm") + pl.lit("|") + pl.col("subitem_norm"))
        .otherwise(pl.col("category_norm"))
        .alias("key"),
    )

    excluded = {"category", "subitem", "category_norm", "subitem_norm", "key"}
    value_columns = tuple(col for col in table_df.columns if col not in excluded)

    def baseline_builder():
        direct_map: Dict[str, Dict[str, float]] = {}
        category_tokens: Dict[str, list[tuple[frozenset[str], Dict[str, float], str]]] = {}
        for row in table_df.to_dicts():
            key = row["key"]
            if not key:
                continue
            payload = {col: float(row[col]) for col in value_columns}
            direct_map[key] = payload
            category_tokens.setdefault(row["category_norm"], []).append(
                (generator._token_set(row["subitem_norm"]), payload, key)
            )
        return direct_map, category_tokens

    baseline_map, _ = baseline_builder()
    vector_map, _, value_matrix = generator._vectorized_feature_maps(table_df, value_columns)

    assert set(baseline_map.keys()) == set(vector_map.keys())
    sample_keys = list(baseline_map.keys())[:10]
    for key in sample_keys:
        idx = vector_map[key]
        payload = generator._build_payload_from_row(value_matrix[idx], value_columns)
        assert payload == baseline_map[key]

    def measure(fn):
        best = float("inf")
        for _ in range(3):
            start = time.perf_counter()
            fn()
            elapsed = time.perf_counter() - start
            best = min(best, elapsed)
        return best

    baseline_time = measure(baseline_builder)
    vectorized_time = measure(lambda: generator._vectorized_feature_maps(table_df, value_columns))

    assert vectorized_time <= baseline_time * 1.1


class DummyRegistry:
    ready = True
    metadata = {"model_hash": "dummy-hash"}

    def predict(self, features):
        return {
            "rigidez": 0.8,
            "estanqueidad": 0.7,
            "energy_kwh": 3.5,
            "water_l": 1.2,
            "crew_min": 15.0,
            "uncertainty": {"rigidez": 0.05},
            "comparisons": {
                "lightgbm_gpu": {
                    "rigidez": 0.78,
                    "estanqueidad": 0.69,
                    "energy_kwh": 3.4,
                    "water_l": 1.1,
                    "crew_min": 14.8,
                }
            },
        }

    def embed(self, features):
        return [0.1, 0.2, 0.3]


def _dummy_process_series() -> pd.Series:
    return pd.Series(
        {
            "process_id": "P01",
            "energy_kwh_per_kg": 1.0,
            "water_l_per_kg": 0.5,
            "crew_min_per_batch": 30.0,
        }
    )


def _collect_single_log_dir(root: Path) -> Path:
    log_root = root / "inference"
    assert log_root.exists(), "Expected inference log directory to be created"
    day_dirs = list(log_root.iterdir())
    assert len(day_dirs) == 1, f"Expected a single log directory, found {day_dirs}"
    return day_dirs[0]


def _read_inference_log(day_dir: Path) -> pd.DataFrame:
    files = sorted(day_dir.glob("*.parquet"))
    assert files, "Expected at least one Parquet log shard"
    frames = [pq.read_table(file).to_pandas() for file in files]
    return pd.concat(frames, ignore_index=True, sort=False)


def test_generate_candidates_uses_parallel_backend(monkeypatch):
    monkeypatch.setattr(generator, "_PARALLEL_THRESHOLD", 1)

    created: list[object] = []

    class DummyExecutor:
        def __init__(self, max_workers: int):
            self.max_workers = max_workers
            self.map_calls = 0
            self.shutdown_called = False
            created.append(self)

        def map(self, func, iterable):
            self.map_calls += 1
            return [func(item) for item in iterable]

        def shutdown(self):
            self.shutdown_called = True

    monkeypatch.setattr(generator, "ThreadPoolExecutor", DummyExecutor)
    monkeypatch.setattr(generator, "prepare_waste_frame", lambda df: df)
    monkeypatch.setattr(generator, "_pick_materials", lambda df, rng, n=2, bias=2.0: pd.DataFrame(
        {
            "kg": [1.0],
            "_source_id": ["A"],
            "_source_category": ["packaging"],
            "_source_flags": [""],
            "_problematic": [0],
            "material": ["foil"],
            "category": ["packaging"],
            "flags": [""],
            "_problematic": [0],
        }
    ))
    monkeypatch.setattr(generator, "lookup_labels", lambda *args, **kwargs: ([], {}))
    monkeypatch.setattr(
        generator,
        "heuristic_props",
        lambda *args, **kwargs: pd.DataFrame(
            {
                "kg": [1.0],
                "pct_mass": [100.0],
                "pct_volume": [100.0],
                "moisture_pct": [0.0],
                "difficulty_factor": [1.0],
                "density_kg_m3": [1.0],
                "category": ["packaging"],
                "flags": [""],
                "_problematic": [0],
                "_source_id": ["A"],
                "_source_category": ["packaging"],
                "_source_flags": [""],
                "material": ["foil"],
            }
        ),
    )
    monkeypatch.setattr(generator, "lookup_labels", lambda *args, **kwargs: ({}, {}))

    draws: list[float] = []

    def fake_build(picks, proc_df, rng, target, crew_time_low, use_ml, tuning):
        value = rng.random()
        draws.append(value)
        return {
            "score": value,
            "props": generator.PredProps(
                rigidity=1.0,
                tightness=1.0,
                mass_final_kg=1.0,
                energy_kwh=0.1,
                water_l=0.1,
                crew_min=1.0,
            ),
        }

    monkeypatch.setattr(generator, "_build_candidate", fake_build)
    base_random_cls = random.Random
    monkeypatch.setattr(generator.random, "Random", lambda *args, **kwargs: base_random_cls(1234))

    monkeypatch.setattr(generator, "lookup_labels", lambda *args, **kwargs: ({}, {}))

    waste_df = pd.DataFrame(
        {
            "material": ["foil"],
            "kg": [1.0],
            "_problematic": [0],
            "_source_id": ["A"],
            "_source_category": ["packaging"],
            "_source_flags": [""],
            "category": ["packaging"],
            "flags": [""],
        }
    )
    proc_df = pd.DataFrame(
        {
            "process_id": ["P01"],
            "name": ["Process"],
            "energy_kwh_per_kg": [1.0],
            "water_l_per_kg": [0.5],
            "crew_min_per_batch": [30.0],
        }
    )
    proc_df = pd.DataFrame(
        {
            "process_id": ["P01"],
            "name": ["Process"],
            "energy_kwh_per_kg": [1.0],
            "water_l_per_kg": [0.5],
            "crew_min_per_batch": [30.0],
        }
    )

    candidates, history = generator.generate_candidates(
        waste_df,
        proc_df,
        target={},
        n=5,
        crew_time_low=False,
        optimizer_evals=0,
        use_ml=False,
    )

    assert len(candidates) == 5
    scores = [cand["score"] for cand in candidates]
    assert scores == sorted(scores, reverse=True)
    assert len(draws) == 5
    assert created and created[0].map_calls >= 1
    assert created[0].shutdown_called is True
    assert history.empty


def test_append_inference_log_appends_without_reads(monkeypatch, tmp_path):
    monkeypatch.setattr(logging_utils, "LOGS_ROOT", tmp_path)
    monkeypatch.setattr(generator, "LOGS_ROOT", tmp_path)
    generator._close_inference_log_writer()

    log_root = tmp_path / "inference"
    if log_root.exists():
        shutil.rmtree(log_root)

    for idx in range(2):
        logging_utils.append_inference_log(
            input_features={"feature": idx},
            prediction={"score": idx},
            uncertainty=None,
            model_registry=None,
        )

    generator._close_inference_log_writer()

    log_dir = _collect_single_log_dir(tmp_path)
    table = _read_inference_log(log_dir)
    assert len(table) == 2


def test_append_inference_log_handles_schema_evolution(monkeypatch, tmp_path):
    monkeypatch.setattr(logging_utils, "LOGS_ROOT", tmp_path)
    monkeypatch.setattr(generator, "LOGS_ROOT", tmp_path)
    generator._close_inference_log_writer()

    log_root = tmp_path / "inference"
    if log_root.exists():
        shutil.rmtree(log_root)

    logging_utils.append_inference_log(
        input_features={"feature": 0},
        prediction={"score": 0},
        uncertainty=None,
        model_registry=None,
    )

    original_prepare = logging_utils.prepare_inference_event

    def prepare_with_session(*args, **kwargs):
        timestamp, payload = original_prepare(*args, **kwargs)
        updated = dict(payload)
        updated["session_id"] = "alpha"
        return timestamp, updated

    monkeypatch.setattr(logging_utils, "prepare_inference_event", prepare_with_session)

    logging_utils.append_inference_log(
        input_features={"feature": 1},
        prediction={"score": 1},
        uncertainty=None,
        model_registry=None,
    )

    generator._close_inference_log_writer()

    log_dir = _collect_single_log_dir(tmp_path)
    log_df = _read_inference_log(log_dir)
    assert "session_id" in log_df.columns
    assert log_df["session_id"].isna().sum() == 1
    assert set(log_df["session_id"].dropna()) == {"alpha"}


def test_generate_candidates_appends_inference_log(monkeypatch, tmp_path):
    monkeypatch.setattr(generator, "MODEL_REGISTRY", DummyRegistry())
    monkeypatch.setattr(logging_utils, "LOGS_ROOT", tmp_path)
    monkeypatch.setattr(generator, "lookup_labels", lambda *args, **kwargs: ({}, {}))
    generator._close_inference_log_writer()

    log_root = tmp_path / "inference"
    if log_root.exists():
        shutil.rmtree(log_root)

    waste_df = pd.DataFrame(
        {
            "id": ["W1", "W2", "W3"],
            "category": ["packaging", "eva", "metal"],
            "material": ["plastic", "foam", "aluminum"],
            "kg": [1.0, 2.0, 0.5],
            "volume_l": [10.0, 5.0, 2.5],
            "flags": ["", "ctb", ""],
        }
    )
    proc_df = pd.DataFrame(
        {
            "process_id": ["P01"],
            "name": ["Demo"],
            "energy_kwh_per_kg": [1.0],
            "water_l_per_kg": [0.5],
            "crew_min_per_batch": [30.0],
        }
    )

    candidates, history = generator.generate_candidates(waste_df, proc_df, target={}, n=1)

    assert candidates, "Expected at least one candidate to be generated"
    assert history.empty

    generator._close_inference_log_writer()

    log_dir = _collect_single_log_dir(tmp_path)
    log_df = _read_inference_log(log_dir).sort_values("timestamp")
    for column in ["timestamp", "input_features", "prediction", "uncertainty", "model_hash"]:
        assert column in log_df.columns

    last_event = log_df.iloc[-1]
    assert last_event["model_hash"] == "dummy-hash"

    prediction_payload = json.loads(last_event["prediction"])
    assert prediction_payload["rigidez"] == 0.8

    uncertainty_payload = json.loads(last_event["uncertainty"])
    assert "rigidez" in uncertainty_payload

    cand = candidates[0]
    breakdown = cand.get("score_breakdown")
    assert isinstance(breakdown, dict)
    assert "contributions" in breakdown
    assert "auxiliary" in breakdown
    assert pytest.approx(breakdown.get("total", cand["score"]), rel=1e-2) == cand["score"]
    auxiliary = cand.get("auxiliary")
    assert isinstance(auxiliary, dict)
    assert "passes_seal" in auxiliary
    assert "lightgbm_gpu" in cand.get("model_variants", {})

    shutil.rmtree(log_dir.parent, ignore_errors=True)


def test_generate_candidates_heuristic_mode_skips_ml(monkeypatch, tmp_path):
    calls: list[str] = []

    class NoCallRegistry:
        ready = True
        metadata = {"model_hash": "noop"}

        def predict(self, features):
            calls.append("predict")
            return {}

        def embed(self, features):
            return []

    monkeypatch.setattr(generator, "MODEL_REGISTRY", NoCallRegistry())
    monkeypatch.setattr(logging_utils, "LOGS_ROOT", tmp_path)
    monkeypatch.setattr(generator, "LOGS_ROOT", tmp_path)
    generator._close_inference_log_writer()

    log_root = tmp_path / "inference"
    if log_root.exists():
        shutil.rmtree(log_root)
    monkeypatch.setattr(generator, "lookup_labels", lambda *args, **kwargs: ({}, {}))

    log_dir = logging_utils.LOGS_ROOT
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"inference_{datetime.utcnow().strftime('%Y%m%d')}.parquet"
    log_path.unlink(missing_ok=True)
    waste_df = pd.DataFrame(
        {
            "id": ["W1", "W2"],
            "category": ["packaging", "eva"],
            "material": ["plastic", "foam"],
            "kg": [1.0, 2.0],
            "volume_l": [10.0, 5.0],
            "flags": ["", "ctb"],
        }
    )
    proc_df = pd.DataFrame(
        {
            "process_id": ["P01"],
            "name": ["Demo"],
            "energy_kwh_per_kg": [1.0],
            "water_l_per_kg": [0.5],
            "crew_min_per_batch": [30.0],
        }
    )

    candidates, history = generator.generate_candidates(
        waste_df, proc_df, target={}, n=1, use_ml=False
    )

    assert candidates, "Expected heuristic candidate even when ML disabled"
    assert history.empty
    assert not calls, "ML predict should not be invoked in heuristic mode"
    assert (not log_root.exists()) or (not any(log_root.iterdir())), (
        "Inference log should not be created in heuristic mode"
    )

    cand = candidates[0]
    assert "score_breakdown" in cand
    assert "auxiliary" in cand


def test_prepare_waste_frame_direct_match_overrides_official_fields():
    waste_df = pd.DataFrame(
        {
            "id": ["W1"],
            "category": ["Foam Packaging"],
            "material": ["Zotek F30 (PVDF foam)"],
            "kg": [10.0],
            "volume_l": [0.0],
            "flags": [""],
        }
    )

    prepared = generator.prepare_waste_frame(waste_df)
    row = prepared.iloc[0]

    assert pytest.approx(row["difficulty_factor"], rel=1e-6) == 3.0
    assert pytest.approx(row["PVDF_pct"], rel=1e-6) == 100.0
    assert pytest.approx(row["moisture_pct"], rel=1e-6) == 0.0
    assert pytest.approx(row["density_kg_m3"], rel=1e-2) == 100.0


def test_prepare_waste_frame_includes_reference_columns(reference_dataset_tables):
    expected_columns: set[str] = set()
    for prefix, table in reference_dataset_tables.items():
        for column in table.columns:
            if column in {"category", "subitem"}:
                continue
            expected_columns.add(f"{prefix}_{generator._slugify(column)}")

    waste_df = pd.DataFrame(
        {
            "id": ["W_summary"],
            "category": ["Fabrics"],
            "material": ["Clothing"],
            "kg": [5.0],
            "volume_l": [2.0],
            "flags": [""],
        }
    )

    prepared = generator.prepare_waste_frame(waste_df)
    missing = expected_columns - set(prepared.columns)

    assert not missing, f"Missing reference columns: {sorted(missing)}"

    summary_df = reference_dataset_tables["summary"].to_pandas()
    clothing_row = summary_df[
        (summary_df["category"] == "Fabrics")
        & (summary_df["subitem"] == "Clothing")
    ]
    assert not clothing_row.empty
    expected_total = float(clothing_row["total_mass_kg"].iloc[0])

    assert prepared.loc[0, "summary_total_mass_kg"] == pytest.approx(
        expected_total, rel=1e-6
    )


def test_prepare_waste_frame_token_match_applies_composition():
    waste_df = pd.DataFrame(
        {
            "id": ["W2"],
            "category": ["Food Packaging"],
            "material": ["Rehydratable Pouch"],
            "kg": [5.0],
            "volume_l": [0.0],
            "flags": [""],
        }
    )

    prepared = generator.prepare_waste_frame(waste_df)
    row = prepared.iloc[0]

    assert pytest.approx(row["Nylon_pct"], rel=1e-6) == 41.0
    assert pytest.approx(row["EVOH_pct"], rel=1e-6) == 11.0
    assert pytest.approx(row["Polyethylene_pct"], rel=1e-6) == 33.0
    assert pytest.approx(row["moisture_pct"], rel=1e-6) == 4.0
    assert pytest.approx(row["density_kg_m3"], rel=1e-2) == 100.0


def test_compute_feature_vector_blends_official_and_keyword_sources():
    waste_df = pd.DataFrame(
        {
            "id": ["A", "B"],
            "category": ["Food Packaging", "Unknown"],
            "material": ["Rehydratable Pouch", "High density polyethylene liner"],
            "kg": [7.0, 3.0],
            "volume_l": [0.0, 4.0],
            "flags": ["", ""],
        }
    )

    prepared = generator.prepare_waste_frame(waste_df)
    process = _dummy_process_series()
    features = generator.compute_feature_vector(
        prepared,
        [0.7, 0.3],
        process,
        regolith_pct=0.0,
    )

    assert features["polyethylene_frac"] > 0.2
    assert features["gas_recovery_index"] > 0.0
    assert features["moisture_frac"] == pytest.approx(0.028, rel=1e-6)

    batched = _batched_feature_vectors(prepared, [0.7, 0.3], process, 0.0)
    assert len(batched) == 2
    for candidate in batched:
        assert candidate["polyethylene_frac"] == pytest.approx(features["polyethylene_frac"], rel=1e-6)
        assert candidate["gas_recovery_index"] == pytest.approx(features["gas_recovery_index"], rel=1e-6)
        assert candidate["moisture_frac"] == pytest.approx(features["moisture_frac"], rel=1e-6)


def test_compute_feature_vector_keyword_fallback_triggers_polyethylene():
    waste_df = pd.DataFrame(
        {
            "id": ["C"],
            "category": ["Unknown"],
            "material": ["High density polyethylene film"],
            "kg": [5.0],
            "volume_l": [5.0],
            "flags": [""],
        }
    )

    prepared = generator.prepare_waste_frame(waste_df)
    process = _dummy_process_series()
    features = generator.compute_feature_vector(
        prepared,
        [1.0],
        process,
        regolith_pct=0.0,
    )

    assert features["polyethylene_frac"] > 0.5
    assert features["gas_recovery_index"] > 0.0

    batched = _batched_feature_vectors(prepared, [1.0], process, 0.0)
    assert len(batched) == 2
    for candidate in batched:
        assert candidate["polyethylene_frac"] == pytest.approx(features["polyethylene_frac"], rel=1e-6)
        assert candidate["gas_recovery_index"] == pytest.approx(features["gas_recovery_index"], rel=1e-6)


def test_compute_feature_vector_dataframe_matches_tensor_batch():
    waste_df = pd.DataFrame(
        {
            "id": ["X", "Y", "Z"],
            "category": ["Food Packaging", "Food Packaging", "Logistics"],
            "material": [
                "Rehydratable Pouch",
                "Nomex shipping bag",
                "Polyethylene foam block",
            ],
            "kg": [7.0, 2.0, 4.0],
            "volume_l": [0.0, 1.0, 8.0],
            "flags": ["", "multilayer", ""],
        }
    )

    prepared = generator.prepare_waste_frame(waste_df)
    process = _dummy_process_series()
    weights = [0.5, 0.3, 0.2]
    regolith_pct = 0.15

    dataframe_features = generator.compute_feature_vector(
        prepared, weights, process, regolith_pct
    )

    tensor_batch = generator.build_feature_tensor_batch(
        [prepared], [weights], [process], [regolith_pct]
    )
    tensor_features = generator.compute_feature_vector(tensor_batch)

    assert isinstance(tensor_features, list) and tensor_features, "Tensor batch returned no features"
    tensor_features = tensor_features[0]

    assert set(tensor_features) == set(dataframe_features)
    for key, value in dataframe_features.items():
        lhs = tensor_features[key]
        if isinstance(value, numbers.Real):
            assert lhs == pytest.approx(value, rel=1e-6, abs=1e-8)
        else:
            assert lhs == value


def test_compute_feature_vector_accepts_polars_dataframe():
    waste_df = pd.DataFrame(
        {
            "id": ["A", "B"],
            "category": ["Packaging", "Tools"],
            "material": ["Polyethylene film", "Aluminum Wrench"],
            "kg": [4.0, 1.5],
            "volume_l": [2.0, 0.5],
            "flags": ["", ""],
        }
    )

    prepared = generator.prepare_waste_frame(waste_df)
    polars_prepared = pl.from_pandas(prepared)
    process = _dummy_process_series()
    weights = np.array([0.7, 0.3], dtype=float)
    regolith_pct = 0.05

    pandas_features = generator.compute_feature_vector(
        prepared, weights, process, regolith_pct
    )
    polars_features = generator.compute_feature_vector(
        polars_prepared, weights, process, regolith_pct
    )

    assert set(polars_features) == set(pandas_features)
    for key, value in pandas_features.items():
        lhs = polars_features[key]
        if isinstance(value, numbers.Real):
            assert lhs == pytest.approx(value, rel=1e-6, abs=1e-8)
        else:
            assert lhs == value


def test_compute_feature_vectors_batch_matches_individual():
    waste_a = pd.DataFrame(
        {
            "id": ["A1", "A2"],
            "category": ["Packaging", "Logistics"],
            "material": ["Polyethylene wrap", "Nomex bag"],
            "kg": [4.0, 3.0],
            "volume_l": [6.0, 2.0],
            "flags": ["", "multilayer"],
        }
    )
    waste_b = pd.DataFrame(
        {
            "id": ["B1"],
            "category": ["Foam"],
            "material": ["Closed cell foam"],
            "kg": [2.5],
            "volume_l": [4.0],
            "flags": [""],
        }
    )

    prepared_a = generator.prepare_waste_frame(waste_a)
    prepared_b = generator.prepare_waste_frame(waste_b)
    process_a = _dummy_process_series()
    process_b = _dummy_process_series().copy(deep=True)
    process_b["process_id"] = "P02"

    weights_a = [0.6, 0.4]
    weights_b = [1.0]
    regolith_a = 0.1
    regolith_b = 0.0

    batched = generator.compute_feature_vectors_batch(
        [prepared_a, prepared_b],
        [weights_a, weights_b],
        [process_a, process_b],
        [regolith_a, regolith_b],
    )
    singles = [
        generator.compute_feature_vector(prepared_a, weights_a, process_a, regolith_a),
        generator.compute_feature_vector(prepared_b, weights_b, process_b, regolith_b),
    ]

    assert len(batched) == len(singles)
    for combined, expected in zip(batched, singles, strict=True):
        assert set(combined) == set(expected)
        for key, value in expected.items():
            lhs = combined[key]
            if isinstance(value, numbers.Real):
                assert lhs == pytest.approx(value, rel=1e-6, abs=1e-8)
            else:
                assert lhs == value


def test_compute_feature_vector_includes_mission_metrics(monkeypatch):
    # Ensure cached bundles from other tests do not leak.
    data_sources.official_features_bundle.cache_clear()
    generator._official_features_bundle.cache_clear()

    match_key = "food packaging|rehydratable pouch"
    dummy_matrix = np.array([[1.0]], dtype=np.float64)
    mass_by_key = {
        match_key: {"gateway_i": 200.0},
        "food packaging": {"gateway_i": 300.0},
    }
    mission_totals = {"gateway_i": 1000.0}
    (
        mission_reference_keys,
        mission_reference_index,
        mission_reference_matrix,
        mission_names,
        mission_totals_vector,
    ) = generator._build_mission_reference_tables(mass_by_key, mission_totals)

    dummy_bundle = generator._OfficialFeaturesBundle(
        value_columns=("dummy_col",),
        composition_columns=(),
        direct_map={match_key: 0},
        category_tokens={
            "food packaging": (
                np.array([frozenset({"rehydratable", "pouch"})], dtype=object),
                np.array([match_key], dtype=object),
                np.array([0], dtype=np.int32),
            )
        },
        table=pl.DataFrame(
            {
                "category_norm": ["food packaging"],
                "subitem_norm": ["rehydratable pouch"],
                "dummy_col": [1.0],
            }
        ),
        value_matrix=dummy_matrix,
        mission_mass=mass_by_key,
        mission_totals=mission_totals,
        mission_reference_keys=mission_reference_keys,
        mission_reference_index=mission_reference_index,
        mission_reference_matrix=mission_reference_matrix,
        mission_names=mission_names,
        mission_totals_vector=mission_totals_vector,
        processing_metrics={"gateway_i": {"processing_o2_ch4_yield_kg": 5.0}},
        leo_mass_savings={"gateway_i": {"leo_mass_savings_kg": 120.0}},
        propellant_benefits={"gateway_i": {"propellant_delta_v_m_s": 35.0}},
        l2l_constants={},
        l2l_category_features={},
        l2l_item_features={},
        l2l_hints={},
    )

    def fake_bundle():
        return dummy_bundle

    fake_bundle.cache_clear = lambda: None  # type: ignore[attr-defined]
    monkeypatch.setattr(generator, "official_features_bundle", fake_bundle)
    monkeypatch.setattr(generator, "_official_features_bundle", fake_bundle)

    waste_df = pd.DataFrame(
        {
            "id": ["M1"],
            "category": ["Food Packaging"],
            "material": ["Rehydratable Pouch"],
            "kg": [10.0],
            "volume_l": [5.0],
        }
    )

    prepared = generator.prepare_waste_frame(waste_df)
    process = _dummy_process_series()
    features = generator.compute_feature_vector(prepared, [1.0], process, regolith_pct=0.0)

    assert features["mission_similarity_gateway_i"] == pytest.approx(0.2, rel=1e-6)
    assert features["mission_reference_mass_gateway_i"] == pytest.approx(200.0, rel=1e-6)
    assert features["mission_scaled_mass_gateway_i"] == pytest.approx(2.0, rel=1e-6)
    assert features["mission_official_mass_gateway_i"] == pytest.approx(200.0, rel=1e-6)
    assert features["mission_similarity_total"] == pytest.approx(0.2, rel=1e-6)

    # Aggregated NASA references should appear as weighted expectations.
    assert features["processing_o2_ch4_yield_kg_gateway_i"] == pytest.approx(5.0, rel=1e-6)
    assert features["processing_o2_ch4_yield_kg_expected"] == pytest.approx(1.0, rel=1e-6)
    assert features["leo_mass_savings_kg_expected"] == pytest.approx(24.0, rel=1e-6)
    assert features["propellant_delta_v_m_s_expected"] == pytest.approx(7.0, rel=1e-6)

    batched = _batched_feature_vectors(prepared, [1.0], process, 0.0)
    assert len(batched) == 2
    for candidate in batched:
        assert candidate["mission_similarity_gateway_i"] == pytest.approx(features["mission_similarity_gateway_i"], rel=1e-6)
        assert candidate["mission_reference_mass_gateway_i"] == pytest.approx(features["mission_reference_mass_gateway_i"], rel=1e-6)
        assert candidate["mission_scaled_mass_gateway_i"] == pytest.approx(features["mission_scaled_mass_gateway_i"], rel=1e-6)
        assert candidate["mission_official_mass_gateway_i"] == pytest.approx(features["mission_official_mass_gateway_i"], rel=1e-6)
        assert candidate["mission_similarity_total"] == pytest.approx(features["mission_similarity_total"], rel=1e-6)
        assert candidate["processing_o2_ch4_yield_kg_gateway_i"] == pytest.approx(features["processing_o2_ch4_yield_kg_gateway_i"], rel=1e-6)
        assert candidate["processing_o2_ch4_yield_kg_expected"] == pytest.approx(features["processing_o2_ch4_yield_kg_expected"], rel=1e-6)
        assert candidate["leo_mass_savings_kg_expected"] == pytest.approx(features["leo_mass_savings_kg_expected"], rel=1e-6)
        assert candidate["propellant_delta_v_m_s_expected"] == pytest.approx(features["propellant_delta_v_m_s_expected"], rel=1e-6)


def test_prepare_waste_frame_handles_malformed_l2l(monkeypatch):
    data_sources.official_features_bundle.cache_clear()
    generator._official_features_bundle.cache_clear()
    monkeypatch.setattr(generator, "_L2L_PARAMETERS", {})

    def broken_loader() -> dict:
        return {}

    monkeypatch.setattr(generator, "_load_l2l_parameters", broken_loader)

    waste_df = pd.DataFrame(
        {
            "category": ["Packaging"],
            "material": ["Foam"],
            "kg": [1.0],
        }
    )

    prepared = generator.prepare_waste_frame(waste_df)
    assert isinstance(prepared, pd.DataFrame)
    assert not prepared.empty


def test_prepare_waste_frame_injects_l2l_features(monkeypatch):
    data_sources.official_features_bundle.cache_clear()
    generator._official_features_bundle.cache_clear()

    match_key = "food packaging|rehydratable pouch"
    dummy_matrix = np.array([[1.0]], dtype=np.float64)
    empty_reference = (
        generator.sparse.csr_matrix((0, 0), dtype=np.float64)
        if generator.sparse is not None
        else np.zeros((0, 0), dtype=np.float64)
    )
    dummy_bundle = generator._OfficialFeaturesBundle(
        value_columns=("dummy_col",),
        composition_columns=(),
        direct_map={match_key: 0},
        category_tokens={
            "food packaging": (
                np.array([frozenset({"rehydratable", "pouch"})], dtype=object),
                np.array([match_key], dtype=object),
                np.array([0], dtype=np.int32),
            )
        },
        table=pl.DataFrame(
            {
                "category_norm": ["food packaging"],
                "subitem_norm": ["rehydratable pouch"],
                "dummy_col": [1.0],
            }
        ),
        value_matrix=dummy_matrix,
        mission_mass={},
        mission_totals={},
        mission_reference_keys=(),
        mission_reference_index={},
        mission_reference_matrix=empty_reference,
        mission_names=(),
        mission_totals_vector=np.zeros(0, dtype=np.float64),
        processing_metrics={},
        leo_mass_savings={},
        propellant_benefits={},
        l2l_constants={},
        l2l_category_features={"food packaging": {"l2l_geometry_panel_area_m2": 5.0}},
        l2l_item_features={match_key: {"l2l_ops_random_access_required": 1.0}},
        l2l_hints={
            "l2l_geometry_panel_area_m2": "p.42",
            "l2l_ops_random_access_required": "p.15",
        },
    )

    def fake_bundle():
        return dummy_bundle

    fake_bundle.cache_clear = lambda: None  # type: ignore[attr-defined]
    monkeypatch.setattr(generator, "official_features_bundle", fake_bundle)
    monkeypatch.setattr(generator, "_official_features_bundle", fake_bundle)

    waste_df = pd.DataFrame(
        {
            "id": ["M2"],
            "category": ["Food Packaging"],
            "material": ["Rehydratable Pouch"],
            "kg": [12.0],
            "volume_l": [6.0],
        }
    )

    prepared = generator.prepare_waste_frame(waste_df)
    row = prepared.iloc[0]

    assert "dummy_col" in row.index
    assert pytest.approx(row["dummy_col"], rel=1e-6) == 1.0
    assert pytest.approx(row["l2l_geometry_panel_area_m2"], rel=1e-6) == 5.0
    assert pytest.approx(row["l2l_ops_random_access_required"], rel=1e-6) == 1.0
    assert "_l2l_page_hints" in row.index
    assert "p.42" in row["_l2l_page_hints"]
    assert "p.15" in row["_l2l_page_hints"]


def test_prepare_waste_frame_vectorized_large_inventory():
    rng = np.random.default_rng(42)
    size = 2048

    category_choices = np.array(
        [
            "Food Packaging",
            "Foam Packaging",
            "EVA Waste",
            "Other Packaging Glove",
        ]
    )
    material_choices = np.array(
        [
            "Polyethylene Film",
            "Nomex Fabric",
            "Aluminum Panel",
            "Zotek Foam",
            "Nitrile Glove",
        ]
    )
    family_choices = np.array(
        [
            "PE-PET-AL Laminate",
            "Closed_cell Foam",
            "Polyester Textile",
            "Nylon Mesh",
            "Nomex Weave",
        ]
    )
    flag_choices = np.array(["", "multilayer", "closed_cell", "ctb", "wipe"])

    kg = rng.uniform(0.1, 25.0, size).round(4)
    volume_l = rng.uniform(0.0, 50.0, size).round(4)
    zero_volume_mask = rng.random(size) < 0.35
    volume_l[zero_volume_mask] = 0.0

    category_mass = rng.uniform(5.0, 120.0, size).round(4)
    category_volume = rng.uniform(0.05, 5.0, size).round(4)
    zero_category_volume = rng.random(size) < 0.4
    category_volume[zero_category_volume] = 0.0

    data = {
        "id": [f"W{idx}" for idx in range(size)],
        "category": rng.choice(category_choices, size=size),
        "material": rng.choice(material_choices, size=size),
        "material_family": rng.choice(family_choices, size=size),
        "flags": rng.choice(flag_choices, size=size),
        "kg": kg,
        "volume_l": volume_l,
        "category_total_mass_kg": category_mass,
        "category_total_volume_m3": category_volume,
    }

    for column in generator._COMPOSITION_DENSITY_MAP:
        values = rng.uniform(0.0, 100.0, size).round(3)
        values[rng.random(size) < 0.45] = 0.0
        data[column] = values

    waste_df = pd.DataFrame(data)

    prepared = generator.prepare_waste_frame(waste_df)
    assert prepared.shape[0] == size

    expected_tokens = (
        prepared["material"].astype(str).str.lower()
        + " "
        + prepared["category"].astype(str).str.lower()
        + " "
        + prepared["flags"].astype(str).str.lower()
        + " "
        + prepared["key_materials"].astype(str).str.lower()
    )
    pd.testing.assert_series_equal(prepared["tokens"], expected_tokens, check_names=False)

    expected_problematic = prepared.apply(generator._is_problematic, axis=1)
    pd.testing.assert_series_equal(
        prepared["_problematic"],
        expected_problematic.astype(bool),
        check_names=False,
    )

    mass_series = pd.to_numeric(prepared["kg"], errors="coerce").fillna(0.0)
    volume_series = pd.to_numeric(prepared["volume_l"], errors="coerce") / 1000.0
    expected_density = mass_series.divide(volume_series).where(
        (volume_series > 0) & volume_series.notna()
    )

    missing = expected_density.isna() | ~np.isfinite(expected_density)
    if missing.any():
        fallback = prepared.loc[missing].apply(generator._estimate_density_from_row, axis=1)
        expected_density.loc[missing] = fallback

    default_density = float(
        generator._CATEGORY_DENSITY_DEFAULTS.get("packaging", 500.0)
    )
    expected_density = expected_density.fillna(default_density).clip(lower=20.0, upper=4000.0)

    pd.testing.assert_series_equal(
        prepared["density_kg_m3"],
        expected_density,
        check_names=False,
    )


def test_compute_feature_vector_uses_l2l_packaging_ratio(monkeypatch):
    data_sources.official_features_bundle.cache_clear()
    generator._official_features_bundle.cache_clear()

    dummy_matrix = np.empty((0, 0), dtype=np.float64)
    empty_reference = (
        generator.sparse.csr_matrix((0, 0), dtype=np.float64)
        if generator.sparse is not None
        else np.zeros((0, 0), dtype=np.float64)
    )
    dummy_bundle = generator._OfficialFeaturesBundle(
        value_columns=("dummy_col",),
        composition_columns=(),
        direct_map={},
        category_tokens={},
        table=pl.DataFrame(
            schema={
                "category_norm": pl.Utf8,
                "subitem_norm": pl.Utf8,
                "dummy_col": pl.Float64,
            },
            data=[],
        ),
        value_matrix=dummy_matrix,
        mission_mass={},
        mission_totals={},
        mission_reference_keys=(),
        mission_reference_index={},
        mission_reference_matrix=empty_reference,
        mission_names=(),
        mission_totals_vector=np.zeros(0, dtype=np.float64),
        processing_metrics={},
        leo_mass_savings={},
        propellant_benefits={},
        l2l_constants={"l2l_logistics_packaging_per_goods_ratio": 0.2},
        l2l_category_features={},
        l2l_item_features={},
        l2l_hints={},
    )

    def fake_bundle():
        return dummy_bundle

    fake_bundle.cache_clear = lambda: None  # type: ignore[attr-defined]
    monkeypatch.setattr(generator, "official_features_bundle", fake_bundle)
    monkeypatch.setattr(generator, "_official_features_bundle", fake_bundle)

    waste_df = pd.DataFrame(
        {
            "id": ["P1"],
            "category": ["Packaging"],
            "material": ["Polyethylene wrap"],
            "kg": [5.0],
            "volume_l": [8.0],
            "flags": [""],
        }
    )

    prepared = generator.prepare_waste_frame(waste_df)
    process = _dummy_process_series()
    features = generator.compute_feature_vector(prepared, [1.0], process, regolith_pct=0.0)

    assert features["l2l_logistics_packaging_per_goods_ratio"] == pytest.approx(0.2, rel=1e-6)
    packaging_term = features.get("packaging_frac", 0.0) + 0.5 * features.get("eva_frac", 0.0)
    expected = min(2.0, packaging_term / 0.2 if 0.2 else 0.0)
    assert features["logistics_reuse_index"] == pytest.approx(expected, rel=1e-6)

    batched = _batched_feature_vectors(prepared, [1.0], process, 0.0)
    assert len(batched) == 2
    for candidate in batched:
        assert candidate["l2l_logistics_packaging_per_goods_ratio"] == pytest.approx(
            features["l2l_logistics_packaging_per_goods_ratio"], rel=1e-6
        )
        assert candidate["logistics_reuse_index"] == pytest.approx(features["logistics_reuse_index"], rel=1e-6)
