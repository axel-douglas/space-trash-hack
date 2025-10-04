from __future__ import annotations

import json
import logging
import math
import numbers
import random
import shutil
import time
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

from typing import Any, Dict

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest

from app.modules import data_sources, execution, label_mapper, logging_utils
from app.modules.generator import GeneratorService
from app.modules.generator import service as generator
from app.modules.process_planner import choose_process

pl = generator.pl


def _assert_feature_mapping_equal(lhs: Dict[str, Any], rhs: Dict[str, Any]) -> None:
    bundle = data_sources.load_material_reference_bundle()
    ignore = set(bundle.property_columns) | {"material_reference_name"}
    lhs_keys = set(lhs) - ignore
    rhs_keys = set(rhs) - ignore
    assert lhs_keys == rhs_keys
    for key in lhs_keys:
        left = lhs[key]
        right = rhs.get(key)
        if isinstance(left, numbers.Real) and isinstance(right, numbers.Real):
            if math.isnan(left) and math.isnan(right):
                continue
            assert left == pytest.approx(right, rel=1e-6, abs=1e-8)
        else:
            assert left == right


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


@pytest.fixture
def generator_service() -> GeneratorService:
    return GeneratorService()


def test_extend_category_synonyms_updates_normalization(monkeypatch):
    synonyms_copy = data_sources._CATEGORY_SYNONYMS.copy()
    monkeypatch.setattr(data_sources, "_CATEGORY_SYNONYMS", synonyms_copy)
    monkeypatch.setattr(generator, "_CATEGORY_SYNONYMS", synonyms_copy, raising=False)

    data_sources.extend_category_synonyms({"Experimental Packaging": "Packaging"})

    assert generator.normalize_category("Experimental Packaging") == "packaging"


def test_load_regolith_vector_matches_data_sources():
    polars_vector = generator._load_regolith_vector()
    pandas_vector = data_sources._load_regolith_vector()

    assert set(polars_vector) == set(pandas_vector)
    for key, expected in pandas_vector.items():
        assert polars_vector[key] == pytest.approx(expected, rel=1e-9, abs=1e-9)

    assert sum(polars_vector.values()) == pytest.approx(1.0, rel=1e-9)


def test_choose_process_filters_and_scores():
    catalog = pd.DataFrame(
        [
            {
                "process_id": "P01",
                "name": "Shredder (low RPM)",
                "crew_min_per_batch": 6,
                "energy_kwh_per_kg": 0.12,
                "water_l_per_kg": 0.0,
            },
            {
                "process_id": "P02",
                "name": "Press & Heat Lamination",
                "crew_min_per_batch": 18,
                "energy_kwh_per_kg": 0.55,
                "water_l_per_kg": 0.0,
            },
            {
                "process_id": "P03",
                "name": "Sinter with MGS-1",
                "crew_min_per_batch": 25,
                "energy_kwh_per_kg": 0.70,
                "water_l_per_kg": 0.1,
            },
            {
                "process_id": "P04",
                "name": "CTB Kit Reconfig",
                "crew_min_per_batch": 12,
                "energy_kwh_per_kg": 0.05,
                "water_l_per_kg": 0.0,
            },
        ]
    )

    result = choose_process(
        "EVA bag foam",
        catalog,
        scenario="Residence Renovations",
        crew_time_low=True,
    )

    assert list(result["process_id"]) == ["P04", "P02", "P03"]
    assert result["match_score"].is_monotonic_decreasing
    assert result.iloc[0]["process_id"] == "P04"
    assert isinstance(result.iloc[0]["match_reason"], str) and result.iloc[0]["match_reason"]


def test_append_inference_log_reuses_daily_writer(monkeypatch, tmp_path):
    logging_utils.shutdown_inference_logging()
    monkeypatch.setattr(logging_utils, "LOGS_ROOT", tmp_path)

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

    def fake_writer(path: str, schema: Any, **_kwargs: Any) -> WriterSpy:
        writer = WriterSpy(path, schema)
        created_paths.append(writer.path)
        return writer

    monkeypatch.setattr(logging_utils.pq, "ParquetWriter", fake_writer)

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

    monkeypatch.setattr(logging_utils, "prepare_inference_event", fake_prepare)

    manager = logging_utils.get_inference_log_manager()
    assert manager is logging_utils.get_inference_log_manager()

    generator.append_inference_log({}, {}, {}, None)
    generator.append_inference_log({}, {}, {}, None)

    assert len(created_paths) == 1
    expected_path = (
        tmp_path
        / "inference"
        / "20240504"
        / "inference_20240504.parquet"
    )
    assert created_paths[0] == expected_path
    assert [count for _, count in write_calls] == [1, 2]

    logging_utils.shutdown_inference_logging()


def test_append_inference_log_skips_parquet_reads(monkeypatch, tmp_path):
    """Ensure append operations never trigger a Parquet read."""

    logging_utils.shutdown_inference_logging()
    monkeypatch.setattr(logging_utils, "LOGS_ROOT", tmp_path)

    created_paths: list[Path] = []
    write_counts: list[int] = []

    class WriterSpy:
        def __init__(self, path: str, schema: Any) -> None:  # pragma: no cover - helper
            self.path = Path(path)
            self.schema = schema
            self.count = 0

        def write_table(self, table: Any) -> None:  # pragma: no cover - helper
            self.count += 1
            write_counts.append(self.count)

        def close(self) -> None:  # pragma: no cover - helper
            pass

    def fake_writer(path: str, schema: Any, **_kwargs: Any) -> WriterSpy:
        writer = WriterSpy(path, schema)
        created_paths.append(writer.path)
        return writer

    monkeypatch.setattr(logging_utils.pq, "ParquetWriter", fake_writer)

    def fail_read(*_args: Any, **_kwargs: Any) -> None:
        pytest.fail("append_inference_log should not read existing shards")

    if hasattr(logging_utils.pq, "read_table"):
        monkeypatch.setattr(logging_utils.pq, "read_table", fail_read)

    base = datetime(2024, 6, 1, 8, 30, tzinfo=UTC)
    events = [
        (
            base,
            {
                "timestamp": base.isoformat(),
                "input_features": "{}",
                "prediction": "{}",
                "uncertainty": "{}",
                "model_hash": "alpha",
            },
        ),
        (
            base.replace(hour=21),
            {
                "timestamp": base.replace(hour=21).isoformat(),
                "input_features": "{\"foo\": 1}",
                "prediction": "{\"bar\": 2}",
                "uncertainty": "{}",
                "model_hash": "bravo",
            },
        ),
    ]

    def fake_prepare(*_args: Any, **_kwargs: Any) -> tuple[datetime, Dict[str, str | None]]:
        ts, payload = events.pop(0)
        return ts, payload

    monkeypatch.setattr(logging_utils, "prepare_inference_event", fake_prepare)

    logging_utils.get_inference_log_manager()

    generator.append_inference_log({}, {}, {}, None)
    generator.append_inference_log({}, {}, {}, None)

    expected_path = (
        tmp_path / "inference" / "20240601" / "inference_20240601.parquet"
    )
    assert created_paths == [expected_path]
    assert write_counts == [1, 2]

    logging_utils.shutdown_inference_logging()


def test_append_inference_log_rotates_daily(monkeypatch, tmp_path):
    logging_utils.shutdown_inference_logging()
    monkeypatch.setattr(logging_utils, "LOGS_ROOT", tmp_path)

    created_paths: list[Path] = []
    closed_paths: list[Path] = []

    class WriterSpy:
        def __init__(self, path: str, schema: Any) -> None:  # pragma: no cover - test helper
            self.path = Path(path)

        def write_table(self, table: Any) -> None:  # pragma: no cover - test helper
            pass

        def close(self) -> None:  # pragma: no cover - test helper
            closed_paths.append(self.path)

    def fake_writer(path: str, schema: Any, **_kwargs: Any) -> WriterSpy:
        writer = WriterSpy(path, schema)
        created_paths.append(writer.path)
        return writer

    monkeypatch.setattr(logging_utils.pq, "ParquetWriter", fake_writer)

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

    monkeypatch.setattr(logging_utils, "prepare_inference_event", fake_prepare)

    logging_utils.get_inference_log_manager()

    generator.append_inference_log({}, {}, {}, None)
    assert len(created_paths) == 1
    assert closed_paths == []

    generator.append_inference_log({}, {}, {}, None)
    assert len(created_paths) == 2
    assert closed_paths == [
        tmp_path / "inference" / "20240504" / "inference_20240504.parquet"
    ]

    logging_utils.shutdown_inference_logging()
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
    monkeypatch.setattr(generator, "_load_official_features_bundle", data_sources.official_features_bundle)
    generator._official_features_bundle.cache_clear()
    vector_bundle = generator._official_features_bundle()
    vector_idx = vector_bundle.direct_map.get("packaging|foam")
    if vector_idx is None:
        token_entry = vector_bundle.category_tokens.get("packaging")
        if token_entry:
            _, key_array, indices_array = token_entry
            keys_list = key_array.tolist() if hasattr(key_array, "tolist") else list(key_array)
            idx_list = indices_array.tolist() if hasattr(indices_array, "tolist") else list(indices_array)
            for key_candidate, idx_candidate in zip(keys_list, idx_list, strict=False):
                if str(key_candidate) == "packaging|foam":
                    vector_idx = int(idx_candidate)
                    break
    assert vector_idx is not None
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


def test_official_features_bundle_material_metrics(tmp_path, monkeypatch):
    official_path = tmp_path / "official.csv"
    official_path.write_text(
        "\n".join(
            [
                "category,subitem",
                "Fabrics,Composite Panel",
                "Structural Elements,Aluminium Panel",
            ]
        )
        + "\n"
    )

    summary_path = tmp_path / "nasa_waste_summary.csv"
    summary_path.write_text(
        "\n".join(
            [
                "category,subitem,total_mass_kg",
                "Fabrics,Composite Panel,10",
                "Structural Elements,Aluminium Panel,5",
            ]
        )
        + "\n"
    )

    processing_path = tmp_path / "nasa_waste_processing_products.csv"
    processing_path.write_text("category,subitem,output_kg\nFabrics,Composite Panel,2\n")

    leo_path = tmp_path / "nasa_leo_mass_savings.csv"
    leo_path.write_text("category,subitem,savings_pct\nFabrics,Composite Panel,5\n")

    propellant_path = tmp_path / "nasa_propellant_benefits.csv"
    propellant_path.write_text("category,subitem,benefit\nFabrics,Composite Panel,1\n")

    density_path = tmp_path / "polymer_composite_density.csv"
    density_path.write_text(
        "category,subitem,sample_label,density_g_per_cm3\n"
        "Fabrics,Composite Panel,Original 0 %,1.3\n"
    )

    mechanics_path = tmp_path / "polymer_composite_mechanics.csv"
    mechanics_path.write_text(
        "category,subitem,sample_label,stress_mpa,modulus_gpa\n"
        "Fabrics,Composite Panel,Original 0 %,420,32\n"
    )

    thermal_path = tmp_path / "polymer_composite_thermal.csv"
    thermal_path.write_text(
        "category,subitem,sample_label,glass_transition_c\n"
        "Fabrics,Composite Panel,Original 0 %,125\n"
    )

    ignition_path = tmp_path / "polymer_composite_ignition.csv"
    ignition_path.write_text(
        "category,subitem,sample_label,ignition_temperature_c,burn_time_min\n"
        "Fabrics,Composite Panel,Original 0 %,780,4.5\n"
    )

    aluminium_path = tmp_path / "aluminium_alloys.csv"
    aluminium_path.write_text(
        "category,subitem,processing_route,class_id,tensile_strength_mpa,yield_strength_mpa,elongation_pct\n"
        "Structural Elements,Aluminium Panel,Solutionised + Artificially peak aged,2,650,580,15\n"
    )

    file_map = {
        "nasa_waste_summary.csv": summary_path,
        "nasa_waste_processing_products.csv": processing_path,
        "nasa_leo_mass_savings.csv": leo_path,
        "nasa_propellant_benefits.csv": propellant_path,
        "polymer_composite_density.csv": density_path,
        "polymer_composite_mechanics.csv": mechanics_path,
        "polymer_composite_thermal.csv": thermal_path,
        "polymer_composite_ignition.csv": ignition_path,
        "aluminium_alloys.csv": aluminium_path,
    }

    monkeypatch.setattr(data_sources, "_OFFICIAL_FEATURES_PATH", official_path)
    monkeypatch.setattr(data_sources, "resolve_dataset_path", lambda name: file_map.get(name))
    data_sources.official_features_bundle.cache_clear()

    bundle = data_sources.official_features_bundle()

    assert "pc_density_density_g_per_cm3" in bundle.value_columns
    assert "pc_mechanics_modulus_gpa" in bundle.value_columns
    assert "pc_thermal_glass_transition_c" in bundle.value_columns
    assert "pc_ignition_ignition_temperature_c" in bundle.value_columns
    assert "aluminium_tensile_strength_mpa" in bundle.value_columns

    assert "pc_density_original_0" in bundle.reference_metrics
    assert bundle.reference_metrics["pc_density_original_0"]["pc_density_density_g_per_cm3"] == pytest.approx(1.3)

    generator._official_features_bundle.cache_clear()
    monkeypatch.setattr(generator, "_load_official_features_bundle", data_sources.official_features_bundle)
    generator._official_features_bundle.cache_clear()

    polymer_row = pd.Series(
        {
            "category": "Fabrics",
            "material": "Composite Panel",
            "material_family": "",
            "key_materials": "",
        }
    )

    payload, match_key = generator._lookup_official_feature_values(polymer_row)
    assert match_key == "fabric|composite panel"
    assert payload["official_density_kg_m3"] == pytest.approx(1300.0)
    assert payload["official_tensile_strength_mpa"] == pytest.approx(420.0)
    assert payload["official_modulus_gpa"] == pytest.approx(32.0)
    assert payload["official_ignition_temperature_c"] == pytest.approx(780.0)

    aluminium_row = pd.Series(
        {
            "category": "Structural Elements",
            "material": "Aluminium Panel",
            "material_family": "",
            "key_materials": "",
        }
    )

    payload_al, key_al = generator._lookup_official_feature_values(aluminium_row)
    assert key_al == "structural elements|aluminium panel"
    assert payload_al["official_tensile_strength_mpa"] == pytest.approx(650.0)
    assert payload_al["official_yield_strength_mpa"] == pytest.approx(580.0)
    assert payload_al["official_elongation_pct"] == pytest.approx(15.0)

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


def test_generate_candidates_uses_parallel_backend(
    monkeypatch: pytest.MonkeyPatch, generator_service: GeneratorService
) -> None:
    monkeypatch.setattr(generator, "_PARALLEL_THRESHOLD", 1)

    class DummyBackend(execution.ExecutionBackend):
        def __init__(self):
            super().__init__(max_workers=4)
            self.map_calls = 0
            self.shutdown_called = False

        def map(self, func, iterable):
            self.map_calls += 1
            return [func(item) for item in iterable]

        def submit(self, func, *args, **kwargs):
            raise AssertionError("submit should not be called")

        def shutdown(self):
            self.shutdown_called = True

    backend = DummyBackend()
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

    def fake_build(picks, proc_df, rng, target, crew_time_low, use_ml, tuning, registry):
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

    candidates, history = generator_service.generate_candidates(
        waste_df,
        proc_df,
        target={},
        n=5,
        crew_time_low=False,
        optimizer_evals=0,
        use_ml=False,
        backend=backend,
    )

    assert len(candidates) == 5
    scores = [cand["score"] for cand in candidates]
    assert scores == sorted(scores, reverse=True)
    assert len(draws) == 5
    assert backend.map_calls >= 1
    assert backend.shutdown_called is False
    assert history.empty


def test_generate_candidates_parallel_is_deterministic(
    monkeypatch: pytest.MonkeyPatch, generator_service: GeneratorService
) -> None:
    monkeypatch.setattr(generator, "_PARALLEL_THRESHOLD", 1)
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

    def fake_build(picks, proc_df, rng, target, crew_time_low, use_ml, tuning, registry):
        value = round(rng.random(), 6)
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

    def deterministic_random(seed: int | None = None):
        if seed is None:
            seed = 1234
        return base_random_cls(seed)

    monkeypatch.setattr(generator.random, "Random", deterministic_random)

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

    def run_once() -> list[float]:
        backend = execution.ThreadPoolBackend(max_workers=4)
        try:
            candidates, _ = generator_service.generate_candidates(
                waste_df,
                proc_df,
                target={},
                n=4,
                crew_time_low=False,
                optimizer_evals=0,
                use_ml=False,
                backend=backend,
            )
        finally:
            backend.shutdown()
        return [cand["score"] for cand in candidates]

    scores_a = run_once()
    scores_b = run_once()

    assert scores_a == scores_b


def test_generate_candidates_seed_reproducible(
    monkeypatch: pytest.MonkeyPatch, generator_service: GeneratorService
) -> None:
    monkeypatch.delenv("REXAI_GENERATOR_SEED", raising=False)
    monkeypatch.setattr(generator, "_PARALLEL_THRESHOLD", 1)
    monkeypatch.setattr(generator, "prepare_waste_frame", lambda df: df)
    monkeypatch.setattr(generator, "lookup_labels", lambda *args, **kwargs: ({}, {}))
    monkeypatch.setattr(generator, "append_inference_log", lambda *args, **kwargs: None)
    monkeypatch.setattr(generator, "_create_candidate_components", lambda *args, **kwargs: None)

    def fake_pick(df, rng, n=2, bias=2.0):
        order = list(range(len(df)))
        rng.shuffle(order)
        take = max(1, min(int(n), len(order)))
        return df.iloc[order[:take]].reset_index(drop=True)

    monkeypatch.setattr(generator, "_pick_materials", fake_pick)

    def fake_build(picks, proc_df, rng, target, crew_time_low, use_ml, tuning, registry):
        value = round(rng.random(), 6)
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
            "picked": sorted(picks["_source_id"].tolist()),
        }

    monkeypatch.setattr(generator, "_build_candidate", fake_build)

    waste_df = pd.DataFrame(
        {
            "material": ["foil", "foam", "tape"],
            "kg": [1.0, 0.5, 0.75],
            "_problematic": [0, 0, 0],
            "_source_id": ["A", "B", "C"],
            "_source_category": ["packaging", "eva", "packaging"],
            "_source_flags": ["", "", ""],
            "category": ["packaging", "eva", "packaging"],
            "flags": ["", "", ""],
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

    def run_once(seed_value: int | None) -> list[float]:
        backend = execution.SynchronousBackend()
        try:
            candidates, _ = generator_service.generate_candidates(
                waste_df,
                proc_df,
                target={},
                n=4,
                crew_time_low=False,
                optimizer_evals=0,
                use_ml=False,
                backend=backend,
                seed=seed_value,
            )
        finally:
            backend.shutdown()
        return [cand["score"] for cand in candidates]

    scores_seed_a = run_once(1234)
    scores_seed_b = run_once(1234)
    assert scores_seed_a == scores_seed_b

    scores_seed_c = run_once(4321)
    assert scores_seed_c != scores_seed_a

    monkeypatch.setenv("REXAI_GENERATOR_SEED", "2024")
    scores_env_a = run_once(None)
    scores_env_b = run_once(None)
    assert scores_env_a == scores_env_b

    monkeypatch.delenv("REXAI_GENERATOR_SEED", raising=False)

def test_append_inference_log_appends_without_reads(monkeypatch, tmp_path):
    monkeypatch.setattr(logging_utils, "LOGS_ROOT", tmp_path)
    logging_utils.shutdown_inference_logging()

    log_root = tmp_path / "inference"
    if log_root.exists():
        shutil.rmtree(log_root)

    for idx in range(2):
        generator.append_inference_log(
            input_features={"feature": idx},
            prediction={"score": idx},
            uncertainty=None,
            model_registry=None,
        )

    logging_utils.shutdown_inference_logging()

    log_dir = _collect_single_log_dir(tmp_path)
    table = _read_inference_log(log_dir)
    assert len(table) == 2


def test_append_inference_log_handles_schema_evolution(monkeypatch, tmp_path):
    monkeypatch.setattr(logging_utils, "LOGS_ROOT", tmp_path)
    logging_utils.shutdown_inference_logging()

    log_root = tmp_path / "inference"
    if log_root.exists():
        shutil.rmtree(log_root)

    generator.append_inference_log(
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

    generator.append_inference_log(
        input_features={"feature": 1},
        prediction={"score": 1},
        uncertainty=None,
        model_registry=None,
    )

    logging_utils.shutdown_inference_logging()

    log_dir = _collect_single_log_dir(tmp_path)
    log_df = _read_inference_log(log_dir)
    assert "session_id" in log_df.columns
    assert log_df["session_id"].isna().sum() == 1
    assert set(log_df["session_id"].dropna()) == {"alpha"}


def test_generate_candidates_appends_inference_log(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    generator_service: GeneratorService,
) -> None:
    dummy_registry = DummyRegistry()
    monkeypatch.setattr(generator, "get_model_registry", lambda: dummy_registry)
    monkeypatch.setattr(logging_utils, "LOGS_ROOT", tmp_path)
    monkeypatch.setattr(generator, "lookup_labels", lambda *args, **kwargs: ({}, {}))
    logging_utils.shutdown_inference_logging()

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

    candidates, history = generator_service.generate_candidates(waste_df, proc_df, target={}, n=1)

    assert candidates, "Expected at least one candidate to be generated"
    assert history.empty

    logging_utils.shutdown_inference_logging()

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


def test_generate_candidates_heuristic_mode_skips_ml(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    generator_service: GeneratorService,
) -> None:
    calls: list[str] = []

    class NoCallRegistry:
        ready = True
        metadata = {"model_hash": "noop"}

        def predict(self, features):
            calls.append("predict")
            return {}

        def embed(self, features):
            return []

    no_call_registry = NoCallRegistry()

    monkeypatch.setattr(generator, "get_model_registry", lambda: no_call_registry)
    monkeypatch.setattr(logging_utils, "LOGS_ROOT", tmp_path)
    logging_utils.shutdown_inference_logging()

    log_root = tmp_path / "inference"
    shutil.rmtree(log_root, ignore_errors=True)
    monkeypatch.setattr(generator, "lookup_labels", lambda *args, **kwargs: ({}, {}))

    log_dir = logging_utils.LOGS_ROOT
    log_dir.mkdir(parents=True, exist_ok=True)
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

    candidates, history = generator_service.generate_candidates(
        waste_df, proc_df, target={}, n=1, use_ml=False
    )

    logging_utils.shutdown_inference_logging()

    assert candidates, "Expected heuristic candidate even when ML disabled"
    assert history.empty
    assert not calls, "ML predict should not be invoked in heuristic mode"
    assert (not log_root.exists()) or (not any(log_root.iterdir())), (
        "Inference log should not be created in heuristic mode"
    )

    cand = candidates[0]
    assert "score_breakdown" in cand
    assert "auxiliary" in cand


def test_generate_candidates_handles_missing_curated_labels(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    generator_service: GeneratorService,
) -> None:
    caplog.set_level(logging.WARNING, logger="app.modules.label_mapper")
    monkeypatch.setattr(label_mapper, "_LABELS_CACHE", None, raising=False)
    monkeypatch.setattr(label_mapper, "_LABELS_CACHE_PATH", None, raising=False)

    missing_labels_path = tmp_path / "missing" / "labels.parquet"
    monkeypatch.setattr(label_mapper, "GOLD_LABELS_PATH", missing_labels_path, raising=False)

    calls: list[tuple[str, tuple, dict]] = []

    def boom(*args, **kwargs):
        calls.append(("ensure", args, kwargs))
        raise RuntimeError("boom")

    monkeypatch.setattr(
        "app.modules.data_build.ensure_gold_dataset",
        boom,
    )

    picks_template = pd.DataFrame(
        {
            "kg": [1.0, 0.5],
            "_source_id": ["A", "B"],
            "_source_category": ["packaging", "eva"],
            "_source_flags": ["", ""],
            "_problematic": [0, 0],
            "material": ["aluminum foil", "eva foam"],
            "category": ["packaging", "eva"],
            "flags": ["", ""],
            "moisture_pct": [5.0, 10.0],
            "difficulty_factor": [1.0, 2.0],
        }
    )

    monkeypatch.setattr(generator, "prepare_waste_frame", lambda df: df)
    monkeypatch.setattr(
        generator,
        "_pick_materials",
        lambda df, rng, n=2, bias=2.0: picks_template.copy(),
    )
    monkeypatch.setattr(
        generator,
        "build_feature_tensor_batch",
        lambda *args, **kwargs: object(),
    )
    monkeypatch.setattr(
        generator,
        "_compute_features_from_batch",
        lambda batch: [{"process_id": "P01"}],
    )
    monkeypatch.setattr(generator, "get_model_registry", lambda: None)

    waste_df = picks_template.copy()
    proc_df = pd.DataFrame(
        {
            "process_id": ["P01"],
            "name": ["Process"],
            "energy_kwh_per_kg": [1.0],
            "water_l_per_kg": [0.5],
            "crew_min_per_batch": [30.0],
        }
    )

    candidates, history = generator_service.generate_candidates(waste_df, proc_df, target={}, n=1)

    assert calls, "ensure_gold_dataset should have been invoked"
    assert candidates, "Expected heuristic candidates even when gold labels fail"

    features = candidates[0].get("features", {})
    assert features.get("curated_label_targets") == {}
    assert features.get("curated_label_metadata") == {}
    assert features.get("prediction_mode") == "heuristic"
    assert history.empty

    assert any(
        record.levelno == logging.WARNING
        and "Failed to load curated labels" in record.getMessage()
        for record in caplog.records
    ), "Expected warning about curated label loading failure"

    cache = label_mapper._LABELS_CACHE
    assert isinstance(cache, pd.DataFrame)
    assert cache.empty
    assert label_mapper._LABELS_CACHE_PATH == missing_labels_path


def test_generate_candidates_marks_empirical_prediction(monkeypatch):
    monkeypatch.setattr(generator, "get_model_registry", lambda: None)
    monkeypatch.setattr(label_mapper, "lookup_labels", lambda *args, **kwargs: ({}, {}))
    monkeypatch.setattr(
        "app.modules.data_build.ensure_gold_dataset",
        lambda *args, **kwargs: (data_sources.DATA_ROOT / "gold" / "features.parquet", data_sources.DATA_ROOT / "gold" / "labels.parquet"),
    )

    waste_df = pd.DataFrame(
        {
            "id": ["Z1"],
            "category": ["EVA Waste"],
            "material": ["Nomex 410"],
            "kg": [8.0],
            "volume_l": [6.0],
            "moisture_pct": [2.0],
            "difficulty_factor": [1.5],
        }
    )

    proc_df = pd.DataFrame(
        {
            "process_id": ["P01"],
            "name": ["Process"],
            "energy_kwh_per_kg": [0.4],
            "water_l_per_kg": [0.2],
            "crew_min_per_batch": [20.0],
        }
    )

    service = GeneratorService()
    candidates, _ = service.generate_candidates(waste_df, proc_df, target={}, n=1, use_ml=False)
    assert candidates
    features = candidates[0]["features"]
    assert features.get("prediction_mode") == "empirical"
    assert features.get("prediction_model") == "zenodo_reference"


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
            expected_columns.add(f"{prefix}_{data_sources.slugify(column)}")

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
    expected_gateway = float(clothing_row["gateway_phase_i_mass_kg"].iloc[0])
    expected_mars = float(clothing_row["mars_transit_mass_kg"].iloc[0])

    assert prepared.loc[0, "summary_total_mass_kg"] == pytest.approx(
        expected_total, rel=1e-6
    )
    assert prepared.loc[0, "summary_gateway_phase_i_mass_kg"] == pytest.approx(
        expected_gateway, rel=1e-6
    )
    assert prepared.loc[0, "summary_mars_transit_mass_kg"] == pytest.approx(
        expected_mars, rel=1e-6
    )

    l2l_params = data_sources.load_l2l_parameters()
    constant_name = "l2l_geometry_ctb_small_volume_value"
    assert constant_name in l2l_params.constants
    assert constant_name in prepared.columns
    assert prepared.loc[0, constant_name] == pytest.approx(
        l2l_params.constants[constant_name], rel=1e-6
    )


def test_heuristic_props_prefers_material_reference():
    waste_df = pd.DataFrame(
        {
            "id": ["M1"],
            "category": ["EVA Waste"],
            "material": ["Nomex 410"],
            "kg": [12.0],
            "volume_l": [8.0],
            "moisture_pct": [3.0],
            "difficulty_factor": [2.0],
        }
    )
    prepared = generator.prepare_waste_frame(waste_df)
    proc = pd.Series(
        {
            "process_id": "P01",
            "energy_kwh_per_kg": 0.4,
            "water_l_per_kg": 0.1,
            "crew_min_per_batch": 25.0,
        }
    )

    weights = np.ones(len(prepared), dtype=float)
    props = generator.heuristic_props(prepared, proc, weights, 0.0)
    assert props.source == "zenodo_reference"
    assert props.energy_kwh > 0
    assert props.water_l > 0

    features = generator.compute_feature_vector(prepared, [1.0], proc, 0.0)
    bundle = data_sources.load_material_reference_bundle()
    for column in bundle.property_columns:
        assert column in features


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
        [prepared], [weights], [process], [regolith_pct], backend="numpy"
    )
    tensor_features = generator.compute_feature_vector(tensor_batch)

    assert isinstance(tensor_features, list) and tensor_features, "Tensor batch returned no features"
    tensor_features = tensor_features[0]

    _assert_feature_mapping_equal(dataframe_features, tensor_features)

    tensor_mapping = {
        field: getattr(tensor_batch, field)
        for field in generator.FeatureTensorBatch.__annotations__
    }
    mapping_features = generator.compute_feature_vector(tensor_mapping)
    assert isinstance(mapping_features, list) and mapping_features
    mapping_features = mapping_features[0]
    _assert_feature_mapping_equal(tensor_features, mapping_features)


def test_compute_feature_vector_emits_regolith_characterization():
    waste_df = pd.DataFrame(
        {
            "id": ["R1"],
            "category": ["Soil"],
            "material": ["MGS-1 simulant"],
            "kg": [5.0],
            "volume_l": [2.0],
            "flags": [""],
        }
    )

    prepared = generator.prepare_waste_frame(waste_df)
    process = _dummy_process_series()
    regolith_pct = 0.25

    features = generator.compute_feature_vector(prepared, [1.0], process, regolith_pct)

    for name, baseline in data_sources.REGOLITH_CHARACTERIZATION.feature_items:
        assert name in features
        assert features[name] == pytest.approx(regolith_pct * float(baseline), rel=1e-6)

    zero_features = generator.compute_feature_vector(prepared, [1.0], process, 0.0)
    for name, _ in data_sources.REGOLITH_CHARACTERIZATION.feature_items:
        assert zero_features[name] == pytest.approx(0.0, abs=1e-8)


def test_compute_feature_vector_sequence_matches_batch():
    waste_a = pd.DataFrame(
        {
            "id": ["A1", "A2"],
            "category": ["Packaging", "Tools"],
            "material": ["Polyethylene wrap", "Aluminum wrench"],
            "kg": [4.0, 1.0],
            "volume_l": [6.0, 0.5],
            "flags": ["", ""],
        }
    )
    waste_b = pd.DataFrame(
        {
            "id": ["B1"],
            "category": ["Logistics"],
            "material": ["Nomex bag"],
            "kg": [2.5],
            "volume_l": [3.0],
            "flags": ["multilayer"],
        }
    )

    prepared_a = generator.prepare_waste_frame(waste_a)
    prepared_b = generator.prepare_waste_frame(waste_b)
    process_a = _dummy_process_series()
    process_b = _dummy_process_series().copy(deep=True)
    process_b["process_id"] = "P02"

    weights_a = [0.7, 0.3]
    weights_b = [1.0]
    regolith_a = 0.1
    regolith_b = 0.05

    vectorized = generator.compute_feature_vector(
        [prepared_a, prepared_b],
        weights=[weights_a, weights_b],
        process=[process_a, process_b],
        regolith_pct=[regolith_a, regolith_b],
    )
    assert isinstance(vectorized, list)
    assert len(vectorized) == 2

    expected = generator.compute_feature_vectors_batch(
        [prepared_a, prepared_b],
        [weights_a, weights_b],
        [process_a, process_b],
        [regolith_a, regolith_b],
    )

    for combined, batch_expected in zip(vectorized, expected, strict=True):
        _assert_feature_mapping_equal(combined, batch_expected)


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

    _assert_feature_mapping_equal(pandas_features, polars_features)


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
        _assert_feature_mapping_equal(combined, expected)


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
        mission_reference_dense=np.asarray(mission_reference_matrix.todense())
        if generator.sparse is not None and generator.sparse.issparse(mission_reference_matrix)
        else np.asarray(mission_reference_matrix, dtype=np.float64),
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
        mission_reference_dense=np.asarray(empty_reference.todense())
        if generator.sparse is not None and generator.sparse.issparse(empty_reference)
        else np.asarray(empty_reference, dtype=np.float64),
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

    base_flags = prepared.get("_source_flags", prepared["flags"]).astype(str).str.lower()
    expected_tokens = (
        prepared["material"].astype(str).str.lower()
        + " "
        + prepared["category"].astype(str).str.lower()
        + " "
        + base_flags
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

    cat_mass = pd.to_numeric(prepared.get("category_total_mass_kg"), errors="coerce")
    if not isinstance(cat_mass, pd.Series):
        cat_mass = pd.Series(cat_mass, index=prepared.index, dtype=float)
    cat_volume = pd.to_numeric(prepared.get("category_total_volume_m3"), errors="coerce")
    if not isinstance(cat_volume, pd.Series):
        cat_volume = pd.Series(cat_volume, index=prepared.index, dtype=float)
    cat_density = cat_mass.divide(cat_volume).where((cat_volume > 0) & cat_volume.notna())
    expected_density = expected_density.fillna(cat_density)

    bundle = generator._official_features_bundle()
    selected_columns: list[str] = []
    if getattr(bundle, "composition_columns", None):
        selected_columns.extend(
            [
                column
                for column in bundle.composition_columns
                if column in prepared.columns
                and column in generator._COMPOSITION_DENSITY_MAP
            ]
        )
    fallback_columns = [
        column
        for column in generator._COMPOSITION_DENSITY_MAP
        if column in prepared.columns and column not in selected_columns
    ]
    selected_columns.extend(fallback_columns)

    if selected_columns:
        composition_numeric = {
            column: pd.to_numeric(prepared[column], errors="coerce").fillna(0.0)
            for column in selected_columns
        }
        composition_frac = pd.DataFrame(composition_numeric, index=prepared.index).div(100.0)
        frac_total = composition_frac.sum(axis=1)
        density_lookup = pd.Series(
            {
                column: float(generator._COMPOSITION_DENSITY_MAP[column])
                for column in selected_columns
            },
            index=selected_columns,
            dtype=float,
        )
        weighted_density = composition_frac.multiply(density_lookup, axis=1).sum(axis=1)
        weighted_density = weighted_density.divide(frac_total).where(frac_total > 0)
    else:
        weighted_density = pd.Series(np.nan, index=prepared.index, dtype=float)

    normalized_category = generator._vectorized_normalize_category(prepared["category"])
    foam_mask = normalized_category == "foam packaging"
    foam_default = generator._CATEGORY_DENSITY_DEFAULTS.get("foam packaging")
    if foam_default is not None:
        weighted_density = weighted_density.where(
            ~foam_mask,
            weighted_density.clip(upper=float(foam_default)),
        )

    expected_density = expected_density.fillna(weighted_density)

    logistic_density_map: Dict[str, float] = {}
    category_features = getattr(bundle, "l2l_category_features", None)
    if isinstance(category_features, dict):
        for key, features in category_features.items():
            if not isinstance(features, dict):
                continue
            for name, value in features.items():
                if not isinstance(value, (int, float)) or not np.isfinite(value):
                    continue
                if "density" not in str(name).lower():
                    continue
                normalized_key = generator.normalize_category(key)
                if normalized_key:
                    logistic_density_map.setdefault(normalized_key, float(value))
                break

    if logistic_density_map:
        logistic_series = normalized_category.map(logistic_density_map)
        logistic_series = pd.to_numeric(logistic_series, errors="coerce")
        expected_density = expected_density.fillna(logistic_series)

    category_defaults = normalized_category.map(generator._CATEGORY_DENSITY_DEFAULTS).astype(float)
    expected_density = expected_density.fillna(category_defaults)

    default_density = float(
        generator._CATEGORY_DENSITY_DEFAULTS.get("packaging", 500.0)
    )
    expected_density = expected_density.fillna(default_density).clip(lower=20.0, upper=4000.0)

    pd.testing.assert_series_equal(
        prepared["density_kg_m3"],
        expected_density,
        check_names=False,
    )


def test_prepare_waste_frame_density_missing_volume(monkeypatch):
    data_sources.official_features_bundle.cache_clear()
    generator._official_features_bundle.cache_clear()

    dummy_table = pl.DataFrame(
        schema={"category_norm": pl.Utf8, "subitem_norm": pl.Utf8},
        data=[],
    )

    dummy_bundle = generator._OfficialFeaturesBundle(
        value_columns=(),
        composition_columns=("Aluminum_pct", "Plastic_Resin_pct", "approx_moisture_pct"),
        direct_map={},
        category_tokens={},
        table=dummy_table,
        value_matrix=np.empty((0, 0), dtype=np.float64),
        mission_mass={},
        mission_totals={},
        mission_reference_keys=(),
        mission_reference_index={},
        mission_reference_matrix=np.zeros((0, 0), dtype=np.float64),
        mission_reference_dense=np.zeros((0, 0), dtype=np.float64),
        mission_names=(),
        mission_totals_vector=np.zeros(0, dtype=np.float64),
        processing_metrics={},
        leo_mass_savings={},
        propellant_benefits={},
        l2l_constants={},
        l2l_category_features={
            "food packaging": {"l2l_packaging_density_kg_m3": 630.0},
            "foam packaging": {"l2l_packaging_density_kg_m3": 95.0},
        },
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
            "id": ["A", "B", "C", "D", "E"],
            "category": [
                "Food Packaging",
                "Structural Elements",
                "Food Packaging",
                "Foam Packaging",
                "Other Packaging",
            ],
            "kg": [12.0, 5.0, 1.0, 0.8, 0.5],
            "volume_l": [np.nan, np.nan, np.nan, np.nan, np.nan],
            "category_total_mass_kg": [180.0, np.nan, np.nan, np.nan, np.nan],
            "category_total_volume_m3": [0.25, np.nan, np.nan, np.nan, np.nan],
            "Aluminum_pct": [0.0, 60.0, 0.0, 0.0, 0.0],
            "Plastic_Resin_pct": [0.0, 40.0, 0.0, 0.0, 0.0],
            "approx_moisture_pct": [0.0, 0.0, 0.0, 0.0, 0.0],
        }
    )

    prepared = generator.prepare_waste_frame(waste_df)

    densities = prepared["density_kg_m3"].to_numpy()

    assert densities[0] == pytest.approx(720.0, rel=1e-6)
    assert densities[1] == pytest.approx(2000.0, rel=1e-6)
    assert densities[2] == pytest.approx(630.0, rel=1e-6)
    assert densities[3] == pytest.approx(95.0, rel=1e-6)
    assert densities[4] == pytest.approx(
        generator._CATEGORY_DENSITY_DEFAULTS["other packaging"], rel=1e-6
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
        mission_reference_dense=np.asarray(empty_reference.todense())
        if generator.sparse is not None and generator.sparse.issparse(empty_reference)
        else np.asarray(empty_reference, dtype=np.float64),
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
