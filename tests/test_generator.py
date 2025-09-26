from __future__ import annotations

import json
from datetime import datetime

import pandas as pd
import pytest

from app.modules import generator


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
        generator,
        "_resolve_dataset_path",
        lambda name: summary_path if name == "nasa_waste_summary.csv" else None,
    )

    summary = generator._load_waste_summary_data()

    assert summary.mission_totals["artemis"] == pytest.approx(13.0)
    assert summary.mission_totals["gateway"] == pytest.approx(7.0)
    assert summary.mass_by_key["other packaging glove"]["artemis"] == pytest.approx(13.0)
    assert summary.mass_by_key["other packaging glove|foam"]["gateway"] == pytest.approx(5.0)


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
        generator,
        "_resolve_dataset_path",
        lambda name: metrics_path if name == "nasa_waste_processing_products.csv" else None,
    )

    aggregated = generator._extract_grouped_metrics(
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

    monkeypatch.setattr(generator, "_OFFICIAL_FEATURES_PATH", official_path)
    monkeypatch.setattr(generator, "_resolve_dataset_path", lambda name: file_map.get(name))
    generator._official_features_bundle.cache_clear()

    bundle = generator._official_features_bundle()

    assert "value_kg" in bundle.value_columns
    assert "processing_output_kg" in bundle.value_columns
    assert "leo_savings_pct" in bundle.value_columns
    assert "propellant_benefit" in bundle.value_columns
    assert bundle.direct_map["other packaging glove|foam"]["value_kg"] == pytest.approx(2.5)
    assert bundle.mission_totals["artemis"] == pytest.approx(13.0)
    assert bundle.processing_metrics["processing"]["processing_output_kg"] == pytest.approx(3.0)

    generator._official_features_bundle.cache_clear()

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


def test_generate_candidates_appends_inference_log(monkeypatch):
    monkeypatch.setattr(generator, "MODEL_REGISTRY", DummyRegistry())
    monkeypatch.setattr(generator, "lookup_labels", lambda *args, **kwargs: ({}, {}))

    log_dir = generator.LOGS_ROOT
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"inference_{datetime.utcnow().strftime('%Y%m%d')}.parquet"
    if log_path.exists():
        log_path.unlink()

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

    assert log_path.exists(), "Inference log parquet file was not created"

    log_df = pd.read_parquet(log_path)
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

    log_path.unlink(missing_ok=True)

def test_generate_candidates_heuristic_mode_skips_ml(monkeypatch):
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
    monkeypatch.setattr(generator, "lookup_labels", lambda *args, **kwargs: ({}, {}))

    log_dir = generator.LOGS_ROOT
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
    assert not log_path.exists(), "Inference log should not be created in heuristic mode"

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


def test_compute_feature_vector_includes_mission_metrics(monkeypatch):
    # Ensure cached bundles from other tests do not leak.
    generator._official_features_bundle.cache_clear()

    match_key = "food packaging|rehydratable pouch"
    dummy_bundle = generator._OfficialFeaturesBundle(
        value_columns=("dummy_col",),
        composition_columns=(),
        direct_map={match_key: {"dummy_col": 1.0}},
        category_tokens={
            "food packaging": [
                (frozenset({"rehydratable", "pouch"}), {"dummy_col": 1.0}, match_key)
            ]
        },
        mission_mass={
            match_key: {"gateway_i": 200.0},
            "food packaging": {"gateway_i": 300.0},
        },
        mission_totals={"gateway_i": 1000.0},
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


def test_prepare_waste_frame_injects_l2l_features(monkeypatch):
    generator._official_features_bundle.cache_clear()

    match_key = "food packaging|rehydratable pouch"
    dummy_bundle = generator._OfficialFeaturesBundle(
        value_columns=("dummy_col",),
        composition_columns=(),
        direct_map={match_key: {"dummy_col": 1.0}},
        category_tokens={
            "food packaging": [
                (frozenset({"rehydratable", "pouch"}), {"dummy_col": 1.0}, match_key)
            ]
        },
        mission_mass={},
        mission_totals={},
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

    assert pytest.approx(row["l2l_geometry_panel_area_m2"], rel=1e-6) == 5.0
    assert pytest.approx(row["l2l_ops_random_access_required"], rel=1e-6) == 1.0
    assert "_l2l_page_hints" in row.index
    assert "p.42" in row["_l2l_page_hints"]
    assert "p.15" in row["_l2l_page_hints"]


def test_compute_feature_vector_uses_l2l_packaging_ratio(monkeypatch):
    generator._official_features_bundle.cache_clear()

    dummy_bundle = generator._OfficialFeaturesBundle(
        value_columns=("dummy_col",),
        composition_columns=(),
        direct_map={},
        category_tokens={},
        mission_mass={},
        mission_totals={},
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
