"""Unit tests for regolith helper loaders in :mod:`app.modules.data_sources`."""

from __future__ import annotations

import warnings

import pandas as pd
import polars as pl
import pytest
from polars.testing import assert_frame_equal as assert_pl_frame_equal

from app.modules import data_sources

from app.modules.data_sources import (
    RegolithCharacterization,
    REGOLITH_CHARACTERIZATION,
    load_material_reference_bundle,
    load_regolith_characterization,
    load_regolith_particle_size,
    load_regolith_spectra,
    load_regolith_thermogravimetry,
)
from app.modules.generator.assembly import CandidateAssembler


def test_merge_reference_dataset_lazyframe_matches_eager(tmp_path, monkeypatch) -> None:
    base = pl.DataFrame(
        {
            "category": ["Fabrics", "Metals"],
            "subitem": ["Composite Panel", "Aluminium"],
            "mass": [1.0, 2.0],
        }
    )
    extra = pl.DataFrame(
        {
            "category": ["Fabrics", "Metals"],
            "subitem": ["Composite Panel", "Aluminium"],
            "value": [10.0, 20.0],
            "mass": [100.0, 200.0],
            "New Metric": [0.5, 0.75],
        }
    )
    dataset_path = tmp_path / "extra.csv"
    extra.write_csv(dataset_path)

    monkeypatch.setattr(
        data_sources,
        "resolve_dataset_path",
        lambda name: dataset_path if name == "extra.csv" else None,
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        lazy_result = data_sources.merge_reference_dataset(base.lazy(), "extra.csv", "extra")

    eager_result = data_sources.merge_reference_dataset(base, "extra.csv", "extra")

    assert not any(w.category.__name__ == "PerformanceWarning" for w in caught)

    if isinstance(lazy_result, pl.LazyFrame):
        lazy_df = lazy_result.collect()
    elif isinstance(lazy_result, pl.DataFrame):
        lazy_df = lazy_result
    elif isinstance(lazy_result, pd.DataFrame):
        lazy_df = pl.from_pandas(lazy_result)
    else:  # pragma: no cover - defensive
        raise AssertionError(f"Unexpected result type: {type(lazy_result)!r}")

    if isinstance(eager_result, pl.DataFrame):
        eager_df = eager_result
    elif isinstance(eager_result, pd.DataFrame):
        eager_df = pl.from_pandas(eager_result)
    elif isinstance(eager_result, pl.LazyFrame):
        eager_df = eager_result.collect()
    else:  # pragma: no cover - defensive
        raise AssertionError(f"Unexpected result type: {type(eager_result)!r}")

    assert_pl_frame_equal(lazy_df, eager_df)
    assert eager_df.columns == [
        "category",
        "subitem",
        "mass",
        "extra_value",
        "extra_new_metric",
    ]
    assert eager_df.get_column("extra_value").to_list() == [10.0, 20.0]
    assert eager_df.get_column("extra_new_metric").to_list() == [0.5, 0.75]


def test_material_reference_bundle_exposes_properties() -> None:
    bundle = load_material_reference_bundle.cache_clear() or load_material_reference_bundle()

    assert bundle.table.height > 0
    assert "material_density_kg_m3" in bundle.property_columns

    extended_columns = {
        "material_service_temperature_short_c",
        "material_service_temperature_long_c",
        "material_service_temperature_min_c",
        "material_coefficient_thermal_expansion_per_k_min",
        "material_coefficient_thermal_expansion_per_k_max",
        "material_ball_indentation_hardness_mpa",
        "material_shore_d_hardness",
        "material_rockwell_m_hardness",
        "material_surface_resistivity_ohm",
        "material_volume_resistivity_ohm_cm",
        "material_dielectric_strength_kv_mm",
        "material_relative_permittivity_low_freq",
        "material_relative_permittivity_high_freq",
        "material_dielectric_loss_tan_delta_low_freq",
        "material_dielectric_loss_tan_delta_high_freq",
        "material_comparative_tracking_index_cti",
    }
    assert extended_columns.issubset(set(bundle.property_columns))
    assert extended_columns.issubset(set(bundle.table.columns))

    slug = data_sources.slugify(data_sources.normalize_item("Nomex 410"))
    assert bundle.alias_map.get(slug)

    assert "pvdf_alpha_160c" in bundle.spectral_curves
    assert not bundle.spectral_curves["pvdf_alpha_160c"].empty

    metadata = bundle.metadata.get("pvdf_alpha_160c")
    assert metadata and "source" in metadata

    poly_slug = data_sources.slugify(data_sources.normalize_item("polyethylene"))
    poly_key = bundle.alias_map[poly_slug]
    poly_props = bundle.properties[poly_key]
    assert poly_props["material_service_temperature_short_c"] == pytest.approx(90.0)
    assert poly_props["material_dielectric_strength_kv_mm"] == pytest.approx(50.0)

    nylon_props = bundle.properties["nylon_6_6"]
    assert nylon_props["material_relative_permittivity_low_freq"] == pytest.approx(5.6)
    assert nylon_props["material_dielectric_loss_tan_delta_low_freq"] == pytest.approx(0.0715, rel=1e-4)


def test_material_reference_bundle_includes_mixing_information() -> None:
    bundle = load_material_reference_bundle.cache_clear() or load_material_reference_bundle()

    mixing = bundle.mixing_rules
    assert "pe_evoh_multilayer_film" in mixing

    pe_rule = mixing["pe_evoh_multilayer_film"]
    assert pe_rule["rule"] == "series"
    variants = pe_rule.get("variants") or []
    assert variants
    first_variant = variants[0]
    composition = first_variant.get("composition") or {}
    assert composition.get("hdpe_natural") == pytest.approx(0.95, rel=1e-2)
    assert composition.get("ethylene_vinyl_alcohol") == pytest.approx(0.05, rel=1e-2)

    compatibility = bundle.compatibility_matrix
    assert "pe_evoh_multilayer_film" in compatibility
    assert "mgs_1_regolith" in compatibility

    reg_entry = compatibility["pe_evoh_multilayer_film"]["mgs_1_regolith"]
    assert reg_entry["rule"] == "parallel"
    assert reg_entry["sources"]


def test_candidate_assembler_resolves_mixing_profile_aliases() -> None:
    bundle = load_material_reference_bundle()
    assembler = CandidateAssembler(material_reference=bundle)

    picks = pd.DataFrame([
        {"material": "PE/EVOH multilayer film", "kg": 4.0},
    ])

    profile = assembler.build_mixing_profile(picks, regolith_pct=0.2)
    assert profile

    composites = {entry["material_key"] for entry in profile.get("composites", [])}
    assert "pe_evoh_multilayer_film" in composites

    compatibility_pairs = {
        tuple(sorted(pair["materials"])) for pair in profile.get("compatibility_pairs", [])
    }
    assert ("mgs_1_regolith", "pe_evoh_multilayer_film") in compatibility_pairs


def test_candidate_assembler_aggregates_extended_metrics() -> None:
    bundle = load_material_reference_bundle()
    assembler = CandidateAssembler(material_reference=bundle)

    table = bundle.table.to_pandas()
    sample = table.loc[table["material_key"] == "hdpe_natural"].copy()
    assert not sample.empty
    sample["kg"] = 5.0

    aggregated = assembler.aggregate_material_properties(sample, [1.0])
    assert "material_service_temperature_short_c" in aggregated
    assert aggregated["material_service_temperature_short_c"] == pytest.approx(90.0)
    assert aggregated["material_coefficient_thermal_expansion_per_k_min"] == pytest.approx(1.3e-4, rel=1e-6)


def test_candidate_assembler_applies_mixing_rules_for_components() -> None:
    bundle = load_material_reference_bundle()
    assembler = CandidateAssembler(material_reference=bundle)

    assembler._alias_map = dict(assembler._alias_map)
    assembler._mixing_rules = dict(assembler._mixing_rules)

    components = {
        "hdpe_natural": {"canonical_key": "hdpe_natural"},
        "nylon_6_6": {"canonical_key": "nylon_6_6"},
    }
    composition = {"hdpe_natural": 0.6, "nylon_6_6": 0.4}

    parallel_key = "custom_parallel_laminate"
    series_key = "custom_series_laminate"
    parallel_slug = data_sources.slugify(data_sources.normalize_item("Custom Parallel Laminate"))
    series_slug = data_sources.slugify(data_sources.normalize_item("Custom Series Laminate"))
    assembler._alias_map[parallel_slug] = parallel_key
    assembler._alias_map[series_slug] = series_key

    assembler._mixing_rules[parallel_key] = {
        "rule": "parallel",
        "components": components,
        "variants": [{"composition": composition}],
    }
    assembler._mixing_rules[series_key] = {
        "rule": "series",
        "components": components,
        "variants": [{"composition": composition}],
    }

    parallel_pick = pd.DataFrame(
        [{"_material_reference_key": parallel_key, "kg": 1.0}]
    )
    aggregates_parallel = assembler.aggregate_material_properties(parallel_pick, [1.0])

    hdpe = bundle.properties["hdpe_natural"]
    nylon = bundle.properties["nylon_6_6"]
    expected_density = 0.6 * hdpe["material_density_kg_m3"] + 0.4 * nylon["material_density_kg_m3"]
    assert aggregates_parallel["material_density_kg_m3"] == pytest.approx(
        expected_density, rel=1e-6
    )

    series_pick = pd.DataFrame(
        [{"_material_reference_key": series_key, "kg": 1.0}]
    )
    aggregates_series = assembler.aggregate_material_properties(series_pick, [1.0])
    expected_conductivity = 1.0 / (
        (0.6 / hdpe["material_thermal_conductivity_w_mk"])
        + (0.4 / nylon["material_thermal_conductivity_w_mk"])
    )
    assert aggregates_series["material_thermal_conductivity_w_mk"] == pytest.approx(
        expected_conductivity, rel=1e-6
    )


def test_particle_size_loader_produces_expected_metrics() -> None:
    frame, metrics = load_regolith_particle_size.cache_clear() or load_regolith_particle_size()

    expected_columns = {
        "diameter_microns",
        "percent_retained",
        "percent_channel",
        "fraction_channel",
        "cumulative_percent_finer",
        "cumulative_percent_retained",
        "percent_finer_than",
    }

    assert expected_columns.issubset(set(frame.columns))
    assert metrics["d10_microns"] < metrics["d50_microns"] < metrics["d90_microns"]
    assert metrics["d50_microns"] == pytest.approx(83.8, rel=0.02)
    assert metrics["log_slope_fraction_finer"] == pytest.approx(-0.69, rel=0.05)


def test_spectral_loader_returns_slopes_and_means() -> None:
    frame, metrics = load_regolith_spectra.cache_clear() or load_regolith_spectra()

    assert frame.height > 0
    assert frame.columns == [
        "wavelength_nm",
        "reflectance_mms1",
        "reflectance_mms2",
        "reflectance_jsc_mars_1",
        "reflectance_mgs_1",
    ]

    for key in (
        "mean_reflectance_mms1",
        "mean_reflectance_mms2",
        "mean_reflectance_jsc_mars_1",
        "mean_reflectance_mgs_1",
    ):
        assert metrics[key] > 0

    slope_key = "slope_reflectance_mgs_1_700_1000"
    assert slope_key in metrics
    assert metrics[slope_key] == pytest.approx(-9.3e-05, rel=0.1)


def test_thermogravimetry_loader_tracks_mass_loss_and_peaks() -> None:
    tg_frame, ega_frame, thermal_metrics, ega_metrics = (
        load_regolith_thermogravimetry.cache_clear() or load_regolith_thermogravimetry()
    )

    assert tg_frame.columns == ["temperature_c", "mass_percent"]
    assert ega_frame.columns[0] == "temperature_c"
    assert thermal_metrics["mass_loss_total_percent"] == pytest.approx(3.63, rel=0.03)
    assert thermal_metrics["mass_loss_30_200_c"] == pytest.approx(2.06, rel=0.05)

    assert ega_metrics["peak_temperature_m_z_18_h2o"] == pytest.approx(405.6, rel=0.02)
    assert ega_metrics["peak_temperature_m_z_32_o2"] == pytest.approx(963.0, rel=1e-3)


def test_regolith_characterization_bundle_is_cached() -> None:
    bundle = load_regolith_characterization.cache_clear() or load_regolith_characterization()
    cached_again = load_regolith_characterization()

    assert isinstance(bundle, RegolithCharacterization)
    assert bundle is cached_again
    assert bundle.particle_metrics["d50_microns"] == pytest.approx(83.8, rel=0.02)
    assert "mass_loss_total_percent" in bundle.thermal_metrics
    assert "peak_temperature_m_z_44_co2" in bundle.gas_release_peaks
    assert bundle.feature_items == REGOLITH_CHARACTERIZATION.feature_items
    assert {name for name, _ in bundle.feature_items} == {
        "regolith_d50_um",
        "regolith_spectral_slope_1um",
        "regolith_mass_loss_400c",
        "regolith_h2o_peak_c",
    }
    assert REGOLITH_CHARACTERIZATION.particle_size.shape == bundle.particle_size.shape
    assert REGOLITH_CHARACTERIZATION.spectra.shape == bundle.spectra.shape
    assert (
        REGOLITH_CHARACTERIZATION.thermogravimetry.shape
        == bundle.thermogravimetry.shape
    )


def test_lookup_official_feature_values_maps_material_metrics(tmp_path, monkeypatch):
    official_path = tmp_path / "official.csv"
    official_path.write_text("category,subitem\nFabrics,Composite Panel\n")

    summary_path = tmp_path / "nasa_waste_summary.csv"
    summary_path.write_text("category,subitem,total_mass_kg\nFabrics,Composite Panel,5\n")

    density_path = tmp_path / "polymer_composite_density.csv"
    density_path.write_text(
        "category,subitem,sample_label,density_g_per_cm3\n"
        "Fabrics,Composite Panel,Original 0 %,1.25\n"
    )

    mechanics_path = tmp_path / "polymer_composite_mechanics.csv"
    mechanics_path.write_text(
        "category,subitem,sample_label,stress_mpa,modulus_gpa\n"
        "Fabrics,Composite Panel,Original 0 %,410,31\n"
    )

    ignition_path = tmp_path / "polymer_composite_ignition.csv"
    ignition_path.write_text(
        "category,subitem,sample_label,ignition_temperature_c\n"
        "Fabrics,Composite Panel,Original 0 %,770\n"
    )

    file_map = {
        "nasa_waste_summary.csv": summary_path,
        "polymer_composite_density.csv": density_path,
        "polymer_composite_mechanics.csv": mechanics_path,
        "polymer_composite_ignition.csv": ignition_path,
    }

    monkeypatch.setattr(data_sources, "_OFFICIAL_FEATURES_PATH", official_path)
    monkeypatch.setattr(data_sources, "resolve_dataset_path", lambda name: file_map.get(name))
    data_sources.official_features_bundle.cache_clear()

    row = pd.Series({"category": "Fabrics", "material": "Composite Panel"})
    payload, key = data_sources.lookup_official_feature_values(row)

    assert key == "fabric|composite panel"
    assert payload["official_density_kg_m3"] == pytest.approx(1250.0)
    assert payload["official_tensile_strength_mpa"] == pytest.approx(410.0)
    assert payload["official_modulus_gpa"] == pytest.approx(31.0)
    assert payload["official_ignition_temperature_c"] == pytest.approx(770.0)

    data_sources.official_features_bundle.cache_clear()
