"""Unit tests for regolith helper loaders in :mod:`app.modules.data_sources`."""

from __future__ import annotations

import pytest

from app.modules.data_sources import (
    RegolithCharacterization,
    REGOLITH_CHARACTERIZATION,
    load_regolith_characterization,
    load_regolith_particle_size,
    load_regolith_spectra,
    load_regolith_thermogravimetry,
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
