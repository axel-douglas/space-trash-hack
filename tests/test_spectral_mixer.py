import numpy as np
import pandas as pd
import polars as pl
import pytest

from app.modules import data_sources as ds
from app.modules.generator import GeneratorService
from app.modules.spectral_mixer import solve_spectral_recipe


@pytest.fixture
def spectral_bundle(monkeypatch):
    wavenumbers = np.linspace(500.0, 1500.0, 10)
    base_a = pd.DataFrame(
        {
            "wavenumber_cm_1": wavenumbers,
            "absorbance": np.linspace(0.1, 0.9, wavenumbers.size),
        }
    )
    base_b = pd.DataFrame(
        {
            "wavenumber_cm_1": wavenumbers,
            "absorbance": np.linspace(0.9, 0.1, wavenumbers.size),
        }
    )
    bundle = ds.MaterialReferenceBundle(
        pl.DataFrame(),
        {},
        {},
        {},
        tuple(),
        {
            "spec_a": base_a,
            "spec_b": base_b,
        },
        {
            "spec_a": {"material": "Specimen A"},
            "spec_b": {"material": "Specimen B"},
        },
        {},
        {},
    )
    monkeypatch.setattr(ds, "load_material_reference_bundle", lambda: bundle)
    return bundle


def test_solve_spectral_recipe_recovers_known_mix(spectral_bundle):
    target_curve = spectral_bundle.spectral_curves["spec_a"].copy()
    target_curve["absorbance"] = 0.6 * target_curve["absorbance"] + 0.3 * spectral_bundle.spectral_curves["spec_b"]["absorbance"]

    availability = pd.DataFrame({"spectral_key": ["spec_a", "spec_b"]})
    result = solve_spectral_recipe(target_curve, availability, constraints={"max_fraction": 1.0})

    coefficients = dict(zip(result.basis, result.coefficients, strict=False))
    assert coefficients["spec_a"] == pytest.approx(0.6, rel=1e-3, abs=1e-3)
    assert coefficients["spec_b"] == pytest.approx(0.3, rel=1e-3, abs=1e-3)
    assert result.error_metrics["mae"] == pytest.approx(0.0, abs=1e-9)


def test_generator_service_propose_spectral_mix_returns_payload(spectral_bundle):
    generator = GeneratorService()
    target_curve = spectral_bundle.spectral_curves["spec_a"].copy()
    target_curve["absorbance"] = 0.7 * target_curve["absorbance"] + 0.2 * spectral_bundle.spectral_curves["spec_b"]["absorbance"]

    stock_df = pd.DataFrame(
        {
            "spectral_key": ["spec_a", "spec_b"],
            "kg": [2.0, 1.0],
        }
    )

    payload = generator.propose_spectral_mix(target_curve, stock_df, constraints={"max_fraction": 1.0})
    assert "coefficients" in payload and payload["coefficients"]
    coeff_map = {row["material"]: row["fraction"] for row in payload["coefficients"]}
    assert coeff_map["spec_a"] == pytest.approx(min(0.7, 2.0 / 3.0), rel=1e-3, abs=1e-3)
    assert coeff_map["spec_b"] == pytest.approx(0.2, rel=1e-2, abs=3e-2)
    error = payload.get("error", {})
    assert error["mae"] == pytest.approx(0.0, abs=2e-2)
