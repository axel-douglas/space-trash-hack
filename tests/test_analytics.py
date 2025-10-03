"""Tests para utilidades de analytics."""

import time

import numpy as np
import pandas as pd

from app.modules.analytics import pareto_front


def _slow_pareto(matrix: np.ndarray) -> np.ndarray:
    """Implementación de referencia O(n²) para comparar resultados."""

    dominated = np.zeros(len(matrix), dtype=bool)
    for i in range(len(matrix)):
        if dominated[i]:
            continue
        for j in range(len(matrix)):
            if i == j:
                continue
            if np.all(matrix[j] <= matrix[i]) and np.any(matrix[j] < matrix[i]):
                dominated[i] = True
                break
    return np.nonzero(~dominated)[0]


def test_pareto_front_expected_indices():
    df = pd.DataFrame(
        {
            "Energía (kWh)": [10, 8, 12, 9],
            "Agua (L)": [5, 6, 4, 7],
            "Crew (min)": [60, 55, 65, 58],
            "Score": [80, 82, 78, 81],
        },
        index=["a", "b", "c", "d"],
    )

    # "d" es dominada por "b", el resto permanecen en el frente de Pareto.
    expected = ["a", "b", "c"]
    assert pareto_front(df) == expected


def test_pareto_front_large_dataframe_matches_baseline():
    rng = np.random.default_rng(42)
    size = 250
    df = pd.DataFrame(
        {
            "Energía (kWh)": rng.uniform(0, 100, size),
            "Agua (L)": rng.uniform(0, 100, size),
            "Crew (min)": rng.uniform(30, 120, size),
            "Score": rng.uniform(0, 1, size),
        }
    )

    result = pareto_front(df)
    matrix = np.column_stack(
        [
            df["Energía (kWh)"].to_numpy(),
            df["Agua (L)"].to_numpy(),
            df["Crew (min)"].to_numpy(),
            -df["Score"].to_numpy(),
        ]
    )
    baseline = _slow_pareto(matrix)
    assert set(result) == set(baseline)


def test_pareto_front_large_dataframe_performance():
    rng = np.random.default_rng(123)
    size = 2000
    df = pd.DataFrame(
        {
            "Energía (kWh)": rng.uniform(0, 100, size),
            "Agua (L)": rng.uniform(0, 100, size),
            "Crew (min)": rng.uniform(30, 120, size),
            "Score": rng.uniform(0, 1, size),
        }
    )

    start = time.perf_counter()
    result = pareto_front(df)
    duration = time.perf_counter() - start

    # Evitar regresiones severas: la implementación vectorizada debe completar
    # en menos de 1.5 segundos sobre un DataFrame de 2000 filas.
    assert duration < 1.5, f"pareto_front demasiado lento ({duration:.2f}s)"
    assert result  # el frente no debe estar vacío
