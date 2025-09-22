"""Utilities para optimizar combinaciones multiobjetivo.

Este módulo implementa una capa ligera de optimización inspirada en
Bayesian Optimization/MILP. Trabaja con los candidatos generados y un
sampler callable que produce nuevas combinaciones. El resultado es un
conjunto Pareto y métricas de convergencia (hipervolumen y dominancia).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np
import pandas as pd

try:  # pragma: no cover - optional dependency
    from ax.service.ax_client import AxClient

    AX_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency missing
    AxClient = None  # type: ignore[assignment]
    AX_AVAILABLE = False


@dataclass
class OptimizationSummary:
    iteration: int
    score: float
    penalty: float
    hypervolume: float
    dominance_ratio: float
    pareto_size: int


Candidate = dict
Sampler = Callable[[dict[str, float] | None], Candidate | None]


def optimize_candidates(
    initial_candidates: Iterable[Candidate],
    sampler: Sampler,
    target: dict,
    n_evals: int = 30,
    process_ids: list[str] | None = None,
) -> tuple[list[Candidate], pd.DataFrame]:
    """Ejecuta un barrido de optimización multiobjetivo.

    Parameters
    ----------
    initial_candidates:
        Colección inicial (semillas) generadas por heurística.
    sampler:
        Callable que devuelve nuevos candidatos a evaluar.
    target:
        Diccionario con restricciones y objetivos (rigidez, estanqueidad,
        máximos de recursos, etc.).
    n_evals:
        Número de evaluaciones adicionales a ejecutar.

    Returns
    -------
    tuple[list[Candidate], pd.DataFrame]
        La lista de soluciones no dominadas y el historial de métricas
        (hipervolumen, dominancia, etc.).
    """

    evaluated: list[Candidate] = [cand for cand in initial_candidates if cand]
    history: list[OptimizationSummary] = []

    pareto = _pareto_front(evaluated, target)
    hv = _hypervolume(pareto, evaluated, target)
    dom_ratio = _dominance_ratio(pareto, evaluated)
    history.append(
        OptimizationSummary(
            iteration=0,
            score=float("nan"),
            penalty=float("nan"),
            hypervolume=hv,
            dominance_ratio=dom_ratio,
            pareto_size=len(pareto),
        )
    )

    if n_evals > 0:
        if AX_AVAILABLE and AxClient is not None:
            _run_bayesian_optimization(
                evaluated,
                history,
                sampler,
                target,
                n_evals,
                process_ids or [],
            )
        else:
            for i in range(1, n_evals + 1):
                candidate = sampler({})
                if candidate:
                    evaluated.append(candidate)
                pareto = _pareto_front(evaluated, target)
                hv = _hypervolume(pareto, evaluated, target)
                dom_ratio = _dominance_ratio(pareto, evaluated)
                score = float(candidate["score"]) if candidate else float("nan")
                penalty = _penalty(candidate, target) if candidate else float("nan")
                history.append(
                    OptimizationSummary(
                        iteration=i,
                        score=score,
                        penalty=penalty,
                        hypervolume=hv,
                        dominance_ratio=dom_ratio,
                        pareto_size=len(pareto),
                    )
                )

    pareto_sorted = sorted(pareto, key=lambda c: c.get("score", 0.0), reverse=True)
    history_df = pd.DataFrame(history)
    return pareto_sorted, history_df


def _pareto_front(candidates: list[Candidate], target: dict) -> list[Candidate]:
    if not candidates:
        return []

    front: list[Candidate] = []
    for cand in candidates:
        dominated = False
        metrics_c = _metrics(cand, target)
        remove: list[Candidate] = []
        for other in front:
            metrics_o = _metrics(other, target)
            if _dominates(metrics_o, metrics_c):
                dominated = True
                break
            if _dominates(metrics_c, metrics_o):
                remove.append(other)
        if not dominated:
            front.append(cand)
            for r in remove:
                if r in front:
                    front.remove(r)
    return front


def _metrics(candidate: Candidate, target: dict) -> dict[str, float]:
    props = candidate.get("props")
    energy = getattr(props, "energy_kwh", 0.0)
    water = getattr(props, "water_l", 0.0)
    crew = getattr(props, "crew_min", 0.0)
    score = float(candidate.get("score", 0.0))

    def _norm(value: float, limit: float, eps: float = 1e-6) -> float:
        limit = max(limit, eps)
        return max(0.0, value) / limit

    energy_n = _norm(energy, float(target.get("max_energy_kwh", 1.0)))
    water_n = _norm(water, float(target.get("max_water_l", 1.0)))
    crew_n = _norm(crew, float(target.get("max_crew_min", 1.0)))
    penalty = np.mean([energy_n, water_n, crew_n])
    return {
        "score": score,
        "energy": energy_n,
        "water": water_n,
        "crew": crew_n,
        "penalty": penalty,
    }


def _dominates(metrics_a: dict[str, float], metrics_b: dict[str, float]) -> bool:
    better_or_equal = (
        metrics_a["score"] >= metrics_b["score"]
        and metrics_a["energy"] <= metrics_b["energy"]
        and metrics_a["water"] <= metrics_b["water"]
        and metrics_a["crew"] <= metrics_b["crew"]
    )
    strictly_better = (
        metrics_a["score"] > metrics_b["score"]
        or metrics_a["energy"] < metrics_b["energy"]
        or metrics_a["water"] < metrics_b["water"]
        or metrics_a["crew"] < metrics_b["crew"]
    )
    return better_or_equal and strictly_better


def _hypervolume(
    pareto: list[Candidate],
    evaluated: list[Candidate],
    target: dict,
) -> float:
    if not pareto:
        return 0.0

    scores = [float(c.get("score", 0.0)) for c in evaluated]
    score_min = min(scores)
    score_max = max(scores)
    span = max(score_max - score_min, 1e-6)

    points = []
    for cand in pareto:
        metrics = _metrics(cand, target)
        score_norm = (metrics["score"] - score_min) / span
        quality = max(0.0, 1.0 - metrics["penalty"])
        points.append((np.clip(score_norm, 0.0, 1.0), np.clip(quality, 0.0, 1.0)))

    points.sort(key=lambda p: p[0])
    hv = 0.0
    prev_score = 0.0
    prev_height = 0.0
    for score_norm, height in points:
        width = max(0.0, score_norm - prev_score)
        prev_height = max(prev_height, height)
        hv += width * prev_height
        prev_score = score_norm
    hv += max(0.0, 1.0 - prev_score) * prev_height
    return float(np.clip(hv, 0.0, 1.0))


def _dominance_ratio(pareto: list[Candidate], evaluated: list[Candidate]) -> float:
    if not evaluated:
        return 0.0
    dominated = max(len(evaluated) - len(pareto), 0)
    return dominated / len(evaluated)


def _penalty(candidate: Candidate, target: dict) -> float:
    if not candidate:
        return float("nan")
    metrics = _metrics(candidate, target)
    return float(metrics["penalty"])


def _run_bayesian_optimization(
    evaluated: list[Candidate],
    history: list[OptimizationSummary],
    sampler: Sampler,
    target: dict,
    n_evals: int,
    process_ids: list[str],
) -> None:
    if AxClient is None:
        return

    processes = sorted({str(pid) for pid in process_ids}) or ["P02", "P03", "P04"]
    parameters = [
        {"name": "problematic_bias", "type": "range", "bounds": [1.0, 5.0]},
        {"name": "regolith_pct", "type": "range", "bounds": [0.0, 0.6]},
        {"name": "process_choice", "type": "choice", "values": processes},
    ]

    ax_client = AxClient(enforce_sequential_optimization=True)
    ax_client.create_experiment(
        name="rexai_bo",
        parameters=parameters,
        objective_name="score",
        minimize=False,
    )

    for _ in range(n_evals):
        params, trial_index = ax_client.get_next_trial()
        override = {
            "problematic_bias": float(params.get("problematic_bias", 2.0)),
            "regolith_pct": float(params.get("regolith_pct", 0.0)),
            "process_choice": params.get("process_choice"),
        }
        candidate = sampler(override)
        if not candidate:
            candidate = sampler({})

        score = float(candidate["score"]) if candidate else float("nan")
        ax_client.complete_trial(trial_index=trial_index, raw_data=score)

        if candidate:
            evaluated.append(candidate)

        pareto = _pareto_front(evaluated, target)
        hv = _hypervolume(pareto, evaluated, target)
        dom_ratio = _dominance_ratio(pareto, evaluated)
        penalty = _penalty(candidate, target) if candidate else float("nan")
        history.append(
            OptimizationSummary(
                iteration=len(history),
                score=score,
                penalty=penalty,
                hypervolume=hv,
                dominance_ratio=dom_ratio,
                pareto_size=len(pareto),
            )
        )
