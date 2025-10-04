"""Utilities para optimizar combinaciones multiobjetivo.

Este módulo implementa una capa ligera de optimización inspirada en
Bayesian Optimization/MILP. Trabaja con los candidatos generados y un
sampler callable que produce nuevas combinaciones. El resultado es un
conjunto Pareto y métricas de convergencia (hipervolumen y dominancia).
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Callable, Iterable, Mapping

import numpy as np
import pandas as pd

from app.modules.execution import (
    DEFAULT_PARALLEL_THRESHOLD,
    ExecutionBackend,
    create_backend,
)
from app.modules.property_planner import parse_property_constraints

try:  # pragma: no cover - optional dependency during tests without bundle
    from app.modules.generator import CandidateAssembler
except Exception:  # pragma: no cover - defensive guard
    CandidateAssembler = None  # type: ignore[assignment]

try:  # pragma: no cover - optional dependency
    from ax.service.ax_client import AxClient

    AX_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency missing
    AxClient = None  # type: ignore[assignment]
    AX_AVAILABLE = False


_PARALLEL_THRESHOLD = DEFAULT_PARALLEL_THRESHOLD


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


@lru_cache(maxsize=1)
def _property_columns() -> tuple[str, ...]:
    if CandidateAssembler is None:
        return tuple()
    try:
        assembler = CandidateAssembler()
    except Exception:  # pragma: no cover - optional bundle missing
        return tuple()
    columns = getattr(assembler.material_reference, "property_columns", ())
    return tuple(columns) if columns else tuple()


def _safe_float(value: object) -> float | None:
    try:
        candidate = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if np.isnan(candidate):
        return None
    return float(candidate)


def optimize_candidates(
    initial_candidates: Iterable[Candidate],
    sampler: Sampler,
    target: dict,
    n_evals: int = 30,
    process_ids: list[str] | None = None,
    backend: ExecutionBackend | None = None,
    backend_kind: str | None = None,
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

    parallel_tasks = max(len(evaluated), n_evals, 1)
    local_backend = backend
    owns_backend = False
    if local_backend is None:
        local_backend = create_backend(
            parallel_tasks,
            preferred=backend_kind,
            threshold=_PARALLEL_THRESHOLD,
        )
        owns_backend = True

    try:
        pareto = _pareto_front(evaluated, target, backend=local_backend)
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
            iteration = 0
            if AX_AVAILABLE and AxClient is not None:
                _run_bayesian_optimization(
                    evaluated,
                    history,
                    sampler,
                    target,
                    n_evals,
                    process_ids or [],
                    backend=local_backend,
                    start_iteration=len(history) - 1,
                )
            else:
                batch_size = max(1, local_backend.max_workers)
                produced = 0
                while produced < n_evals:
                    current_batch = min(batch_size, n_evals - produced)
                    overrides = [{} for _ in range(current_batch)]
                    results = local_backend.map(lambda override: sampler(override), overrides)
                    for candidate in results:
                        produced += 1
                        iteration += 1
                        if candidate:
                            evaluated.append(candidate)
                        pareto = _pareto_front(evaluated, target, backend=local_backend)
                        hv = _hypervolume(pareto, evaluated, target)
                        dom_ratio = _dominance_ratio(pareto, evaluated)
                        score = float(candidate["score"]) if candidate else float("nan")
                        penalty = _penalty(candidate, target) if candidate else float("nan")
                        history.append(
                            OptimizationSummary(
                                iteration=iteration,
                                score=score,
                                penalty=penalty,
                                hypervolume=hv,
                                dominance_ratio=dom_ratio,
                                pareto_size=len(pareto),
                            )
                        )
            pareto = _pareto_front(evaluated, target, backend=local_backend)

        pareto_sorted = sorted(pareto, key=lambda c: c.get("score", 0.0), reverse=True)
        history_df = pd.DataFrame(history)
        return pareto_sorted, history_df
    finally:
        if owns_backend and local_backend is not None:
            local_backend.shutdown()


def _pareto_front(
    candidates: list[Candidate],
    target: dict,
    backend: ExecutionBackend | None = None,
) -> list[Candidate]:
    if not candidates:
        return []

    front: list[Candidate] = []
    if backend is not None and backend.max_workers > 1:
        metrics_iter = backend.map(lambda c: _metrics(c, target), candidates)
    else:
        metrics_iter = (_metrics(c, target) for c in candidates)
    metrics_map = {id(cand): metrics for cand, metrics in zip(candidates, metrics_iter)}
    for cand in candidates:
        dominated = False
        metrics_c = metrics_map[id(cand)]
        remove: list[Candidate] = []
        for other in front:
            metrics_o = metrics_map[id(other)]
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
    penalty_components: list[float] = [energy_n, water_n, crew_n]

    features: Mapping[str, object] | None = candidate.get("features")  # type: ignore[assignment]
    constraint_penalties: dict[str, float] = {}
    if isinstance(features, Mapping):
        constraints = parse_property_constraints(target, _property_columns() or None)
        for column, (minimum, maximum) in constraints.items():
            value = _safe_float(features.get(column))
            violation = 0.0
            if value is None:
                violation = 1.0
            else:
                if minimum is not None and value < minimum:
                    denom = max(abs(minimum), 1e-6)
                    violation = max(violation, (minimum - value) / denom)
                if maximum is not None and value > maximum:
                    denom = max(abs(maximum), 1e-6)
                    violation = max(violation, (value - maximum) / denom)
            violation = float(max(0.0, violation))
            constraint_penalties[column] = violation
            penalty_components.append(violation)

    penalty = float(np.mean(penalty_components)) if penalty_components else 0.0

    metrics: dict[str, float] = {
        "score": score,
        "energy": energy_n,
        "water": water_n,
        "crew": crew_n,
        "penalty": penalty,
    }

    if constraint_penalties:
        metrics["constraint_penalty"] = float(np.mean(list(constraint_penalties.values())))
        for column, value in constraint_penalties.items():
            metrics[f"constraint::{column}"] = value
    else:
        metrics["constraint_penalty"] = 0.0

    metrics["meta::constraint_violations"] = constraint_penalties
    return metrics


def _dominates(metrics_a: dict[str, float], metrics_b: dict[str, float]) -> bool:
    if "score" not in metrics_a or "score" not in metrics_b:
        return False

    keys = set(metrics_a) & set(metrics_b)
    exclude = {"score", "penalty", "constraint_penalty", "meta::constraint_violations"}
    cost_keys = [key for key in keys if key not in exclude]

    if not cost_keys:
        return metrics_a["score"] > metrics_b["score"]

    better_or_equal = metrics_a["score"] >= metrics_b["score"] and all(
        metrics_a[key] <= metrics_b[key] for key in cost_keys
    )
    strictly_better = metrics_a["score"] > metrics_b["score"] or any(
        metrics_a[key] < metrics_b[key] for key in cost_keys
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


def candidate_metrics(candidate: Candidate, target: Mapping[str, object]) -> dict[str, object]:
    """Public wrapper returning the constraint-aware metrics for *candidate*."""

    return _metrics(candidate, dict(target))


def pareto_front(
    candidates: Iterable[Candidate],
    target: Mapping[str, object],
    *,
    backend: ExecutionBackend | None = None,
) -> list[Candidate]:
    """Expose the Pareto front helper for visualisations and analytics."""

    return _pareto_front(list(candidates), dict(target), backend=backend)


def _run_bayesian_optimization(
    evaluated: list[Candidate],
    history: list[OptimizationSummary],
    sampler: Sampler,
    target: dict,
    n_evals: int,
    process_ids: list[str],
    backend: ExecutionBackend | None = None,
    start_iteration: int = 0,
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

    iteration = start_iteration
    for _ in range(n_evals):
        params, trial_index = ax_client.get_next_trial()
        override = {
            "problematic_bias": float(params.get("problematic_bias", 2.0)),
            "regolith_pct": float(params.get("regolith_pct", 0.0)),
            "process_choice": params.get("process_choice"),
        }
        if backend is not None:
            future = backend.submit(sampler, override)
            candidate = future.result()
        else:
            candidate = sampler(override)
        if not candidate:
            if backend is not None:
                candidate = backend.submit(sampler, {}).result()
            else:
                candidate = sampler({})

        score = float(candidate["score"]) if candidate else float("nan")
        ax_client.complete_trial(trial_index=trial_index, raw_data=score)

        if candidate:
            evaluated.append(candidate)

        iteration += 1
        pareto = _pareto_front(evaluated, target, backend=backend)
        hv = _hypervolume(pareto, evaluated, target)
        dom_ratio = _dominance_ratio(pareto, evaluated)
        penalty = _penalty(candidate, target) if candidate else float("nan")
        history.append(
            OptimizationSummary(
                iteration=iteration,
                score=score,
                penalty=penalty,
                hypervolume=hv,
                dominance_ratio=dom_ratio,
                pareto_size=len(pareto),
            )
        )
