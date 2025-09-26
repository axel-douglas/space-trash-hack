# app/modules/process_planner.py
from __future__ import annotations

import re
from typing import Iterable

import pandas as pd

# Reglas simples de idoneidad por residuo (pueden expandirse)
SUITABILITY = {
    "pouches": ["P02"],                 # Press & Heat Lamination
    "foam": ["P02", "P03"],             # laminar o sinter con MGS-1
    "EVA_bag": ["P04", "P02"],          # reutilizar CTB / laminar
    "glove": ["P01", "P02"],            # triturar + laminar
    "aluminum": ["P04"],                # reconfiguración herrajes/struts
    "textiles": ["P02"],                # laminar en capas
}

FLAG_BOOST = {
    "multilayer": ["P02"],
    "thermal": ["P02"],
    "ctb": ["P04"],
    "closed_cell": ["P03", "P02"],
    "nitrile": ["P01", "P02"],
    "struts": ["P04"],
}

SCENARIO_HINTS = {
    "residence renovations": {"P04": 4.0, "P02": 1.5, "P03": 1.0},
    "cosmic celebrations": {"P02": 3.5, "P04": 1.5, "P03": 0.8},
    "daring discoveries": {"P03": 3.8, "P01": 1.8, "P02": 0.8},
}

SUITABILITY_WEIGHT = 3.0
SUITABILITY_DECAY = 0.6
FLAG_WEIGHT = 2.0
FLAG_DECAY = 0.5


def _tokenize(text: str | None) -> set[str]:
    if not text:
        return set()
    tokens = [t for t in re.split(r"[^0-9a-zA-Z]+", str(text).lower()) if t]
    return set(tokens)


def _iter_pref_weights(process_ids: Iterable[str], base: float, decay: float) -> Iterable[tuple[str, float]]:
    for idx, proc_id in enumerate(process_ids):
        yield str(proc_id), max(0.0, base - decay * idx)


def choose_process(
    target_name: str,
    proc_df: pd.DataFrame,
    scenario: str | None = None,
    crew_time_low: bool = False,
) -> pd.DataFrame:
    if proc_df is None or proc_df.empty:
        return proc_df.copy() if proc_df is not None else pd.DataFrame()

    df = proc_df.copy()
    df["process_id"] = df["process_id"].astype(str)
    df = df.drop_duplicates("process_id").set_index("process_id", drop=False)
    df.index.name = "_process_index"

    # Preparar columnas auxiliares
    crew_minutes = pd.to_numeric(df.get("crew_min_per_batch"), errors="coerce").fillna(0.0)
    if crew_time_low:
        max_minutes = float(max(crew_minutes.max(), 1.0))
        df["crew_bias"] = 1.0 - crew_minutes / max_minutes
    else:
        df["crew_bias"] = 0.0

    df["match_score"] = 0.0
    df["_reasons"] = [[] for _ in range(len(df))]

    tokens = _tokenize(target_name)
    tokens_text = " ".join(sorted(tokens))
    matched_ids: set[str] = set()

    def _add_score(process_id: str, value: float, reason: str) -> None:
        if process_id not in df.index:
            return
        df.at[process_id, "match_score"] += float(value)
        df.at[process_id, "_reasons"].append(reason)
        matched_ids.add(process_id)

    # Matches by suitability (prioridad según orden en tabla)
    for key, proc_ids in SUITABILITY.items():
        key_tokens = _tokenize(key)
        if not key_tokens:
            continue
        if key.lower() in tokens_text or key_tokens.issubset(tokens) or (tokens & key_tokens):
            for proc_id, weight in _iter_pref_weights(proc_ids, SUITABILITY_WEIGHT, SUITABILITY_DECAY):
                if weight <= 0:
                    continue
                _add_score(proc_id, weight, f"Adecuado para {key}")

    # Matches by flags (boosts menores)
    for key, proc_ids in FLAG_BOOST.items():
        key_tokens = _tokenize(key)
        if not key_tokens:
            continue
        if key.lower() in tokens_text or key_tokens.issubset(tokens) or (tokens & key_tokens):
            for proc_id, weight in _iter_pref_weights(proc_ids, FLAG_WEIGHT, FLAG_DECAY):
                if weight <= 0:
                    continue
                _add_score(proc_id, weight, f"Flag prioritario: {key}")

    # Scenario hints (opcional)
    if scenario:
        scenario_key = str(scenario).strip().lower()
        for proc_id, bonus in SCENARIO_HINTS.get(scenario_key, {}).items():
            if bonus <= 0:
                continue
            _add_score(proc_id, bonus, f"Escenario {scenario}")

    if not matched_ids:
        # Si no hubo matches, usar todos los procesos disponibles
        matched_ids = set(df.index)

    df = df[df["process_id"].isin(matched_ids)].copy()
    df["match_score"] = df["match_score"] + df["crew_bias"]
    df["match_reason"] = df["_reasons"].apply(
        lambda reasons: " · ".join(dict.fromkeys(str(r) for r in reasons if r))
    )
    df = df.drop(columns=["_reasons"], errors="ignore")

    df = df.sort_values(
        by=["match_score", "crew_min_per_batch", "process_id"],
        ascending=[False, True, True],
    )

    return df.reset_index(drop=True)
