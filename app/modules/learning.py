"""Utilities for learning from feedback logs and guiding experimentation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from .analytics import pareto_front

FEATURE_COLUMNS = [
    "score",
    "pred_rigidity",
    "pred_tightness",
    "mass_final_kg",
    "energy_kwh",
    "water_l",
    "crew_min",
    "regolith_pct",
]


@dataclass
class LearningBundle:
    dataset: pd.DataFrame
    merged: pd.DataFrame
    baseline_score: float
    has_supervision: bool


def _coerce_numeric(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def prepare_learning_bundle(impact_df: pd.DataFrame, feedback_df: pd.DataFrame) -> LearningBundle:
    if impact_df is None or feedback_df is None or impact_df.empty or feedback_df.empty:
        return LearningBundle(pd.DataFrame(), pd.DataFrame(), 0.0, False)

    keys = ["scenario", "target_name", "option_idx"]
    merged = feedback_df.merge(impact_df, on=keys, how="inner", suffixes=("_fb", ""))
    if merged.empty:
        return LearningBundle(pd.DataFrame(), pd.DataFrame(), 0.0, False)

    # Normalizar columnas necesarias
    merged = _coerce_numeric(
        merged,
        [
            "score",
            "pred_rigidity",
            "pred_tightness",
            "mass_final_kg",
            "energy_kwh",
            "water_l",
            "crew_min",
            "regolith_pct",
            "overall",
            "porosity",
            "surface",
            "bonding",
        ],
    )

    if "regolith_pct" not in merged.columns:
        merged["regolith_pct"] = 0.0
    merged["regolith_pct"] = merged["regolith_pct"].fillna(0.0)

    merged["overall"] = merged["overall"].fillna(0.0)
    merged["porosity"] = merged["porosity"].fillna(0.0)

    merged["accepted"] = (
        merged["rigidity_ok"].astype(bool)
        & merged["ease_ok"].astype(bool)
        & (merged["overall"] >= 7)
        & (merged["porosity"] <= 6)
    ).astype(int)

    merged["sample_weight"] = np.clip(merged["overall"] / 7.0, 0.2, 3.0)

    dataset = merged[[c for c in FEATURE_COLUMNS if c in merged.columns]].copy()
    for col in FEATURE_COLUMNS:
        if col not in dataset.columns:
            dataset[col] = 0.0
    dataset = dataset[FEATURE_COLUMNS]
    dataset["accepted"] = merged["accepted"].astype(int)
    dataset["sample_weight"] = merged["sample_weight"].fillna(1.0)

    baseline_score = (
        merged.loc[merged["accepted"] == 1, "score"].max()
        if (merged["accepted"] == 1).any()
        else merged["score"].max()
    )
    if pd.isna(baseline_score):
        baseline_score = 0.0

    has_supervision = dataset["accepted"].nunique() >= 2
    return LearningBundle(dataset, merged, float(baseline_score), has_supervision)


def train_acceptance_model(dataset: pd.DataFrame) -> Pipeline | None:
    if dataset is None or dataset.empty:
        return None
    if dataset["accepted"].nunique() < 2:
        return None

    X = dataset[FEATURE_COLUMNS]
    y = dataset["accepted"].astype(int)
    weights = dataset.get("sample_weight", None)

    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]
    )
    pipe.fit(X, y, sample_weight=weights)
    return pipe


def score_candidates(
    candidates: list[dict],
    model: Pipeline | None,
    baseline_score: float,
    learning_ready: bool,
) -> list[dict]:
    if not candidates:
        return []

    rows = []
    for cand in candidates:
        props = cand.get("props")
        rows.append(
            {
                "score": float(cand.get("score", 0.0)),
                "pred_rigidity": float(getattr(props, "rigidity", 0.0) if props else 0.0),
                "pred_tightness": float(getattr(props, "tightness", 0.0) if props else 0.0),
                "mass_final_kg": float(getattr(props, "mass_final_kg", 0.0) if props else 0.0),
                "energy_kwh": float(getattr(props, "energy_kwh", 0.0) if props else 0.0),
                "water_l": float(getattr(props, "water_l", 0.0) if props else 0.0),
                "crew_min": float(getattr(props, "crew_min", 0.0) if props else 0.0),
                "regolith_pct": float(cand.get("regolith_pct", 0.0)),
            }
        )

    feat_df = pd.DataFrame(rows)
    for col in FEATURE_COLUMNS:
        if col not in feat_df.columns:
            feat_df[col] = 0.0

    if model is not None:
        prob = model.predict_proba(feat_df[FEATURE_COLUMNS])[:, 1]
    else:
        scores = feat_df["score"]
        if scores.max() == scores.min():
            prob = np.full(len(scores), 0.5)
        else:
            scaled = (scores - scores.min()) / (scores.max() - scores.min())
            prob = 0.35 + 0.6 * scaled

    prob = np.clip(prob, 0.01, 0.99)
    uncertainty = 1.0 - np.abs(prob - 0.5) * 2.0
    improvement = np.maximum(0.0, feat_df["score"] - baseline_score) * prob

    results = []
    for idx, cand in enumerate(candidates):
        results.append(
            {
                "acceptance_prob": float(prob[idx]),
                "uncertainty": float(uncertainty[idx]),
                "expected_improvement": float(improvement.iloc[idx]),
            }
        )

    if results and learning_ready:
        improve_vals = np.array([r["expected_improvement"] for r in results])
        if improve_vals.max() > 0:
            top_mask = improve_vals >= np.percentile(improve_vals, 70)
        else:
            top_mask = np.zeros_like(improve_vals, dtype=bool)
        for res, unc, top in zip(results, uncertainty, top_mask):
            res["active_priority"] = bool((unc >= 0.6) or top)
    elif results:
        for res in results:
            res["active_priority"] = False
    return results


def attach_learning_signals(
    candidates: list[dict], signals: list[dict]
) -> list[dict]:
    if not candidates or not signals:
        return candidates
    out = []
    for cand, sig in zip(candidates, signals):
        cand = cand.copy()
        cand.update(sig)
        out.append(cand)
    out.sort(key=lambda x: x.get("expected_improvement", 0.0), reverse=True)
    return out


def pareto_shift_data(merged: pd.DataFrame) -> dict[str, pd.DataFrame]:
    if merged is None or merged.empty:
        return {}

    base_cols = {
        "energy_kwh": "Energía (kWh)",
        "water_l": "Agua (L)",
        "crew_min": "Crew (min)",
        "score": "Score",
    }
    scatter = merged[list(base_cols.keys()) + ["accepted"]].rename(columns=base_cols)
    scatter["accepted"] = scatter["accepted"].map({1: "Aceptado", 0: "Rechazado"})

    fronts = {}
    for label, subset in {
        "Histórico": scatter,
        "Aceptado": scatter[scatter["accepted"] == "Aceptado"],
    }.items():
        if subset.empty:
            continue
        idx = pareto_front(
            subset,
            minimize_cols=("Energía (kWh)", "Agua (L)", "Crew (min)"),
            maximize_cols=("Score",),
        )
        front_df = subset.iloc[idx].copy()
        front_df["set"] = label
        fronts[label] = front_df

    return {"scatter": scatter, "fronts": fronts}
