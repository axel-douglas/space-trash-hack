from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
import pandas as pd
import json
from datetime import datetime

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
LOG_FILE = DATA_DIR / "impact_log.csv"
FEEDBACK_FILE = DATA_DIR / "feedback_log.csv"

@dataclass
class ImpactEntry:
    ts_iso: str
    scenario: str
    target_name: str
    option_idx: int
    materials: str
    weights: str
    process_id: str
    process_name: str
    mass_final_kg: float
    energy_kwh: float
    water_l: float
    crew_min: float
    score: float
    pred_rigidity: float
    pred_tightness: float
    regolith_pct: float = 0.0
    extra: str = ""

@dataclass
class FeedbackEntry:
    ts_iso: str
    astronaut: str
    scenario: str
    target_name: str
    option_idx: int
    rigidity_ok: bool
    ease_ok: bool
    issues: str
    notes: str
    overall: float = 0.0
    porosity: float = 0.0
    surface: float = 0.0
    bonding: float = 0.0
    failure_mode: str = "-"
    extra: str = ""

def _ensure_files():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not LOG_FILE.exists():
        pd.DataFrame(columns=list(ImpactEntry.__annotations__.keys())).to_csv(LOG_FILE, index=False)
    if not FEEDBACK_FILE.exists():
        pd.DataFrame(columns=list(FeedbackEntry.__annotations__.keys())).to_csv(FEEDBACK_FILE, index=False)

def _ensure_columns(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    missing = [c for c in cols if c not in df.columns]
    for c in missing:
        df[c] = pd.NA
    return df

def append_impact(entry: ImpactEntry):
    _ensure_files()
    df = pd.read_csv(LOG_FILE)
    cols = list(ImpactEntry.__annotations__.keys())
    df = _ensure_columns(df, cols)
    record = asdict(entry)
    df.loc[len(df), cols] = [record.get(c, pd.NA) for c in cols]
    df.to_csv(LOG_FILE, index=False)

def append_feedback(entry: FeedbackEntry):
    _ensure_files()
    df = pd.read_csv(FEEDBACK_FILE)
    cols = list(FeedbackEntry.__annotations__.keys())
    df = _ensure_columns(df, cols)
    record = asdict(entry)
    df.loc[len(df), cols] = [record.get(c, pd.NA) for c in cols]
    df.to_csv(FEEDBACK_FILE, index=False)

def load_impact_df() -> pd.DataFrame:
    _ensure_files()
    df = pd.read_csv(LOG_FILE)
    df = _ensure_columns(df, list(ImpactEntry.__annotations__.keys()))
    for col in ["mass_final_kg", "energy_kwh", "water_l", "crew_min", "score",
                "pred_rigidity", "pred_tightness", "regolith_pct"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "regolith_pct" in df.columns and df["regolith_pct"].isna().all() and "extra" in df.columns:
        extracted = df["extra"].astype(str).str.extract(r"regolith_pct=([0-9\.]+)")
        df.loc[df["regolith_pct"].isna(), "regolith_pct"] = pd.to_numeric(extracted[0], errors="coerce")
    if "option_idx" in df.columns:
        df["option_idx"] = pd.to_numeric(df["option_idx"], errors="coerce").fillna(0).astype(int)
    return df

def load_feedback_df() -> pd.DataFrame:
    _ensure_files()
    df = pd.read_csv(FEEDBACK_FILE)
    df = _ensure_columns(df, list(FeedbackEntry.__annotations__.keys()))
    bool_cols = ["rigidity_ok", "ease_ok"]
    for col in bool_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.lower().isin(["true", "1", "yes", "t"])
    num_cols = ["overall", "porosity", "surface", "bonding"]
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "option_idx" in df.columns:
        df["option_idx"] = pd.to_numeric(df["option_idx"], errors="coerce").fillna(0).astype(int)
    return df

def summarize_impact(df: pd.DataFrame) -> dict:
    if df.empty:
        return {"runs":0, "kg":0.0, "kwh":0.0, "water_l":0.0, "crew_min":0.0}
    return {
        "runs": int(len(df)),
        "kg": float(df["mass_final_kg"].sum()),
        "kwh": float(df["energy_kwh"].sum()),
        "water_l": float(df["water_l"].sum()),
        "crew_min": float(df["crew_min"].sum())
    }
