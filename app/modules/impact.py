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
    materials: str
    weights: str
    process_id: str
    process_name: str
    mass_final_kg: float
    energy_kwh: float
    water_l: float
    crew_min: float
    score: float

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

def _ensure_files():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not LOG_FILE.exists():
        pd.DataFrame(columns=list(ImpactEntry.__annotations__.keys())).to_csv(LOG_FILE, index=False)
    if not FEEDBACK_FILE.exists():
        pd.DataFrame(columns=list(FeedbackEntry.__annotations__.keys())).to_csv(FEEDBACK_FILE, index=False)

def append_impact(entry: ImpactEntry):
    _ensure_files()
    df = pd.read_csv(LOG_FILE)
    df.loc[len(df)] = [
        entry.ts_iso, entry.scenario, entry.target_name, entry.materials, entry.weights,
        entry.process_id, entry.process_name, entry.mass_final_kg, entry.energy_kwh,
        entry.water_l, entry.crew_min, entry.score
    ]
    df.to_csv(LOG_FILE, index=False)

def append_feedback(entry: FeedbackEntry):
    _ensure_files()
    df = pd.read_csv(FEEDBACK_FILE)
    df.loc[len(df)] = [
        entry.ts_iso, entry.astronaut, entry.scenario, entry.target_name, entry.option_idx,
        entry.rigidity_ok, entry.ease_ok, entry.issues, entry.notes
    ]
    df.to_csv(FEEDBACK_FILE, index=False)

def load_impact_df() -> pd.DataFrame:
    _ensure_files()
    return pd.read_csv(LOG_FILE)

def load_feedback_df() -> pd.DataFrame:
    _ensure_files()
    return pd.read_csv(FEEDBACK_FILE)

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
