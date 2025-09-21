from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
import pandas as pd

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
    extra: str = ""

def _ensure_files():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    impact_columns = list(ImpactEntry.__annotations__.keys())
    feedback_columns = list(FeedbackEntry.__annotations__.keys())

    def _ensure_file(path: Path, columns: list[str]):
        if not path.exists():
            pd.DataFrame(columns=columns).to_csv(path, index=False)
            return
        df = pd.read_csv(path)
        updated = False
        missing = [col for col in columns if col not in df.columns]
        if missing:
            for col in missing:
                df[col] = ""
            updated = True
        # Reordenamos columnas para mantener consistencia
        ordered = df.reindex(columns=columns)
        if not ordered.equals(df):
            df = ordered
            updated = True
        if updated:
            df.to_csv(path, index=False)

    _ensure_file(LOG_FILE, impact_columns)
    _ensure_file(FEEDBACK_FILE, feedback_columns)

def append_impact(entry: ImpactEntry):
    _ensure_files()
    df = pd.read_csv(LOG_FILE)
    if "extra" not in df.columns:
        df["extra"] = ""
    payload = asdict(entry)
    df.loc[len(df)] = [payload.get(col, "") for col in df.columns]
    df.to_csv(LOG_FILE, index=False)

def append_feedback(entry: FeedbackEntry):
    _ensure_files()
    df = pd.read_csv(FEEDBACK_FILE)
    if "extra" not in df.columns:
        df["extra"] = ""
    payload = asdict(entry)
    df.loc[len(df)] = [payload.get(col, "") for col in df.columns]
    df.to_csv(FEEDBACK_FILE, index=False)

def load_impact_df() -> pd.DataFrame:
    _ensure_files()
    df = pd.read_csv(LOG_FILE)
    if "extra" not in df.columns:
        df["extra"] = ""
    return df

def load_feedback_df() -> pd.DataFrame:
    _ensure_files()
    df = pd.read_csv(FEEDBACK_FILE)
    if "extra" not in df.columns:
        df["extra"] = ""
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
