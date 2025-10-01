from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd

from app.modules import impact


def _setup_tmp_logs(monkeypatch, tmp_path):
    logs_dir = tmp_path / "logs"
    monkeypatch.setattr(impact, "DATA_DIR", tmp_path)
    monkeypatch.setattr(impact, "LOGS_DIR", logs_dir)
    return logs_dir


def test_append_impact_preserves_extra_columns(tmp_path, monkeypatch):
    logs_dir = _setup_tmp_logs(monkeypatch, tmp_path)
    ts = datetime.utcnow().replace(microsecond=0).isoformat()
    entry = impact.ImpactEntry(
        ts_iso=ts,
        scenario="moon_base",
        target_name="panel",
        materials="A|B",
        weights="0.4|0.6",
        process_id="proc-1",
        process_name="Sintering",
        mass_final_kg=10.5,
        energy_kwh=3.2,
        water_l=1.1,
        crew_min=45.0,
        score=0.87,
        extra={"custom_metric": 123, "regolith_pct": 0.5},
    )

    run_id = impact.append_impact(entry)
    assert isinstance(run_id, str) and run_id

    files = list(logs_dir.glob("impact_*.parquet"))
    assert len(files) == 1
    df = pd.read_parquet(files[0])
    assert "run_id" in df.columns
    assert df.loc[0, "run_id"] == run_id
    assert "extra_custom_metric" in df.columns
    assert "extra_regolith_pct" in df.columns

    loaded = impact.load_impact_df()
    assert loaded.loc[0, "extra"]["custom_metric"] == 123
    assert loaded.loc[0, "extra"]["regolith_pct"] == 0.5
    assert "extra_custom_metric" in loaded.columns
    assert "extra_regolith_pct" in loaded.columns


def test_feedback_logging_concatenates_daily(tmp_path, monkeypatch):
    logs_dir = _setup_tmp_logs(monkeypatch, tmp_path)
    ts = datetime.utcnow().replace(microsecond=0)
    entries = [
        impact.FeedbackEntry(
            ts_iso=ts.isoformat(),
            astronaut="astro",
            scenario="moon",
            target_name="tile",
            option_idx=1,
            rigidity_ok=True,
            ease_ok=False,
            issues="cracks",
            notes="",
            extra={"overall": 9, "porosity": 2},
        ),
        impact.FeedbackEntry(
            ts_iso=(ts + timedelta(days=1)).isoformat(),
            astronaut="astro2",
            scenario="moon",
            target_name="tile",
            option_idx=2,
            rigidity_ok=False,
            ease_ok=True,
            issues="",
            notes="note",
            extra={"overall": 6, "unknown_field": "x"},
        ),
    ]

    run_ids = [impact.append_feedback(entry) for entry in entries]
    assert all(run_ids)

    files = sorted(logs_dir.glob("feedback_*.parquet"))
    assert len(files) == 2
    df0 = pd.read_parquet(files[0])
    assert "run_id" in df0.columns
    assert "scenario" in df0.columns
    assert df0.loc[0, "scenario"] == "moon"
    assert "extra_overall" in df0.columns

    loaded = impact.load_feedback_df()
    assert len(loaded) == 2
    assert set(loaded["run_id"]) == set(run_ids)
    assert "extra_unknown_field" in loaded.columns
    assert loaded.loc[1, "extra"]["unknown_field"] == "x"
