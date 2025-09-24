from pathlib import Path

import pandas as pd

from app.modules import model_training
from scripts import ingest_feedback


def _build_feedback_row(recipe_id: str, process_id: str, seed: float) -> dict:
    row: dict[str, object] = {
        "recipe_id": recipe_id,
        "process_id": process_id,
        "label_source": "mission",
        "label_weight": 5.0,
        "rigidez": 0.75 + 0.02 * seed,
        "estanqueidad": 0.7 + 0.01 * seed,
        "energy_kwh": 3.0 + seed,
        "water_l": 1.5 + 0.2 * seed,
        "crew_min": 18.0 + seed,
        "tightness_pass": 1,
        "rigidity_level": 3,
        "conf_lo_rigidez": 0.7,
        "conf_hi_rigidez": 0.85,
        "conf_lo_estanqueidad": 0.65,
        "conf_hi_estanqueidad": 0.78,
        "conf_lo_energy_kwh": 2.8,
        "conf_hi_energy_kwh": 3.4,
        "conf_lo_water_l": 1.2,
        "conf_hi_water_l": 1.9,
        "conf_lo_crew_min": 16.5,
        "conf_hi_crew_min": 19.5,
        "notes": "Validated in thermal chamber",
        "measurement_ts": "2025-09-24T12:00:00Z",
        "operator_id": "astro-A",
    }

    for column in model_training.FEATURE_COLUMNS:
        if column == "process_id":
            row[column] = process_id
        elif column == "regolith_pct":
            row[column] = 0.35 + 0.01 * seed
        elif column == "num_items":
            row[column] = 3
        elif column == "density_kg_m3":
            row[column] = 0.9
        elif column == "moisture_frac":
            row[column] = 0.05
        elif column.endswith("_frac") or column.endswith("_index"):
            row[column] = 0.1
        elif column.endswith("_per_kg"):
            row[column] = 1.0
        elif column.startswith("target_"):
            row[column] = 0.8
        else:
            row.setdefault(column, 0.0)

    return row


def test_ingest_feedback_updates_gold_dataset(tmp_path: Path):
    gold_dir = tmp_path / "gold"
    gold_dir.mkdir(parents=True, exist_ok=True)

    existing_row = _build_feedback_row("R-EXISTING", "P02", seed=0.0)
    existing_row["rigidez"] = 0.4
    features_df = pd.DataFrame([{col: existing_row.get(col) for col in model_training.FEATURE_COLUMNS}])
    features_df.insert(0, "process_id", features_df.pop("process_id"))
    features_df.insert(0, "recipe_id", "R-EXISTING")
    features_df.to_parquet(gold_dir / "features.parquet", index=False)

    labels_df = pd.DataFrame(
        [
            {
                "recipe_id": "R-EXISTING",
                "process_id": "P02",
                "label_source": "mission",
                "label_weight": 5.0,
                "rigidez": 0.4,
                "estanqueidad": 0.6,
                "energy_kwh": 3.2,
                "water_l": 1.4,
                "crew_min": 18.0,
                "tightness_pass": 1,
                "rigidity_level": 3,
            }
        ]
    )
    labels_df.to_parquet(gold_dir / "labels.parquet", index=False)

    feedback_rows = [
        _build_feedback_row("R-EXISTING", "P02", seed=1.0),
        _build_feedback_row("R-NEW", "P03", seed=2.0),
    ]
    feedback_path = tmp_path / "feedback.parquet"
    pd.DataFrame(feedback_rows).to_parquet(feedback_path, index=False)

    summary = ingest_feedback.ingest_feedback([str(feedback_path)], gold_dir=gold_dir)

    features_after = pd.read_parquet(gold_dir / "features.parquet")
    labels_after = pd.read_parquet(gold_dir / "labels.parquet")

    assert set(features_after["recipe_id"]) == {"R-EXISTING", "R-NEW"}
    assert set(labels_after["recipe_id"]) == {"R-EXISTING", "R-NEW"}

    updated = labels_after.set_index("recipe_id").loc["R-EXISTING"]
    assert updated["rigidez"] > 0.4
    assert "ingested_at" in labels_after.columns

    assert summary["rows_ingested"] == 2
    assert summary["total_features"] == 2
    assert summary["total_labels"] == 2
