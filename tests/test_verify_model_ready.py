import json
from pathlib import Path

import pytest

from scripts import verify_model_ready as verifier


def test_collect_regression_metrics_extracts_numeric_values() -> None:
    metadata = {
        "artifacts": {
            "xgboost": {
                "metrics": {
                    "crew_min": {"mae": "1.5", "rmse": 2.0, "r2": 0.9},
                    "energy_kwh": {"mae": None, "rmse": "3.1"},
                    "invalid": "not-a-metric",
                }
            },
            "bad": "ignore",
        }
    }

    metrics = verifier._collect_regression_metrics(metadata)
    assert metrics == {
        "xgboost": {
            "crew_min": {"mae": 1.5, "rmse": 2.0},
            "energy_kwh": {"rmse": 3.1},
        }
    }


def test_collect_classifier_curves_validates_paths(tmp_path: Path) -> None:
    roc = tmp_path / "roc.png"
    pr = tmp_path / "pr.png"
    roc.write_bytes(b"binary")
    pr.write_bytes(b"binary")

    payload = {
        "demo": {
            "roc_curve_path": roc,
            "pr_curve_path": pr,
        },
        "noop": {},
    }

    curves = verifier._collect_classifier_curves(payload)
    assert curves == {
        "demo": {
            "roc_curve_path": roc.as_posix(),
            "pr_curve_path": pr.as_posix(),
        }
    }

    with pytest.raises(SystemExit):
        verifier._collect_classifier_curves({"demo": {"roc_curve_path": tmp_path / "missing.png"}})


def test_main_emits_metrics_payload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    metadata = {
        "artifacts": {
            "xgboost": {
                "metrics": {
                    "crew_min": {"mae": 1.0, "rmse": 2.0},
                },
                "path": "data/models/rexai_regressor.joblib",
            }
        },
        "residual_std": {"crew_min": 0.1},
        "residuals_summary": {"crew_min": {"mean": 0.0}},
        "residuals_by_label_source": {"training": {"count": 1}},
        "targets": ["crew_min"],
        "trained_on": "fixtures.csv",
        "trained_at": "2024-01-01T00:00:00Z",
    }

    class DummyRegistry:
        ready = True
        feature_names = ["f1"]

        def __init__(self, *args, **kwargs) -> None:
            self.metadata = metadata

        def trained_label(self) -> str:
            return "crew_min"

    class DummyModels:
        TARGET_COLUMNS = ["crew_min"]
        DATA_ROOT = Path(tmp_path)
        PIPELINE_PATH = tmp_path / "pipeline.joblib"
        METADATA_PATH = tmp_path / "metadata.json"
        LEGACY_METADATA_PATH = tmp_path / "legacy.json"
        ModelRegistry = DummyRegistry

    DummyModels.PIPELINE_PATH.write_text("{}", encoding="utf-8")
    DummyModels.METADATA_PATH.write_text(json.dumps(metadata), encoding="utf-8")

    monkeypatch.setattr(verifier, "ml_models", DummyModels)

    verifier.main()

    output = json.loads(capsys.readouterr().out)

    assert output["regression_metrics"]["xgboost"]["crew_min"]["mae"] == pytest.approx(1.0)
    assert output["mae_mean"] == pytest.approx(1.0)
