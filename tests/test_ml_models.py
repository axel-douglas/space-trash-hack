"""Tests for model bundle bootstrap in ModelRegistry."""

from __future__ import annotations

import hashlib
import io
import json
import zipfile
from pathlib import Path

import joblib
import numpy as np
import pytest
import responses

from app.modules import ml_models


class DummyPipeline:
    """Minimal pipeline object for joblib serialization in tests."""

    def __init__(self) -> None:
        self.named_steps = {"preprocess": None}

    def predict(self, frame):  # pragma: no cover - not exercised in this test
        return [[0.0 for _ in ml_models.TARGET_COLUMNS]]


@pytest.fixture(autouse=True)
def reset_bundle_env(monkeypatch):
    monkeypatch.delenv("MODEL_BUNDLE_URL", raising=False)
    monkeypatch.delenv("MODEL_BUNDLE_SHA256", raising=False)


def _prepare_bundle(tmp_path: Path) -> tuple[bytes, str]:
    bundle_root = tmp_path / "bundle_root"
    bundle_root.mkdir()

    joblib.dump(DummyPipeline(), bundle_root / "rexai_regressor.joblib")
    (bundle_root / "metadata_gold.json").write_text(json.dumps({"trained_on": "test"}), encoding="utf-8")

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for item in bundle_root.iterdir():
            archive.write(item, item.name)
    payload = buffer.getvalue()
    digest = hashlib.sha256(payload).hexdigest()
    return payload, digest


def _write_constant_lightgbm_model(path: Path, values: list[float]) -> None:
    onnx = pytest.importorskip("onnx")
    from onnx import TensorProto, helper

    tensor = helper.make_tensor(
        name="const_values",
        data_type=TensorProto.FLOAT,
        dims=[len(values)],
        vals=[float(v) for v in values],
    )
    const_node = helper.make_node("Constant", inputs=[], outputs=["const"], value=tensor)
    add_node = helper.make_node("Add", inputs=["input", "const"], outputs=["sum"])
    graph = helper.make_graph(
        nodes=[const_node, add_node],
        name="ConstAdd",
        inputs=[helper.make_tensor_value_info("input", TensorProto.FLOAT, ["N", len(values)])],
        outputs=[helper.make_tensor_value_info("sum", TensorProto.FLOAT, ["N", len(values)])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=10)
    onnx.save(model, path)


@responses.activate
def test_model_bundle_download_skips_bootstrap(monkeypatch, tmp_path):
    models_dir = tmp_path / "models"
    artifacts = ml_models.resolve_artifact_paths(models_dir)

    payload, digest = _prepare_bundle(tmp_path)
    url = "https://example.com/model.zip"
    responses.add(responses.GET, url, body=payload, status=200, content_type="application/zip")

    monkeypatch.setenv("MODEL_BUNDLE_URL", url)
    monkeypatch.setenv("MODEL_BUNDLE_SHA256", digest)

    bootstrap_calls: list[bool] = []

    def fake_bootstrap():
        bootstrap_calls.append(True)
        return str(artifacts.pipeline)

    monkeypatch.setattr("app.modules.model_training.bootstrap_demo_model", fake_bootstrap)

    registry = ml_models.ModelRegistry(model_dir=models_dir)

    assert registry.ready is True
    assert isinstance(registry.pipeline, DummyPipeline)
    assert registry.artifacts.pipeline == artifacts.pipeline
    assert artifacts.pipeline.exists()
    assert registry.metadata.get("trained_on") == "test"
    assert not bootstrap_calls
    responses.assert_call_count(url, 1)


def test_model_registry_parses_label_summary(tmp_path):
    models_dir = tmp_path / "models"
    models_dir.mkdir()

    artifacts = ml_models.resolve_artifact_paths(models_dir)

    joblib.dump(DummyPipeline(), artifacts.pipeline)

    metadata_path = artifacts.metadata
    metadata_payload = {
        "trained_at": "2025-01-01T00:00:00Z",
        "trained_on": "gold_v1",
        "trained_label": "gold_v1",
        "n_samples": 10,
        "feature_means": {},
        "feature_stds": {},
        "post_transform_features": [],
        "targets": ml_models.TARGET_COLUMNS,
        "random_forest": {"feature_importance": {"average": []}},
        "labeling": {
            "columns": {"source": "label_source", "weight": "label_weight"},
            "summary": {
                "mission": {
                    "count": 3,
                    "mean_weight": 1.0,
                    "min_weight": 1.0,
                    "max_weight": 1.0,
                },
                "simulated": {
                    "count": 2,
                    "mean_weight": 0.6,
                    "min_weight": 0.5,
                    "max_weight": 0.7,
                },
            },
        },
    }
    metadata_path.write_text(json.dumps(metadata_payload), encoding="utf-8")

    registry = ml_models.ModelRegistry(model_dir=models_dir)

    assert registry.label_columns == {"source": "label_source", "weight": "label_weight"}
    assert registry.label_summary["mission"]["count"] == 3
    assert registry.label_summary["simulated"]["mean_weight"] == 0.6
    label_text = registry.label_distribution_label()
    assert "mission×3" in label_text
    assert "simulated×2" in label_text


@pytest.mark.skipif(not ml_models.HAS_ONNXRUNTIME, reason="onnxruntime not available")
def test_model_registry_uses_lightgbm_variant(tmp_path):
    models_dir = tmp_path / "models"
    models_dir.mkdir()

    artifacts = ml_models.resolve_artifact_paths(models_dir)

    joblib.dump(DummyPipeline(), artifacts.pipeline)

    onnx_path = artifacts.lightgbm
    constant_output = [0.72, 0.68, 4.5, 1.8, 18.0]
    _write_constant_lightgbm_model(onnx_path, constant_output)

    metadata_path = artifacts.metadata
    metadata_payload = {
        "trained_at": "2025-01-01T00:00:00Z",
        "trained_on": "synthetic_v0",
        "post_transform_features": [f"f{i}" for i in range(len(ml_models.TARGET_COLUMNS))],
        "random_forest": {"feature_importance": {"average": []}},
        "artifacts": {
            "lightgbm_gpu": {
                "path": onnx_path.name,
                "metrics": {"overall": {"mae": 0.5}},
                "backend": "gpu",
                "format": "onnx",
            }
        },
    }
    metadata_path.write_text(json.dumps(metadata_payload), encoding="utf-8")

    registry = ml_models.ModelRegistry(model_dir=models_dir)

    assert registry.lightgbm_session is not None
    assert registry.lightgbm_meta.get("backend") == "gpu"
    assert registry.lightgbm_input_name

    matrix = np.zeros((1, len(ml_models.TARGET_COLUMNS)), dtype=float)
    variants = registry._predict_variants(matrix)
    assert "lightgbm_gpu" in variants
    for idx, target in enumerate(ml_models.TARGET_COLUMNS):
        assert variants["lightgbm_gpu"][target] == pytest.approx(constant_output[idx])
