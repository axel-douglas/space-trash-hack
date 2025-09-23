"""CLI utility to assert that the Rex-AI model bundle is ready for release."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from app.modules import ml_models


def _check_residuals(metadata: Dict[str, Any]) -> Dict[str, Any]:
    required = set(ml_models.TARGET_COLUMNS)
    residual_std = metadata.get("residual_std", {})
    if required - residual_std.keys():
        missing = sorted(required - residual_std.keys())
        raise SystemExit(f"Missing residual_std entries for: {', '.join(missing)}")

    summary = metadata.get("residuals_summary")
    if not isinstance(summary, dict) or required - summary.keys():
        raise SystemExit("Metadata.residuals_summary must contain entries for all targets")

    by_source = metadata.get("residuals_by_label_source")
    if not isinstance(by_source, dict) or not by_source:
        raise SystemExit("Metadata.residuals_by_label_source is required for auditability")

    return {
        "residual_std": residual_std,
        "residuals_summary": summary,
        "sources": {source: payload.get("count", 0) for source, payload in by_source.items()},
    }


def main() -> None:
    registry = ml_models.ModelRegistry()
    if not registry.ready:
        raise SystemExit("ModelRegistry.ready is False — did you generate data/models/*?")

    pipeline_path = ml_models.PIPELINE_PATH
    if not pipeline_path.exists():
        raise SystemExit(f"Pipeline artefact missing: {pipeline_path}")

    metadata_path = ml_models.METADATA_PATH if ml_models.METADATA_PATH.exists() else ml_models.LEGACY_METADATA_PATH
    if metadata_path is None or not metadata_path.exists():
        raise SystemExit("Metadata file not found. Run the training pipeline before releasing.")

    metadata = registry.metadata

    trained_on = metadata.get("trained_on")
    if not isinstance(trained_on, str) or not trained_on.strip():
        raise SystemExit("Metadata.trained_on is required to describe training provenance")

    residuals = _check_residuals(metadata)

    artefacts = {
        "pipeline": str(pipeline_path.relative_to(ml_models.DATA_ROOT.parent)),
    }
    xgb = metadata.get("artifacts", {}).get("xgboost", {})
    if xgb.get("path"):
        artefact_path = Path(xgb["path"])
        artefacts["xgboost"] = artefact_path.as_posix()

    classifiers = metadata.get("classifiers", {})
    classifier_paths = {
        name: info.get("path") for name, info in classifiers.items() if info.get("path")
    }

    payload = {
        "ready": registry.ready,
        "trained_on": trained_on,
        "trained_at": metadata.get("trained_at"),
        "n_features": len(registry.feature_names),
        "n_targets": len(metadata.get("targets", [])),
        "trained_label": registry.trained_label(),
        "artefacts": artefacts,
        "classifiers": classifier_paths,
        "residuals": residuals,
    }

    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":  # pragma: no cover - script entry point
    main()
