"""CLI utility to assert that the Rex-AI model bundle is ready for release."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping

from app.modules import ml_models


def _check_residuals(metadata: Mapping[str, Any]) -> Dict[str, Any]:
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


def _collect_regression_metrics(metadata: Mapping[str, Any]) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Extract MAE/RMSE metrics for each artefact and target."""

    metrics_payload: Dict[str, Dict[str, Dict[str, float]]] = {}
    artifacts = metadata.get("artifacts", {}) if isinstance(metadata, Mapping) else {}
    if not isinstance(artifacts, Mapping):
        return metrics_payload

    for name, info in artifacts.items():
        if not isinstance(info, Mapping):
            continue
        metric_block = info.get("metrics")
        if not isinstance(metric_block, Mapping):
            continue
        targets: Dict[str, Dict[str, float]] = {}
        for target, values in metric_block.items():
            if not isinstance(values, Mapping):
                continue
            entry: Dict[str, float] = {}
            for key in ("mae", "rmse"):
                value = values.get(key)
                if value is None:
                    continue
                try:
                    entry[key] = float(value)
                except (TypeError, ValueError):
                    continue
            if entry:
                targets[str(target)] = entry
        if targets:
            metrics_payload[str(name)] = targets
    return metrics_payload


def _collect_classifier_curves(classifiers: Mapping[str, Any]) -> Dict[str, Dict[str, str]]:
    """Return verified ROC/PR curve paths for each classifier."""

    curves: Dict[str, Dict[str, str]] = {}
    for name, payload in classifiers.items():
        if not isinstance(payload, Mapping):
            continue
        curve_paths: Dict[str, str] = {}
        for key in ("roc_curve_path", "pr_curve_path"):
            raw_path = payload.get(key)
            if not raw_path:
                continue
            candidate = Path(raw_path)
            if not candidate.exists():
                raise SystemExit(f"Classifier {name} declares {key} but file is missing: {candidate}")
            curve_paths[key] = candidate.as_posix()
        if curve_paths:
            curves[str(name)] = curve_paths
    return curves


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

    regression_metrics = _collect_regression_metrics(metadata)

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
        "regression_metrics": regression_metrics,
    }

    if regression_metrics:
        overall_mae = []
        overall_rmse = []
        for targets in regression_metrics.values():
            for values in targets.values():
                mae = values.get("mae")
                rmse = values.get("rmse")
                if mae is not None:
                    overall_mae.append(mae)
                if rmse is not None:
                    overall_rmse.append(rmse)
        if overall_mae:
            payload["mae_mean"] = sum(overall_mae) / len(overall_mae)
        if overall_rmse:
            payload["rmse_mean"] = sum(overall_rmse) / len(overall_rmse)

    classifier_curves = _collect_classifier_curves(classifiers)
    if classifier_curves:
        payload["classifier_curves"] = classifier_curves

    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":  # pragma: no cover - script entry point
    main()
