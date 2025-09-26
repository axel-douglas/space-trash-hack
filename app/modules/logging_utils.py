"""Logging helpers shared by generator and analytics modules.

The original ``generator.py`` bundled inference telemetry writers alongside
candidate feature engineering which made the surface area difficult to test.
This module owns the durable write path for inference logs and exposes a small
API used by the generator and analytics tests.
"""

from __future__ import annotations

import json
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict, Tuple

try:  # Optional heavy dependencies; gracefully disable logging if missing
    import pyarrow as pa
except Exception:  # pragma: no cover - pyarrow is expected in production
    pa = None  # type: ignore[assignment]

try:  # ``deltalake`` provides lightweight Delta transactions
    from deltalake.writer import write_deltalake
except Exception:  # pragma: no cover - deltalake is expected in production
    write_deltalake = None  # type: ignore[assignment]

LOGS_ROOT = Path(__file__).resolve().parents[2] / "data" / "logs"

_INFERENCE_LOG_LOCK = threading.Lock()

__all__ = [
    "LOGS_ROOT",
    "resolve_inference_log_dir",
    "prepare_inference_event",
    "append_inference_log",
]


def resolve_inference_log_dir(timestamp: datetime) -> Path:
    """Return the Delta Lake directory for the given *timestamp*."""

    return LOGS_ROOT / "inference" / timestamp.strftime("%Y%m%d")


def _to_serializable(value: Any) -> Any:
    """Convert *value* into a JSON-serializable structure."""

    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _to_serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_serializable(v) for v in value]
    try:
        import numpy as np

        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return [_to_serializable(v) for v in value.tolist()]
    except Exception:  # pragma: no cover - numpy optional at runtime
        pass
    return str(value)


def prepare_inference_event(
    input_features: Dict[str, Any],
    prediction: Dict[str, Any] | None,
    uncertainty: Dict[str, Any] | None,
    model_registry: Any | None,
    timestamp: datetime | None = None,
) -> tuple[datetime, Dict[str, str | None]]:
    """Build the serializable payload for an inference log event."""

    now = timestamp or datetime.now(UTC)

    model_hash = ""
    if model_registry is not None:
        metadata = getattr(model_registry, "metadata", {}) or {}
        if isinstance(metadata, dict):
            model_hash = str(metadata.get("model_hash") or metadata.get("checksum") or "")
        if not model_hash:
            for attr in ("model_hash", "checksum", "pipeline_checksum", "pipeline_hash"):
                value = getattr(model_registry, attr, None)
                if value:
                    model_hash = str(value)
                    break

    payload: Dict[str, str | None] = {
        "timestamp": now.isoformat(timespec="microseconds"),
        "input_features": json.dumps(_to_serializable(input_features or {}), sort_keys=True),
        "prediction": json.dumps(_to_serializable(prediction or {}), sort_keys=True),
        "uncertainty": json.dumps(_to_serializable(uncertainty or {}), sort_keys=True),
        "model_hash": model_hash or None,
    }

    return now, payload


def append_inference_log(
    input_features: Dict[str, Any],
    prediction: Dict[str, Any] | None,
    uncertainty: Dict[str, Any] | None,
    model_registry: Any | None,
) -> None:
    """Persist an inference event using Delta transactions to avoid read-modify-write."""

    if pa is None or write_deltalake is None:  # pragma: no cover - dependencies should exist
        return

    try:
        LOGS_ROOT.mkdir(parents=True, exist_ok=True)
    except Exception:
        return

    event_time, event_payload = prepare_inference_event(
        input_features=input_features,
        prediction=prediction,
        uncertainty=uncertainty,
        model_registry=model_registry,
    )

    log_dir = resolve_inference_log_dir(event_time)

    try:
        log_dir.parent.mkdir(parents=True, exist_ok=True)
    except Exception:
        return

    data = {
        key: pa.array([value], type=pa.string())
        for key, value in event_payload.items()
    }

    table = pa.table(data)

    try:
        with _INFERENCE_LOG_LOCK:
            write_deltalake(
                str(log_dir),
                table,
                mode="append",
                schema_mode="merge",
                engine="rust",
            )
    except Exception:
        return
