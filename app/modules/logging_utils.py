"""Logging helpers shared by generator and analytics modules.

The original ``generator.py`` bundled inference telemetry writers alongside
candidate feature engineering which made the surface area difficult to test.
This module owns the durable write path for inference logs and exposes a small
API used by the generator and analytics tests.

Inference events are persisted as Parquet files compressed with Zstandard by
default. Call :func:`configure_inference_parquet_writer` to override the codec
or other :class:`pyarrow.parquet.ParquetWriter` keyword arguments.
"""

from __future__ import annotations

import atexit
import json
import threading
from datetime import UTC, datetime
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, Mapping, Tuple

try:  # Optional heavy dependencies; gracefully disable logging if missing
    import pyarrow as pa
    import pyarrow.parquet as pq
except Exception:  # pragma: no cover - pyarrow is expected in production
    pa = None  # type: ignore[assignment]
    pq = None  # type: ignore[assignment]

from .paths import LOGS_DIR

LOGS_ROOT = LOGS_DIR

_DEFAULT_PARQUET_WRITER_KWARGS: Dict[str, Any] = {
    "compression": "zstd",
    "use_dictionary": True,
    "version": "2.6",
}
_PARQUET_WRITER_KWARGS: Dict[str, Any] = dict(_DEFAULT_PARQUET_WRITER_KWARGS)


@dataclass
class _InferenceWriterState:
    """Track the active Parquet writer along with its schema metadata."""

    date_token: str
    path: Path
    schema: "pa.Schema"
    writer: "pq.ParquetWriter"


class _InferenceLogWriterManager:
    """Manage a single append-only Parquet writer with daily rotation."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._state: _InferenceWriterState | None = None

    def close(self) -> None:
        """Close the active writer, if any."""

        if pq is None:
            return

        with self._lock:
            self._close_locked()

    def _close_locked(self) -> None:
        state = self._state
        if state is None:
            return

        try:
            state.writer.close()
        except Exception:  # pragma: no cover - best effort cleanup
            pass

        self._state = None

    def _open_locked(
        self, timestamp: datetime, field_names: Iterable[str]
    ) -> _InferenceWriterState | None:
        if pa is None or pq is None:
            return None

        desired_fields = sorted(set(str(name) for name in field_names))
        if not desired_fields:
            return None

        log_dir = resolve_inference_log_dir(timestamp)
        try:
            log_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            return None

        date_token = timestamp.strftime("%Y%m%d")
        path = self._resolve_log_path(log_dir, date_token)
        schema = pa.schema(pa.field(name, pa.string()) for name in desired_fields)

        try:
            writer = pq.ParquetWriter(
                str(path),
                schema=schema,
                **_PARQUET_WRITER_KWARGS,
            )
        except Exception:
            return None

        state = _InferenceWriterState(
            date_token=date_token,
            path=path,
            schema=schema,
            writer=writer,
        )
        self._state = state
        return state

    def _resolve_log_path(self, log_dir: Path, date_token: str) -> Path:
        """Return a Parquet path for *date_token* avoiding overwriting shards."""

        base = log_dir / f"inference_{date_token}.parquet"
        if not base.exists():
            return base

        counter = 1
        while True:
            candidate = log_dir / f"inference_{date_token}_{counter:04d}.parquet"
            if not candidate.exists():
                return candidate
            counter += 1

    def _ensure_state_locked(
        self, timestamp: datetime, field_names: Iterable[str]
    ) -> _InferenceWriterState | None:
        state = self._state
        desired_fields = sorted(set(str(name) for name in field_names))
        if not desired_fields:
            return None

        date_token = timestamp.strftime("%Y%m%d")
        if state is not None:
            if state.date_token != date_token or set(state.schema.names) != set(
                desired_fields
            ):
                self._close_locked()
                state = None

        if state is None:
            state = self._open_locked(timestamp, desired_fields)

        return state

    def write_event(self, timestamp: datetime, payload: Mapping[str, str | None]) -> None:
        if pa is None or pq is None:
            return

        with self._lock:
            state = self._ensure_state_locked(timestamp, payload.keys())
            if state is None:
                return

            arrays = [
                pa.array([payload.get(field.name)], type=field.type)
                for field in state.schema
            ]
            table = pa.Table.from_arrays(arrays, schema=state.schema)

            try:
                state.writer.write_table(table)
            except Exception:
                self._close_locked()


_INFERENCE_LOG_MANAGER = _InferenceLogWriterManager()
_SHUTDOWN_HOOK: Callable[[], None] | None = None
_SHUTDOWN_LOCK = threading.Lock()


def configure_inference_parquet_writer(**kwargs: Any) -> None:
    """Update the keyword arguments used when constructing Parquet writers.

    Parameters mirror :class:`pyarrow.parquet.ParquetWriter`. Passing an empty
    set of keyword arguments resets the defaults (Zstandard compression,
    dictionary encoding, and Parquet v2.6 metadata).
    """

    global _PARQUET_WRITER_KWARGS
    if kwargs:
        _PARQUET_WRITER_KWARGS = dict(_DEFAULT_PARQUET_WRITER_KWARGS)
        _PARQUET_WRITER_KWARGS.update(kwargs)
    else:
        _PARQUET_WRITER_KWARGS = dict(_DEFAULT_PARQUET_WRITER_KWARGS)

__all__ = [
    "LOGS_ROOT",
    "configure_inference_parquet_writer",
    "resolve_inference_log_dir",
    "prepare_inference_event",
    "append_inference_log",
    "get_inference_log_manager",
    "shutdown_inference_logging",
]


def resolve_inference_log_dir(timestamp: datetime) -> Path:
    """Return the directory storing Parquet logs for *timestamp*."""

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
    """Persist an inference event using a streaming Parquet writer."""

    if pa is None or pq is None:  # pragma: no cover - dependencies should exist
        return

    event_time, event_payload = prepare_inference_event(
        input_features=input_features,
        prediction=prediction,
        uncertainty=uncertainty,
        model_registry=model_registry,
    )

    get_inference_log_manager().write_event(event_time, event_payload)


def get_inference_log_manager() -> _InferenceLogWriterManager:
    """Return the singleton inference log manager, registering shutdown hooks."""

    if pq is None:  # pragma: no cover - dependencies should exist in prod
        return _INFERENCE_LOG_MANAGER

    global _SHUTDOWN_HOOK
    if _SHUTDOWN_HOOK is None:
        with _SHUTDOWN_LOCK:
            if _SHUTDOWN_HOOK is None:
                def _close_manager() -> None:
                    _INFERENCE_LOG_MANAGER.close()

                atexit.register(_close_manager)
                _SHUTDOWN_HOOK = _close_manager

    return _INFERENCE_LOG_MANAGER


def shutdown_inference_logging() -> None:
    """Close the inference log manager and unregister the exit hook."""

    global _SHUTDOWN_HOOK

    hook = _SHUTDOWN_HOOK
    if hook is not None:
        with _SHUTDOWN_LOCK:
            hook = _SHUTDOWN_HOOK
            if hook is not None:
                try:
                    atexit.unregister(hook)
                except Exception:  # pragma: no cover - best effort cleanup
                    pass
                _SHUTDOWN_HOOK = None

    _INFERENCE_LOG_MANAGER.close()
