from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict

import pytest

from app.modules import logging_utils


def test_append_inference_log_uses_cached_writer(monkeypatch, tmp_path):
    """Ensure sequential appends reuse the same Parquet writer without reads."""

    logging_utils.shutdown_inference_logging()
    monkeypatch.setattr(logging_utils, "LOGS_ROOT", tmp_path)

    created_paths: list[Path] = []
    write_counts: list[int] = []

    class WriterSpy:
        def __init__(
            self, path: str, schema: Any, **kwargs: Any
        ) -> None:  # pragma: no cover - helper
            self.path = Path(path)
            self.schema = schema
            self.count = 0
            self.kwargs = kwargs

        def write_table(self, table: Any) -> None:  # pragma: no cover - helper
            self.count += 1
            write_counts.append(self.count)

        def close(self) -> None:  # pragma: no cover - helper
            pass

    def fake_writer(path: str, schema: Any, **kwargs: Any) -> WriterSpy:
        writer = WriterSpy(path, schema, **kwargs)
        created_paths.append(writer.path)
        return writer

    monkeypatch.setattr(logging_utils.pq, "ParquetWriter", fake_writer)

    def fail_read(*_args: Any, **_kwargs: Any) -> None:
        pytest.fail("append_inference_log should not trigger Parquet reads")

    if hasattr(logging_utils.pq, "read_table"):
        monkeypatch.setattr(logging_utils.pq, "read_table", fail_read)

    base = datetime(2024, 5, 4, 12, 0, tzinfo=UTC)
    events = [
        (
            base,
            {
                "timestamp": base.isoformat(),
                "input_features": "{}",
                "prediction": "{}",
                "uncertainty": "{}",
                "model_hash": "abc",
            },
        ),
        (
            base.replace(hour=18),
            {
                "timestamp": base.replace(hour=18).isoformat(),
                "input_features": "{\"foo\": 1}",
                "prediction": "{\"bar\": 2}",
                "uncertainty": "{}",
                "model_hash": "def",
            },
        ),
    ]

    def fake_prepare(*_args: Any, **_kwargs: Any) -> tuple[datetime, Dict[str, str | None]]:
        timestamp, payload = events.pop(0)
        return timestamp, payload

    monkeypatch.setattr(logging_utils, "prepare_inference_event", fake_prepare)

    manager = logging_utils.get_inference_log_manager()
    assert manager is logging_utils.get_inference_log_manager()

    logging_utils.append_inference_log({}, {}, {}, None)
    logging_utils.append_inference_log({}, {}, {}, None)

    expected_path = (
        tmp_path / "inference" / "20240504" / "inference_20240504.parquet"
    )
    assert created_paths == [expected_path]
    assert write_counts == [1, 2]

    logging_utils.shutdown_inference_logging()


def test_configure_inference_parquet_writer_passes_options(monkeypatch, tmp_path):
    logging_utils.shutdown_inference_logging()
    monkeypatch.setattr(logging_utils, "LOGS_ROOT", tmp_path)

    captured_kwargs: list[Dict[str, Any]] = []

    class DummyWriter:
        def write_table(self, table: Any) -> None:  # pragma: no cover - helper
            pass

        def close(self) -> None:  # pragma: no cover - helper
            pass

    def fake_writer(path: str, schema: Any, **kwargs: Any) -> DummyWriter:
        captured_kwargs.append(kwargs)
        return DummyWriter()

    monkeypatch.setattr(logging_utils.pq, "ParquetWriter", fake_writer)

    logging_utils.configure_inference_parquet_writer()
    logging_utils.append_inference_log({}, {}, {}, None)

    assert captured_kwargs[-1]["compression"] == "zstd"
    assert captured_kwargs[-1]["use_dictionary"] is True
    assert captured_kwargs[-1]["version"] == "2.6"

    logging_utils.shutdown_inference_logging()
    captured_kwargs.clear()

    logging_utils.configure_inference_parquet_writer(
        compression="brotli",
        use_dictionary=False,
    )

    logging_utils.append_inference_log({}, {}, {}, None)

    assert captured_kwargs[-1]["compression"] == "brotli"
    assert captured_kwargs[-1]["use_dictionary"] is False
    assert captured_kwargs[-1]["version"] == "2.6"

    logging_utils.configure_inference_parquet_writer()
    logging_utils.shutdown_inference_logging()
