from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict

import pytest

from app.modules import logging_utils


def test_append_inference_log_uses_cached_writer(monkeypatch, tmp_path):
    """Ensure sequential appends reuse the same Parquet writer without reads."""

    logging_utils._INFERENCE_LOG_MANAGER.close()

    manager = logging_utils._InferenceLogWriterManager()
    monkeypatch.setattr(logging_utils, "_INFERENCE_LOG_MANAGER", manager)
    monkeypatch.setattr(logging_utils, "LOGS_ROOT", tmp_path)

    created_paths: list[Path] = []
    write_counts: list[int] = []

    class WriterSpy:
        def __init__(self, path: str, schema: Any) -> None:  # pragma: no cover - helper
            self.path = Path(path)
            self.schema = schema
            self.count = 0

        def write_table(self, table: Any) -> None:  # pragma: no cover - helper
            self.count += 1
            write_counts.append(self.count)

        def close(self) -> None:  # pragma: no cover - helper
            pass

    def fake_writer(path: str, schema: Any) -> WriterSpy:
        writer = WriterSpy(path, schema)
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

    logging_utils.append_inference_log({}, {}, {}, None)
    logging_utils.append_inference_log({}, {}, {}, None)

    expected_path = (
        tmp_path / "inference" / "20240504" / "inference_20240504.parquet"
    )
    assert created_paths == [expected_path]
    assert write_counts == [1, 2]

    manager.close()
