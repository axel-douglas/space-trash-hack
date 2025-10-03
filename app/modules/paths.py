"""Centralized filesystem locations for application artifacts."""

from __future__ import annotations

import os
from pathlib import Path


_ENV_DATA_ROOT = "REXAI_DATA_ROOT"
_ENV_MODELS_DIR = "REXAI_MODELS_DIR"

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _normalise_path(value: str | Path) -> Path:
    """Return an absolute ``Path`` while being forgiving with inputs."""

    candidate = Path(value).expanduser()
    try:
        return candidate.resolve()
    except RuntimeError:
        # ``resolve`` can raise on recursive symlinks; fall back to ``absolute``.
        return candidate.absolute()


def _path_from_env(var_name: str, default: Path) -> Path:
    """Load ``var_name`` from the environment, normalising it when available."""

    raw_value = os.environ.get(var_name)
    if raw_value is None:
        return default

    stripped = raw_value.strip()
    if not stripped:
        return default

    return _normalise_path(stripped)


# Shared data locations -----------------------------------------------------

DATA_ROOT = _path_from_env(_ENV_DATA_ROOT, _normalise_path(_REPO_ROOT / "data"))
"""Directory containing curated datasets and generated artifacts."""

MODELS_DIR = _path_from_env(_ENV_MODELS_DIR, DATA_ROOT / "models")
"""Directory storing trained model bundles shipped with the app."""

LOGS_DIR = DATA_ROOT / "logs"
"""Directory where runtime telemetry such as inference logs is persisted."""

GOLD_DIR = DATA_ROOT / "gold"
"""Default location for the curated gold feature/label parquet files."""


__all__ = ["DATA_ROOT", "MODELS_DIR", "LOGS_DIR", "GOLD_DIR"]
