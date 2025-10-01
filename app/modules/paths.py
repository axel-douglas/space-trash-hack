"""Centralized filesystem locations for application artifacts."""

from __future__ import annotations

from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[2]

# Shared data locations -----------------------------------------------------

DATA_ROOT = _REPO_ROOT / "data"
"""Directory containing curated datasets and generated artifacts."""

MODELS_DIR = DATA_ROOT / "models"
"""Directory storing trained model bundles shipped with the app."""

LOGS_DIR = DATA_ROOT / "logs"
"""Directory where runtime telemetry such as inference logs is persisted."""

GOLD_DIR = DATA_ROOT / "gold"
"""Default location for the curated gold feature/label parquet files."""


__all__ = ["DATA_ROOT", "MODELS_DIR", "LOGS_DIR", "GOLD_DIR"]
