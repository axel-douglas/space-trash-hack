"""Regression tests ensuring the legacy charts helper stays removed."""
from __future__ import annotations

from pathlib import Path


def test_no_legacy_charts_helper_references() -> None:
    """Ensure the deprecated charts helper is not referenced anywhere."""
    repo_root = Path(__file__).resolve().parents[1]
    targets = ["predictions_ci_" + "chart", "modules." + "charts"]

    offenders: list[tuple[Path, str]] = []
    for path in repo_root.rglob("*.py"):
        text = path.read_text(encoding="utf-8", errors="ignore")
        for target in targets:
            if target in text:
                offenders.append((path, target))

    assert not offenders, f"legacy helper references found: {offenders}"
