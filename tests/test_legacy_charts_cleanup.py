"""Regression tests ensuring removed legacy helpers stay gone."""
from __future__ import annotations

from pathlib import Path


def test_no_legacy_helper_references() -> None:
    """Ensure deprecated helpers are not referenced anywhere in the codebase."""
    repo_root = Path(__file__).resolve().parents[1]
    targets = [
        "predictions_ci_" + "chart",
        "modules." + "charts",
        "modules." + "branding",
        "modules." + "embeddings",
        "from app.modules import branding",
        "from app.modules import charts",
        "from app.modules import embeddings",
        "import app.modules.branding",
        "import app.modules.charts",
        "import app.modules.embeddings",
    ]

    offenders: list[tuple[Path, str]] = []
    for path in repo_root.rglob("*.py"):
        if path == Path(__file__):
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for target in targets:
            if target in text:
                offenders.append((path, target))

    assert not offenders, f"legacy helper references found: {offenders}"
