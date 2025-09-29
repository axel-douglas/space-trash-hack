from __future__ import annotations

import importlib
import sys
from pathlib import Path

from app.modules import paths


def test_path_constants_align_with_repository_structure() -> None:
    """Smoke test ensuring filesystem constants remain in sync."""

    assert paths.DATA_ROOT.is_dir()
    assert paths.MODELS_DIR.is_dir()
    assert paths.LOGS_DIR.is_dir()
    assert paths.LOGS_DIR.parent == paths.DATA_ROOT


def test_bootstrap_allows_app_import_from_app_directory(monkeypatch) -> None:
    """Importing ``_bootstrap`` should make ``app`` available even when cwd is ``app/``."""

    monkeypatch.chdir(Path("app"))
    monkeypatch.setattr(sys, "path", [str(Path.cwd())])
    sys.modules.pop("app", None)
    sys.modules.pop("_bootstrap", None)

    importlib.import_module("_bootstrap")
    module = importlib.import_module("app")

    assert Path(module.__file__).resolve().name == "__init__.py"
