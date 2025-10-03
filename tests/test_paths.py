from __future__ import annotations

import importlib
from pathlib import Path
from types import ModuleType

import pytest

from app.modules import paths, ui_blocks


@pytest.fixture
def reload_paths(monkeypatch):
    """Reload ``app.modules.paths`` after adjusting environment variables."""

    def _reload(**env: str) -> ModuleType:
        for key in ("REXAI_DATA_ROOT", "REXAI_MODELS_DIR"):
            monkeypatch.delenv(key, raising=False)
        for key, value in env.items():
            monkeypatch.setenv(key, value)
        return importlib.reload(paths)

    yield _reload

    for key in ("REXAI_DATA_ROOT", "REXAI_MODELS_DIR"):
        monkeypatch.delenv(key, raising=False)
    importlib.reload(paths)


def test_path_constants_align_with_repository_structure(reload_paths) -> None:
    """Smoke test ensuring filesystem constants remain in sync."""

    module = reload_paths()
    repo_root = Path(__file__).resolve().parents[1]

    assert module.DATA_ROOT == (repo_root / "data").resolve()
    assert module.DATA_ROOT.is_dir()
    assert module.MODELS_DIR == module.DATA_ROOT / "models"
    assert module.MODELS_DIR.is_dir()
    assert module.LOGS_DIR.parent == module.DATA_ROOT
    assert module.LOGS_DIR.name == "logs"


def test_initialise_frontend_invokes_theme_helpers(monkeypatch) -> None:
    """``initialise_frontend`` should cascade to theme loaders in order."""

    calls: list[str] = []

    def fake_load_theme() -> None:
        calls.append("load")

    def fake_apply() -> None:
        calls.append("apply")

    monkeypatch.setattr(ui_blocks, "load_theme", fake_load_theme)
    monkeypatch.setattr(ui_blocks, "apply_global_visual_theme", fake_apply)

    ui_blocks.initialise_frontend()

    assert calls == ["load", "apply"]


def test_data_root_environment_variable_overrides_default(tmp_path, reload_paths) -> None:
    """``REXAI_DATA_ROOT`` should take precedence when defined."""

    custom_root = tmp_path / "custom-root"
    custom_root.mkdir()

    module = reload_paths(REXAI_DATA_ROOT=str(custom_root))

    assert module.DATA_ROOT == custom_root.resolve()
    assert module.MODELS_DIR == module.DATA_ROOT / "models"
    assert module.LOGS_DIR == module.DATA_ROOT / "logs"
    assert module.GOLD_DIR == module.DATA_ROOT / "gold"


def test_models_dir_environment_variable_overrides_default(tmp_path, reload_paths) -> None:
    """``REXAI_MODELS_DIR`` should replace the derived models directory."""

    custom_models = tmp_path / "custom-models"
    custom_models.mkdir()

    module = reload_paths(REXAI_MODELS_DIR=str(custom_models))

    repo_root = Path(__file__).resolve().parents[1]
    assert module.DATA_ROOT == (repo_root / "data").resolve()
    assert module.MODELS_DIR == custom_models.resolve()
