"""Ensure Streamlit entrypoints can resolve ``app`` imports when run as scripts."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


class _EarlyStreamlitExit(RuntimeError):
    """Signal that the patched Streamlit function was invoked."""


_ENTRYPOINT_FILES = [
    Path("app/Home.py"),
    *sorted(Path("app/pages").glob("*.py")),
]


@pytest.mark.parametrize("entrypoint", _ENTRYPOINT_FILES, ids=lambda path: path.name)
def test_streamlit_entrypoints_import_without_repo_on_path(entrypoint: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Import each entrypoint with the repository root temporarily removed."""

    repo_root = Path(__file__).resolve().parents[2]
    clean_sys_path: list[str] = []
    for item in sys.path:
        try:
            resolved = Path(item).resolve()
        except TypeError:  # pragma: no cover - defensive for unusual path entries
            clean_sys_path.append(item)
            continue
        if resolved == repo_root:
            continue
        clean_sys_path.append(item)
    monkeypatch.setattr(sys, "path", clean_sys_path)

    streamlit = pytest.importorskip("streamlit")

    def _exit_on_config(*_args, **_kwargs) -> None:
        raise _EarlyStreamlitExit("set_page_config")

    monkeypatch.setattr(streamlit, "set_page_config", _exit_on_config)

    module_name = f"_entrypoints_{entrypoint.stem.replace('-', '_').replace('.', '_')}"
    spec = importlib.util.spec_from_file_location(module_name, entrypoint)
    assert spec is not None and spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except _EarlyStreamlitExit:
        # Reaching the stub indicates the module was imported successfully and
        # executed far enough to touch Streamlit, which is more than enough for
        # this regression test.
        pass
    finally:
        sys.modules.pop(module_name, None)
