"""Ensure Streamlit entrypoints can be imported without path hacks."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


class _StreamlitAbort(RuntimeError):
    """Sentinel exception raised when ``st.stop`` is invoked."""


ENTRYPOINT_FILES = [
    Path("app/Home.py"),
    *sorted(Path("app/pages").glob("*.py")),
]


def _sanitise_module_name(path: Path) -> str:
    stem = path.stem
    sanitized = "".join(char if char.isalnum() or char == "_" else "_" for char in stem)
    if sanitized and sanitized[0].isdigit():
        sanitized = f"page_{sanitized}"
    return f"tests.ui.entrypoints.{sanitized}"


@pytest.mark.parametrize("module_path", ENTRYPOINT_FILES)
def test_entrypoint_imports_without_module_not_found(
    module_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Import each Streamlit entrypoint and ensure dependencies resolve."""

    module_name = _sanitise_module_name(module_path)
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec is not None, f"Expected import spec for '{module_path}'"
    assert spec.loader is not None, f"Expected loader in import spec for '{module_path}'"
    sys.modules.pop(module_name, None)
    module = importlib.util.module_from_spec(spec)
    try:
        import streamlit as st
    except ModuleNotFoundError:
        pytest.skip("streamlit is required for entrypoint imports")

    def _stop() -> None:
        raise _StreamlitAbort

    monkeypatch.setattr(st, "stop", _stop, raising=False)
    try:
        spec.loader.exec_module(module)
    except ModuleNotFoundError as error:  # pragma: no cover - explicit failure path
        pytest.fail(f"Unexpected ModuleNotFoundError importing '{module_path}': {error}")
    except _StreamlitAbort:
        pass
    assert module.__name__ == module_name

