from __future__ import annotations

import sys
from pathlib import Path

from app.bootstrap import ensure_streamlit_entrypoint


def _assert_app_package_present(root: Path) -> None:
    app_package = root / "app"
    assert app_package.is_dir()
    assert (app_package / "__init__.py").is_file()


def test_ensure_streamlit_entrypoint_for_home_includes_app(monkeypatch) -> None:
    monkeypatch.setattr(sys, "path", [])

    root = ensure_streamlit_entrypoint(Path("app/Home.py"))

    assert sys.path[0] == str(root)
    _assert_app_package_present(root)


def test_ensure_streamlit_entrypoint_for_nested_page_includes_app(monkeypatch) -> None:
    monkeypatch.setattr(sys, "path", [])

    root = ensure_streamlit_entrypoint(Path("app/pages/x.py"))

    assert sys.path[0] == str(root)
    _assert_app_package_present(root)


def test_ensure_streamlit_entrypoint_accepts_string_path(monkeypatch) -> None:
    monkeypatch.setattr(sys, "path", [])

    root = ensure_streamlit_entrypoint("app/pages/another_page.py")

    assert sys.path[0] == str(root)
    _assert_app_package_present(root)
