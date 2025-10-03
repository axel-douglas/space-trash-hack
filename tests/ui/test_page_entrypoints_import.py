"""Ensure Streamlit entrypoints are discoverable without path hacks."""

from __future__ import annotations

import importlib
import importlib.util
import sys

import pytest


PAGE_MODULES = [
    "app.Home",
    "app.pages.0_Mission_Overview",
    "app.pages.2_Target_Designer",
    "app.pages.3_Generator",
    "app.pages.4_Results_and_Tradeoffs",
    "app.pages.5_Compare_and_Explain",
    "app.pages.6_Pareto_and_Export",
    "app.pages.7_Scenario_Playbooks",
    "app.pages.8_Feedback_and_Impact",
    "app.pages.9_Capacity_Simulator",
]


@pytest.mark.parametrize("module_name", PAGE_MODULES)
def test_page_module_importable(module_name: str) -> None:
    """Import the module ensuring ``ModuleNotFoundError`` is not raised."""

    spec = importlib.util.find_spec(module_name)
    assert spec is not None, f"Expected to discover module '{module_name}'"
    assert spec.loader is not None, f"Expected loader for module '{module_name}'"
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as error:  # pragma: no cover - explicit failure path
        pytest.fail(f"Unexpected ModuleNotFoundError importing '{module_name}': {error}")
    except Exception:
        module = sys.modules.get(module_name)
    if module is not None:
        assert module.__name__ == module_name
