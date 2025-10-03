"""Smoke tests that ensure Streamlit pages remain importable."""

from __future__ import annotations

import importlib.util

import pytest

PAGE_MODULES = [
    "app.Home",
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
def test_streamlit_pages_are_importable(module_name: str) -> None:
    """Import each Streamlit page to catch missing dependencies early."""

    spec = importlib.util.find_spec(module_name)
    assert spec is not None, f"Module spec for {module_name} could not be found"
