"""Smoke tests for the luxe components module exports."""

import importlib


def test_luxe_components_exports_key_symbols() -> None:
    module = importlib.import_module("app.modules.luxe_components")

    assert hasattr(module, "__all__"), "luxe_components must define __all__"
    for symbol in ("TeslaHero", "target_configurator", "mission_briefing", "ChipRow"):
        assert symbol in module.__all__, f"{symbol} missing from __all__"
