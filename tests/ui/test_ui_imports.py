"""Smoke tests for key UI modules to guard against accidental regressions."""

import importlib


def _get_module(name: str):
    """Import *name* and return the module object.

    Keeping this logic in a helper means we get a nicer assertion message if the
    import fails, while ensuring the import only happens once per test function.
    """

    module = importlib.import_module(name)
    return module


def test_luxe_components_exports_expected_helpers() -> None:
    module = _get_module("app.modules.luxe_components")

    expected_symbols = {
        "TeslaHero",
        "ChipRow",
        "mission_briefing",
        "target_configurator",
    }

    assert hasattr(module, "__all__"), "luxe_components must define __all__"
    exported = set(module.__all__)

    missing_from_attributes = [symbol for symbol in expected_symbols if not hasattr(module, symbol)]
    assert not missing_from_attributes, f"Missing attributes: {missing_from_attributes}"

    missing_from_all = expected_symbols - exported
    assert not missing_from_all, f"Symbols not listed in __all__: {missing_from_all}"


def test_ui_blocks_exposes_primary_helpers() -> None:
    module = _get_module("app.modules.ui_blocks")

    expected_callables = ["load_theme", "inject_css", "surface", "use_token"]

    for name in expected_callables:
        value = getattr(module, name, None)
        assert value is not None, f"ui_blocks missing helper: {name}"
        assert callable(value), f"{name} should be callable"

    # `surface` is defined via ``@contextmanager`` so calling it should yield a context manager.
    surface_cm = module.surface()
    assert hasattr(surface_cm, "__enter__") and hasattr(surface_cm, "__exit__"), "surface() should return a context manager"
