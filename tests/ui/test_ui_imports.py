import importlib


def _get_module(name: str):
    module = importlib.import_module(name)
    return module


def test_ui_blocks_exports_expected_helpers() -> None:
    module = _get_module("app.modules.ui_blocks")

    expected_callables = {
        "load_theme",
        "chipline",
        "pill",
        "action_button",
        "layout_stack",
    }

    missing = [name for name in expected_callables if not hasattr(module, name)]
    assert not missing, f"ui_blocks missing helpers: {missing}"


def test_ui_blocks_does_not_expose_legacy_helpers() -> None:
    module = _get_module("app.modules.ui_blocks")

    for legacy_name in ("enable_reveal_animation", "surface", "glass_card"):
        assert not hasattr(
            module, legacy_name
        ), f"ui_blocks should no longer expose {legacy_name}"


def test_target_limits_exports_constants() -> None:
    module = _get_module("app.modules.target_limits")

    for name in (
        "CREW_SIZE_BASELINE",
        "CREW_MINUTES_PER_MEMBER",
        "ENERGY_KWH_PER_KG_BASELINE",
        "WATER_L_PER_VOLUME_L_BASELINE",
        "compute_target_limits",
    ):
        assert hasattr(module, name), f"target_limits should export {name}"
