from pathlib import Path

from app.modules import mars_control


def test_load_mars_scenegraph_exposes_static_asset():
    payload = mars_control.load_mars_scenegraph()

    assert isinstance(payload, dict)
    assert payload["url"].endswith("24881_Mars_1_6792.glb")

    static_path = Path(payload["path"])
    assert static_path.is_file()
    assert static_path.name == "24881_Mars_1_6792.glb"

    scale = payload["scale"]
    orientation = payload["orientation"]
    translation = payload["translation"]

    assert len(scale) == 3
    assert all(float(value) > 0 for value in scale)
    assert len(orientation) == 3
    assert len(translation) == 3
