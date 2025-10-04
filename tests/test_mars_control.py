import pandas as pd
import pytest

from app.modules.mars_control import load_jezero_bitmap
from app.modules.mars_control_center import MarsControlCenterService


def test_load_jezero_bitmap_payload_structure():
    payload = load_jezero_bitmap()

    assert set(payload.keys()) >= {"image_uri", "image", "bounds", "center", "metadata"}
    assert payload["image_uri"].startswith("data:image/")

    bounds = payload["bounds"]
    assert isinstance(bounds, tuple)
    assert len(bounds) == 4

    min_lon, min_lat, max_lon, max_lat = bounds
    assert min_lon < max_lon
    assert min_lat < max_lat

    metadata = payload["metadata"]
    assert isinstance(metadata, dict)
    assert "attribution" in metadata
    assert metadata.get("mime_type") in {"image/jpeg", "image/png"}


def test_build_map_payload_includes_bitmap_and_bounds():
    service = MarsControlCenterService()
    flights_df = pd.DataFrame()

    payload = service.build_map_payload(flights_df)

    bitmap = payload.get("bitmap_layer")
    assert isinstance(bitmap, dict)
    assert bitmap.get("image")
    assert payload.get("map_bounds") == bitmap.get("bounds")

    view_state = payload.get("map_view_state")
    assert isinstance(view_state, dict)
    assert pytest.approx(bitmap["center"]["latitude"], rel=1e-3) == view_state["latitude"]
    assert pytest.approx(bitmap["center"]["longitude"], rel=1e-3) == view_state["longitude"]
    assert 5.0 < view_state["zoom"] < 16.5
