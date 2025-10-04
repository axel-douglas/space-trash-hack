import pandas as pd
import pytest

from app.modules.mars_control import (
    load_jezero_bitmap,
    load_jezero_ortho_bitmap,
    load_jezero_slope_bitmap,
)
from app.modules.mars_control_center import MarsControlCenterService


def test_load_jezero_bitmap_payload_structure():
    payload = load_jezero_bitmap()

    assert set(payload.keys()) >= {"image_uri", "image", "bounds", "center", "metadata"}
    assert payload["image_uri"].startswith("data:image/")
    image_payload = payload["image"]
    assert isinstance(image_payload, dict)
    assert image_payload.get("url", "").endswith(".jpg")

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
    assert metadata.get("asset_url") == image_payload.get("url")
    assert metadata.get("license")
    assert metadata.get("provenance")


def test_build_map_payload_includes_bitmap_and_bounds():
    service = MarsControlCenterService()
    flights_df = pd.DataFrame()

    payload = service.build_map_payload(flights_df)

    bitmap = payload.get("bitmap_layer")
    assert isinstance(bitmap, dict)
    assert isinstance(bitmap.get("image"), dict)
    assert "static/mars/" in bitmap.get("image", {}).get("url", "")
    assert payload.get("map_bounds") == bitmap.get("bounds")

    view_state = payload.get("map_view_state")
    assert isinstance(view_state, dict)
    assert pytest.approx(bitmap["center"]["latitude"], rel=1e-3) == view_state["latitude"]
    assert pytest.approx(bitmap["center"]["longitude"], rel=1e-3) == view_state["longitude"]
    assert 5.0 < view_state["zoom"] < 16.5


def test_load_jezero_slope_bitmap_contains_legend():
    payload = load_jezero_slope_bitmap()

    metadata = payload.get("metadata")
    assert isinstance(metadata, dict)
    legend = metadata.get("legend")
    assert isinstance(legend, dict)
    assert legend.get("description")
    assert isinstance(legend.get("ticks"), list) and legend["ticks"]
    assert payload.get("image_uri").startswith("data:image/")
    assert isinstance(payload.get("image"), dict)
    assert payload["image"].get("url")


def test_load_jezero_ortho_bitmap_metadata():
    payload = load_jezero_ortho_bitmap()

    metadata = payload.get("metadata")
    assert isinstance(metadata, dict)
    assert "Ortofoto" in metadata.get("label", "")
    assert payload.get("bounds")


def test_overlay_flags_toggle_layers():
    service = MarsControlCenterService()
    flights_df = pd.DataFrame()

    payload_without_overlays = service.build_map_payload(
        flights_df,
        include_slope=False,
        include_ortho=False,
    )
    assert payload_without_overlays.get("slope_layer") is None
    assert payload_without_overlays.get("ortho_layer") is None

    payload_with_slope = service.build_map_payload(
        flights_df,
        include_slope=True,
        include_ortho=False,
        slope_opacity=0.55,
    )
    slope_layer = payload_with_slope.get("slope_layer")
    assert isinstance(slope_layer, dict)
    assert pytest.approx(slope_layer.get("opacity"), rel=1e-3) == 0.55

    payload_with_all = service.build_map_payload(
        flights_df,
        include_slope=True,
        include_ortho=True,
    )
    assert isinstance(payload_with_all.get("slope_layer"), dict)
    assert isinstance(payload_with_all.get("ortho_layer"), dict)
    labels = payload_with_all.get("active_overlay_labels")
    assert "Pendiente" in " ".join(str(label) for label in labels)
