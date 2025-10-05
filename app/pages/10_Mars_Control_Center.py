from __future__ import annotations
from datetime import datetime
import base64
import math
import hashlib
import io
import json
from pathlib import Path
import html
import sys
from datetime import datetime
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

try:
    from app.bootstrap import ensure_streamlit_entrypoint
except ModuleNotFoundError:  # pragma: no cover - fallback for direct execution
    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from app.bootstrap import ensure_streamlit_entrypoint

ensure_streamlit_entrypoint(__file__)

from typing import Any, Iterable, Mapping, Sequence

import pandas as pd
import plotly.graph_objects as go
import pydeck as pdk
from PIL import Image
import streamlit as st
import streamlit.components.v1 as components

from app.modules import data_sources as ds
from app.modules import mars_control

from app.modules.generator import GeneratorService
from app.modules.manifest_loader import (
    build_manifest_template,
    load_manifest_from_upload,
    manifest_template_csv_bytes,
    run_policy_analysis,
)
from app.modules.mars_control_center import (
    MarsControlCenterService,
    summarize_artifacts,
)
from app.modules.mission_overview import _format_metric, render_mission_objective
from app.modules.ui_blocks import (
    badge_group,
    configure_page,
    initialise_frontend,
    micro_divider,
    render_brand_header,
)


configure_page(page_title="Rex-AI • Mars Control Center", page_icon="🛰️")
initialise_frontend()
render_brand_header(tagline="Mars Control Center · Interplanetary Recycling")


_MANUAL_DECISIONS_KEY = "mars_decision_actions"
_BATCH_RESULTS_KEY = "mars_manifest_batch_results"
_BATCH_SIGNATURE_KEY = "mars_manifest_batch_signature"
_SCORE_THRESHOLDS = {"spectral": 0.65, "mechanical": 0.6}
_ACTION_PRESETS = {
    "accept": {"label": "Aceptar plan Rex-AI", "badge": "🟢 Aceptado"},
    "reject": {"label": "Rechazar acción propuesta", "badge": "🔴 Rechazado"},
    "reprioritize": {"label": "Repriorizar envío crítico", "badge": "🟠 Repriorizar"},
}


_MARS_TEXTURE_PATH = (
    Path(__file__).resolve().parent.parent / "static" / "images" / "mars_global_8k.jpg"
)
_MARS_IMAGE_SEARCH_DIRS = [
    Path(__file__).resolve().parents[2] / "datasets" / "mars",
    Path(__file__).resolve().parent.parent / "static" / "images",
]
_MARS_IMAGE_PATTERNS = ("*.jpg", "*.jpeg", "*.png", "*.tif", "*.tiff")
_MARS_GLOBAL_BOUNDS = [-180.0, -90.0, 180.0, 90.0]
_REMOTE_JEZERO_IMAGE = "https://photojournal.jpl.nasa.gov/jpeg/PIA24688.jpg"
_MARS_TILE_TEMPLATE = (
    "https://planetarymaps.usgs.gov/tiles/"
    "Mars_Viking_MDIM21_ClrMosaic_global_232m/1.0.0/default/default028mm/{z}/{y}/{x}.png"
)
_MARS_TILE_PROBE = _MARS_TILE_TEMPLATE.format(z=5, x=17, y=9)
_MARS_TEXTURE_PATH = Path(__file__).resolve().parent.parent / "static" / "images" / "mars_global_8k.jpg"
_MARS_GLOBAL_BOUNDS = [-180.0, -90.0, 180.0, 90.0]

_CRATER_ASSET_DIR = Path(__file__).resolve().parents[2] / "datasets" / "crater"
_CRATER_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".tif", ".tiff")
_CRATER_MODEL_FILENAME = "jezero_crater_mars.glb"
_CRATER_DEFAULT_BOUNDS = (77.05, 18.15, 77.95, 18.85)
_CRATER_IMAGE_METADATA = {
    "Jezero_Crater_in_3D.jpg": {
        "label": "Relieve orbital coloreado",
        "description": "Modelo digital del terreno con exageración vertical generado por DLR/FU Berlin.",
        "credit": "ESA/DLR/FU Berlin · Mars Express HRSC",
    },
    "Katie_1_-_DLR_Jezero_hi_v2.jpg": {
        "label": "Mosaico científico Mars 2020",
        "description": "Montaje contextual de Jezero elaborado para el aterrizaje de Perseverance.",
        "credit": "NASA/JPL-Caltech · DLR",
    },
    "j03_045994_1986_j03_046060_1986_20m_slope_20m-full.jpg": {
        "label": "Pendientes y relieve (HiRISE)",
        "description": "Mapa de pendientes derivado de pares estéreo HiRISE a 20 m/px.",
        "credit": "NASA/JPL-Caltech/University of Arizona",
    },
    "j03_045994_1986_xn_18n282w_6m_ortho-full.jpg": {
        "label": "Ortoimagen HiRISE 6 m/px",
        "description": "Ortoimagen de precisión para planificar rutas sobre el delta occidental.",
        "credit": "NASA/JPL-Caltech/University of Arizona",
    },
    "m20_jezerocrater_ctxdem_mosaic_20m.jpg": {
        "label": "CTX DEM Mosaic 20 m",
        "description": "Modelo digital de elevación compilado para la misión Mars 2020.",
        "credit": "NASA/JPL-Caltech/MSSS/ASU",
    },
    "m2020_jezerocrater_ctxdem_mosaic-slide.png": {
        "label": "Resumen topográfico Mars 2020",
        "description": "Presentación operativa con realce de zonas prioritarias del cráter.",
        "credit": "NASA/JPL-Caltech/MSSS/USGS",
    },
}


@st.cache_resource(show_spinner=False)
def _load_reference_bundle() -> ds.MaterialReferenceBundle:
    return ds.load_material_reference_bundle()


def _manifest_signature(manifest_df: pd.DataFrame | None) -> str:
    if manifest_df is None or manifest_df.empty:
        return "empty"
    try:
        payload = manifest_df.to_csv(index=False).encode("utf-8")
    except Exception:
        payload = json.dumps(manifest_df.to_dict(orient="records"), sort_keys=True).encode("utf-8")
    return hashlib.sha1(payload).hexdigest()


def _store_flight_snapshot(flights_df: pd.DataFrame) -> None:
    st.session_state["flight_operations_table"] = flights_df
    st.session_state["flight_operations_last_decisions"] = {
        row["flight_id"]: row["ai_decision"]
        for row in flights_df.to_dict(orient="records")
        if row.get("flight_id")
    }


def _apply_manual_overrides(flights_df: pd.DataFrame | None) -> pd.DataFrame | None:
    if flights_df is None or flights_df.empty:
        return flights_df
    overrides = st.session_state.get(_MANUAL_DECISIONS_KEY, {})
    if not overrides:
        return flights_df
    updated = flights_df.copy()
    for manifest_ref, payload in overrides.items():
        if not isinstance(payload, Mapping):
            continue
        mask = updated["manifest_ref"].astype(str) == str(manifest_ref)
        if not mask.any():
            continue
        label = payload.get("label")
        badge = payload.get("badge", "⚙️ Manual")
        timestamp = payload.get("timestamp")
        if label:
            updated.loc[mask, "ai_decision"] = label
        if timestamp:
            updated.loc[mask, "ai_decision_timestamp"] = timestamp
        updated.loc[mask, "decision_indicator"] = badge
        updated.loc[mask, "decision_changed"] = True
    return updated


def _register_manual_action(
    manifest_ref: str,
    action_key: str,
    *,
    label: str | None = None,
    badge: str | None = None,
) -> None:
    presets = _ACTION_PRESETS.get(action_key, {})
    payload = {
        "action": action_key,
        "label": label or presets.get("label") or action_key.title(),
        "badge": badge or presets.get("badge", "⚙️ Manual"),
        "timestamp": datetime.utcnow().isoformat(),
    }
    overrides = st.session_state.setdefault(_MANUAL_DECISIONS_KEY, {})
    overrides[str(manifest_ref)] = payload
    st.session_state[_MANUAL_DECISIONS_KEY] = overrides

    flights_df: pd.DataFrame | None = st.session_state.get("flight_operations_table")
    if isinstance(flights_df, pd.DataFrame):
        updated = _apply_manual_overrides(flights_df)
        _store_flight_snapshot(updated)


def _mars_local_texture_path() -> Path | None:
    candidates: list[Path] = []
    try:
        candidates.append(_MARS_TEXTURE_PATH)
    except Exception:
        pass

    for directory in _MARS_IMAGE_SEARCH_DIRS:
        try:
            if not directory.exists() or not directory.is_dir():
                continue
        except OSError:
            continue
        for pattern in _MARS_IMAGE_PATTERNS:
            for candidate in sorted(directory.glob(pattern)):
                candidates.append(candidate)

    seen: set[Path] = set()
    for candidate in candidates:
        if candidate is None:
            continue
        if candidate in seen:
            continue
        seen.add(candidate)
        try:
            if candidate.exists() and candidate.is_file():
                return candidate
        except OSError:
            continue

@st.cache_data(show_spinner=False, ttl=3600)
def _mars_tile_service_available() -> bool:
    try:
        probe_request = Request(_MARS_TILE_PROBE, method="HEAD")
        with urlopen(probe_request, timeout=4) as response:
            if 200 <= getattr(response, "status", 0) < 400:
                return True
    except TypeError:
        # Python <3.11 compat: fallback to manual method override.
        try:
            probe_request = Request(_MARS_TILE_PROBE)
            probe_request.get_method = lambda: "HEAD"  # type: ignore[attr-defined]
            with urlopen(probe_request, timeout=4) as response:
                if 200 <= getattr(response, "status", 0) < 400:
                    return True
        except Exception:
            pass
    except (HTTPError, URLError, TimeoutError, ValueError):
        pass

    try:
        # Some WMTS endpoints reject HEAD; fetch a byte range instead.
        probe_request = Request(_MARS_TILE_PROBE)
        probe_request.add_header("Range", "bytes=0-0")
        with urlopen(probe_request, timeout=4) as response:
            return 200 <= getattr(response, "status", 0) < 400
    except (HTTPError, URLError, TimeoutError, ValueError):
        return False


def _mars_background_layers() -> list[pdk.Layer]:
    background_layers: list[pdk.Layer] = []

    if _mars_tile_service_available():
        background_layers.append(
            pdk.Layer(
                "TileLayer",
                data=_MARS_TILE_TEMPLATE,
                id="mars-wmts",
                min_zoom=0,
                max_zoom=12,
                tile_size=256,
                pickable=False,
            )
        )
        return background_layers

    local_texture = _mars_local_texture_path()
    if local_texture:
        background_layers.append(
            pdk.Layer(
                "BitmapLayer",
                data=None,
                id="mars-static-texture",
                image=str(local_texture),
                bounds=_MARS_GLOBAL_BOUNDS,
                pickable=False,
            )
        )
        return background_layers

    background_layers.append(
        pdk.Layer(
            "PolygonLayer",
            data=[
                {
                    "coordinates": [
                        [-180.0, -90.0],
                        [-180.0, 90.0],
                        [180.0, 90.0],
                        [180.0, -90.0],
                    ]
                }
            ],
            id="mars-solid-background",
            get_polygon="coordinates",
            stroked=False,
            filled=True,
            get_fill_color=[105, 69, 33],
            pickable=False,
        )
    )
    return background_layers


@st.cache_data(show_spinner=False)
def _crater_image_catalog() -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    try:
        if not _CRATER_ASSET_DIR.is_dir():
            return entries
    except OSError:
        return entries

    for path in sorted(_CRATER_ASSET_DIR.iterdir()):
        if path.suffix.lower() not in _CRATER_IMAGE_EXTENSIONS:
            continue
        metadata = _CRATER_IMAGE_METADATA.get(path.name, {})
        label = metadata.get("label") or path.stem.replace("_", " ")
        entries.append(
            {
                "id": path.name,
                "path": str(path),
                "label": label,
                "description": metadata.get("description"),
                "credit": metadata.get("credit"),
            }
        )
    return entries


@st.cache_data(show_spinner=False)
def _crater_image_bytes(path_str: str) -> bytes | None:
    try:
        return Path(path_str).read_bytes()
    except OSError:
        return None


@st.cache_data(show_spinner=False)
def _load_crater_preview(
    path_str: str,
    max_px: int = 4096,
) -> tuple[bytes, tuple[int, int], tuple[int, int]]:
    """Load a crater raster and return a resized preview as PNG bytes.

    The helper keeps aspect ratio while ensuring that neither dimension exceeds
    ``max_px``. It returns the preview bytes along with both the preview and the
    original dimensions so callers can keep overlays aligned and show metadata.
    """

    path = Path(path_str)
    if not path.is_file():
        return b"", (0, 0), (0, 0)

    try:
        with Image.open(path) as image:
            rgb_image = image.convert("RGB")
            original_size = rgb_image.size
            max_dimension = max(original_size) if original_size else 0
            preview_image = rgb_image

            if max_dimension > max_px and max_dimension > 0:
                scale = max_px / float(max_dimension)
                new_size = (
                    max(1, int(round(original_size[0] * scale))),
                    max(1, int(round(original_size[1] * scale))),
                )
                resampling = getattr(Image, "Resampling", Image).LANCZOS
                preview_image = rgb_image.resize(new_size, resample=resampling)
            else:
                new_size = preview_image.size

            buffer = io.BytesIO()
            preview_image.save(buffer, format="PNG")
    except OSError:
        return b"", (0, 0), (0, 0)

    return buffer.getvalue(), new_size, original_size


def _guess_image_mime(path_str: str) -> str:
    mapping = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".tif": "image/tiff",
        ".tiff": "image/tiff",
    }
    return mapping.get(Path(path_str).suffix.lower(), "application/octet-stream")


@st.cache_data(show_spinner=False)
def _crater_model_source() -> str | None:
    model_path = _CRATER_ASSET_DIR / _CRATER_MODEL_FILENAME
    try:
        if not model_path.is_file():
            return None
        data = model_path.read_bytes()
    except OSError:
        return None
    encoded = base64.b64encode(data).decode("ascii")
    return f"data:model/gltf-binary;base64,{encoded}"


def _flatten_geo_coordinates(payload: Any) -> Iterable[tuple[float, float]]:
    if isinstance(payload, Sequence) and not isinstance(payload, (str, bytes)):
        if len(payload) >= 2 and all(isinstance(value, (int, float)) for value in payload[:2]):
            yield float(payload[0]), float(payload[1])
        else:
            for item in payload:
                yield from _flatten_geo_coordinates(item)


def _geometry_coordinate_iter(geometry: Mapping[str, Any] | None) -> Iterable[tuple[float, float]]:
    if not isinstance(geometry, Mapping):
        return []
    features = geometry.get("features", [])
    if not isinstance(features, Sequence):
        return []
    for feature in features:
        if not isinstance(feature, Mapping):
            continue
        geom = feature.get("geometry")
        if not isinstance(geom, Mapping):
            continue
        coordinates = geom.get("coordinates")
        if coordinates is None:
            continue
        yield from _flatten_geo_coordinates(coordinates)


def _resolve_crater_bounds(
    geometry: Mapping[str, Any] | None,
    capsules: pd.DataFrame | None,
    zones: pd.DataFrame | None,
) -> tuple[float, float, float, float]:
    lon_min, lat_min, lon_max, lat_max = _CRATER_DEFAULT_BOUNDS
    longitudes: list[float] = []
    latitudes: list[float] = []

    for lon, lat in _geometry_coordinate_iter(geometry):
        longitudes.append(lon)
        latitudes.append(lat)

    if isinstance(capsules, pd.DataFrame) and not capsules.empty:
        longitudes.extend(
            pd.to_numeric(capsules.get("longitude"), errors="coerce").dropna().astype(float).tolist()
        )
        latitudes.extend(
            pd.to_numeric(capsules.get("latitude"), errors="coerce").dropna().astype(float).tolist()
        )

    if isinstance(zones, pd.DataFrame) and not zones.empty:
        longitudes.extend(
            pd.to_numeric(zones.get("longitude"), errors="coerce").dropna().astype(float).tolist()
        )
        latitudes.extend(
            pd.to_numeric(zones.get("latitude"), errors="coerce").dropna().astype(float).tolist()
        )

    if longitudes and latitudes:
        lon_min = min(longitudes)
        lon_max = max(longitudes)
        lat_min = min(latitudes)
        lat_max = max(latitudes)

        if lon_min == lon_max:
            lon_min -= 0.01
            lon_max += 0.01
        if lat_min == lat_max:
            lat_min -= 0.01
            lat_max += 0.01

        lon_margin = max((lon_max - lon_min) * 0.08, 0.01)
        lat_margin = max((lat_max - lat_min) * 0.08, 0.01)
        lon_min -= lon_margin
        lon_max += lon_margin
        lat_min -= lat_margin
        lat_max += lat_margin

    return lon_min, lat_min, lon_max, lat_max


def _project_latlon_to_image(
    longitude: float,
    latitude: float,
    width: int,
    height: int,
    bounds: tuple[float, float, float, float],
) -> tuple[float, float]:
    lon_min, lat_min, lon_max, lat_max = bounds
    if lon_max == lon_min:
        lon_max = lon_min + 1e-6
    if lat_max == lat_min:
        lat_max = lat_min + 1e-6

    lon_ratio = (longitude - lon_min) / (lon_max - lon_min)
    lat_ratio = (latitude - lat_min) / (lat_max - lat_min)
    lon_ratio = max(0.0, min(1.0, lon_ratio))
    lat_ratio = max(0.0, min(1.0, lat_ratio))

    return lon_ratio * width, lat_ratio * height


def _rgba_color(rgb: Sequence[float | int], *, alpha: float = 1.0) -> str:
    channel_values = [int(max(0, min(255, round(float(value))))) for value in rgb[:3]]
    normalized_alpha = max(0.0, min(1.0, float(alpha)))
    if normalized_alpha >= 1.0:
        return f"rgb({channel_values[0]},{channel_values[1]},{channel_values[2]})"
    return (
        f"rgba({channel_values[0]},{channel_values[1]},{channel_values[2]},{normalized_alpha:.3f})"
    )


def _iter_polygon_rings(geometry: Mapping[str, Any]) -> Iterable[Sequence[Sequence[float]]]:
    gtype = str(geometry.get("type", "")).lower()
    coordinates = geometry.get("coordinates", [])
    if gtype == "polygon":
        for ring in coordinates:
            if isinstance(ring, Sequence):
                yield ring
    elif gtype == "multipolygon":
        for polygon in coordinates:
            if not isinstance(polygon, Sequence):
                continue
            for ring in polygon:
                if isinstance(ring, Sequence):
                    yield ring


def _iter_line_paths(geometry: Mapping[str, Any]) -> Iterable[Sequence[Sequence[float]]]:
    gtype = str(geometry.get("type", "")).lower()
    coordinates = geometry.get("coordinates", [])
    if gtype == "linestring" and isinstance(coordinates, Sequence):
        yield coordinates
    elif gtype == "multilinestring":
        for segment in coordinates:
            if isinstance(segment, Sequence):
                yield segment


def _add_geometry_overlays(
    fig: go.Figure,
    geometry: Mapping[str, Any] | None,
    *,
    width: int,
    height: int,
    bounds: tuple[float, float, float, float],
) -> int:
    if not isinstance(geometry, Mapping):
        return 0
    features = geometry.get("features", [])
    if not isinstance(features, Sequence):
        return 0

    polygon_count = 0
    for feature in features:
        if not isinstance(feature, Mapping):
            continue
        geom = feature.get("geometry")
        if not isinstance(geom, Mapping):
            continue
        gtype = str(geom.get("type", "")).lower()
        if gtype not in {"polygon", "multipolygon", "linestring", "multilinestring"}:
            continue
        properties = feature.get("properties") if isinstance(feature.get("properties"), Mapping) else {}
        name = str(properties.get("name", "Perímetro Jezero"))
        base_color = properties.get("color")
        color = base_color if isinstance(base_color, str) else "#38bdf8"

        if gtype in {"polygon", "multipolygon"}:
            for ring in _iter_polygon_rings(geom):
                xs: list[float] = []
                ys: list[float] = []
                for point in ring:
                    if not isinstance(point, Sequence) or len(point) < 2:
                        continue
                    lon, lat = float(point[0]), float(point[1])
                    projected = _project_latlon_to_image(lon, lat, width, height, bounds)
                    xs.append(projected[0])
                    ys.append(projected[1])
                if len(xs) < 2:
                    continue
                fig.add_trace(
                    go.Scatter(
                        x=xs,
                        y=ys,
                        mode="lines",
                        line=dict(color=color, width=3),
                        hoverinfo="skip",
                        name=name,
                        legendgroup="geometry",
                        showlegend=polygon_count == 0,
                    )
                )
                polygon_count += 1
        else:
            for segment in _iter_line_paths(geom):
                xs = []
                ys = []
                for point in segment:
                    if not isinstance(point, Sequence) or len(point) < 2:
                        continue
                    lon, lat = float(point[0]), float(point[1])
                    projected = _project_latlon_to_image(lon, lat, width, height, bounds)
                    xs.append(projected[0])
                    ys.append(projected[1])
                if len(xs) < 2:
                    continue
                fig.add_trace(
                    go.Scatter(
                        x=xs,
                        y=ys,
                        mode="lines",
                        line=dict(color=color, width=2, dash="dot"),
                        hoverinfo="skip",
                        name=name,
                        legendgroup="geometry",
                        showlegend=polygon_count == 0,
                    )
                )
                polygon_count += 1
    return polygon_count


def _build_zone_circle_points(
    longitude: float,
    latitude: float,
    radius_m: float,
    *,
    width: int,
    height: int,
    bounds: tuple[float, float, float, float],
) -> tuple[list[float], list[float]]:
    if radius_m <= 0:
        return [], []
    steps = 48
    lat_radius = radius_m / 111_320.0
    cos_lat = math.cos(math.radians(latitude))
    if abs(cos_lat) < 1e-6:
        cos_lat = 1e-6
    lon_radius = radius_m / (111_320.0 * cos_lat)

    xs: list[float] = []
    ys: list[float] = []
    for angle in range(0, 360 + int(360 / steps), int(360 / steps)):
        theta = math.radians(angle)
        point_lon = longitude + (lon_radius * math.cos(theta))
        point_lat = latitude + (lat_radius * math.sin(theta))
        projected = _project_latlon_to_image(point_lon, point_lat, width, height, bounds)
        xs.append(projected[0])
        ys.append(projected[1])
    return xs, ys


def _add_zone_overlays(
    fig: go.Figure,
    zones: pd.DataFrame | None,
    *,
    width: int,
    height: int,
    bounds: tuple[float, float, float, float],
) -> int:
    if not isinstance(zones, pd.DataFrame) or zones.empty:
        return 0

    center_x: list[float] = []
    center_y: list[float] = []
    hover_payload: list[str] = []
    marker_colors: list[str] = []
    zone_labels: list[str] = []
    zone_count = 0

    for zone in zones.itertuples():
        latitude = float(getattr(zone, "latitude", 0.0))
        longitude = float(getattr(zone, "longitude", 0.0))
        radius_m = float(getattr(zone, "radius_m", 0.0))
        color_rgb = (
            getattr(zone, "color_r", 180),
            getattr(zone, "color_g", 198),
            getattr(zone, "color_b", 231),
        )
        xs, ys = _build_zone_circle_points(
            longitude,
            latitude,
            radius_m,
            width=width,
            height=height,
            bounds=bounds,
        )
        if xs and ys:
            fig.add_trace(
                go.Scatter(
                    x=xs,
                    y=ys,
                    mode="lines",
                    line=dict(color=_rgba_color(color_rgb, alpha=0.8), width=2),
                    fill="toself",
                    fillcolor=_rgba_color(color_rgb, alpha=0.18),
                    hoverinfo="text",
                    hovertext=getattr(zone, "tooltip", getattr(zone, "name", "Zona")),
                    name=str(getattr(zone, "name", "Zona operativa")),
                    legendgroup="zones",
                    showlegend=False,
                )
            )
        projected = _project_latlon_to_image(longitude, latitude, width, height, bounds)
        center_x.append(projected[0])
        center_y.append(projected[1])
        hover_payload.append(getattr(zone, "tooltip", getattr(zone, "name", "Zona")))
        marker_colors.append(_rgba_color(color_rgb, alpha=0.95))
        zone_labels.append(str(getattr(zone, "name", "Zona")))
        zone_count += 1

    if center_x:
        fig.add_trace(
            go.Scatter(
                x=center_x,
                y=center_y,
                mode="markers",
                marker=dict(
                    size=[18] * len(center_x),
                    symbol="diamond", 
                    color=marker_colors,
                    line=dict(width=1.6, color="rgba(15,23,42,0.65)"),
                ),
                hoverinfo="text",
                hovertext=hover_payload,
                text=zone_labels,
                textposition="top center",
                name="Zonas logísticas",
                legendgroup="zones",
                showlegend=True,
            )
        )

    return zone_count


def _add_capsule_overlays(
    fig: go.Figure,
    capsules: pd.DataFrame | None,
    *,
    width: int,
    height: int,
    bounds: tuple[float, float, float, float],
) -> int:
    if not isinstance(capsules, pd.DataFrame) or capsules.empty:
        return 0

    xs: list[float] = []
    ys: list[float] = []
    marker_sizes: list[float] = []
    marker_colors: list[str] = []
    border_colors: list[str] = []
    hover_texts: list[str] = []

    for capsule in capsules.itertuples():
        latitude = float(getattr(capsule, "latitude", 0.0))
        longitude = float(getattr(capsule, "longitude", 0.0))
        projected = _project_latlon_to_image(longitude, latitude, width, height, bounds)
        xs.append(projected[0])
        ys.append(projected[1])

        category_color = (
            getattr(capsule, "category_color_r", 56),
            getattr(capsule, "category_color_g", 149),
            getattr(capsule, "category_color_b", 255),
        )
        status_color = (
            getattr(capsule, "status_color_r", 15),
            getattr(capsule, "status_color_g", 23),
            getattr(capsule, "status_color_b", 42),
        )
        marker_radius = float(getattr(capsule, "marker_radius_m", 900.0))
        marker_sizes.append(max(12.0, min(marker_radius / 250.0, 34.0)))
        marker_colors.append(_rgba_color(category_color, alpha=0.92))
        border_colors.append(_rgba_color(status_color, alpha=1.0))

        vehicle = getattr(capsule, "vehicle", getattr(capsule, "flight_id", "Cápsula"))
        status = getattr(capsule, "status_badge", getattr(capsule, "status", ""))
        eta_value = getattr(capsule, "eta_minutes", None)
        if eta_value is None or pd.isna(eta_value):
            eta_display = "—"
        else:
            try:
                eta_display = f"{int(float(eta_value))}"
            except (TypeError, ValueError):
                eta_display = str(eta_value)
        materials_value = getattr(capsule, "materials_display", None)
        density = getattr(capsule, "density", None)
        compatibility = getattr(capsule, "compatibility", None)

        hover_lines = [f"<b>{html.escape(str(vehicle))}</b>"]
        if status:
            hover_lines.append(html.escape(str(status)))
        hover_lines.append(f"ETA: {eta_display} min")
        if materials_value is not None and not pd.isna(materials_value):
            hover_lines.append(f"Materiales: {html.escape(str(materials_value))}")
        if density is not None and not pd.isna(density):
            hover_lines.append(f"Densidad: {float(density):.2f} g/cm³")
        if compatibility is not None and not pd.isna(compatibility):
            hover_lines.append(f"Compatibilidad: {float(compatibility):.2f}")
        hover_texts.append("<br/>".join(hover_lines))

    if not xs:
        return 0

    fig.add_trace(
        go.Scatter(
            x=xs,
            y=ys,
            mode="markers",
            marker=dict(
                size=marker_sizes,
                color=marker_colors,
                line=dict(width=2.4, color=border_colors),
                symbol="circle",
            ),
            hoverinfo="text",
            hovertext=hover_texts,
            name="Cápsulas activas",
            legendgroup="capsules",
            showlegend=True,
        )
    )

    return len(xs)


@st.cache_data(show_spinner=False, ttl=3600)
def _load_jezero_crater_image() -> tuple[bytes, Path | None] | None:
    local_texture = _mars_local_texture_path()
    if local_texture:
        try:
            return local_texture.read_bytes(), local_texture
        except Exception:
            pass

    try:
        with urlopen(_REMOTE_JEZERO_IMAGE, timeout=10) as response:
            data = response.read()
        if data:
            return data, None
    except Exception:
        pass

    return None


def _render_legacy_crater_overview() -> None:
    crater_payload = _load_jezero_crater_image()
    if crater_payload is None:
        st.info(
            "Añadí una imagen equirectangular de Jezero en `datasets/mars/` (por ejemplo,"
            " `jezero_reference.jpg`) o en `app/static/images/` para habilitar el briefing"
            " visual del cráter."
        )
        return

    crater_bytes, texture_path = crater_payload
    try:
        crater_image = Image.open(io.BytesIO(crater_bytes)).convert("RGB")
    except Exception:
        st.info(
            "Añadí una imagen equirectangular de Jezero en `datasets/mars/`"
            " (por ejemplo, `jezero_reference.jpg`) o en `app/static/images/`"
            " para habilitar el briefing visual del cráter."
        )
        return

    width, height = crater_image.size

    fig = go.Figure()
    fig.add_layout_image(
        dict(
            source=crater_image,
            x=0,
            y=height,
            sizex=width,
            sizey=height,
            xref="x",
            yref="y",
            sizing="stretch",
            layer="below",
        )
    )

    highlight_paths = [
        {
            "label": "Deltas occidentales",
            "path": [(0.14, 0.8), (0.3, 0.65), (0.46, 0.57)],
            "color": "#fb923c",
            "description": "Canales fluviales que alimentaron el delta fosilizado.",
        },
        {
            "label": "Ridges elevadas",
            "path": [(0.58, 0.4), (0.72, 0.44), (0.86, 0.48)],
            "color": "#f472b6",
            "description": "Crestas con fracturas que podrían concentrar agua subsuperficial.",
        },
    ]

    for overlay in highlight_paths:
        x_coords = [point[0] * width for point in overlay["path"]]
        y_coords = [height - (point[1] * height) for point in overlay["path"]]
        fig.add_trace(
            go.Scatter(
                x=x_coords,
                y=y_coords,
                mode="lines",
                line=dict(color=overlay["color"], width=5),
                hoverinfo="text",
                hovertext=overlay["description"],
                name=overlay["label"],
            )
        )

    rim_points_x: list[float] = []
    rim_points_y: list[float] = []
    rim_center = (0.5, 0.5)
    rim_radius = 0.48
    for angle in range(0, 361, 6):
        theta = math.radians(angle)
        rim_x = (rim_center[0] + rim_radius * math.cos(theta)) * width
        rim_y = height - ((rim_center[1] + rim_radius * math.sin(theta)) * height)
        rim_points_x.append(rim_x)
        rim_points_y.append(rim_y)
    fig.add_trace(
        go.Scatter(
            x=rim_points_x,
            y=rim_points_y,
            mode="lines",
            line=dict(color="#94a3b8", width=2, dash="dash"),
            hoverinfo="skip",
            name="Borde del cráter",
        )
    )

    highlight_points = [
        {
            "label": "Aterrizaje Perseverance",
            "coords": (0.52, 0.47),
            "color": "#38bdf8",
            "description": "Zona de descenso 2021 – referencia para logística robotizada.",
            "textposition": "top right",
        },
        {
            "label": "Corredor logístico",
            "coords": (0.65, 0.58),
            "color": "#facc15",
            "description": "Pasillo de aterrizaje propuesto para cápsulas de reabastecimiento.",
            "textposition": "bottom left",
        },
    ]

    for point in highlight_points:
        x_val = point["coords"][0] * width
        y_val = height - (point["coords"][1] * height)
        fig.add_trace(
            go.Scatter(
                x=[x_val],
                y=[y_val],
                mode="markers+text",
                marker=dict(size=12, color=point["color"], symbol="circle"),
                text=[point["label"]],
                textposition=point.get("textposition", "top left"),
                hovertext=point["description"],
                hoverinfo="text",
                name=point["label"],
            )
        )

    fig.update_xaxes(visible=False, range=[0, width])
    fig.update_yaxes(visible=False, range=[0, height])
    fig.update_layout(
        showlegend=False,
        margin=dict(l=0, r=0, t=40, b=0),
        height=420,
        title=dict(
            text="Jezero Crater · briefing visual",
            x=0.01,
            y=0.98,
            font=dict(size=18),
        ),
    )

    st.plotly_chart(fig, use_container_width=True)

    credit_parts = [
        "Créditos: NASA/JPL-Caltech · PIA24688",
    ]
    if texture_path is not None:
        credit_parts.append(f"Fuente local: `{texture_path.name}`")
    st.caption(" · ".join(credit_parts))


def _render_crater_image_tab(
    assets: Sequence[dict[str, Any]],
    capsules: pd.DataFrame | None,
    zones: pd.DataFrame | None,
    geometry: Mapping[str, Any] | None,
) -> tuple[int, int, int]:
    if not assets:
        return 0, 0, 0

    option_map = {entry["label"]: entry for entry in assets}
    labels = list(option_map.keys())
    selected_label = st.selectbox(
        "Base cartográfica científica",
        labels,
        index=0,
        key="jezero_image_asset",
    )
    selected_asset = option_map[selected_label]
    preview_bytes, preview_size, original_size = _load_crater_preview(
        selected_asset["path"]
    )
    if not preview_bytes:
        st.warning(
            f"No se pudo cargar `{selected_asset['id']}` desde `datasets/crater`."
        )
        return 0, 0, 0

    try:
        crater_image = Image.open(io.BytesIO(preview_bytes)).convert("RGB")
        crater_image.load()
    except OSError:
        st.warning(
            f"No se pudo procesar `{selected_asset['id']}` para visualización."
        )
        return 0, 0, 0

    width, height = crater_image.size
    if not preview_size or preview_size == (0, 0):
        preview_size = (width, height)

    if not original_size or original_size == (0, 0):
        original_size = preview_size

    overlay_options = {
        "Perímetro Jezero": "geometry",
        "Zonas logísticas": "zones",
        "Cápsulas activas": "capsules",
    }
    selected_layers = st.multiselect(
        "Capas de misión",
        list(overlay_options.keys()),
        default=list(overlay_options.keys()),
        key="jezero_overlay_layers",
    )

    capsules_df = capsules if isinstance(capsules, pd.DataFrame) else pd.DataFrame()
    zones_df = zones if isinstance(zones, pd.DataFrame) else pd.DataFrame()
    geometry_payload = geometry if isinstance(geometry, Mapping) else None
    bounds = _resolve_crater_bounds(geometry_payload, capsules_df, zones_df)

    fig = go.Figure()
    fig.add_layout_image(
        dict(
            source=crater_image,
            x=0,
            y=height,
            sizex=width,
            sizey=height,
            xref="x",
            yref="y",
            sizing="stretch",
            layer="below",
        )
    )

    polygon_count = 0
    zone_count = 0
    capsule_count = 0

    if "Perímetro Jezero" in selected_layers:
        polygon_count = _add_geometry_overlays(
            fig,
            geometry_payload,
            width=width,
            height=height,
            bounds=bounds,
        )

    if "Zonas logísticas" in selected_layers:
        zone_count = _add_zone_overlays(
            fig,
            zones_df,
            width=width,
            height=height,
            bounds=bounds,
        )

    if "Cápsulas activas" in selected_layers:
        capsule_count = _add_capsule_overlays(
            fig,
            capsules_df,
            width=width,
            height=height,
            bounds=bounds,
        )

    fig.update_xaxes(visible=False, range=[0, width])
    fig.update_yaxes(visible=False, range=[0, height])
    fig.update_layout(
        margin=dict(l=0, r=0, t=36, b=0),
        height=520,
        plot_bgcolor="#020617",
        paper_bgcolor="rgba(2,6,23,0.92)",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            x=0,
            bgcolor="rgba(2,6,23,0.78)",
            borderwidth=0,
        ),
        font=dict(color="#e2e8f0"),
        title=dict(
            text="Atlas operativo de Jezero",
            x=0.01,
            y=0.98,
            font=dict(size=18),
        ),
    )

    st.plotly_chart(fig, use_container_width=True)

    description = selected_asset.get("description")
    if description:
        st.markdown(f"_{description}_")

    credit_parts: list[str] = []
    credit = selected_asset.get("credit")
    if credit:
        credit_parts.append(str(credit))
    credit_parts.append(f"Fuente: `datasets/crater/{selected_asset['id']}`")
    orig_width, orig_height = original_size
    if preview_size != original_size:
        credit_parts.append(
            f"Resolución original: {orig_width:,} × {orig_height:,} px"
        )
        credit_parts.append(f"Vista previa: {width:,} × {height:,} px")
    else:
        credit_parts.append(f"Resolución: {width:,} × {height:,} px")
    st.caption(" · ".join(credit_parts))

    image_bytes = _crater_image_bytes(selected_asset["path"])
    if image_bytes:
        st.download_button(
            "Descargar imagen original",
            data=image_bytes,
            file_name=selected_asset["id"],
            mime=_guess_image_mime(selected_asset["path"]),
            use_container_width=True,
        )

    metric_cols = st.columns(3)
    metric_cols[0].metric("Cápsulas rastreadas", f"{capsule_count}")
    metric_cols[1].metric("Zonas monitorizadas", f"{zone_count}")
    metric_cols[2].metric("Polígonos activos", f"{polygon_count}")

    if capsule_count == 0 and zone_count == 0 and polygon_count == 0:
        st.info(
            "Sin capas dinámicas todavía. Cargá telemetría o un manifiesto para"
            " poblar el mapa operativo."
        )

    return capsule_count, zone_count, polygon_count


def _render_crater_model_tab(
    model_source: str,
    capsules: pd.DataFrame | None,
    zones: pd.DataFrame | None,
) -> None:
    components.html(  # type: ignore[no-untyped-call]
        """
        <script type="module" src="https://unpkg.com/@google/model-viewer/dist/model-viewer.min.js"></script>
        <model-viewer
            src="{model_src}"
            camera-controls
            auto-rotate
            rotation-per-second="18deg"
            exposure="1.15"
            shadow-intensity="0.6"
            style="width:100%;height:520px;border-radius:18px;background:radial-gradient(circle at 50% 20%, rgba(148,163,184,0.24), rgba(15,23,42,0.95));"
        ></model-viewer>
        """.format(model_src=model_source),
        height=540,
    )

    st.caption(
        "Interactuá con el relieve: arrastrá para orbitar, acercate con la"
        " rueda del mouse o trackpad y mantené Mayús para hacer un paneo lateral."
    )

    capsule_badges: list[str] = []
    if isinstance(capsules, pd.DataFrame) and not capsules.empty:
        for capsule in capsules.head(4).itertuples():
            eta_value = getattr(capsule, "eta_minutes", None)
            if eta_value is None or pd.isna(eta_value):
                eta_display = "—"
            else:
                try:
                    eta_display = f"{int(float(eta_value))}"
                except (TypeError, ValueError):
                    eta_display = str(eta_value)
            vehicle_label = getattr(capsule, "vehicle", getattr(capsule, "flight_id", "Cápsula"))
            capsule_badges.append(f"🛰️ {vehicle_label} · ETA {eta_display} min")

    zone_badges: list[str] = []
    if isinstance(zones, pd.DataFrame) and not zones.empty:
        for zone in zones.head(3).itertuples():
            zone_badges.append(
                f"🏷️ {getattr(zone, 'name', 'Zona')} · {getattr(zone, 'category', '').replace('_', ' ').title()}"
            )

    if capsule_badges or zone_badges:
        badge_group(capsule_badges + zone_badges)

    st.caption(
        "Modelo GLB: `datasets/crater/jezero_crater_mars.glb` · NASA/JPL-Caltech/MSSS/ASU"
    )


def _render_crater_overview(
    capsules: pd.DataFrame | None,
    zones: pd.DataFrame | None,
    geometry: Mapping[str, Any] | None,
) -> None:
    crater_assets = _crater_image_catalog()
    model_source = _crater_model_source()

    available_tabs: list[str] = []
    if crater_assets:
        available_tabs.append("🗺️ Atlas fotogramétrico")
    if model_source:
        available_tabs.append("🗻 Modelo 3D & relieve")

    if not available_tabs:
        _render_legacy_crater_overview()
        return

    st.markdown("#### Jezero Crater · hub interactivo")
    tabs = st.tabs(available_tabs)

    tab_index = 0
    if crater_assets:
        with tabs[tab_index]:
            _render_crater_image_tab(crater_assets, capsules, zones, geometry)
        tab_index += 1

    if model_source:
        with tabs[tab_index]:
            _render_crater_model_tab(model_source, capsules, zones)

def _demo_event_severity(severity: str | None) -> str:
    normalized = str(severity or "info").lower()
    if normalized in {"critical", "alert", "danger", "severe"}:
        return "critical"
    if normalized in {"warning", "caution", "warn"}:
        return "warning"
    return "info"


def _format_demo_timestamp(event: mars_control.DemoEvent) -> str:
    if event.emitted_at is None:
        return "En cola"
    try:
        return f"{event.emitted_at.strftime('%H:%M:%S')} UTC"
    except Exception:
        return str(event.emitted_at)


def _render_demo_event_card(event: mars_control.DemoEvent) -> str:
    severity = _demo_event_severity(event.severity)
    icon = html.escape(event.icon or "🛰️")
    category = html.escape(event.category.title())
    timestamp = html.escape(_format_demo_timestamp(event))
    metadata_html = ""
    if event.metadata:
        tags = []
        for key, value in event.metadata.items():
            key_label = html.escape(str(key).replace("_", " ").title())
            value_label = html.escape(str(value))
            tags.append(
                f"<span class='demo-event-card__tag'><strong>{key_label}</strong>: {value_label}</span>"
            )
        metadata_html = (
            "<div class='demo-event-card__meta-tags'>" + " · ".join(tags) + "</div>"
        )
    title = html.escape(event.title)
    message = html.escape(event.message)
    return (
        "<div class='demo-event-card demo-event-card--"
        + severity
        + "'>"
        + f"<div class='demo-event-card__icon'>{icon}</div>"
        + "<div class='demo-event-card__content'>"
        + "<div class='demo-event-card__header'>"
        + f"<span class='demo-event-card__category'>{category}</span>"
        + f"<span class='demo-event-card__timestamp'>{timestamp}</span>"
        + "</div>"
        + f"<div class='demo-event-card__title'>{title}</div>"
        + f"<div class='demo-event-card__message'>{message}</div>"
        + metadata_html
        + "</div></div>"
    )


def _render_demo_ticker(events: list[mars_control.DemoEvent]) -> str:
    if not events:
        return ""
    items: list[str] = []
    for event in events:
        severity = _demo_event_severity(event.severity)
        icon = html.escape(event.icon or "🛰️")
        title = html.escape(event.title)
        timestamp = html.escape(
            event.emitted_at.strftime("%H:%M:%S") if event.emitted_at else "—"
        )
        items.append(
            "<div class='demo-event-ticker__item demo-event-ticker__item--"
            + severity
            + "'>"
            + f"<span class='demo-event-ticker__icon'>{icon}</span>"
            + f"<span class='demo-event-ticker__text'>{title}</span>"
            + f"<span class='demo-event-ticker__time'>{timestamp}</span>"
            + "</div>"
        )
    return "<div class='demo-event-ticker'>" + "".join(items) + "</div>"


def _ensure_manifest_batch(
    service: GeneratorService, manifest_df: pd.DataFrame | None
) -> list[dict[str, Any]]:
    if manifest_df is None or manifest_df.empty:
        st.session_state.pop(_BATCH_RESULTS_KEY, None)
        st.session_state.pop(_BATCH_SIGNATURE_KEY, None)
        return []

    signature = _manifest_signature(manifest_df)
    if st.session_state.get(_BATCH_SIGNATURE_KEY) != signature:
        batch = mars_control.score_manifest_batch(service, [manifest_df])
        st.session_state[_BATCH_RESULTS_KEY] = batch
        st.session_state[_BATCH_SIGNATURE_KEY] = signature
    return st.session_state.get(_BATCH_RESULTS_KEY, [])


def _resolve_spectral_curve(
    material_key: str | None,
    item_name: str | None,
) -> tuple[str | None, pd.DataFrame | None, Mapping[str, Any]]:
    bundle = _load_reference_bundle()
    alias_map = bundle.alias_map
    spectral_curves = bundle.spectral_curves
    metadata = bundle.metadata

    candidates = [material_key, item_name]
    for candidate in candidates:
        if not candidate:
            continue
        text = str(candidate)
        if text in spectral_curves:
            return text, spectral_curves[text], metadata.get(text, {})
        slug = ds.slugify(ds.normalize_item(text))
        canonical = alias_map.get(slug)
        if canonical and canonical in spectral_curves:
            return canonical, spectral_curves[canonical], metadata.get(canonical, {})
    return None, None, {}


def _synthetic_spectral_curve(
    spectral_score: float, mechanical_score: float
) -> pd.DataFrame:
    import numpy as np

    wavenumbers = np.linspace(500, 4000, 40)
    base = 0.45 + (1.0 - spectral_score) * 0.35
    modulation = 0.1 + (1.0 - mechanical_score) * 0.25
    transmittance = 100.0 * (base + modulation * np.sin(np.linspace(0, 3.5, 40)))
    frame = pd.DataFrame(
        {
            "wavenumber_cm_1": wavenumbers,
            "transmittance_pct": transmittance.clip(lower=5.0, upper=95.0),
        }
    )
    return frame


def _score_radar_chart(spectral: float, mechanical: float) -> go.Figure:
    thresholds = [
        _SCORE_THRESHOLDS["spectral"],
        _SCORE_THRESHOLDS["mechanical"],
    ]
    values = [spectral, mechanical]
    categories = ["Espectral", "Mecánico"]

    def _close(payload: list[float]) -> list[float]:
        return payload + payload[:1]

    fig = go.Figure()
    fig.add_trace(
        go.Scatterpolar(
            r=_close(values),
            theta=_close(categories),
            fill="toself",
            name="Score",
            line=dict(color="#38bdf8"),
            fillcolor="rgba(56, 189, 248, 0.3)",
        )
    )
    fig.add_trace(
        go.Scatterpolar(
            r=_close(thresholds),
            theta=_close(categories),
            fill="toself",
            name="Umbral",
            line=dict(color="#f97316", dash="dash"),
            fillcolor="rgba(249, 115, 22, 0.1)",
        )
    )
    fig.update_layout(
        showlegend=False,
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        margin=dict(l=0, r=0, t=20, b=20),
        height=220,
    )
    return fig


def _compute_severity(row: Mapping[str, Any]) -> float:
    spectral = max(float(row.get("spectral_score", 0.0) or 0.0), 0.0)
    mechanical = max(float(row.get("mechanical_score", 0.0) or 0.0), 0.0)
    utility = max(float(row.get("material_utility_score", 0.0) or 0.0), 0.0)
    gaps = [
        max(0.0, _SCORE_THRESHOLDS["spectral"] - spectral),
        max(0.0, _SCORE_THRESHOLDS["mechanical"] - mechanical),
        max(0.0, 0.5 - utility),
    ]
    return max(gaps)


def _standardise_spectral_curve(curve: pd.DataFrame) -> pd.DataFrame:
    if curve is None or curve.empty:
        return pd.DataFrame(columns=["wavenumber_cm_1", "transmittance_pct"])

    working = curve.copy()
    if "transmittance_pct" in working.columns:
        working["transmittance_pct"] = pd.to_numeric(
            working["transmittance_pct"], errors="coerce"
        )
    else:
        value_column = None
        for candidate in ("absorbance_norm_1um", "absorbance", "intensity", "signal"):
            if candidate in working.columns:
                value_column = candidate
                break
        if value_column is None:
            for column in working.columns:
                if column != "wavenumber_cm_1":
                    value_column = column
                    break
        values = pd.to_numeric(working.get(value_column), errors="coerce")
        if value_column and "absorb" in value_column:
            min_val = float(values.min()) if not values.isna().all() else 0.0
            max_val = float(values.max()) if not values.isna().all() else 1.0
            span = max(max_val - min_val, 1e-6)
            working["transmittance_pct"] = (1.0 - (values - min_val) / span) * 100.0
        else:
            working["transmittance_pct"] = values

    working["transmittance_pct"] = working["transmittance_pct"].interpolate().fillna(0.0)
    working = working.loc[:, [col for col in working.columns if col in {"wavenumber_cm_1", "transmittance_pct"}]]
    return working.dropna(subset=["wavenumber_cm_1", "transmittance_pct"])


def _traffic_color(score: float) -> str:
    if score >= 0.75:
        return "#22c55e"
    if score >= 0.5:
        return "#facc15"
    return "#ef4444"


def _traffic_label(score: float) -> str:
    if score >= 0.75:
        return "Alto"
    if score >= 0.5:
        return "Medio"
    return "Bajo"


def _render_metric(label: str, score: float, help_text: str | None = None) -> None:
    color = _traffic_color(score)
    status = _traffic_label(score)
    with st.container():
        st.markdown(
            (
                "<div style='border-radius:12px;padding:1rem;background:{color};color:white;'>"
                "<div style='font-size:0.85rem;text-transform:uppercase;opacity:0.85;'>{label}</div>"
                "<div style='font-size:1.8rem;font-weight:700;'>{value:.2f}</div>"
                "<div style='font-size:0.9rem;'>Nivel {status}</div>"
                "</div>"
            ).format(color=color, label=label, value=score, status=status),
            unsafe_allow_html=True,
        )
        if help_text:
            st.caption(help_text)


st.title("🛰️ Centro de control marciano")
st.markdown(
    """
    Consolida vuelos, inventario, decisiones automáticas y planificación diaria
    en una sola consola. Cada pestaña se alimenta de telemetría en tiempo real
    para que operaciones priorice acciones críticas y documente resultados.
    """
)

generator_service = GeneratorService()
telemetry_service = MarsControlCenterService()

analysis_state: dict[str, Any] | None = st.session_state.get("policy_analysis")
manifest_df: pd.DataFrame | None = st.session_state.get("uploaded_manifest_df")

st.session_state.setdefault(_MANUAL_DECISIONS_KEY, {})

tabs = st.tabs(
    [
        "🛰️ Flight Radar / Mapa",
        "📦 Inventario vivo",
        "🤖 Decisiones IA",
        "🗺️ Planner",
        "🎛️ Modo Demo",
    ]
)


with tabs[0]:
    st.subheader("Flight Radar · logística interplanetaria")
    passport: Mapping[str, Any] | None = None
    if analysis_state:
        passport = analysis_state.get("material_passport")

    manifest_signature = "baseline"
    if passport:
        manifest_signature = (
            f"{passport.get('generated_at', 'baseline')}"
            f":{passport.get('total_items', 0)}:{passport.get('total_mass_kg', 0)}"
        )
    elif isinstance(manifest_df, pd.DataFrame) and not manifest_df.empty:
        manifest_signature = f"uploaded:{manifest_df.shape[0]}:{','.join(manifest_df.columns)}"

    flights_df: pd.DataFrame | None = st.session_state.get("flight_operations_table")
    previous_signature = st.session_state.get("flight_operations_signature")
    if flights_df is None or previous_signature != manifest_signature:
        flights_df = telemetry_service.flight_operations_overview(
            passport,
            manifest_df=manifest_df,
            analysis_state=analysis_state,
        )
        flights_df = _apply_manual_overrides(flights_df)
        _store_flight_snapshot(flights_df)
        st.session_state["flight_operations_signature"] = manifest_signature
        st.session_state.setdefault("flight_operations_recent_events", [])
        st.session_state.setdefault("flight_operations_recent_changes", [])

    if flights_df is None or flights_df.empty:
        st.info("Aún no hay vuelos registrados. Cargá un manifiesto para sincronizar la carga.")
    else:
        control_cols = st.columns([2, 1])
        with control_cols[0]:
            auto_tick = st.toggle(
                "Tick automático cada 20 s",
                value=st.session_state.get("mars_auto_tick_toggle", False),
                key="mars_auto_tick_toggle",
            )
        with control_cols[1]:
            manual_tick = st.button("Avanzar simulación", use_container_width=True)

        tick_triggered = bool(manual_tick)
        if auto_tick:
            tick_count = st.autorefresh(
                interval=20000,
                limit=None,
                key="mars_auto_tick_counter",
            )
            previous_count = st.session_state.get("mars_auto_tick_prev", 0)
            if tick_count > previous_count:
                st.session_state["mars_auto_tick_prev"] = tick_count
                if tick_count > 0:
                    tick_triggered = True

        previous_decisions: Mapping[str, str] = st.session_state.get(
            "flight_operations_last_decisions", {}
        )
        if tick_triggered:
            flights_df, events, changed_flights = telemetry_service.advance_timeline(
                flights_df,
                manifest_df=manifest_df,
                analysis_state=analysis_state,
                previous_decisions=previous_decisions,
            )
            flights_df = _apply_manual_overrides(flights_df)
            _store_flight_snapshot(flights_df)
            st.session_state["flight_operations_recent_events"] = events
            st.session_state["flight_operations_recent_changes"] = list(changed_flights)
        else:
            events = st.session_state.get("flight_operations_recent_events", [])
            changed_flights = set(
                st.session_state.get("flight_operations_recent_changes", [])
            )

        map_payload = telemetry_service.build_map_payload(flights_df)
        capsule_data = map_payload["capsules"]
        zone_data = map_payload["zones"]
        geometry = map_payload["geometry"]

        st.markdown("#### Jezero Crater · orientación visual")
        _render_crater_overview(capsule_data, zone_data, geometry)

        layers: list[pdk.Layer] = list(_mars_background_layers())
        if geometry and isinstance(geometry, Mapping) and geometry.get("features"):
            layers.append(
                pdk.Layer(
                    "GeoJsonLayer",
                    geometry,
                    id="jezero-boundary",
                    stroked=True,
                    filled=False,
                    get_line_color=[180, 198, 231],
                    line_width_min_pixels=2,
                )
            )
        if isinstance(zone_data, pd.DataFrame) and not zone_data.empty:
            layers.append(
                pdk.Layer(
                    "ScatterplotLayer",
                    zone_data,
                    id="zones",
                    get_position="[longitude, latitude]",
                    get_radius="radius_m",
                    get_fill_color="[color_r, color_g, color_b]",
                    get_line_color="[color_r, color_g, color_b]",
                    pickable=True,
                    stroked=True,
                    opacity=0.25,
                    radius_units="meters",
                )
            )
        if isinstance(capsule_data, pd.DataFrame) and not capsule_data.empty:
            layers.append(
                pdk.Layer(
                    "ScatterplotLayer",
                    capsule_data,
                    id="capsules",
                    get_position="[longitude, latitude]",
                    get_radius="marker_radius_m",
                    get_fill_color="[category_color_r, category_color_g, category_color_b]",
                    get_line_color="[status_color_r, status_color_g, status_color_b]",
                    radius_units="meters",
                    pickable=True,
                    stroked=True,
                    auto_highlight=True,
                )
            )

        tooltip = {
            "html": (
                "<div style='font-size:14px;font-weight:600;'>{vehicle}</div>"
                "<div>{status}</div>"
                "<div>ETA: {eta_minutes} min</div>"
                "<div>Materiales: {materials_tooltip}</div>"
                "<div>Espectro: {material_spectrum}</div>"
                "<div>Densidad: {density} g/cm³ · Compatibilidad: {compatibility}</div>"
                "<div>{tooltip}</div>"
            ),
            "style": {"backgroundColor": "#0f172a", "color": "white"},
        }

        view_state = pdk.ViewState(
            latitude=18.43,
            longitude=77.58,
            zoom=9.1,
            pitch=45,
            bearing=25,
        )

        st.pydeck_chart(
            pdk.Deck(
                layers=layers,
                initial_view_state=view_state,
                tooltip=tooltip,
                map_style=None,
                map_provider=None,
            ),
            use_container_width=True,
        )
        st.caption("Mapa operacional de Jezero: cápsulas, zonas clave y perímetro de seguridad.")

        micro_divider()
        display_df = flights_df[
            [
                "flight_id",
                "vehicle",
                "status_label",
                "eta_minutes",
                "key_materials_display",
                "ai_decision",
                "decision_indicator",
            ]
        ].rename(
            columns={
                "flight_id": "Vuelo",
                "vehicle": "Vehículo",
                "status_label": "Estado",
                "eta_minutes": "ETA (min)",
                "key_materials_display": "Materiales clave",
                "ai_decision": "Decisión IA",
                "decision_indicator": "Decisión ∆",
            }
        )

        def _status_style(series: pd.Series) -> list[str]:
            colors = flights_df.loc[series.index, "status_color"]
            return [
                f"background-color: {color}; color: white; font-weight:600" for color in colors
            ]

        def _decision_style(series: pd.Series) -> list[str]:
            flags = flights_df.loc[series.index, "decision_indicator"].astype(str)
            return [
                "background-color: #facc15; color: #1f2937; font-weight:700" if flag else ""
                for flag in flags
            ]

        styled_df = (
            display_df.style.apply(_status_style, subset=["Estado"])
            .apply(_decision_style, subset=["Decisión ∆"])
            .format({"ETA (min)": "{:,.0f}"})
        )

        st.dataframe(
            styled_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "ETA (min)": st.column_config.NumberColumn("ETA (min)", format="%d min"),
                "Materiales clave": st.column_config.TextColumn(
                    "Materiales clave",
                    help="Top materiales declarados por masa para la cápsula.",
                ),
                "Decisión IA": st.column_config.TextColumn(
                    "Decisión IA",
                    help="Última directriz activa para la misión.",
                ),
                "Decisión ∆": st.column_config.TextColumn(
                    "∆",
                    help="Indicador de cambios recientes en decisiones automáticas.",
                ),
            },
        )

        editor_df = flights_df[
            [
                "flight_id",
                "vehicle",
                "status_badge",
                "eta_minutes",
                "ai_decision",
                "key_materials_display",
                "material_spectrum",
                "material_density",
                "compatibility_index",
            ]
        ].rename(
            columns={
                "flight_id": "Vuelo",
                "vehicle": "Vehículo",
                "status_badge": "Estado",
                "eta_minutes": "ETA (min)",
                "ai_decision": "Decisión IA",
                "key_materials_display": "Materiales clave",
                "material_spectrum": "Espectro",
                "material_density": "Densidad (g/cm³)",
                "compatibility_index": "Compatibilidad",
            }
        )

        editor_result = st.data_editor(
            editor_df,
            hide_index=True,
            use_container_width=True,
            column_config={
                "Vuelo": st.column_config.TextColumn("Vuelo", disabled=True),
                "Vehículo": st.column_config.TextColumn("Vehículo", disabled=True),
                "Estado": st.column_config.TextColumn("Estado", disabled=True),
                "ETA (min)": st.column_config.NumberColumn("ETA (min)", format="%d min", disabled=True),
                "Materiales clave": st.column_config.TextColumn(
                    "Materiales clave",
                    disabled=True,
                ),
                "Espectro": st.column_config.TextColumn("Espectro", disabled=True),
                "Densidad (g/cm³)": st.column_config.NumberColumn(
                    "Densidad (g/cm³)", format="%.2f", disabled=True
                ),
                "Compatibilidad": st.column_config.NumberColumn(
                    "Compatibilidad", format="%.2f", disabled=True
                ),
                "Decisión IA": st.column_config.TextColumn(
                    "Decisión IA",
                    help="Podés forzar una decisión manual que se mantendrá hasta el próximo tick.",
                ),
            },
            key="flight_ops_editor",
        )

        if not editor_result.equals(editor_df):
            for idx in editor_result.index:
                new_value = editor_result.loc[idx, "Decisión IA"]
                flights_df.at[idx, "ai_decision"] = new_value
            flights_df["decision_changed"] = False
            flights_df["decision_indicator"] = ""
            st.session_state["flight_operations_table"] = flights_df
            st.session_state["flight_operations_last_decisions"] = {
                row["flight_id"]: row["ai_decision"]
                for row in flights_df.to_dict(orient="records")
            }
            st.session_state["flight_operations_recent_changes"] = []
            st.success("Decisiones actualizadas manualmente.")

        micro_divider()

        timeline_df = telemetry_service.timeline_history()
        if isinstance(timeline_df, pd.DataFrame) and not timeline_df.empty:
            st.markdown("#### Timeline de eventos logísticos")
            st.dataframe(
                timeline_df[[
                    "tick",
                    "category",
                    "title",
                    "details",
                    "capsule_id",
                    "mass_delta",
                ]],
                hide_index=True,
                use_container_width=True,
                column_config={
                    "tick": st.column_config.NumberColumn("Tick"),
                    "category": st.column_config.TextColumn("Categoría"),
                    "title": st.column_config.TextColumn("Evento"),
                    "details": st.column_config.TextColumn("Detalles"),
                    "capsule_id": st.column_config.TextColumn("Cápsula"),
                    "mass_delta": st.column_config.NumberColumn("Δ Masa (kg)", format="%.1f"),
                },
            )

        if events:
            st.markdown("**Eventos recientes**")
            for event in events:
                tick = event.get("tick")
                title = event.get("title")
                category = event.get("category")
                st.markdown(f"• Tick {tick}: {title} ({category})")
                details = event.get("details")
                if details:
                    st.caption(details)


with tabs[1]:
    st.subheader("Inventario vivo")
    try:
        inventory_df, metrics, category_payload = telemetry_service.inventory_snapshot()
    except Exception as exc:
        st.error(f"No se pudo cargar el inventario en vivo: {exc}")
    else:
        render_mission_objective(metrics)

        problematic = int(metrics.get("problematic_count", 0))
        st.caption(
            "Residuos problemáticos detectados: "
            f"{problematic}. Coordiná protocolos especiales según severidad."
        )

        category_stats = category_payload.get("categories")
        flows_df = category_payload.get("flows")
        palette = category_payload.get("group_palette", {})
        group_labels = category_payload.get("material_groups", {})
        destination_info = category_payload.get("destinations", {})

        has_breakdown = isinstance(category_stats, pd.DataFrame) and not category_stats.empty

        if has_breakdown:
            micro_divider()
            st.markdown("**Flujos circulares por categoría**")

            badge_emojis = {
                "polimeros": "🟦",
                "metales": "🟧",
                "textiles": "🟩",
                "espumas": "🟪",
                "mixtos": "⬜",
            }
            legend_labels: list[str] = []
            for key in ("polimeros", "metales", "textiles", "espumas", "mixtos"):
                label = group_labels.get(key, key.title())
                color = palette.get(key)
                emoji = badge_emojis.get(key, "•")
                legend_labels.append(
                    f"{emoji} {label}{f' · {color}' if color else ''}"
                )
            badge_group(legend_labels)
            st.caption("Colores sincronizados con el bubble chart según familia de material.")

            sankey_col, bubble_col = st.columns((1.1, 1))

            if isinstance(flows_df, pd.DataFrame) and not flows_df.empty:
                categories = category_stats["category"].astype(str).tolist()
                destination_keys = list(destination_info.keys())
                node_labels = categories + [
                    destination_info[key]["display"] for key in destination_keys
                ]
                node_colors = [
                    palette.get(group, "#94a3b8")
                    for group in category_stats["material_group"].astype(str)
                ] + [destination_info[key]["color"] for key in destination_keys]

                link_sources: list[int] = []
                link_targets: list[int] = []
                link_values: list[float] = []
                link_custom: list[list[Any]] = []

                for _, flow_row in flows_df.iterrows():
                    destination_key = str(flow_row.get("destination_key"))
                    if destination_key not in destination_keys:
                        continue
                    category = str(flow_row.get("category"))
                    try:
                        source_index = categories.index(category)
                        target_index = len(categories) + destination_keys.index(destination_key)
                    except ValueError:
                        continue
                    mass_value = float(flow_row.get("mass_kg", 0.0))
                    if mass_value <= 0:
                        continue
                    link_sources.append(source_index)
                    link_targets.append(target_index)
                    link_values.append(mass_value)
                    display = destination_info[destination_key]["label"]
                    link_custom.append([category, display, mass_value])

                if link_values:
                    sankey_fig = go.Figure(
                        data=[
                            go.Sankey(
                                arrangement="snap",
                                node=dict(
                                    pad=16,
                                    thickness=18,
                                    label=node_labels,
                                    color=node_colors,
                                ),
                                link=dict(
                                    source=link_sources,
                                    target=link_targets,
                                    value=link_values,
                                    customdata=link_custom,
                                    hovertemplate=(
                                        "<b>%{customdata[0]}</b> → <b>%{customdata[1]}</b>"
                                        "<br>Masa: %{customdata[2]:,.1f} kg<extra></extra>"
                                    ),
                                ),
                            )
                        ]
                    )
                    sankey_fig.update_layout(
                        margin=dict(t=30, b=10, l=10, r=10),
                        font=dict(color="#e2e8f0"),
                        paper_bgcolor="rgba(12,18,28,1)",
                    )
                    sankey_col.plotly_chart(sankey_fig, use_container_width=True)
                else:
                    sankey_col.info("No hay datos suficientes para el flujo Sankey.")
            else:
                sankey_col.info("No hay datos suficientes para el flujo Sankey.")

            stock_sizes = category_stats["stock_mass_kg"].astype(float).fillna(0.0)
            max_stock = float(stock_sizes.max()) if not stock_sizes.empty else 0.0
            if max_stock <= 0:
                marker_sizes = [15.0 for _ in stock_sizes]
                size_reference = 1.0
            else:
                min_size = max_stock * 0.1
                marker_sizes = stock_sizes.clip(lower=min_size).tolist()
                size_reference = 2.0 * max(marker_sizes) / (32.0**2)

            bubble_colors = [
                palette.get(group, "#94a3b8")
                for group in category_stats["material_group"].astype(str)
            ]
            bubble_custom = [
                [
                    float(row.stock_mass_kg),
                    float(row.water_l),
                    float(row.energy_kwh),
                    float(row.cross_contamination_risk),
                ]
                for row in category_stats.itertuples()
            ]

            bubble_fig = go.Figure(
                data=[
                    go.Scatter3d(
                        x=category_stats["water_l"].astype(float),
                        y=category_stats["energy_kwh"].astype(float),
                        z=category_stats["purity_index"].astype(float),
                        text=category_stats["category"].astype(str),
                        mode="markers",
                        customdata=bubble_custom,
                        marker=dict(
                            size=marker_sizes,
                            sizemode="area",
                            sizeref=size_reference,
                            opacity=0.85,
                            color=bubble_colors,
                            line=dict(color="rgba(255,255,255,0.55)", width=1),
                        ),
                        hovertemplate=(
                            "<b>%{text}</b><br>Stock estratégico: %{customdata[0]:,.1f} kg"
                            "<br>Agua recuperable: %{x:,.1f} L"
                            "<br>Energía estimada: %{y:,.1f} kWh"
                            "<br>Pureza: %{z:.1f}% · Contaminación: %{customdata[3]:.1f}%"
                            "<extra></extra>"
                        ),
                    )
                ]
            )
            bubble_fig.update_layout(
                margin=dict(t=30, b=10, l=0, r=0),
                paper_bgcolor="rgba(12,18,28,1)",
                plot_bgcolor="rgba(12,18,28,1)",
                font=dict(color="#e2e8f0"),
                scene=dict(
                    xaxis=dict(
                        title="Agua recuperable (L)",
                        backgroundcolor="rgba(15,23,42,0.18)",
                        gridcolor="rgba(148,163,184,0.3)",
                    ),
                    yaxis=dict(
                        title="Energía estimada (kWh)",
                        backgroundcolor="rgba(15,23,42,0.18)",
                        gridcolor="rgba(148,163,184,0.3)",
                    ),
                    zaxis=dict(
                        title="Índice de pureza (%)",
                        range=[0, 100],
                        backgroundcolor="rgba(15,23,42,0.18)",
                        gridcolor="rgba(148,163,184,0.3)",
                    ),
                ),
            )
            bubble_col.plotly_chart(bubble_fig, use_container_width=True)

            top_categories = (
                category_stats.sort_values("stock_mass_kg", ascending=False).head(3)
            )
            if not top_categories.empty:
                st.markdown("**Indicadores clave por stock estratégico**")
                indicator_cols = st.columns(len(top_categories))
                for column, (_, row) in zip(indicator_cols, top_categories.iterrows()):
                    column.metric(
                        str(row["category"]),
                        _format_metric(float(row["stock_mass_kg"]), "kg"),
                        delta=(
                            f"Pureza {row['purity_index']:.0f}% · Contam. "
                            f"{row['cross_contamination_risk']:.0f}%"
                        ),
                    )
                    column.caption(
                        " · ".join(
                            [
                                _format_metric(float(row["energy_kwh"]), "kWh"),
                                _format_metric(float(row["water_l"]), "L"),
                            ]
                        )
                    )

        micro_divider()
        st.dataframe(
            inventory_df,
            use_container_width=True,
            hide_index=True,
        )


with tabs[2]:
    st.subheader("Decisiones IA y reporting")
    st.caption(
        "Cargá el manifiesto actualizado para que Rex-AI evalúe compatibilidad, penalizaciones y artefactos de reporting."
    )

    template_bytes = manifest_template_csv_bytes()
    template_df = build_manifest_template()
    col_template, col_preview = st.columns(2)
    with col_template:
        st.download_button(
            "Descargar plantilla CSV",
            template_bytes,
            file_name="manifiesto_plantilla.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with col_preview:
        st.dataframe(
            template_df,
            column_config={col: st.column_config.Column(col.replace("_", " ").title()) for col in template_df.columns},
            hide_index=True,
            use_container_width=True,
        )

    uploaded_file = st.file_uploader(
        "Manifiesto (CSV)",
        type=["csv"],
        accept_multiple_files=False,
        key="manifest_uploader",
    )
    include_pdf = st.checkbox("Generar Material Passport en PDF", value=False)

    if st.button("Evaluar manifiesto", use_container_width=True):
        if uploaded_file is None:
            st.warning("Subí un archivo CSV para iniciar el análisis.")
        else:
            manifest_df = load_manifest_from_upload(uploaded_file)
            with st.spinner("Analizando manifiesto con heurísticas de política..."):
                analysis_state = run_policy_analysis(
                    generator_service, manifest_df, include_pdf=include_pdf
                )
            st.session_state["policy_analysis"] = analysis_state
            st.session_state["uploaded_manifest_df"] = manifest_df

    if not analysis_state:
        st.info(
            "Esperando un manifiesto. Cuando se procese uno verás métricas en vivo, recomendaciones y descargas."
        )
    else:
        summary = telemetry_service.summarize_decisions(analysis_state)

        cols = st.columns(3)
        with cols[0]:
            _render_metric("Puntaje promedio", summary["score"])
        with cols[1]:
            st.metric("Masa total (kg)", f"{summary['total_mass']:.1f}")
        with cols[2]:
            st.metric("Total ítems", f"{summary['item_count']}")

        passport = analysis_state.get("material_passport", {})

        manifest_source_df: pd.DataFrame | None = None
        if isinstance(manifest_df, pd.DataFrame) and not manifest_df.empty:
            manifest_source_df = manifest_df
        else:
            candidate_manifest = analysis_state.get("manifest")
            if isinstance(candidate_manifest, pd.DataFrame):
                manifest_source_df = candidate_manifest
            elif isinstance(candidate_manifest, Mapping):
                manifest_source_df = pd.DataFrame(candidate_manifest)

        batch_results = _ensure_manifest_batch(generator_service, manifest_source_df)
        if batch_results:
            batch_entry = batch_results[0]
            scored_manifest = batch_entry.get("scored_manifest", pd.DataFrame())
            compatibility = batch_entry.get("compatibility", pd.DataFrame())
            recommendations = batch_entry.get("policy_recommendations", pd.DataFrame())
        else:
            scored_manifest = summary.get("manifest", pd.DataFrame())
            compatibility = summary.get("compatibility", pd.DataFrame())
            recommendations = summary.get("recommendations", pd.DataFrame())

        flights_df = st.session_state.get("flight_operations_table")
        if (
            (flights_df is None or flights_df.empty)
            and manifest_source_df is not None
            and not manifest_source_df.empty
        ):
            flights_df = telemetry_service.flight_operations_overview(
                passport,
                manifest_df=manifest_source_df,
                analysis_state=analysis_state,
            )
            flights_df = _apply_manual_overrides(flights_df)
            _store_flight_snapshot(flights_df)

        working_manifest = (
            scored_manifest.copy() if isinstance(scored_manifest, pd.DataFrame) else pd.DataFrame()
        )
        if not working_manifest.empty:
            for column in ("spectral_score", "mechanical_score", "material_utility_score", "mass_kg"):
                if column in working_manifest.columns:
                    working_manifest[column] = pd.to_numeric(
                        working_manifest[column], errors="coerce"
                    ).fillna(0.0)
                else:
                    working_manifest[column] = 0.0
            working_manifest["severity"] = working_manifest.apply(_compute_severity, axis=1)

        flight_lookup: dict[str, Mapping[str, Any]] = {}
        if isinstance(flights_df, pd.DataFrame) and not flights_df.empty:
            for record in flights_df.to_dict(orient="records"):
                manifest_ref = str(record.get("manifest_ref"))
                if manifest_ref:
                    flight_lookup[manifest_ref] = record

        if not working_manifest.empty:
            def _resolve_shipment(row: pd.Series) -> str:
                item_text = str(row.get("item") or "").lower()
                material_text = str(row.get("material_key") or "").lower()
                for manifest_ref, record in flight_lookup.items():
                    materials = record.get("key_materials") or []
                    for material in materials:
                        token = str(material).lower()
                        if token and (token in item_text or token in material_text):
                            return manifest_ref
                if flight_lookup:
                    return next(iter(flight_lookup.keys()))
                return "manifest-alpha"

            working_manifest["shipment_ref"] = working_manifest.apply(_resolve_shipment, axis=1)
        else:
            working_manifest["shipment_ref"] = "manifest-alpha"

        shipments: list[dict[str, Any]] = []
        decisions_records: list[dict[str, Any]] = []

        if not working_manifest.empty:
            grouped = working_manifest.groupby("shipment_ref", dropna=False)
            for manifest_ref, group in grouped:
                manifest_key = str(manifest_ref or "manifest-alpha")
                flight_info = flight_lookup.get(manifest_key, {})
                severity = float(group["severity"].max()) if not group.empty else 0.0
                critical = group[group["severity"] > 0.0].sort_values(
                    "material_utility_score"
                )
                focus_pool = critical if not critical.empty else group
                focus_row = focus_pool.sort_values("material_utility_score").iloc[0]
                compatibility_score = float(group["material_utility_score"].mean())
                spectral_avg = float(group["spectral_score"].mean())
                mechanical_avg = float(group["mechanical_score"].mean())
                critical_count = int((group["severity"] > 0.0).sum())

                rec_subset = pd.DataFrame()
                if (
                    isinstance(recommendations, pd.DataFrame)
                    and not recommendations.empty
                    and "item_name" in recommendations.columns
                ):
                    rec_subset = recommendations[recommendations["item_name"].isin(
                        group["item"].astype(str)
                    )]

                comp_subset = pd.DataFrame()
                if (
                    isinstance(compatibility, pd.DataFrame)
                    and not compatibility.empty
                    and "material_key" in compatibility.columns
                ):
                    comp_subset = compatibility[compatibility["material_key"].isin(
                        group["material_key"].astype(str)
                    )]

                spectral_key, spectral_curve, spectral_meta = _resolve_spectral_curve(
                    focus_row.get("material_key"), focus_row.get("item")
                )
                synthetic_curve = False
                if spectral_curve is None or spectral_curve.empty:
                    spectral_curve = _synthetic_spectral_curve(
                        float(focus_row.get("spectral_score") or 0.0),
                        float(focus_row.get("mechanical_score") or 0.0),
                    )
                    synthetic_curve = True

                overrides = st.session_state.get(_MANUAL_DECISIONS_KEY, {})
                manual_payload = overrides.get(manifest_key) or overrides.get(
                    str(manifest_key), {}
                )
                order_label = manual_payload.get("label") or flight_info.get(
                    "ai_decision"
                ) or "Monitoreo nominal"
                order_badge = (
                    manual_payload.get("badge")
                    or flight_info.get("decision_indicator")
                    or "Orden Rex-AI"
                )

                if rec_subset is not None and not rec_subset.empty:
                    top_rec = rec_subset.sort_values(
                        "recommended_score", ascending=False
                    ).iloc[0]
                    recommendation_text = (
                        f"{top_rec.get('action')} → {top_rec.get('recommended_material_key')}"
                    )
                else:
                    recommendation_text = "Sin cambios"

                detected_text = (
                    f"{focus_row.get('item')} · Score {float(focus_row.get('material_utility_score') or 0.0):.2f}"
                )
                compatibility_text = f"{compatibility_score:.2f}"

                radar_fig = _score_radar_chart(spectral_avg, mechanical_avg)

                spectral_curve = _standardise_spectral_curve(spectral_curve).sort_values(
                    "wavenumber_cm_1", ascending=False
                )
                spectral_caption = (
                    "Curva sintetizada a partir de los puntajes de compatibilidad: "
                    "el bundle FTIR no expone este material."
                    if synthetic_curve
                    else f"FTIR de referencia · {spectral_meta.get('material', spectral_key)}"
                )

                shipments.append(
                    {
                        "manifest_ref": manifest_key,
                        "flight": flight_info,
                        "severity": severity,
                        "critical_count": critical_count,
                        "radar_fig": radar_fig,
                        "spectral_curve": spectral_curve,
                        "spectral_caption": spectral_caption,
                        "critical_table": critical[[
                            "item",
                            "material_key",
                            "material_utility_score",
                            "spectral_score",
                            "mechanical_score",
                            "mass_kg",
                        ]].head(6)
                        if not critical.empty
                        else group[[
                            "item",
                            "material_key",
                            "material_utility_score",
                            "spectral_score",
                            "mechanical_score",
                            "mass_kg",
                        ]].sort_values("material_utility_score").head(6),
                        "compatibility_subset": comp_subset,
                        "badges": [
                            f"Detectado · {detected_text}",
                            f"Compatibilidad · {compatibility_text}",
                            f"Sugerencia Rex-AI · {recommendation_text}",
                            f"{order_badge} · {order_label}",
                        ],
                        "order_label": order_label,
                        "order_badge": order_badge,
                    }
                )

                decisions_records.append(
                    {
                        "manifest_ref": manifest_key,
                        "flight_id": flight_info.get("flight_id"),
                        "vehicle": flight_info.get("vehicle"),
                        "critical_items": critical_count,
                        "worst_item": focus_row.get("item"),
                        "mean_score": compatibility_score,
                        "manual_action": manual_payload.get("action"),
                        "order_label": order_label,
                        "severity": severity,
                    }
                )

        shipments.sort(
            key=lambda payload: (-payload.get("severity", 0.0), payload.get("critical_count", 0))
        )

        micro_divider()
        st.markdown("### Prioridades por envío")
        if shipments:
            for shipment in shipments:
                flight_info = shipment["flight"]
                flight_label = flight_info.get("flight_id") or shipment["manifest_ref"]
                with st.container():
                    header_cols = st.columns([3, 1])
                    header_cols[0].markdown(f"**{flight_label} · {shipment['manifest_ref']}**")
                    badge_group(shipment["badges"], parent=header_cols[0])
                    header_cols[1].metric(
                        "Severidad",
                        f"{shipment['severity']:.2f}",
                        delta=f"{shipment['critical_count']} críticos",
                    )

                    chart_cols = st.columns([1, 1])
                    chart_cols[0].plotly_chart(
                        shipment["radar_fig"], use_container_width=True
                    )
                    spectral_fig = go.Figure(
                        data=[
                            go.Scatter(
                                x=shipment["spectral_curve"]["wavenumber_cm_1"],
                                y=shipment["spectral_curve"]["transmittance_pct"],
                                mode="lines",
                                line=dict(color="#facc15"),
                            )
                        ]
                    )
                    spectral_fig.update_layout(
                        margin=dict(t=10, b=0, l=0, r=0),
                        xaxis_title="Número de onda (cm⁻¹)",
                        yaxis_title="Transmittancia (%)",
                        height=220,
                    )
                    chart_cols[0].plotly_chart(spectral_fig, use_container_width=True)
                    chart_cols[0].caption(shipment["spectral_caption"])

                    detail_df = shipment["critical_table"]
                    chart_cols[1].markdown("**Ítems críticos**")
                    chart_cols[1].dataframe(
                        detail_df,
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            "material_utility_score": st.column_config.NumberColumn(
                                "Score", format="%.2f"
                            ),
                            "spectral_score": st.column_config.NumberColumn(
                                "Espectral", format="%.2f"
                            ),
                            "mechanical_score": st.column_config.NumberColumn(
                                "Mecánico", format="%.2f"
                            ),
                            "mass_kg": st.column_config.NumberColumn("Masa (kg)", format="%.2f"),
                        },
                    )

                    if isinstance(shipment["compatibility_subset"], pd.DataFrame) and not shipment[
                        "compatibility_subset"
                    ].empty:
                        chart_cols[1].markdown("**Compatibilidad documentada**")
                        bullets = []
                        for _, row in shipment["compatibility_subset"].iterrows():
                            bullets.append(
                                f"- `{row.get('material_key')}` ↔ `{row.get('partner_key')}` · {row.get('rule')}"
                            )
                        chart_cols[1].markdown("\n".join(bullets))
                    else:
                        chart_cols[1].caption("Sin trazas de compatibilidad adicionales.")

                    action_cols = st.columns(3)
                    if action_cols[0].button(
                        "Aceptar", key=f"accept_{shipment['manifest_ref']}"
                    ):
                        _register_manual_action(shipment["manifest_ref"], "accept")
                        st.success("Orden manual registrada.")
                    if action_cols[1].button(
                        "Rechazar", key=f"reject_{shipment['manifest_ref']}"
                    ):
                        _register_manual_action(shipment["manifest_ref"], "reject")
                        st.warning("La orden fue rechazada manualmente.")
                    if action_cols[2].button(
                        "Repriorizar", key=f"reprioritize_{shipment['manifest_ref']}"
                    ):
                        _register_manual_action(shipment["manifest_ref"], "reprioritize")
                        st.info("El envío fue marcado para repriorización.")
        else:
            st.success(
                "El lote evaluado no contiene alertas críticas: Rex-AI mantiene monitoreo nominal."
            )

        if decisions_records:
            export_df = pd.DataFrame(decisions_records)
            csv_buffer = io.StringIO()
            export_df.to_csv(csv_buffer, index=False)
            json_payload = json.dumps(
                export_df.to_dict(orient="records"), ensure_ascii=False, indent=2
            )
            export_cols = st.columns(2)
            with export_cols[0]:
                st.download_button(
                    "Descargar decisiones (CSV)",
                    csv_buffer.getvalue().encode("utf-8"),
                    file_name="decisiones_mars_control.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
            with export_cols[1]:
                st.download_button(
                    "Descargar decisiones (JSON)",
                    json_payload.encode("utf-8"),
                    file_name="decisiones_mars_control.json",
                    mime="application/json",
                    use_container_width=True,
                )

        micro_divider()
        st.markdown("**Detalle de ítems evaluados**")
        if not working_manifest.empty:
            st.dataframe(
                working_manifest,
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.caption("Sin datos de manifiesto evaluados en esta corrida.")

        micro_divider()
        st.markdown("**Recomendaciones de política**")
        if isinstance(recommendations, pd.DataFrame) and not recommendations.empty:
            st.dataframe(recommendations, use_container_width=True)
        else:
            st.success(
                "No se identificaron acciones prioritarias: todos los ítems superan el umbral de utilidad."
            )

        micro_divider()
        st.markdown("**Trazabilidad de compatibilidad**")
        if isinstance(compatibility, pd.DataFrame) and not compatibility.empty:
            st.dataframe(compatibility, use_container_width=True)
        else:
            st.caption("Sin datos de compatibilidad asociados al manifiesto.")

        micro_divider()
        st.markdown("**Material Passport**")
        st.json(passport)

        micro_divider()
        st.markdown("### Descargas")
        artifacts = summarize_artifacts(analysis_state)
        col_a, col_b, col_c = st.columns(3)
        policy_path = artifacts.get("policy_recommendations_csv")
        if isinstance(policy_path, Path) and policy_path.exists():
            with col_a:
                st.download_button(
                    "Recomendaciones (CSV)",
                    policy_path.read_bytes(),
                    file_name=policy_path.name,
                    mime="text/csv",
                    use_container_width=True,
                )
        compat_path = artifacts.get("compatibility_matrix_parquet")
        if isinstance(compat_path, Path) and compat_path.exists():
            with col_b:
                st.download_button(
                    "Compatibilidad (Parquet)",
                    compat_path.read_bytes(),
                    file_name=compat_path.name,
                    mime="application/octet-stream",
                    use_container_width=True,
                )
        passport_path = artifacts.get("material_passport_json")
        if isinstance(passport_path, Path) and passport_path.exists():
            with col_c:
                st.download_button(
                    "Material Passport (JSON)",
                    passport_path.read_bytes(),
                    file_name=passport_path.name,
                    mime="application/json",
                    use_container_width=True,
                )

        pdf_path = artifacts.get("material_passport_pdf")
        if isinstance(pdf_path, Path) and pdf_path.exists():
            st.download_button(
                "Material Passport (PDF)",
                pdf_path.read_bytes(),
                file_name=pdf_path.name,
                mime="application/pdf",
                use_container_width=True,
            )


with tabs[3]:
    st.subheader("Planner operacional")
    if not analysis_state:
        st.info("Cargá un manifiesto para que el planner recomiende procesos prioritarios.")
    else:
        summary = telemetry_service.summarize_decisions(analysis_state)
        planner_df = telemetry_service.build_planner_schedule(summary.get("manifest"))
        if planner_df.empty:
            st.caption("Sin procesos asignados todavía. Ajustá el manifiesto para generar recomendaciones.")
        else:
            st.caption(
                "Top procesos sugeridos por residuo crítico (ordenados por masa declarada)."
            )
            st.dataframe(
                planner_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "item": st.column_config.TextColumn("Residuo"),
                    "category": st.column_config.TextColumn("Categoría"),
                    "process_id": st.column_config.TextColumn("Proceso"),
                    "match_score": st.column_config.NumberColumn("Score", format="%.2f"),
                    "match_reason": st.column_config.TextColumn("Racional"),
                },
            )


with tabs[4]:
    st.subheader("Modo Demo")
    st.caption(
        "Activá este modo cuando presentes la misión: mantiene un guion sintético sincronizado con la telemetría."
    )

    loop_cols = st.columns([3, 2, 2])
    default_auto = st.session_state.get("demo_event_auto", False)
    default_interval = int(st.session_state.get("demo_event_interval", 20))
    with loop_cols[0]:
        auto_loop = st.checkbox(
            "Loop automático",
            value=default_auto,
            key="demo_event_auto_checkbox",
            help="Genera un evento demo cada n segundos",
        )
    with loop_cols[1]:
        interval_seconds = st.slider(
            "Intervalo (s)",
            min_value=5,
            max_value=60,
            value=default_interval,
            step=5,
            key="demo_event_interval_slider",
        )
    with loop_cols[2]:
        reset_script = st.button("Reiniciar script", use_container_width=True)

    trigger_next = st.button("Emitir siguiente evento", use_container_width=True)

    st.session_state["demo_event_interval"] = interval_seconds

    if reset_script:
        mars_control.reset_demo_events()
        st.session_state.pop("demo_last_event", None)
        st.success("Script demo reiniciado")
        st.experimental_rerun()

    new_event: mars_control.DemoEvent | None = None
    if trigger_next:
        new_event = mars_control.generate_demo_event(interval_seconds, force=True)

    if auto_loop:
        st.session_state["demo_event_auto"] = True
        st.autorefresh(
            interval=int(interval_seconds * 1000),
            limit=None,
            key="demo_event_autorefresh",
        )
        new_event = new_event or mars_control.generate_demo_event(interval_seconds)
    else:
        st.session_state["demo_event_auto"] = False

    history = mars_control.get_demo_event_history(limit=8)
    latest_event = new_event or (history[0] if history else None)
    if latest_event:
        st.session_state["demo_last_event"] = latest_event

    last_event: mars_control.DemoEvent | None = st.session_state.get(
        "demo_last_event"
    )

    if last_event:
        st.markdown(_render_demo_event_card(last_event), unsafe_allow_html=True)
        if last_event.audio_bytes:
            st.audio(last_event.audio_bytes, format="audio/wav")
        elif last_event.audio_path:
            st.audio(last_event.audio_path, format="audio/wav")
    else:
        st.info(
            "Aún no se emitieron eventos demo. Activá el loop o generá uno manualmente."
        )

    if history:
        ticker_events = list(reversed(history))
        st.markdown(_render_demo_ticker(ticker_events), unsafe_allow_html=True)

        log_rows: list[dict[str, str]] = []
        for event in history:
            metadata = " · ".join(
                f"{key}: {value}" for key, value in event.metadata.items()
            )
            log_rows.append(
                {
                    "Hora": _format_demo_timestamp(event),
                    "Evento": event.title,
                    "Detalle": event.message,
                    "Nivel": event.severity.upper(),
                    "Metadata": metadata,
                }
            )
        st.dataframe(
            pd.DataFrame(log_rows),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.caption("El ticker se completará a medida que se emitan eventos del guion demo.")

    micro_divider()
    st.markdown("#### Inyectar manifiesto demo")
    st.caption(
        "Seleccioná un manifiesto preconfigurado para ver cómo las decisiones IA cambian en vivo."
    )

    catalogue = mars_control.demo_manifest_catalogue()
    if catalogue:
        options = {entry["label"]: entry for entry in catalogue if entry.get("label")}
        if not options:
            options = {entry["key"]: entry for entry in catalogue}
        labels = list(options.keys())
        default_label = st.session_state.get("demo_manifest_selected", labels[0])
        selection = st.selectbox(
            "Seleccioná un manifiesto de prueba",
            options=labels,
            index=labels.index(default_label) if default_label in labels else 0,
        )
        st.session_state["demo_manifest_selected"] = selection
        selected_entry = options[selection]
        if selected_entry.get("description"):
            st.caption(selected_entry["description"])
        preview_df = pd.DataFrame(selected_entry.get("rows", []))
        st.dataframe(preview_df, use_container_width=True, hide_index=True)

        if st.button(
            "Inyectar manifiesto demo",
            use_container_width=True,
            key="inject_demo_manifest_button",
        ):
            manifest_df = mars_control.load_demo_manifest(selected_entry["key"])
            with st.spinner("Generando decisiones Rex-AI para el manifiesto demo..."):
                analysis = run_policy_analysis(
                    generator_service, manifest_df, include_pdf=False
                )
            st.session_state["policy_analysis"] = analysis
            st.session_state["uploaded_manifest_df"] = manifest_df
            st.success(
                "Manifiesto demo procesado. Revisá Decisiones IA y Flight Radar para ver los cambios."
            )
            st.experimental_rerun()
    else:
        st.warning("No se encontraron manifiestos demo preconfigurados.")

    with st.expander("Guion demo predefinido"):
        script_entries = mars_control.demo_event_script()
        script_df = pd.DataFrame(
            [
                {
                    "Evento": entry.title,
                    "Categoría": entry.category,
                    "Nivel": entry.severity,
                    "Mensaje": entry.message,
                }
                for entry in script_entries
            ]
        )
        st.dataframe(script_df, use_container_width=True, hide_index=True)

