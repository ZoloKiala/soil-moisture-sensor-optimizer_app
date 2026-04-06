# sensor_optimizer/views.py
from __future__ import annotations

import base64
import io
import json
import math
import os
import re
import tempfile
from typing import Optional

import pandas as pd
import matplotlib.pyplot as plt

from django.conf import settings
from django.core.mail import send_mail
from django.http import HttpResponse, HttpResponseBadRequest
from django.shortcuts import render, redirect
from django.utils.safestring import mark_safe
from datetime import date

import folium
from folium.plugins import Draw, LocateControl
from branca.colormap import LinearColormap

from shapely.geometry import shape, box, LineString, mapping
from shapely.ops import transform as shp_transform
from shapely.prepared import prep
from pyproj import Transformer

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle

from .optimizer_core import run_sensor_optimization  # (summary_df, best_row, centroid_data)
from .gee_soil_moisture import predict_sm_on_grid


import base64
import json
import tempfile
import time
import uuid
from pathlib import Path

from django.http import Http404
from django.shortcuts import render, redirect
from django.urls import reverse

from django.contrib import messages


from django.http import JsonResponse
from django.views.decorators.http import require_POST
from django.views.decorators.csrf import csrf_protect

from .forms import FeedbackForm


RUNS_DIR = Path(tempfile.gettempdir()) / "smso_runs"
CV_THRESHOLD_PERCENT = 10.0


def theoretical_sensors_per_ha(cell_size_m: int | float | None) -> float | None:
    try:
        cell_size = float(cell_size_m)
    except Exception:
        return None
    if cell_size <= 0:
        return None
    return 10000.0 / (cell_size * cell_size)


def selection_rule_label(best_row: dict | None) -> str:
    if not isinstance(best_row, dict):
        return ""
    try:
        cvp = float(best_row.get("cv_percent"))
    except Exception:
        return ""
    if cvp < CV_THRESHOLD_PERCENT:
        return "fewest sensors with CV below 10% and at least 2 sensors"
    return "minimum CV fallback with at least 2 sensors"

def _cleanup_old_runs(max_age_seconds=6 * 3600):
    """Delete old runs from /tmp to avoid unlimited growth."""
    try:
        if not RUNS_DIR.exists():
            return
        now = time.time()
        for p in RUNS_DIR.iterdir():
            if p.is_dir() and (now - p.stat().st_mtime) > max_age_seconds:
                for f in p.glob("*"):
                    try:
                        f.unlink()
                    except Exception:
                        pass
                try:
                    p.rmdir()
                except Exception:
                    pass
    except Exception:
        pass

def _save_run_payload(ctx: dict) -> str:
    _cleanup_old_runs()
    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    run_id = uuid.uuid4().hex
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save plot as png (avoid huge JSON/session)
    plot_b64 = (ctx.get("plot_png_base64") or "").strip()
    if plot_b64:
        (run_dir / "plot.png").write_bytes(base64.b64decode(plot_b64))
    ctx2 = dict(ctx)
    ctx2.pop("plot_png_base64", None)

    # Save everything else as JSON (GeoJSON dicts are fine)
    (run_dir / "data.json").write_text(json.dumps(ctx2, default=str), encoding="utf-8")
    return run_id

def _load_run_payload(run_id: str) -> dict:
    run_dir = RUNS_DIR / run_id
    data_path = run_dir / "data.json"
    if not data_path.exists():
        raise Http404("Result not found (expired). Please run optimization again.")

    ctx = json.loads(data_path.read_text(encoding="utf-8"))
    plot_path = run_dir / "plot.png"
    if plot_path.exists():
        ctx["plot_png_base64"] = base64.b64encode(plot_path.read_bytes()).decode("utf-8")
    else:
        ctx["plot_png_base64"] = ""
    ctx["run_id"] = run_id
    return ctx

def results_view(request, run_id: str):
    ctx = _load_run_payload(run_id)
    best_row = ctx.get("best_row") or {}
    if "selection_rule_used" not in ctx:
        ctx["selection_rule_used"] = selection_rule_label(best_row)
    # Hydrate session so POST-only endpoints (map/report) work after refresh/share.
    request.session["last_run_id"] = run_id
    request.session["preview_10m_run_id"] = None
    request.session["preview_10m_centroid_data"] = None
    for k in (
        "date_target",
        "summary_rows",
        "summary_records",
        "best_row",
        "centroid_data",
        "aoi_geojson",
        "area_m2",
        "area_ha",
    ):
        if k in ctx:
            request.session[k] = ctx[k]
    return render(request, "sensor_optimizer/results.html", ctx)




# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def compute_aoi_area(geojson_path: str) -> tuple[float, float]:
    """
    Returns (area_m2, area_ha) for the AOI polygon.
    Assumes GeoJSON is EPSG:4326 lon/lat.
    """
    geom_ll = _read_first_geom_from_geojson(geojson_path)
    lon0, lat0 = float(geom_ll.centroid.x), float(geom_ll.centroid.y)
    epsg = _utm_epsg_from_lonlat(lon0, lat0)

    geom_m = _transform_geom_epsg(geom_ll, "EPSG:4326", f"EPSG:{epsg}")
    area_m2 = float(geom_m.area)
    area_ha = area_m2 / 10000.0
    return area_m2, area_ha


def fig_to_base64_png(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def safe_int(x, default=None):
    if x is None:
        return default
    m = re.search(r"\d+", str(x))
    return int(m.group(0)) if m else default


def _save_geojson_upload(uploaded_file) -> str:
    fd, path = tempfile.mkstemp(prefix="aoi_", suffix=".geojson")
    os.close(fd)
    with open(path, "wb") as f:
        for chunk in uploaded_file.chunks():
            f.write(chunk)
    return path


def _save_geojson_text(geojson_text: str) -> str:
    fd, path = tempfile.mkstemp(prefix="aoi_draw_", suffix=".geojson")
    os.close(fd)
    with open(path, "w", encoding="utf-8") as f:
        f.write(geojson_text)
    return path


def _save_geojson_dict(geojson_obj: dict) -> str:
    fd, path = tempfile.mkstemp(prefix="aoi_saved_", suffix=".geojson")
    os.close(fd)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(geojson_obj, f)
    return path


def _aoi_geojson_dict_from_path(geojson_path: str) -> dict:
    with open(geojson_path, "r", encoding="utf-8") as f:
        return json.load(f)


# -----------------------------------------------------------------------------
# Geo helpers
# -----------------------------------------------------------------------------

def _read_first_geom_from_geojson(geojson_path: str):
    """Read first Polygon/MultiPolygon geometry from a GeoJSON file."""
    with open(geojson_path, "r", encoding="utf-8") as f:
        gj = json.load(f)

    geom = None

    if gj.get("type") == "FeatureCollection":
        for ft in (gj.get("features") or []):
            g = (ft or {}).get("geometry")
            if g and g.get("type") in ("Polygon", "MultiPolygon"):
                geom = g
                break

    elif gj.get("type") == "Feature":
        g = gj.get("geometry")
        if g and g.get("type") in ("Polygon", "MultiPolygon"):
            geom = g

    elif gj.get("type") in ("Polygon", "MultiPolygon"):
        geom = gj

    if geom is None:
        raise ValueError("GeoJSON must contain a Polygon or MultiPolygon geometry.")

    return shape(geom)  # shapely geometry in lon/lat


def _utm_epsg_from_lonlat(lon: float, lat: float) -> int:
    zone = int((lon + 180) // 6) + 1
    return (32600 + zone) if lat >= 0 else (32700 + zone)


def _transform_geom_epsg(geom, src_epsg: str, dst_epsg: str):
    tr = Transformer.from_crs(src_epsg, dst_epsg, always_xy=True)
    return shp_transform(lambda x, y, z=None: tr.transform(x, y), geom)


# -----------------------------------------------------------------------------
# Auto-select grid sizes (under the hood)
# -----------------------------------------------------------------------------
def auto_cell_sizes_for_polygon(
    geojson_path: str,
    candidates: list[int] | None = None,
    min_sensors: int = 2,     # ✅ allow 50m for ~0.5 ha
    max_sensors: int = 800,
) -> list[int]:
    """
    10 m interval candidates. Filters to sizes that fit AOI and keep sensor count reasonable.
    """
    if candidates is None:
        candidates = list(range(10, 501, 10))  # ✅ 10 m interval: 10..500

    geom_ll = _read_first_geom_from_geojson(geojson_path)  # EPSG:4326
    lon0, lat0 = float(geom_ll.centroid.x), float(geom_ll.centroid.y)
    epsg = _utm_epsg_from_lonlat(lon0, lat0)

    aoi_m = _transform_geom_epsg(geom_ll, "EPSG:4326", f"EPSG:{epsg}")
    area_m2 = float(aoi_m.area)

    minx, miny, maxx, maxy = aoi_m.bounds
    width = float(maxx - minx)
    height = float(maxy - miny)
    min_dim = max(1.0, min(width, height))

    out: list[int] = []
    for cs in candidates:
        cs = int(cs)
        if cs <= 0:
            continue

        # must fit the field scale (avoid cells larger than narrowest dimension)
        if cs > min_dim:
            continue

        est_n = area_m2 / float(cs * cs)

        if est_n < min_sensors or est_n > max_sensors:
            continue

        out.append(cs)

    # make sure 50m is included when it fits and gives >= 2 cells
    if 50 <= min_dim:
        est_50 = area_m2 / 2500.0
        if est_50 >= min_sensors and 50 not in out:
            out.append(50)

    return sorted(set(out))
# -----------------------------------------------------------------------------
# Candidate grid overlays for results map (strict inside cells OR clipped gridlines)
# -----------------------------------------------------------------------------

def centroids_geojson_by_size(centroid_data: dict) -> dict:
    """
    centroid_data: dict[str(cell_size)] -> rows with lon/lat/sm_pred/...
    returns dict[str] -> FeatureCollection (Points)
    """
    out = {}
    for cs, rows in (centroid_data or {}).items():
        feats = []
        for i, r in enumerate(rows or [], start=1):
            lon = r.get("lon")
            lat = r.get("lat")
            if lon is None or lat is None:
                continue
            try:
                lon = float(lon)
                lat = float(lat)
            except Exception:
                continue

            feats.append(
                {
                    "type": "Feature",
                    "properties": {
                        "cell_size_m": safe_int(cs),
                        "sensor_id": safe_int(r.get("sensor_id"), default=i),
                        "sm_pred": r.get("sm_pred"),
                        "comp_date": r.get("comp_date"),
                        "time_diff_days": r.get("time_diff_days"),
                        "n_images": r.get("n_images"),
                    },
                    "geometry": {"type": "Point", "coordinates": [lon, lat]},
                }
            )
        out[str(cs)] = {"type": "FeatureCollection", "features": feats}
    return out


def representative_sensor_by_size(centroid_data: dict, aoi_geojson: dict | None) -> dict:
    """
    Compute the center of gravity of the sensor set itself for each cell size.
    """
    out = {}
    for cs, rows in (centroid_data or {}).items():
        valid_rows = []
        for i, r in enumerate(rows or [], start=1):
            try:
                lon = float(r.get("lon"))
                lat = float(r.get("lat"))
            except Exception:
                continue
            valid_rows.append((i, r, lon, lat))

        if not valid_rows:
            continue

        avg_lon = sum(item[2] for item in valid_rows) / len(valid_rows)
        avg_lat = sum(item[3] for item in valid_rows) / len(valid_rows)
        epsg = _utm_epsg_from_lonlat(avg_lon, avg_lat)
        to_m = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)

        xy_rows = []
        for i, r, lon, lat in valid_rows:
            x, y = to_m.transform(lon, lat)
            xy_rows.append((i, r, lon, lat, x, y))

        cog_x = sum(item[4] for item in xy_rows) / len(xy_rows)
        cog_y = sum(item[5] for item in xy_rows) / len(xy_rows)
        to_ll = Transformer.from_crs(f"EPSG:{epsg}", "EPSG:4326", always_xy=True)
        cog_lon, cog_lat = to_ll.transform(cog_x, cog_y)

        nearest_sensor_id = None
        nearest_sensor_sm = None
        nearest_d2 = None
        for i, r, lon, lat, x, y in xy_rows:
            d2 = (x - cog_x) ** 2 + (y - cog_y) ** 2
            if nearest_d2 is None or d2 < nearest_d2:
                nearest_d2 = d2
                nearest_sensor_id = safe_int(r.get("sensor_id"), default=i)
                nearest_sensor_sm = r.get("sm_pred")

        out[str(cs)] = {
            "cell_size_m": safe_int(cs),
            "lon": cog_lon,
            "lat": cog_lat,
            "cog_lon": cog_lon,
            "cog_lat": cog_lat,
            "nearest_sensor_id": nearest_sensor_id,
            "nearest_sensor_sm": nearest_sensor_sm,
            "distance_to_nearest_sensor_m": float(nearest_d2 ** 0.5) if nearest_d2 is not None else None,
            "method": "sensor_set_center_of_gravity",
        }

    return out


def grid_overlays_geojson_by_size(
    geojson_path: str,
    cell_sizes: list[int],
    max_cells: int = 1500,
) -> dict:
    """
    For each cell size:
      - If estimated grid cell count <= max_cells: return square cells FULLY inside AOI
      - Else: return grid lines clipped to AOI (fast)

    Output is EPSG:4326 GeoJSON so Leaflet can display it.
    """
    geom_ll = _read_first_geom_from_geojson(geojson_path)
    lon0, lat0 = float(geom_ll.centroid.x), float(geom_ll.centroid.y)
    epsg = _utm_epsg_from_lonlat(lon0, lat0)

    aoi_m = _transform_geom_epsg(geom_ll, "EPSG:4326", f"EPSG:{epsg}")
    aoi_prep = prep(aoi_m)

    minx, miny, maxx, maxy = aoi_m.bounds

    to_ll = Transformer.from_crs(f"EPSG:{epsg}", "EPSG:4326", always_xy=True)

    def _to_ll_geom(g):
        return shp_transform(lambda x, y, z=None: to_ll.transform(x, y), g)

    out = {}

    for cs in cell_sizes:
        cs = int(cs)
        if cs <= 0:
            continue

        x0 = math.floor(minx / cs) * cs
        y0 = math.floor(miny / cs) * cs
        nx = int(math.ceil((maxx - x0) / cs))
        ny = int(math.ceil((maxy - y0) / cs))
        est_cells = nx * ny

        feats = []

        if est_cells <= max_cells:
            # Full square cells strictly inside AOI
            for ix in range(nx):
                x = x0 + ix * cs
                for iy in range(ny):
                    y = y0 + iy * cs
                    cell = box(x, y, x + cs, y + cs)
                    if not aoi_prep.contains(cell):
                        continue
                    cell_ll = _to_ll_geom(cell)
                    feats.append(
                        {
                            "type": "Feature",
                            "properties": {"cell_size_m": cs, "kind": "cell"},
                            "geometry": mapping(cell_ll),
                        }
                    )
        else:
            # Clipped grid lines
            for ix in range(nx + 1):
                x = x0 + ix * cs
                ln = LineString([(x, y0), (x, y0 + ny * cs)])
                clip = ln.intersection(aoi_m)
                if clip.is_empty:
                    continue
                clip_ll = _to_ll_geom(clip)
                feats.append(
                    {
                        "type": "Feature",
                        "properties": {"cell_size_m": cs, "kind": "gridline"},
                        "geometry": mapping(clip_ll),
                    }
                )

            for iy in range(ny + 1):
                y = y0 + iy * cs
                ln = LineString([(x0, y), (x0 + nx * cs, y)])
                clip = ln.intersection(aoi_m)
                if clip.is_empty:
                    continue
                clip_ll = _to_ll_geom(clip)
                feats.append(
                    {
                        "type": "Feature",
                        "properties": {"cell_size_m": cs, "kind": "gridline"},
                        "geometry": mapping(clip_ll),
                    }
                )

        out[str(cs)] = {"type": "FeatureCollection", "features": feats}

    return out


# -----------------------------------------------------------------------------
# Folium drawer map (header/body/script + map name) — robust
# -----------------------------------------------------------------------------

def make_drawer_map_html(center_lat: float = -23.0, center_lon: float = 30.0, zoom: int = 7):
    """
    Returns header/body/script/name for embedding folium into Django template.

    Robustness notes:
      - Do NOT embed <script> tags inside root.script
      - window[map_name] can temporarily refer to the DIV element (id-global),
        so we poll until it's a Leaflet map object.
    """
    fig = folium.Figure()
    m = folium.Map(location=[center_lat, center_lon], zoom_start=zoom, tiles=None, control_scale=True)
    m.add_to(fig)

    sat_layer = folium.TileLayer(
        tiles="https://services.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri, Maxar, Earthstar Geographics",
        name="Satellite",
        control=True,
        show=True,
    ).add_to(m)
    folium.TileLayer(
        tiles="https://{s}.basemaps.cartocdn.com/light_only_labels/{z}/{x}/{y}{r}.png",
        attr="&copy; OpenStreetMap contributors &copy; CARTO",
        name="Hybrid labels",
        overlay=True,
        control=False,
        show=True,
    ).add_to(m)
    osm_layer = folium.TileLayer("OpenStreetMap", name="OpenStreetMap", control=True, show=False).add_to(m)
    topo_layer = folium.TileLayer(
        tiles="https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png",
        attr="Map data: &copy; OpenStreetMap contributors, SRTM | Map style: &copy; OpenTopoMap (CC-BY-SA)",
        name="OpenTopoMap",
        control=True,
        show=False,
    ).add_to(m)
    light_layer = folium.TileLayer(
        tiles="https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png",
        attr="&copy; OpenStreetMap contributors &copy; CARTO",
        name="Carto Light",
        control=True,
        show=False,
    ).add_to(m)

    Draw(
        export=False,
        position="topleft",
        draw_options={
            "polyline": False,
            "rectangle": False,
            "circle": False,
            "circlemarker": False,
            "marker": False,
            "polygon": True,
        },
        edit_options={"edit": True, "remove": True},
    ).add_to(m)

    LocateControl(
        auto_start=True,
        position="topright",
        strings={"title": "My location"},
        flyTo=True,
        keepCurrentZoomLevel=False,
        drawCircle=True,
        drawMarker=True,
    ).add_to(m)

    folium.LayerControl().add_to(m)

    root = fig.get_root()
    map_js_name = m.get_name()
    osm_js_name = osm_layer.get_name()
    topo_js_name = topo_layer.get_name()
    light_js_name = light_layer.get_name()

    root.render()

    expose_js_raw = f"""
      (function() {{
        function isLeafletMap(x) {{
          return x && typeof x.on === "function" &&
                 typeof x.addLayer === "function" &&
                 typeof x.getCenter === "function";
        }}

        var tries = 0;
        var maxTries = 300;

        function tick() {{
          var cand = window["{map_js_name}"];
          if (!isLeafletMap(cand)) {{
            tries++;
            if (tries < maxTries) return setTimeout(tick, 50);
            return;
          }}

          var map = cand;
          window.__SENSOR_OPTIMIZER_MAP__ = map;

          var osm = window["{osm_js_name}"];
          var topo = window["{topo_js_name}"];
          var light = window["{light_js_name}"];
          if (osm && topo && light) {{
            map.on("tileerror", function() {{
              // If satellite tiles fail, keep standard basemaps available.
              try {{ map.addLayer(osm); }} catch(e) {{}}
            }});
          }}
        }}

        tick();
      }})();
    """
    root.script.add_child(folium.Element(expose_js_raw))

    header_html = root.header.render()
    body_html = root.html.render()
    script_html = root.script.render()

    return (
        mark_safe(header_html),
        mark_safe(body_html),
        mark_safe(script_html),
        map_js_name,
    )


# -----------------------------------------------------------------------------
# Plot helper
# -----------------------------------------------------------------------------

def build_cv_plot(summary_df: pd.DataFrame, best_row: dict | None):
    df = summary_df.copy()
    if df.empty:
        fig, ax = plt.subplots(figsize=(6.5, 3.8))
        ax.set_title("CV vs Number of Sensors")
        ax.set_xlabel("Number of sensors")
        ax.set_ylabel("CV (%)")
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center")
        return fig

    df["n_sensors"] = pd.to_numeric(df["n_sensors"], errors="coerce")
    df["cv_percent"] = pd.to_numeric(df["cv_percent"], errors="coerce")
    df["cell_size_m"] = df["cell_size_m"].apply(lambda v: safe_int(v, default=None))
    df = df.dropna(subset=["n_sensors", "cv_percent"]).sort_values("n_sensors")

    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    ax.plot(df["n_sensors"].values, df["cv_percent"].values, marker="o")
    ax.set_title("CV vs Number of Sensors")
    ax.set_xlabel("Number of sensors")
    ax.set_ylabel("CV (%)")
    ax.grid(True, alpha=0.25)
    ax.axhline(CV_THRESHOLD_PERCENT, color="#f39c12", linestyle="--", linewidth=1.2)

    for _, r in df.iterrows():
        cs = r.get("cell_size_m")
        if cs is not None:
            ax.annotate(
                f"{int(cs)}m",
                (r["n_sensors"], r["cv_percent"]),
                xytext=(6, 4),
                textcoords="offset points",
                fontsize=9,
            )

    if isinstance(best_row, dict) and best_row:
        bn = safe_int(best_row.get("n_sensors"))
        bc = best_row.get("cv_percent")
        try:
            bc = float(bc)
        except Exception:
            bc = None
        if bn is not None and bc is not None:
            ax.scatter([bn], [bc], marker="*", s=200)
            ax.annotate("chosen", (bn, bc), xytext=(8, -12), textcoords="offset points", fontsize=10)

    return fig


# -----------------------------------------------------------------------------
# FIXED Soil-moisture preview map (grid cells + red dots + compact legend + fit bounds)
# -----------------------------------------------------------------------------

def _prepare_map_df(df_coords: pd.DataFrame) -> pd.DataFrame:
    df = df_coords.copy()
    if not {"lon", "lat", "sm_pred"}.issubset(df.columns):
        raise ValueError("df_coords must contain lon, lat, sm_pred columns.")

    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["sm_pred"] = pd.to_numeric(df["sm_pred"], errors="coerce")
    df = df.dropna(subset=["lon", "lat", "sm_pred"]).copy()
    if df.empty:
        raise ValueError("No valid rows after cleaning (lon/lat/sm_pred).")
    return df


def _build_sm_map(
    surface_coords: pd.DataFrame,
    surface_cell_size_m: int,
    sensor_coords: pd.DataFrame | None = None,
    sensor_cell_size_m: int | None = None,
    aoi_geojson: dict | None = None,
):
    surface_df = _prepare_map_df(surface_coords)
    sensor_df = _prepare_map_df(sensor_coords) if sensor_coords is not None else surface_df.copy()

    center_lat = float(surface_df["lat"].mean())
    center_lon = float(surface_df["lon"].mean())

    m = folium.Map(location=[center_lat, center_lon], zoom_start=16, tiles=None, control_scale=True)

    folium.TileLayer(
        tiles="https://services.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri, Maxar, Earthstar Geographics",
        name="Satellite",
        control=True,
        show=False,
    ).add_to(m)
    folium.TileLayer(
        tiles="https://{s}.basemaps.cartocdn.com/light_only_labels/{z}/{x}/{y}{r}.png",
        attr="&copy; OpenStreetMap contributors &copy; CARTO",
        name="Hybrid labels",
        overlay=True,
        control=False,
        show=True,
    ).add_to(m)
    folium.TileLayer(
        "OpenStreetMap",
        name="OpenStreetMap",
        control=True,
        show=False,
    ).add_to(m)
    folium.TileLayer(
        tiles="https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png",
        attr="Map data: &copy; OpenStreetMap contributors, SRTM | Map style: &copy; OpenTopoMap (CC-BY-SA)",
        name="OpenTopoMap",
        control=True,
        show=False,
    ).add_to(m)
    folium.TileLayer(
        tiles="https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png",
        attr="&copy; OpenStreetMap contributors &copy; CARTO",
        name="Carto Light",
        control=True,
        show=True,
    ).add_to(m)

    sm_min = float(surface_df["sm_pred"].min())
    sm_max = float(surface_df["sm_pred"].max())
    if sm_min == sm_max:
        sm_min -= 0.1
        sm_max += 0.1

    # Dry -> wet ramp (red -> green) for clearer interpretation.
    colormap = LinearColormap(
        colors=["#b10026", "#f46d43", "#fee08b", "#a6d96a", "#1a9850"],
        vmin=sm_min,
        vmax=sm_max,
    )

    epsg = _utm_epsg_from_lonlat(center_lon, center_lat)
    to_m = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
    to_ll = Transformer.from_crs(f"EPSG:{epsg}", "EPSG:4326", always_xy=True)

    half = float(surface_cell_size_m) / 2.0

    xs, ys = [], []
    cell_features = []

    for _, r in surface_df.iterrows():
        lon = float(r["lon"])
        lat = float(r["lat"])
        sm = float(r["sm_pred"])

        x, y = to_m.transform(lon, lat)
        xs.append(x)
        ys.append(y)

        ring_m = [
            (x - half, y - half),
            (x + half, y - half),
            (x + half, y + half),
            (x - half, y + half),
            (x - half, y - half),
        ]
        ring_ll = [to_ll.transform(xx, yy) for (xx, yy) in ring_m]  # (lon, lat)

        cell_features.append(
            {
                "type": "Feature",
                "properties": {"sm_pred": sm},
                "geometry": {"type": "Polygon", "coordinates": [[list(pt) for pt in ring_ll]]},
            }
        )

    cells_fc = {"type": "FeatureCollection", "features": cell_features}

    surface_label = f"Soil moisture map ({int(surface_cell_size_m)} m)"
    grid_group = folium.FeatureGroup(name=surface_label, show=True)
    folium.GeoJson(
        cells_fc,
        style_function=lambda feat: {
            "fill": True,
            "fillColor": colormap(float(feat["properties"]["sm_pred"])),
            "color": "#475569",
            "weight": 0.15,
            "opacity": 0.18,
            "fillOpacity": 0.96,
        },
        tooltip=folium.GeoJsonTooltip(fields=["sm_pred"], aliases=["Predicted SM (%)"], localize=True),
    ).add_to(grid_group)
    grid_group.add_to(m)

    sensor_label = "Optimal soil moisture sensors"
    if sensor_cell_size_m:
        sensor_label = f"Optimal soil moisture sensors ({int(sensor_cell_size_m)} m layout)"
    sensors_group = folium.FeatureGroup(name=sensor_label, show=True)
    grid_group = folium.FeatureGroup(name="Optimal sensor grid", show=False)
    for _, r in sensor_df.iterrows():
        lon = float(r["lon"])
        lat = float(r["lat"])
        folium.CircleMarker(
            location=[lat, lon],
            radius=6,
            color="#ef4444",
            fill=True,
            fill_color="#ef4444",
            fill_opacity=0.95,
            weight=1,
        ).add_to(sensors_group)
        if sensor_cell_size_m:
            x, y = to_m.transform(lon, lat)
            grid_half = float(sensor_cell_size_m) / 2.0
            ring_m = [
                (x - grid_half, y - grid_half),
                (x + grid_half, y - grid_half),
                (x + grid_half, y + grid_half),
                (x - grid_half, y + grid_half),
                (x - grid_half, y - grid_half),
            ]
            ring_ll = [to_ll.transform(xx, yy) for (xx, yy) in ring_m]
            folium.Polygon(
                locations=[[pt[1], pt[0]] for pt in ring_ll],
                color="#38bdf8",
                weight=2,
                fill=False,
                opacity=0.95,
            ).add_to(grid_group)
    grid_group.add_to(m)
    sensors_group.add_to(m)

    # AOI outline + bounds to full extent
    bounds = None
    if aoi_geojson:
        try:
            aoi_layer = folium.GeoJson(
                aoi_geojson,
                name="AOI",
                style_function=lambda f: {"color": "#06b6d4", "weight": 2, "fillOpacity": 0.0},
            )
            aoi_layer.add_to(m)

            geom_ll = None
            if aoi_geojson.get("type") == "FeatureCollection":
                for ft in aoi_geojson.get("features", []):
                    g = (ft or {}).get("geometry")
                    if g and g.get("type") in ("Polygon", "MultiPolygon"):
                        geom_ll = shape(g)
                        break
            elif aoi_geojson.get("type") == "Feature":
                g = aoi_geojson.get("geometry")
                if g and g.get("type") in ("Polygon", "MultiPolygon"):
                    geom_ll = shape(g)
            elif aoi_geojson.get("type") in ("Polygon", "MultiPolygon"):
                geom_ll = shape(aoi_geojson)

            if geom_ll is not None:
                minx, miny, maxx, maxy = geom_ll.bounds  # lon/lat
                bounds = [[miny, minx], [maxy, maxx]]     # lat/lon
        except Exception:
            bounds = None

    # fallback: grid footprint
    if bounds is None:
        try:
            minx_m = min(xs) - half
            maxx_m = max(xs) + half
            miny_m = min(ys) - half
            maxy_m = max(ys) + half
            lon1, lat1 = to_ll.transform(minx_m, miny_m)
            lon2, lat2 = to_ll.transform(maxx_m, maxy_m)
            bounds = [[min(lat1, lat2), min(lon1, lon2)], [max(lat1, lat2), max(lon1, lon2)]]
        except Exception:
            bounds = None

    if bounds is not None:
        m.fit_bounds(bounds, padding=(30, 30))

    folium.LayerControl(collapsed=True).add_to(m)

    # Compact legend (fixed)
    n = 9
    stops = []
    for i in range(n):
        v = sm_min + (sm_max - sm_min) * (i / (n - 1))
        stops.append(f"{colormap(v)} {100*(i/(n-1)):.0f}%")
    gradient = ", ".join(stops)

    ticks = [round(sm_min + (sm_max - sm_min) * (i / 7), 1) for i in range(8)]
    ticks_desc = list(reversed(ticks))
    tick_rows_html = "".join([f"<span>{t}</span>" for t in ticks_desc])

    legend_html = f"""
    <style>
      #smso-legend {{
        position:absolute;
        top:50%;
        right:16px;
        transform:translateY(-50%);
        z-index:10001;
        background:rgba(255,255,255,0.97);
        padding:14px 14px;
        border-radius: 14px;
        border: 1px solid rgba(0,0,0,0.2);
        box-shadow: 0 8px 24px rgba(0,0,0,0.28);
        width: 112px;
        pointer-events:auto;
      }}
      #smso-legend .legend-wrap {{
        display:flex;
        align-items:stretch;
        gap:10px;
      }}
      #smso-legend .bar {{
        height: 260px;
        width: 18px;
        border-radius: 6px;
        border: 1px solid rgba(0,0,0,0.45);
        background: linear-gradient(to top, {gradient});
        flex: 0 0 auto;
      }}
      #smso-legend .ticks {{
        display:flex;
        flex-direction:column;
        justify-content:space-between;
        font-size: 13px;
        font-weight: 600;
        color:#111827;
        line-height: 1;
      }}
      #smso-legend .label {{
        margin-top: 10px;
        font-size: 14px;
        font-weight: 700;
        color:#0f172a;
        line-height: 1.25;
      }}
      .leaflet-control-layers-expanded {{
        font-size: 14px;
        min-width: 230px;
      }}
      .leaflet-control-layers label {{
        font-size: 14px;
        line-height: 1.45;
      }}
      @media (max-width: 900px) {{
        #smso-legend {{
          top: auto;
          bottom: 16px;
          right: 12px;
          transform: none;
          width: 96px;
          padding: 10px 10px;
        }}
        #smso-legend .bar {{
          height: 180px;
          width: 16px;
        }}
        #smso-legend .ticks {{
          font-size: 11px;
        }}
        #smso-legend .label {{
          font-size: 12px;
        }}
      }}
      @media (max-width: 640px) {{
        #smso-legend {{
          bottom: 12px;
          right: 10px;
          width: 88px;
          padding: 8px 8px;
        }}
        #smso-legend .bar {{
          height: 140px;
          width: 14px;
        }}
        #smso-legend .ticks {{
          font-size: 10px;
        }}
        #smso-features-legend {{
          left: 10px;
          bottom: 10px;
          padding: 10px 12px;
          font-size: 12px;
          max-width: min(240px, calc(100% - 120px));
        }}
      }}
    </style>
    <div id="smso-legend">
      <div class="legend-wrap">
        <div class="bar"></div>
        <div class="ticks">{tick_rows_html}</div>
      </div>
      <div class="label">Predicted soil moisture (%)</div>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    features_legend_html = """
    <div
      id="smso-features-legend"
      style="
        position:absolute;
        bottom:16px;
        left:16px;
        z-index:10010;
        background:rgba(255,255,255,0.96);
        color:#111827;
        border:1px solid rgba(0,0,0,0.25);
        border-radius:14px;
        padding:14px 16px;
        font-size:14px;
        line-height:1.45;
        box-shadow:0 8px 22px rgba(0,0,0,0.24);
        max-width:min(360px, calc(100% - 32px));
        pointer-events:auto;
      "
    >
      <div style="font-weight:700;font-size:15px;margin-bottom:8px;">Map features</div>
      <div style="display:flex;align-items:center;gap:10px;">
        <span
          style="
            width:12px;
            height:12px;
            border-radius:50%;
            background:#ef4444;
            border:1px solid rgba(0,0,0,0.35);
            flex:0 0 auto;
            display:inline-block;
          "
        ></span>
        <span>__SENSOR_LABEL__</span>
      </div>
      <div style="display:flex;align-items:center;gap:10px;margin-top:8px;">
        <span
          style="
            width:14px;
            height:14px;
            background:transparent;
            border:2px solid #38bdf8;
            border-radius:3px;
            flex:0 0 auto;
            display:inline-block;
          "
        ></span>
        <span>Optimal sensor grid</span>
      </div>
    </div>
    """
    features_legend_html = features_legend_html.replace("__SENSOR_LABEL__", str(sensor_label))
    m.get_root().html.add_child(folium.Element(features_legend_html))

    return m


# -----------------------------------------------------------------------------
# Views
# -----------------------------------------------------------------------------

from datetime import date
import json
import pandas as pd

# ... your other imports + helpers stay the same ...

def sensor_optimizer_view(request):
    """
    GET  -> show form + inline folium drawer map
    POST -> run optimization using upload OR drawn polygon
           grid sizes are auto-selected under the hood based on AOI polygon
    """
    drawer_map_header, drawer_map_body, drawer_map_script, drawer_map_name = make_drawer_map_html()

    if request.method == "GET":
        return render(
            request,
            "sensor_optimizer/form.html",
            {
                "drawer_map_header": drawer_map_header,
                "drawer_map_body": drawer_map_body,
                "drawer_map_script": drawer_map_script,
                "drawer_map_name": drawer_map_name,
                # ✅ FIX: you had date_target twice (today then ""), so it always became ""
                "date_target": date.today().isoformat(),
                "cell_sizes": "AUTO",
            },
        )

    # POST
    date_target = (request.POST.get("date_target") or "").strip()
    if not date_target:
        return render(
            request,
            "sensor_optimizer/form.html",
            {
                "drawer_map_header": drawer_map_header,
                "drawer_map_body": drawer_map_body,
                "drawer_map_script": drawer_map_script,
                "drawer_map_name": drawer_map_name,
                "error": "Please provide a target date.",
                "date_target": date_target,
                "cell_sizes": "AUTO",
            },
        )

    # AOI input: upload OR drawn polygon
    geojson_tmp_path = None

    uploaded = request.FILES.get("geojson_file")
    if uploaded and getattr(uploaded, "size", 0) > 0:
        try:
            geojson_tmp_path = _save_geojson_upload(uploaded)
        except Exception as e:
            return render(
                request,
                "sensor_optimizer/form.html",
                {
                    "drawer_map_header": drawer_map_header,
                    "drawer_map_body": drawer_map_body,
                    "drawer_map_script": drawer_map_script,
                    "drawer_map_name": drawer_map_name,
                    "error": f"Could not save uploaded GeoJSON: {e}",
                    "date_target": date_target,
                    "cell_sizes": "AUTO",
                },
            )
    else:
        drawn_geojson_str = (request.POST.get("drawn_geojson") or "").strip()
        if drawn_geojson_str:
            try:
                json.loads(drawn_geojson_str)
                geojson_tmp_path = _save_geojson_text(drawn_geojson_str)
            except Exception as e:
                return render(
                    request,
                    "sensor_optimizer/form.html",
                    {
                        "drawer_map_header": drawer_map_header,
                        "drawer_map_body": drawer_map_body,
                        "drawer_map_script": drawer_map_script,
                        "drawer_map_name": drawer_map_name,
                        "error": f"Could not parse drawn polygon GeoJSON: {e}",
                        "date_target": date_target,
                        "cell_sizes": "AUTO",
                    },
                )

    if geojson_tmp_path is None:
        return render(
            request,
            "sensor_optimizer/form.html",
            {
                "drawer_map_header": drawer_map_header,
                "drawer_map_body": drawer_map_body,
                "drawer_map_script": drawer_map_script,
                "drawer_map_name": drawer_map_name,
                "error": "Please draw a polygon on the map OR upload a GeoJSON file.",
                "date_target": date_target,
                "cell_sizes": "AUTO",
            },
        )

    # ✅ NEW: enforce AOI area limits (min 100 m², max 30 ha)
    MIN_AREA_M2 = 100.0
    MAX_AREA_M2 = 30.0 * 10000.0  # 30 ha = 300,000 m²
    try:
        area_m2, area_ha = compute_aoi_area(geojson_tmp_path)
    except Exception as e:
        return render(
            request,
            "sensor_optimizer/form.html",
            {
                "drawer_map_header": drawer_map_header,
                "drawer_map_body": drawer_map_body,
                "drawer_map_script": drawer_map_script,
                "drawer_map_name": drawer_map_name,
                "error": f"Could not compute field area from polygon: {e}",
                "date_target": date_target,
                "cell_sizes": "AUTO",
            },
        )

    if area_m2 < MIN_AREA_M2 or area_m2 > MAX_AREA_M2:
        return render(
            request,
            "sensor_optimizer/form.html",
            {
                "drawer_map_header": drawer_map_header,
                "drawer_map_body": drawer_map_body,
                "drawer_map_script": drawer_map_script,
                "drawer_map_name": drawer_map_name,
                "error": (
                    "Field area must be between 100 m² (0.01 ha) and 30 ha. "
                    f"Your polygon is {area_m2:,.0f} m² ({area_ha:,.2f} ha)."
                ),
                "date_target": date_target,
                "cell_sizes": "AUTO",
            },
        )

    # Save AOI in session for preview map outline + fit bounds (keep your behavior)
    try:
        request.session["aoi_geojson"] = _aoi_geojson_dict_from_path(geojson_tmp_path)
    except Exception:
        request.session["aoi_geojson"] = None

    # store area early (useful even if later steps fail)
    request.session["aoi_area_m2"] = float(area_m2)
    request.session["aoi_area_ha"] = float(area_ha)

    # Auto cell sizes
    try:
        cell_sizes_list = auto_cell_sizes_for_polygon(geojson_tmp_path, min_sensors=2, max_sensors=800)
    except Exception as e:
        return render(
            request,
            "sensor_optimizer/form.html",
            {
                "drawer_map_header": drawer_map_header,
                "drawer_map_body": drawer_map_body,
                "drawer_map_script": drawer_map_script,
                "drawer_map_name": drawer_map_name,
                "error": f"Could not derive grid sizes from polygon: {e}",
                "date_target": date_target,
                "cell_sizes": "AUTO",
            },
        )

    if not cell_sizes_list:
        return render(
            request,
            "sensor_optimizer/form.html",
            {
                "drawer_map_header": drawer_map_header,
                "drawer_map_body": drawer_map_body,
                "drawer_map_script": drawer_map_script,
                "drawer_map_name": drawer_map_name,
                "error": "No valid grid sizes found for this polygon. Try a larger polygon.",
                "date_target": date_target,
                "cell_sizes": "AUTO",
            },
        )

    # Run optimization
    try:
        summary_df, best_row, centroid_data = run_sensor_optimization(
            date_target=date_target,
            geojson_path=geojson_tmp_path,
            cell_sizes=cell_sizes_list,
        )

        if not isinstance(summary_df, pd.DataFrame):
            summary_df = pd.DataFrame(summary_df)

        fig = build_cv_plot(summary_df, best_row)
        plot_png_base64 = fig_to_base64_png(fig)

        optimal_cell_size = safe_int(best_row.get("cell_size_m")) if best_row else None
        optimal_n = safe_int(best_row.get("n_sensors")) if best_row else None
        try:
            optimal_cv = float(best_row.get("cv_percent")) if best_row else None
        except Exception:
            optimal_cv = None

        summary_rows = []
        if not summary_df.empty:
            tmp = summary_df.copy()
            tmp["cell_size_m_int"] = tmp["cell_size_m"].apply(lambda v: safe_int(v, default=-1))
            tmp["is_optimal"] = tmp["cell_size_m_int"].apply(
                lambda v: (optimal_cell_size is not None and v == int(optimal_cell_size))
            )
            tmp = tmp.drop(columns=["cell_size_m_int"])
            tmp["n_sensors"] = pd.to_numeric(tmp["n_sensors"], errors="coerce")
            tmp = tmp.sort_values("n_sensors")
            summary_df = tmp.copy()
            summary_rows = tmp.to_dict(orient="records")

        cell_size_options = [{"value": int(v), "label": f"{int(v)} m"} for v in cell_sizes_list]

        rule_used = selection_rule_label(best_row)
        best_row_enriched = dict(best_row or {})
        best_row_enriched["rule_used"] = rule_used

        # Store for map preview + PDF (keep your endpoints working)
        request.session["date_target"] = date_target
        request.session["summary_records"] = summary_df.to_dict(orient="records")
        request.session["best_row"] = best_row_enriched
        request.session["centroid_data"] = centroid_data

        # Optional overlays for results
        aoi_geojson = request.session.get("aoi_geojson") or _aoi_geojson_dict_from_path(geojson_tmp_path)
        grid_geojson_by_size = grid_overlays_geojson_by_size(
            geojson_tmp_path,
            cell_sizes_list,
            max_cells=1500,
        )
        centroids_geojson = centroids_geojson_by_size(centroid_data)
        representative_by_size = representative_sensor_by_size(centroid_data, aoi_geojson)

        # ✅ area_m2/area_ha already computed and stored
        # request.session["aoi_area_m2"] = area_m2
        # request.session["aoi_area_ha"] = area_ha

        # ✅ KEY FIX: PRG (POST -> Redirect -> GET) to avoid "Resubmit the form?"
        ctx = {
            "date_target": date_target,
            "plot_png_base64": plot_png_base64,
            "summary_rows": summary_rows,
            "summary_records": summary_df.to_dict(orient="records"),
            "best_row": best_row_enriched,
            "centroid_data": centroid_data,
            "cell_size_options": cell_size_options,
            "optimal_cell_size": optimal_cell_size,
            "optimal_n": optimal_n,
            "optimal_cv": optimal_cv,
            "selection_rule_used": rule_used,
            "area_m2": area_m2,
            "area_ha": area_ha,
            "aoi_geojson": aoi_geojson,
            "grid_geojson_by_size": grid_geojson_by_size,
            "centroids_geojson_by_size": centroids_geojson,
            "representative_sensor_by_size": representative_by_size,
        }

        run_id = _save_run_payload(ctx)
        return redirect(reverse("sensor_optimizer:sensor_optimizer_results", kwargs={"run_id": run_id}))

    except Exception as e:
        return render(
            request,
            "sensor_optimizer/form.html",
            {
                "drawer_map_header": drawer_map_header,
                "drawer_map_body": drawer_map_body,
                "drawer_map_script": drawer_map_script,
                "drawer_map_name": drawer_map_name,
                "error": f"Error running optimization: {e}",
                "date_target": date_target,
                "cell_sizes": "AUTO",
            },
        )


def centroid_map_view(request):
    """
    GET or POST:
      - if download_report=1 -> return PDF
      - else uses cell_size_m -> show styled map preview
    """
    if request.method not in ("GET", "POST"):
        return redirect("sensor_optimizer:sensor_optimizer")

    params = request.POST if request.method == "POST" else request.GET

    if params.get("download_report") == "1":
        return download_layout_report_view(request)

    date_target = request.session.get("date_target")
    centroids_serial = request.session.get("centroid_data") or {}
    best_row = request.session.get("best_row") or {}
    aoi_geojson = request.session.get("aoi_geojson")
    if not date_target or not centroids_serial or not aoi_geojson:
        return HttpResponseBadRequest("Missing session data; please run optimization again.")

    optimal_cell_size = safe_int(best_row.get("cell_size_m"))
    if optimal_cell_size is None:
        return HttpResponseBadRequest("Missing optimal configuration; please run optimization again.")

    optimal_rows = centroids_serial.get(str(optimal_cell_size))
    if not optimal_rows:
        return HttpResponseBadRequest(
            f"No centroid data found for optimal cell size={optimal_cell_size}"
        )

    current_run_id = params.get("run_id") or request.session.get("last_run_id") or ""
    cached_preview_run_id = request.session.get("preview_10m_run_id")
    preview_rows = request.session.get("preview_10m_centroid_data")
    if cached_preview_run_id != current_run_id:
        preview_rows = None
    preview_resolution_m = 10
    surface_warning = ""

    if not preview_rows:
        geojson_tmp_path = None
        try:
            geojson_tmp_path = _save_geojson_dict(aoi_geojson)
            _cv_pct, preview_df, _geom = predict_sm_on_grid(
                date_target=date_target,
                plot_geojson_path=geojson_tmp_path,
                cell_size_m=preview_resolution_m,
            )
            preview_rows = preview_df.to_dict(orient="records")
            request.session["preview_10m_centroid_data"] = preview_rows
            request.session["preview_10m_run_id"] = current_run_id
        except Exception:
            preview_rows = optimal_rows
            preview_resolution_m = optimal_cell_size
            surface_warning = (
                "The 10 m soil-moisture surface could not be generated for this run, "
                f"so the map is showing the {optimal_cell_size} m optimization grid surface instead."
            )
            request.session["preview_10m_centroid_data"] = None
            request.session["preview_10m_run_id"] = None
        finally:
            if geojson_tmp_path and os.path.exists(geojson_tmp_path):
                try:
                    os.remove(geojson_tmp_path)
                except OSError:
                    pass

    surface_df = pd.DataFrame(preview_rows)
    sensor_df = pd.DataFrame(optimal_rows)
    m = _build_sm_map(
        surface_df,
        surface_cell_size_m=int(preview_resolution_m),
        sensor_coords=sensor_df,
        sensor_cell_size_m=int(optimal_cell_size),
        aoi_geojson=aoi_geojson,
    )
    map_html = m._repr_html_()

    sensor_df = sensor_df.copy()
    if "sensor_id" not in sensor_df.columns:
        sensor_df.insert(0, "sensor_id", range(1, len(sensor_df) + 1))

    return render(
        request,
        "sensor_optimizer/map.html",
        {
                "date_target": date_target,
                "cell_size": int(optimal_cell_size),
                "surface_cell_size": int(preview_resolution_m),
                "surface_warning": surface_warning,
                "cv_percent": best_row.get("cv_percent"),
                "map_html": map_html,
                "coords": sensor_df.to_dict(orient="records"),
                "run_id": current_run_id,
            },
    )


def download_layout_report_view(request):
    """
    PDF report (GET or POST):
      - chosen config
      - optimization summary table
      - coordinates table for chosen config
    """
    date_target = request.session.get("date_target")
    summary_records = request.session.get("summary_records")
    best_row = request.session.get("best_row")
    centroids_serial = request.session.get("centroid_data") or {}
    area_ha = request.session.get("aoi_area_ha")

    if not (date_target and summary_records and best_row and centroids_serial):
        run_id = (
            request.GET.get("run_id")
            or request.POST.get("run_id")
            or request.session.get("last_run_id")
        )
        if not run_id:
            return HttpResponseBadRequest("Missing parameters: run optimization first.")
        try:
            ctx = _load_run_payload(run_id)
        except Http404:
            return HttpResponseBadRequest("Result expired. Run optimization again.")

        date_target = ctx.get("date_target")
        summary_records = ctx.get("summary_records") or ctx.get("summary_rows")
        centroids_serial = ctx.get("centroid_data") or {}
        area_ha = ctx.get("area_ha", area_ha)
        best_row = None
        for r in (summary_records or []):
            if isinstance(r, dict) and r.get("is_optimal"):
                best_row = r
                break
        if best_row is None and summary_records:
            # Fallback to lowest CV row if optimal flag missing.
            try:
                best_row = min(
                    [r for r in summary_records if isinstance(r, dict)],
                    key=lambda r: float(r.get("cv_percent")),
                )
            except Exception:
                best_row = None

        if not (date_target and summary_records and best_row and centroids_serial):
            return HttpResponseBadRequest("Missing parameters: run optimization first.")

    summary_df = pd.DataFrame(summary_records)

    chosen_cell = safe_int(best_row.get("cell_size_m"))
    if chosen_cell is None:
        return HttpResponseBadRequest("Could not determine chosen cell size.")

    coords_rows = centroids_serial.get(str(chosen_cell), [])
    coords_df = pd.DataFrame(coords_rows)

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    elements = []

    elements.append(Paragraph("Soil Moisture Sensor Optimization Report", styles["Heading1"]))
    elements.append(Spacer(1, 0.15 * inch))
    elements.append(Paragraph(f"Target date: <b>{date_target}</b>", styles["Normal"]))

    n_sensors = safe_int(best_row.get("n_sensors"), default=0)
    cvp = best_row.get("cv_percent")
    try:
        cvp = float(cvp)
    except Exception:
        cvp = None
    rule_used = best_row.get("rule_used") or selection_rule_label(best_row)
    elements.append(
        Paragraph(
            (
                f"Chosen grid: <b>{int(chosen_cell)} m</b> | "
                f"Sensors: <b>{int(n_sensors)}</b> | "
                f"CV: <b>{cvp:.2f}%</b>"
            )
            if cvp is not None
            else (
                f"Chosen grid: <b>{int(chosen_cell)} m</b> | "
                f"Sensors: <b>{int(n_sensors)}</b>"
            ),
            styles["Normal"],
        )
    )
    elements.append(Spacer(1, 0.2 * inch))
    elements.append(Paragraph("Methodology", styles["Heading2"]))
    elements.append(
        Paragraph(
            (
                "A 10 m soil-moisture surface is estimated across the field polygon. "
                "Candidate grid sizes are then generated inside that polygon, and each retained grid cell is "
                "treated as a polygon-based soil-moisture unit for aggregation and comparison. The app computes the coefficient "
                f"of variation (CV) across those grid-cell values. The selected layout uses the fewest sensors with CV below {CV_THRESHOLD_PERCENT:.0f}% "
                "and at least 2 sensors; if no candidate satisfies that threshold, the app falls back "
                "to the minimum-CV layout."
            ),
            styles["Normal"],
        )
    )
    details = []
    if rule_used:
        details.append(f"Rule used: <b>{rule_used}</b>.")
    if details:
        elements.append(Paragraph(" ".join(details), styles["Normal"]))
        elements.append(Spacer(1, 0.15 * inch))

    # Summary table
    elements.append(Paragraph("Optimization summary", styles["Heading2"]))
    preferred_cols = [
        "cell_size_m",
        "n_sensors",
        "cv_percent",
    ]
    summary_show = summary_df[[c for c in preferred_cols if c in summary_df.columns]].copy()
    if summary_show.empty:
        summary_show = summary_df.copy()
    for col in ["theoretical_sensors_per_ha", "actual_sensors_per_ha", "cv_percent"]:
        if col in summary_show.columns:
            summary_show[col] = pd.to_numeric(summary_show[col], errors="coerce").round(2)
    tbl = Table([summary_show.columns.tolist()] + summary_show.values.tolist(), repeatRows=1)
    tbl.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e5e7eb")),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ]
        )
    )
    elements.append(tbl)
    elements.append(Spacer(1, 0.25 * inch))

    # Coordinates table
    elements.append(Paragraph("Sensor coordinates (chosen grid)", styles["Heading2"]))
    keep = [c for c in ["sensor_id", "lon", "lat", "sm_pred", "comp_date", "time_diff_days", "n_images"] if c in coords_df.columns]
    coords_show = coords_df[keep].copy() if keep else coords_df.copy()

    if "sensor_id" not in coords_show.columns:
        coords_show.insert(0, "sensor_id", range(1, len(coords_show) + 1))

    for c in ["lon", "lat", "sm_pred", "time_diff_days"]:
        if c in coords_show.columns:
            coords_show[c] = pd.to_numeric(coords_show[c], errors="coerce").round(4)

    tbl2 = Table([coords_show.columns.tolist()] + coords_show.values.tolist(), repeatRows=1)
    tbl2.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e5e7eb")),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
                ("FONTSIZE", (0, 0), (-1, -1), 7),
            ]
        )
    )
    elements.append(tbl2)

    doc.build(elements)
    buffer.seek(0)

    resp = HttpResponse(buffer.getvalue(), content_type="application/pdf")
    resp["Content-Disposition"] = f'attachment; filename="sensor_layout_report_{date_target}.pdf"'
    return resp



@require_POST
@csrf_protect
def feedback_ajax_view(request):
    form = FeedbackForm(request.POST)
    if form.is_valid():
        fb = form.save(commit=False)
        fb.page = (request.POST.get("page") or "")[:200]
        fb.user_agent = (request.META.get("HTTP_USER_AGENT") or "")[:300]
        fb.save()
        mailed = False

        to_email = (getattr(settings, "FEEDBACK_NOTIFY_EMAIL", "") or "").strip()
        if to_email:
            subject = f"New Feedback ({fb.rating}/5)"
            body = (
                f"Name: {fb.name or '-'}\n"
                f"Email: {fb.email or '-'}\n"
                f"Rating: {fb.rating}\n"
                f"Page: {fb.page or '-'}\n"
                f"Submitted: {fb.created_at}\n\n"
                f"Message:\n{fb.message or '-'}\n"
            )
            try:
                send_mail(
                    subject=subject,
                    message=body,
                    from_email=getattr(settings, "DEFAULT_FROM_EMAIL", None),
                    recipient_list=[to_email],
                    fail_silently=False,
                )
                mailed = True
            except Exception:
                mailed = False

        return JsonResponse({"ok": True, "mailed": mailed})

    return JsonResponse({"ok": False, "errors": form.errors}, status=400)
