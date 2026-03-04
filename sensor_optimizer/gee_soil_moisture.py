# sensor_optimizer/gee_soil_moisture.py
# ============================================================
# Earth Engine + ExtraTrees: Soil Moisture on Grid Cells
#   - Loads ExtraTrees model + features from HF Hub
#   - Builds regular grid over AOI GeoJSON
#   - For each grid size:
#       • Builds EE polygons
#       • Finds closest Sentinel-1 composite in time (±MAX_DAYS_DIFF)
#       • Samples mean VV/VH/angle + DEM elev/slope over each cell
#       • Predicts soil moisture with ExtraTrees
#       • Computes CV (%) across grid cells
# ============================================================

import os
import math
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import ee

import geopandas as gpd
from shapely.geometry import box

from huggingface_hub import hf_hub_download

# ------------------------------------------------------------
# Hugging Face model config
# ------------------------------------------------------------

HF_MODEL_REPO = os.environ.get(
    "HF_MODEL_REPO",
    "IWMIHQ/soil-moisture-sensor-optimizer-model",
)

HF_MODEL_FILE = os.environ.get(
    "HF_MODEL_FILE",
    "extratrees_s1_soil_moisture_points.pkl",
)

HF_FEATURES_FILE = os.environ.get(
    "HF_FEATURES_FILE",
    "extratrees_s1_soil_moisture_features.txt",
)


def load_model_and_features():
    """
    Download the ExtraTrees model + feature list from HF Hub, then load them.
    """
    model_path = hf_hub_download(
        repo_id=HF_MODEL_REPO,
        filename=HF_MODEL_FILE,
        repo_type="model",
    )
    features_path = hf_hub_download(
        repo_id=HF_MODEL_REPO,
        filename=HF_FEATURES_FILE,
        repo_type="model",
    )

    model = joblib.load(model_path)
    with open(features_path, "r") as f:
        feature_cols = [ln.strip() for ln in f.readlines() if ln.strip()]

    print(f"[MODEL] Loaded model from {HF_MODEL_REPO}/{HF_MODEL_FILE}")
    print(f"[MODEL] Loaded {len(feature_cols)} feature names.")
    return model, feature_cols


MODEL, FEATURE_COLS = load_model_and_features()

# ------------------------------------------------------------
# Earth Engine auth (service account via env vars)
# ------------------------------------------------------------

SA_EMAIL = os.environ.get("EE_SERVICE_ACCOUNT", "")
PROJECT_ID = os.environ.get("EE_PROJECT_ID", "")
EE_KEY_JSON = os.environ.get("EE_SERVICE_ACCOUNT_KEY")          # full JSON string (optional)
EE_KEY_FILE = os.environ.get("EE_SERVICE_ACCOUNT_KEY_FILE")     # file path (recommended)

_EE_READY = False

def init_earth_engine():
    global _EE_READY
    if _EE_READY:
        return

    if EE_KEY_FILE and os.path.exists(EE_KEY_FILE):
        key_path = EE_KEY_FILE
    elif EE_KEY_JSON:
        key_path = "/tmp/ee-service-account.json"
        if not os.path.exists(key_path):
            with open(key_path, "w", encoding="utf-8") as f:
                f.write(EE_KEY_JSON)
    else:
        raise RuntimeError(
            "EE credentials missing.\n"
            "Set either EE_SERVICE_ACCOUNT_KEY (full JSON string) OR "
            "EE_SERVICE_ACCOUNT_KEY_FILE (path to JSON)."
        )

    if not SA_EMAIL:
        raise RuntimeError("EE_SERVICE_ACCOUNT is not set.")
    if not PROJECT_ID:
        raise RuntimeError("EE_PROJECT_ID is not set.")

    creds = ee.ServiceAccountCredentials(SA_EMAIL, key_path)
    ee.Initialize(creds, project=PROJECT_ID)
    _EE_READY = True


# ------------------------------------------------------------
# Constants (no EE calls here)
# ------------------------------------------------------------

MAX_DAYS_DIFF = 6          # days for closest S1 composite (±)
STEP_DAYS = 6              # composite window
AOI_BUFFER_M = 15000       # buffer for S1 search (m)
SCALE = 20                 # sampling scale (m)

S1_ORBIT_PASS = None       # or "ASCENDING"/"DESCENDING"


# ------------------------------------------------------------
# DEM / terrain helpers (call only AFTER EE is init)
# ------------------------------------------------------------

def get_dem_and_slope():
    """
    Create DEM elevation and slope images.

    Must be called only AFTER init_earth_engine().
    """
    dem_coll = ee.ImageCollection("COPERNICUS/DEM/GLO30")
    dem = dem_coll.mosaic()
    dem_elev = dem.select("DEM").rename("elev")
    dem_slope = ee.Terrain.slope(dem).rename("slope")
    return dem_elev, dem_slope


# ------------------------------------------------------------
# Sentinel-1 helpers
# ------------------------------------------------------------

def get_s1_collection(aoi, orbit_pass=None):
    col = (
        ee.ImageCollection("COPERNICUS/S1_GRD")
        .filterBounds(aoi)
        .filter(ee.Filter.eq("instrumentMode", "IW"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
    )
    if orbit_pass:
        col = col.filter(ee.Filter.eq("orbitProperties_pass", orbit_pass))
    return col


def make_s1_composites(s1_col, start_date, end_date, step_days=6):
    """
    Build step-day median composites (VV, VH, angle) with mid-window time_start.
    """
    start = ee.Date(start_date)
    end = ee.Date(end_date)
    n = end.difference(start, "day").divide(step_days).ceil().int()

    empty = (
        ee.Image.constant([0, 0, 0])
        .rename(["VV", "VH", "angle"])
        .updateMask(ee.Image.constant(0))
    )

    def make_one(i):
        i = ee.Number(i)
        d0 = start.advance(i.multiply(step_days), "day")
        d1 = d0.advance(step_days, "day")
        win = s1_col.filterDate(d0, d1)

        comp = ee.Image(
            ee.Algorithms.If(
                win.size().gt(0),
                win.median().select(["VV", "VH", "angle"]),
                empty,
            )
        )

        mid = d0.advance(ee.Number(step_days).divide(2), "day")

        comp = comp.set(
            {
                "system:time_start": mid.millis(),
                "date": mid.format("YYYY-MM-dd"),
                "n_images": win.size(),
            }
        )
        return comp

    comps = ee.ImageCollection(ee.List.sequence(0, n.subtract(1)).map(make_one))
    comps = comps.filter(ee.Filter.gt("n_images", 0))
    return comps


# ------------------------------------------------------------
# FeatureCollection -> pandas
# ------------------------------------------------------------

def fc_to_pandas(fc, force_columns=None):
    d = fc.getInfo()
    rows = [f.get("properties", {}) for f in d.get("features", [])]
    df = pd.DataFrame(rows)
    print("[FC] rows:", len(df))
    print("[FC] columns:", df.columns.tolist())

    if force_columns:
        for c in force_columns:
            if c not in df.columns:
                df[c] = np.nan
                print(f"[FC] Added missing column '{c}' with NaNs.")
    return df


# ------------------------------------------------------------
# Join S1 composites (closest in time) and sample MEAN over cell
# ------------------------------------------------------------

def attach_s1_nearest_composite_closest_mean_over_cell(
    fc_cells,
    s1_comps,
    max_days_diff=6,
    dem_elev=None,
    dem_slope=None,
):
    """
    Join each grid cell to the *closest* S1 composite in time
    (past OR future), within ±max_days_diff days, and sample
    MEAN VV/VH/angle + DEM elev/slope over the polygon.
    """

    if dem_elev is None or dem_slope is None:
        raise ValueError("dem_elev and dem_slope must be provided.")

    # Add numeric time (ms) for feature date
    def add_t(f):
        return f.set("t", ee.Date(f.get("date")).millis())

    fc = fc_cells.map(add_t)

    max_diff_ms = max_days_diff * 24 * 60 * 60 * 1000

    # Symmetric time window: |t_feature - t_image| <= max_diff_ms
    diff_filter = ee.Filter.maxDifference(
        difference=max_diff_ms,
        leftField="t",
        rightField="system:time_start",
    )
    # NOTE: no "past-only" constraint → closest in either direction
    join_filter = diff_filter

    # saveBest picks the match with the smallest absolute time difference
    join = ee.Join.saveBest(matchKey="best_img", measureKey="time_diff")
    joined = ee.FeatureCollection(join.apply(fc, s1_comps, join_filter))

    matched = joined.filter(ee.Filter.notNull(["best_img"]))

    def sample_one(feat):
        img = ee.Image(feat.get("best_img"))
        full_img = img.addBands(dem_elev).addBands(dem_slope)

        vals = full_img.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=feat.geometry(),
            scale=SCALE,
            maxPixels=1e7,
            bestEffort=True,
        )

        return feat.set(
            {
                "VV": vals.get("VV"),
                "VH": vals.get("VH"),
                "angle": vals.get("angle"),
                "elev": vals.get("elev"),
                "slope": vals.get("slope"),
                "comp_date": img.get("date"),
                "time_diff_ms": feat.get("time_diff"),
                "n_images": img.get("n_images"),
            }
        )

    sampled = matched.map(sample_one)
    return sampled


# ------------------------------------------------------------
# Build grid (GeoPandas + UTM -> EE FeatureCollection)
# ------------------------------------------------------------

def build_plot_grid_centroids(date_str, plot_geojson_path, cell_size_m):
    """
    Build regular grid over AOI (in local UTM), clip to AOI, reproject to EPSG:4326,
    and return as EE FeatureCollection of cell polygons.

    Each feature has:
      - geometry: polygon
      - cell_id
      - lon / lat (centroid in EPSG:4326)
      - date
      - Sheet='plot_grid'
    """
    plot_geojson_path = Path(plot_geojson_path)
    if not plot_geojson_path.exists():
        raise FileNotFoundError(f"Plot GeoJSON not found at {plot_geojson_path}.")

    print(f"[GRID] Read AOI: {plot_geojson_path}")
    aoi = gpd.read_file(plot_geojson_path)

    if aoi.empty:
        raise RuntimeError("AOI file has no features.")

    aoi = aoi.dissolve().reset_index(drop=True)

    utm_crs = aoi.estimate_utm_crs()
    aoi_utm = aoi.to_crs(utm_crs)

    minx, miny, maxx, maxy = aoi_utm.total_bounds
    n_cols = math.ceil((maxx - minx) / cell_size_m)
    n_rows = math.ceil((maxy - miny) / cell_size_m)
    print(f"[GRID] rows={n_rows}, cols={n_cols}, cell={cell_size_m} m")

    grid_polys = []
    cell_ids = []
    for i in range(n_cols):
        x0 = minx + i * cell_size_m
        x1 = x0 + cell_size_m
        for j in range(n_rows):
            y0 = miny + j * cell_size_m
            y1 = y0 + cell_size_m
            grid_polys.append(box(x0, y0, x1, y1))
            cell_ids.append(f"{i:04d}_{j:04d}")

    grid = gpd.GeoDataFrame(
        {"cell_id": cell_ids, "geometry": grid_polys}, crs=utm_crs
    )

    print("[GRID] Clipping to AOI...")
    grid_clip = gpd.overlay(grid, aoi_utm, how="intersection")
    if grid_clip.empty:
        raise RuntimeError(
            f"Clipped grid is empty for cell_size_m={cell_size_m}. "
            "Try a larger cell size or check your AOI geometry."
        )

    aoi_4326 = aoi_utm.to_crs(epsg=4326)
    grid_clip_4326 = grid_clip.to_crs(epsg=4326)

    aoi_union = aoi_4326.geometry.unary_union
    aoi_geojson = aoi_union.__geo_interface__
    geom = ee.Geometry(aoi_geojson)

    features = []
    for _, row in grid_clip_4326.iterrows():
        poly = row.geometry
        if poly is None or poly.is_empty:
            continue

        c = poly.centroid
        lon = float(c.x)
        lat = float(c.y)

        feat = ee.Feature(
            ee.Geometry(poly.__geo_interface__),
            {
                "cell_id": str(row.get("cell_id", "")),
                "lon": lon,
                "lat": lat,
                "date": date_str,
                "Sheet": "plot_grid",
            },
        )
        features.append(feat)

    fc_cells = ee.FeatureCollection(features)
    print(f"[GRID] Built {len(features)} cells in EE FeatureCollection.")
    return fc_cells, geom


# ------------------------------------------------------------
# Main entry point – used by Django view
# ------------------------------------------------------------

def predict_sm_on_grid(
    date_target: str,
    plot_geojson_path: str,
    cell_size_m: int,
):
    """
    Core function used by Django.

    Parameters
    ----------
    date_target : str
        Target date (YYYY-MM-DD).
    plot_geojson_path : str
        Path to AOI GeoJSON file on disk.
    cell_size_m : int
        Grid cell size in metres.

    Returns
    -------
    cv_pct : float
        CV (%) of predicted soil moisture across grid cells.
    df_coords : pandas.DataFrame
        One row per grid cell with at least:
            ['sensor_id', 'lon', 'lat', 'sm_pred'].
    geom : ee.Geometry
        AOI geometry for map centring.
    """
    # Lazily initialise EE FIRST
    init_earth_engine()

    # Build grid of cells inside plot
    fc_cells, geom = build_plot_grid_centroids(
        date_target, plot_geojson_path, cell_size_m
    )
    n_cells = fc_cells.size().getInfo()
    print(f"[INFER] grid cells (cell={cell_size_m} m): {n_cells}")

    if n_cells == 0:
        raise RuntimeError(
            f"No grid cells inside plot for cell_size_m={cell_size_m}."
        )

    # DEM / slope images
    dem_elev, dem_slope = get_dem_and_slope()

    # Buffer AOI for S1 search
    aoi = geom.buffer(AOI_BUFFER_M)

    # Sentinel-1 collection
    s1 = get_s1_collection(aoi, S1_ORBIT_PASS)

    start_wide = (
        ee.Date(date_target)
        .advance(-MAX_DAYS_DIFF, "day")
        .format("YYYY-MM-dd")
        .getInfo()
    )
    end_wide = ee.Date(date_target).format("YYYY-MM-dd").getInfo()
    print(f"[INFER] S1 wide date range: {start_wide} → {end_wide}")

    s1_period = s1.filterDate(start_wide, end_wide)
    n_s1 = s1_period.size().getInfo()
    print("[INFER] S1 images in wide range:", n_s1)
    if n_s1 == 0:
        raise RuntimeError(
            f"No S1 images in period for cell_size_m={cell_size_m}. "
            "Try another date or expand range."
        )

    # Make step-day composites
    comps = make_s1_composites(s1_period, start_wide, end_wide, STEP_DAYS)
    n_comps = comps.size().getInfo()
    print("[INFER] composites (non-empty):", n_comps)
    if n_comps == 0:
        raise RuntimeError(
            f"No non-empty composites for cell_size_m={cell_size_m}."
        )

    # Attach closest composite (past OR future), MEAN over polygon
    fc_cells_s1 = attach_s1_nearest_composite_closest_mean_over_cell(
        fc_cells,
        comps,
        MAX_DAYS_DIFF,
        dem_elev=dem_elev,
        dem_slope=dem_slope,
    )
    n_cells_s1 = fc_cells_s1.size().getInfo()
    print(f"[INFER] cells with S1 match: {n_cells_s1}/{n_cells}")
    if n_cells_s1 == 0:
        raise RuntimeError(
            "No grid cells matched to a Sentinel-1 composite in the closest-date join."
        )

    # Download to pandas
    df = fc_to_pandas(
        fc_cells_s1,
        force_columns=["VV", "VH", "angle", "elev", "slope", "lon", "lat"],
    )
    if len(df) == 0:
        raise RuntimeError("Joined dataframe is empty (no rows).")

    # Basic numeric cleaning
    for col in ["VV", "VH", "angle"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["VV_VH_ratio"] = df["VV"] / df["VH"]
    df["VV_minus_VH"] = df["VV"] - df["VH"]
    df["VV_plus_VH"] = df["VV"] + df["VH"]
    df["VV_dB"] = 10.0 * np.log10(df["VV"] + 1e-6)
    df["VH_dB"] = 10.0 * np.log10(df["VH"] + 1e-6)

    if "time_diff_ms" in df.columns:
        df["time_diff_days"] = pd.to_numeric(
            df["time_diff_ms"], errors="coerce"
        ) / (1000.0 * 60.0 * 60.0 * 24.0)
    if "n_images" in df.columns:
        df["n_images"] = pd.to_numeric(df["n_images"], errors="coerce")

    for col in ["elev", "slope"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Ensure all model features exist / are numeric
    for col in FEATURE_COLS:
        if col not in df.columns:
            df[col] = np.nan
            print(f"[INFER] Added missing feature '{col}' with NaNs.")
        df[col] = pd.to_numeric(df[col], errors="coerce")
        med = df[col].median()
        df[col] = df[col].fillna(med)

    X = df[FEATURE_COLS].values
    if X.shape[0] == 0:
        raise RuntimeError("No samples available for prediction (X has 0 rows).")

    # Predict soil moisture
    df["sm_pred"] = MODEL.predict(X)

    # CV (%)
    mean_sm = df["sm_pred"].mean()
    std_sm = df["sm_pred"].std(ddof=1)
    cv_pct = (std_sm / mean_sm) * 100 if mean_sm != 0 else np.nan

    print(f"[INFER] date={date_target}, cell={cell_size_m} m")
    print(f"[INFER] mean={mean_sm:.2f}, std={std_sm:.2f}, CV={cv_pct:.2f} %")
    print(f"[INFER] N cells={len(df)}")

    # Minimal coordinates table for map / PDF
    df_coords = df.copy()
    df_coords["lon"] = pd.to_numeric(df_coords["lon"], errors="coerce")
    df_coords["lat"] = pd.to_numeric(df_coords["lat"], errors="coerce")
    df_coords["sm_pred"] = pd.to_numeric(df_coords["sm_pred"], errors="coerce")
    df_coords.insert(0, "sensor_id", np.arange(1, len(df_coords) + 1))

    return float(cv_pct), df_coords, geom