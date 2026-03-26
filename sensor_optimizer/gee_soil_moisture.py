# sensor_optimizer/gee_soil_moisture.py
# ============================================================
# Earth Engine + trained model: Soil Moisture on Grid Cells
#   - Loads model + features (prefers local files; HF Hub fallback)
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
import json
from pathlib import Path
from shapely import affinity


import numpy as np
import pandas as pd
import joblib
import ee

import geopandas as gpd
from shapely.geometry import box

from huggingface_hub import hf_hub_download

# ------------------------------------------------------------
# Model config
# ------------------------------------------------------------

HF_MODEL_REPO = os.environ.get(
    "HF_MODEL_REPO",
    "IWMIHQ/soil-moisture-sensor-optimizer-model",
)

HF_MODEL_FILE = os.environ.get(
    "HF_MODEL_FILE",
    "soil_moisture_points_model.pkl",
)

HF_FEATURES_FILE = os.environ.get(
    "HF_FEATURES_FILE",
    "soil_moisture_points_features.txt",
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCAL_MODEL_PATH = Path(
    os.environ.get("SOIL_MOISTURE_MODEL_PATH", str(PROJECT_ROOT / "soil_moisture_points_model.pkl"))
)
LOCAL_FEATURES_PATH = Path(
    os.environ.get("SOIL_MOISTURE_FEATURES_PATH", str(PROJECT_ROOT / "soil_moisture_points_features.txt"))
)


def load_model_and_features():
    """
    Load model + feature list.
    Priority:
      1) Local files in project root (or env override paths)
      2) Hugging Face Hub files
    """
    model_path = LOCAL_MODEL_PATH
    features_path = LOCAL_FEATURES_PATH

    if not (model_path.exists() and features_path.exists()):
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

    print(f"[MODEL] Loaded model from {model_path}")
    print(f"[MODEL] Loaded {len(feature_cols)} feature names.")
    return model, feature_cols


MODEL, FEATURE_COLS = load_model_and_features()

# ------------------------------------------------------------
# Earth Engine auth (service account via env vars)
# ------------------------------------------------------------

SA_EMAIL = (
    os.environ.get("EE_SERVICE_ACCOUNT", "")
    or os.environ.get("EE_SERVICE_ACCOUNT_EMAIL", "")
)
PROJECT_ID = (
    os.environ.get("EE_PROJECT_ID", "")
    or os.environ.get("GOOGLE_CLOUD_PROJECT", "")
)
EE_KEY_JSON = os.environ.get("EE_SERVICE_ACCOUNT_KEY")          # full JSON string (optional)
EE_KEY_FILE = (
    os.environ.get("EE_SERVICE_ACCOUNT_KEY_FILE")                # preferred env name
    or os.environ.get("EE_SERVICE_ACCOUNT_FILE")                 # backward-compatible alias
    or str(PROJECT_ROOT / "tethys-app-1-acc3960d3dd6.json")      # local fallback
)

_EE_READY = False

def init_earth_engine():
    global _EE_READY
    if _EE_READY:
        return

    sa_email = SA_EMAIL
    project_id = PROJECT_ID
    key_meta = {}

    if EE_KEY_FILE and os.path.exists(EE_KEY_FILE):
        key_path = EE_KEY_FILE
        try:
            with open(key_path, "r", encoding="utf-8") as f:
                key_meta = json.load(f)
        except Exception:
            key_meta = {}
    elif EE_KEY_JSON:
        key_path = "/tmp/ee-service-account.json"
        if not os.path.exists(key_path):
            with open(key_path, "w", encoding="utf-8") as f:
                f.write(EE_KEY_JSON)
        try:
            key_meta = json.loads(EE_KEY_JSON)
        except Exception:
            key_meta = {}
    else:
        raise RuntimeError(
            "EE credentials missing.\n"
            "Set either EE_SERVICE_ACCOUNT_KEY (full JSON string) OR "
            "EE_SERVICE_ACCOUNT_KEY_FILE / EE_SERVICE_ACCOUNT_FILE (path to JSON)."
        )

    if not sa_email:
        sa_email = key_meta.get("client_email", "")
    if not project_id:
        project_id = key_meta.get("project_id", "")

    if not sa_email:
        raise RuntimeError("EE_SERVICE_ACCOUNT is not set.")
    if not project_id:
        raise RuntimeError("EE_PROJECT_ID is not set.")

    creds = ee.ServiceAccountCredentials(sa_email, key_path)
    ee.Initialize(creds, project=project_id)
    _EE_READY = True


# ------------------------------------------------------------
# Constants (no EE calls here)
# ------------------------------------------------------------

MAX_DAYS_DIFF = 6          # days for closest S1 composite (±)
STEP_DAYS = 6              # composite window
AOI_BUFFER_M = 15000       # buffer for S1 search (m)
SCALE = 20                 # sampling scale (m)
CV_MIN_PCT = 0.01          # keep CV strictly positive for reporting/selection
S2_NDVI_FALLBACK = float(os.environ.get("S2_NDVI_FALLBACK", "0.35"))

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


def build_plot_grid_centroids(
    date_str,
    plot_geojson_path,
    cell_size_m,
    min_overlap: float = 0.50,          # ✅ Option 1: overlap fraction threshold
    align_to_aoi_axis: bool = False,    # ✅ Option 2: rotate AOI, build grid, rotate back
):
    """
    Build grid cells over AOI and return EE FeatureCollection of *cell polygons*.

    Inclusion rule (Option 1):
      - Keep a cell only if its centroid is inside AOI
      - And overlap(cell ∩ AOI) / area(cell) >= min_overlap

    Alignment (Option 2):
      - If align_to_aoi_axis=True, rotate AOI to its main axis, build grid, rotate cells back.

    Returns:
      fc_cells : ee.FeatureCollection  (geometry is the *clipped* cell polygon = cell ∩ AOI)
      geom     : ee.Geometry (AOI union) for map centering / buffering
    """
    plot_geojson_path = Path(plot_geojson_path)
    if not plot_geojson_path.exists():
        raise FileNotFoundError(f"Plot GeoJSON not found at {plot_geojson_path}.")

    print(f"[GRID] Read AOI: {plot_geojson_path}")
    aoi = gpd.read_file(plot_geojson_path)
    if aoi.empty:
        raise RuntimeError("AOI file has no features.")

    # dissolve to one geometry
    aoi = aoi.dissolve().reset_index(drop=True)

    utm_crs = aoi.estimate_utm_crs()
    aoi_utm = aoi.to_crs(utm_crs)
    aoi_union = aoi_utm.geometry.unary_union

    # ---- Optional: align to AOI main axis (Option 2)
    angle_deg = 0.0
    origin_pt = (aoi_union.centroid.x, aoi_union.centroid.y)
    aoi_for_grid = aoi_union

    if align_to_aoi_axis:
        try:
            mrr = aoi_union.minimum_rotated_rectangle
            coords = list(mrr.exterior.coords)

            # find longest edge of the rotated rectangle -> grid axis
            best_len = -1.0
            best_angle = 0.0
            for i in range(len(coords) - 1):
                (x0, y0) = coords[i]
                (x1, y1) = coords[i + 1]
                dx, dy = (x1 - x0), (y1 - y0)
                L = (dx * dx + dy * dy) ** 0.5
                if L > best_len:
                    best_len = L
                    best_angle = math.degrees(math.atan2(dy, dx))

            angle_deg = best_angle
            # rotate AOI so its long axis becomes horizontal (x-axis)
            aoi_for_grid = affinity.rotate(aoi_union, -angle_deg, origin=origin_pt)
            print(f"[GRID] Align grid to AOI axis: rotate by {-angle_deg:.2f}°")
        except Exception as e:
            print(f"[GRID] Align-to-axis failed, continuing unrotated: {e}")
            angle_deg = 0.0
            aoi_for_grid = aoi_union

    # bounds in the grid-building space (rotated or not)
    minx, miny, maxx, maxy = aoi_for_grid.bounds
    n_cols = math.ceil((maxx - minx) / cell_size_m)
    n_rows = math.ceil((maxy - miny) / cell_size_m)
    print(f"[GRID] rows={n_rows}, cols={n_cols}, cell={cell_size_m} m")

    # build full cells in rotated space
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

    grid = gpd.GeoDataFrame({"cell_id": cell_ids, "geometry": grid_polys}, crs=utm_crs)

    # rotate cells back to original AOI orientation if needed
    if align_to_aoi_axis and abs(angle_deg) > 1e-9:
        grid["geometry"] = grid["geometry"].apply(lambda g: affinity.rotate(g, angle_deg, origin=origin_pt))

    # ---- Option 1 inclusion rule (centroid-in + overlap)
    cell_area = float(cell_size_m) * float(cell_size_m)
    clipped_centroid_min_frac = 0.40

    # First cheap filter: intersects AOI
    grid = grid[grid.intersects(aoi_union)].copy()
    if grid.empty:
        raise RuntimeError(
            f"Grid has no cells intersecting AOI for cell_size_m={cell_size_m}."
        )

    # centroid test
    grid["centroid"] = grid.geometry.centroid
    grid["centroid_in_aoi"] = grid["centroid"].within(aoi_union)

    # overlap fraction (cell ∩ AOI)
    inter_geom = grid.geometry.intersection(aoi_union)
    grid["overlap_area"] = inter_geom.area
    grid["overlap_frac"] = grid["overlap_area"] / cell_area

    # keep cells
    keep = grid["centroid_in_aoi"] & (grid["overlap_frac"] >= float(min_overlap))
    grid_keep = grid.loc[keep].copy()

    if grid_keep.empty:
        raise RuntimeError(
            f"No cells passed centroid/overlap rule for cell_size_m={cell_size_m}. "
            f"Try lower min_overlap (current={min_overlap})."
        )

    # Use clipped geometry for EE sampling (keeps sampling within AOI)
    grid_keep["geometry"] = grid_keep.geometry.intersection(aoi_union)

    # Boundary-crossing cells can use clipped-polygon centroid if clipped area is
    # at least 40% of the full cell area; otherwise keep full-cell centroid.
    grid_keep["clipped_centroid"] = grid_keep.geometry.centroid
    grid_keep["is_boundary_cell"] = grid_keep["overlap_frac"] < (1.0 - 1e-9)
    use_clipped_centroid = (
        grid_keep["is_boundary_cell"]
        & (grid_keep["overlap_frac"] >= clipped_centroid_min_frac)
    )
    grid_keep["centroid_selected"] = grid_keep["centroid"]
    grid_keep.loc[use_clipped_centroid, "centroid_selected"] = grid_keep.loc[
        use_clipped_centroid, "clipped_centroid"
    ]

    # Convert kept cells + centroids to EPSG:4326
    grid_keep_4326 = grid_keep.to_crs(epsg=4326)

    # AOI geometry for EE
    aoi_4326 = gpd.GeoSeries([aoi_union], crs=utm_crs).to_crs(epsg=4326).iloc[0]
    geom = ee.Geometry(aoi_4326.__geo_interface__)

    features = []
    for _, row in grid_keep_4326.iterrows():
        poly = row.geometry
        if poly is None or poly.is_empty:
            continue

        # Use centroid selected by overlap rule:
        # - boundary-crossing cells with overlap >= 40% => clipped centroid
        # - all other cells => full-cell centroid
        c_utm = row["centroid_selected"]
        c_wgs = gpd.GeoSeries([c_utm], crs=utm_crs).to_crs(epsg=4326).iloc[0]

        lon = float(c_wgs.x)
        lat = float(c_wgs.y)

        feat = ee.Feature(
            ee.Geometry(poly.__geo_interface__),
            {
                "cell_id": str(row.get("cell_id", "")),
                "lon": lon,
                "lat": lat,
                "date": date_str,
                "Sheet": "plot_grid",
                "overlap_frac": float(row.get("overlap_frac", 0.0)),
                "is_boundary_cell": bool(row.get("is_boundary_cell", False)),
            },
        )
        features.append(feat)

    fc_cells = ee.FeatureCollection(features)
    print(f"[GRID] Kept {len(features)} cells (min_overlap={min_overlap}, align={align_to_aoi_axis}).")
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
    # fc_cells, geom = build_plot_grid_centroids(
    #     date_target, plot_geojson_path, cell_size_m
    # )
    align_flag = os.environ.get("GRID_ALIGN_TO_FIELD", "0") == "1"      # Option 2
    min_overlap = float(os.environ.get("GRID_MIN_OVERLAP", "0.50"))     # Option 1

    fc_cells, geom = build_plot_grid_centroids(
        date_target,
        plot_geojson_path,
        cell_size_m,
        min_overlap=min_overlap,
        align_to_aoi_axis=align_flag,
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

    # Progressive fallback windows (days around target date) to reduce hard failures
    # in areas/dates with sparse Sentinel-1 coverage.
    extra_windows = os.environ.get("S1_FALLBACK_WINDOWS_DAYS", "12,18,24")
    fallback_windows = []
    for tok in extra_windows.split(","):
        tok = tok.strip()
        if tok:
            try:
                fallback_windows.append(int(tok))
            except Exception:
                pass
    window_days_candidates = [int(MAX_DAYS_DIFF)] + fallback_windows
    window_days_candidates = sorted({d for d in window_days_candidates if d > 0})

    s1_period = None
    n_s1 = 0
    selected_days = int(MAX_DAYS_DIFF)
    start_wide = None
    end_wide = None
    for days in window_days_candidates:
        start_wide = ee.Date(date_target).advance(-days, "day").format("YYYY-MM-dd").getInfo()
        end_wide = ee.Date(date_target).advance(days, "day").format("YYYY-MM-dd").getInfo()
        print(f"[INFER] S1 wide date range (±{days}d): {start_wide} → {end_wide}")
        s1_try = s1.filterDate(start_wide, end_wide)
        n_try = s1_try.size().getInfo()
        print(f"[INFER] S1 images in range (±{days}d): {n_try}")
        if n_try > 0:
            s1_period = s1_try
            n_s1 = n_try
            selected_days = days
            break

    if s1_period is None or n_s1 == 0:
        tried_txt = ", ".join([f"±{d}d" for d in window_days_candidates])
        raise RuntimeError(
            f"No S1 images found for cell_size_m={cell_size_m} after trying windows: {tried_txt}. "
            "Try another date or larger AOI."
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
        selected_days,
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

    vv_raw = df["VV"]
    vh_raw = df["VH"]
    vv_med = float(vv_raw.median(skipna=True)) if vv_raw.notna().any() else np.nan
    vh_med = float(vh_raw.median(skipna=True)) if vh_raw.notna().any() else np.nan
    input_is_db = (np.isfinite(vv_med) and vv_med < 0.0) or (np.isfinite(vh_med) and vh_med < 0.0)

    if input_is_db:
        vv_lin = np.power(10.0, vv_raw / 10.0)
        vh_lin = np.power(10.0, vh_raw / 10.0)
        df["VV_dB"] = vv_raw
        df["VH_dB"] = vh_raw
    else:
        vv_lin = vv_raw.clip(lower=0.0)
        vh_lin = vh_raw.clip(lower=0.0)
        df["VV_dB"] = 10.0 * np.log10(vv_lin + 1e-6)
        df["VH_dB"] = 10.0 * np.log10(vh_lin + 1e-6)

    # Engineered predictors expected by the trained model.
    df["VV_lin"] = vv_lin
    df["VH_lin"] = vh_lin
    df["lin_sum"] = vv_lin + vh_lin
    df["lin_diff"] = vv_lin - vh_lin
    df["VV_VH_ratio"] = vv_lin / (vh_lin + 1e-6)
    df["VV_minus_VH"] = vv_raw - vh_raw
    df["VV_plus_VH"] = vv_raw + vh_raw
    df["RVI"] = 4.0 * vh_lin / (vv_lin + vh_lin + 1e-6)

    angle_med = float(df["angle"].median(skipna=True)) if df["angle"].notna().any() else 0.0
    df["angle_centered"] = df["angle"] - angle_med
    if "s2_ndvi" not in df.columns:
        df["s2_ndvi"] = S2_NDVI_FALLBACK

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
    try:
        cv_pct = float(cv_pct)
    except Exception:
        cv_pct = np.nan
    if not np.isfinite(cv_pct) or cv_pct <= 0:
        cv_pct = float(CV_MIN_PCT)

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
