#!/usr/bin/env python3
"""Sentinel-1 soil moisture training pipeline for local/WSL execution."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split

try:
    import ee
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "Missing dependency 'earthengine-api'. Install required packages first, for example:\n"
        "  pip install earthengine-api pandas scikit-learn joblib openpyxl matplotlib"
    ) from exc


BASE_DIR = Path(__file__).resolve().parent

SA_EMAIL = "zolokiala@tethys-app-1.iam.gserviceaccount.com"
PROJECT_ID = "tethys-app-1"
SOIL_TEXTURE_ASSET = "projects/tethys-app-1/assets/soil_type_code_limpopo_30m"

DEFAULT_XLSX = BASE_DIR / "Sensor_moisture_plot15_coords_date_only.xlsx"
DEFAULT_KEY = Path(os.environ.get("EE_SERVICE_ACCOUNT_KEY", BASE_DIR / "tethys-app-1-acc3960d3dd6.json"))
OUTPUT_DIR = Path(os.environ.get("SOIL_MOISTURE_OUTPUT_DIR", BASE_DIR))

TEST_FRACTION = 0.2
RANDOM_STATE = 42
STEP_DAYS = 6
MAX_DAYS_DIFF = 6
S2_MAX_DAYS_DIFF = 10
SCALE = 20
AOI_BUFFER_M = 15000
S1_ORBIT_PASS = None

INSITU_CLEAN_CSV = OUTPUT_DIR / "insitu_points_clean.csv"
TRAIN_JOIN_CSV = OUTPUT_DIR / "calibration_train_points_s1.csv"
TEST_JOIN_CSV = OUTPUT_DIR / "calibration_test_points_s1.csv"
TEST_EVAL_CSV = OUTPUT_DIR / "calibration_test_predictions_points.csv"
MODEL_PATH = OUTPUT_DIR / "soil_moisture_points_model.pkl"
FEATURES_PATH = OUTPUT_DIR / "soil_moisture_points_features.txt"
PLOT_PATH = OUTPUT_DIR / "predicted_vs_observed_soil_moisture.png"

DEM_ELEV = None
DEM_SLOPE = None
SOIL_TEXTURE = None


def init_earth_engine(key_path: Path) -> None:
    from ee import ServiceAccountCredentials

    global DEM_ELEV, DEM_SLOPE, SOIL_TEXTURE

    if not key_path.exists():
        raise FileNotFoundError(
            f"Missing service account key at {key_path}. "
            "Set EE_SERVICE_ACCOUNT_KEY or pass --key-path."
        )

    credentials = ServiceAccountCredentials(SA_EMAIL, str(key_path))
    ee.Initialize(credentials, project=PROJECT_ID)
    print(f"EE initialized: {SA_EMAIL} | project={PROJECT_ID}")

    dem = ee.ImageCollection("COPERNICUS/DEM/GLO30").mosaic()
    DEM_ELEV = dem.select("DEM").rename("elev")
    DEM_SLOPE = ee.Terrain.slope(dem).rename("slope")
    print("DEM (GLO30) loaded.")

    soil_texture_image = ee.Image(SOIL_TEXTURE_ASSET)
    soil_texture_band = ee.String(soil_texture_image.bandNames().get(0))
    SOIL_TEXTURE = soil_texture_image.select([soil_texture_band], ["soil_texture"])
    print(f"Soil texture asset loaded: {SOIL_TEXTURE_ASSET}")


def load_and_clean_insitu_points(xlsx_path: Path) -> pd.DataFrame:
    if not xlsx_path.exists():
        raise FileNotFoundError(f"Missing Excel file at {xlsx_path}")

    xls = pd.ExcelFile(xlsx_path)
    raw = pd.concat(
        [pd.read_excel(xls, sheet_name=sheet).assign(Sheet=sheet) for sheet in xls.sheet_names],
        ignore_index=True,
    )

    date_col = next((c for c in raw.columns if str(c).strip().lower() == "date"), None)
    lat_col = next((c for c in raw.columns if str(c).strip().lower().startswith("lat")), None)
    lon_col = next(
        (c for c in raw.columns if str(c).strip().lower().startswith(("lon", "long"))),
        None,
    )

    moist_candidates = {
        "% moisture",
        "moisture_pct",
        "soilmoisture_pct",
        "soil_moisture_pct",
        "moisture",
        "sm",
        "sm_pct",
    }
    moist_col = next(
        (c for c in raw.columns if str(c).strip().lower() in moist_candidates),
        None,
    )

    print("Detected columns:", {"date": date_col, "lat": lat_col, "lon": lon_col, "sm": moist_col})
    if None in (date_col, lat_col, lon_col, moist_col):
        raise ValueError("Required columns not found: Date, lat, lon, moisture/sm")

    df = raw[[date_col, lat_col, lon_col, moist_col, "Sheet"]].copy()
    df.columns = ["date", "lat", "lon", "sm", "Sheet"]
    df["date"] = pd.to_datetime(df["date"], errors="coerce", dayfirst=True)
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    df["sm"] = pd.to_numeric(df["sm"], errors="coerce")
    df = df.dropna(subset=["date", "lat", "lon", "sm"]).copy()

    if not ((df["lat"].between(-90, 90)).all() and (df["lon"].between(-180, 180)).all()):
        raise ValueError("Latitude/longitude values are out of range after cleaning.")

    df["date"] = df["date"].dt.strftime("%Y-%m-%d")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(INSITU_CLEAN_CSV, index=False)
    print(f"Clean rows: {len(df)}")
    print(f"Saved: {INSITU_CLEAN_CSV}")
    return df


def build_aoi_from_points(df: pd.DataFrame, buffer_m: int = AOI_BUFFER_M):
    rect = ee.Geometry.Rectangle([df["lon"].min(), df["lat"].min(), df["lon"].max(), df["lat"].max()])
    return rect.buffer(buffer_m)


def get_s1_collection(aoi, orbit_pass: str | None = None):
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


def get_s2_collection(aoi):
    return (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(aoi)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 60))
    )


def make_s1_composites(s1_col, start_date: str, end_date: str, step_days: int = STEP_DAYS):
    start = ee.Date(start_date)
    end = ee.Date(end_date)
    n = end.difference(start, "day").divide(step_days).ceil().int()
    empty = ee.Image.constant([0, 0, 0]).rename(["VV", "VH", "angle"]).updateMask(ee.Image.constant(0))

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
        return comp.set(
            {
                "system:time_start": mid.millis(),
                "date": mid.format("YYYY-MM-dd"),
                "n_images": win.size(),
            }
        )

    return ee.ImageCollection(ee.List.sequence(0, n.subtract(1)).map(make_one)).filter(
        ee.Filter.gt("n_images", 0)
    )


def df_to_fc_points(df: pd.DataFrame):
    return ee.FeatureCollection(
        [
            ee.Feature(
                ee.Geometry.Point([float(row["lon"]), float(row["lat"])]),
                {"date": str(row["date"]), "sm": float(row["sm"]), "Sheet": str(row.get("Sheet", ""))},
            )
            for _, row in df.iterrows()
        ]
    )


def attach_s1_nearest_composite(fc_obs, s1_comps, max_days_diff: int = MAX_DAYS_DIFF, s2_col=None, s2_max_days: int = S2_MAX_DAYS_DIFF):
    if DEM_ELEV is None or DEM_SLOPE is None or SOIL_TEXTURE is None:
        raise RuntimeError("Static predictors not initialized.")

    def add_t(feature):
        return feature.set("t", ee.Date(feature.get("date")).millis())

    fc = fc_obs.map(add_t)
    joined = ee.FeatureCollection(
        ee.Join.saveBest(matchKey="best_img", measureKey="time_diff").apply(
            fc,
            s1_comps,
            ee.Filter.maxDifference(
                difference=max_days_diff * 24 * 60 * 60 * 1000,
                leftField="t",
                rightField="system:time_start",
            ),
        )
    )
    matched = joined.filter(ee.Filter.notNull(["best_img"]))

    empty_ndvi = ee.Image.constant(0).rename("s2_ndvi").updateMask(ee.Image.constant(0))

    def sample_one(feature):
        image = ee.Image(feature.get("best_img"))
        full_image = image.addBands(DEM_ELEV).addBands(DEM_SLOPE).addBands(SOIL_TEXTURE)
        if s2_col is not None:
            comp_date = ee.Date(image.get("date"))
            s2_window = s2_col.filterDate(
                comp_date.advance(-s2_max_days, "day"),
                comp_date.advance(s2_max_days, "day"),
            )
            s2_image = ee.Image(
                ee.Algorithms.If(
                    s2_window.size().gt(0),
                    s2_window.median().select(["B4", "B8"]),
                    ee.Image.constant([0, 0]).rename(["B4", "B8"]).updateMask(ee.Image.constant(0)),
                )
            )
            full_image = full_image.addBands(s2_image.normalizedDifference(["B8", "B4"]).rename("s2_ndvi"))
        else:
            full_image = full_image.addBands(empty_ndvi)

        vals = full_image.reduceRegion(
            reducer=ee.Reducer.first(),
            geometry=feature.geometry(),
            scale=SCALE,
            maxPixels=1e7,
        )
        return feature.set(
            {
                "VV": vals.get("VV"),
                "VH": vals.get("VH"),
                "angle": vals.get("angle"),
                "elev": vals.get("elev"),
                "slope": vals.get("slope"),
                "soil_texture": vals.get("soil_texture"),
                "s2_ndvi": vals.get("s2_ndvi"),
                "comp_date": image.get("date"),
                "time_diff_ms": feature.get("time_diff"),
                "n_images": image.get("n_images"),
            }
        )

    return ee.FeatureCollection(matched.map(sample_one))


def fc_to_pandas(fc, force_columns: list[str] | None = None) -> pd.DataFrame:
    rows = [feature.get("properties", {}) for feature in fc.getInfo().get("features", [])]
    df = pd.DataFrame(rows)
    if force_columns:
        for col in force_columns:
            if col not in df.columns:
                df[col] = np.nan
    print("Downloaded rows:", len(df))
    return df


def train_eval_save(df_train_s1: pd.DataFrame, df_test_s1: pd.DataFrame) -> None:
    required = ["VV", "VH", "angle", "sm"]
    df_train_s1 = df_train_s1.dropna(subset=required).copy()
    df_test_s1 = df_test_s1.dropna(subset=required).copy()
    if len(df_train_s1) == 0 or len(df_test_s1) == 0:
        raise RuntimeError("No valid sampled rows remain after dropping missing VV/VH/angle/sm.")

    for frame in (df_train_s1, df_test_s1):
        # Sentinel-1 GRD backscatter is already in dB. Derive linear-power features from it
        # instead of applying log10 again to negative dB values.
        frame["VV_lin"] = np.power(10.0, frame["VV"] / 10.0)
        frame["VH_lin"] = np.power(10.0, frame["VH"] / 10.0)
        frame["VV_VH_ratio"] = frame["VV_lin"] / (frame["VH_lin"] + 1e-9)
        frame["VV_minus_VH"] = frame["VV"] - frame["VH"]
        frame["VV_plus_VH"] = frame["VV"] + frame["VH"]
        frame["RVI"] = 4.0 * frame["VH_lin"] / (frame["VV_lin"] + frame["VH_lin"] + 1e-9)
        frame["lin_sum"] = frame["VV_lin"] + frame["VH_lin"]
        frame["lin_diff"] = frame["VV_lin"] - frame["VH_lin"]
        frame["angle_centered"] = frame["angle"] - frame["angle"].median()
        if "time_diff_ms" in frame.columns:
            frame["time_diff_days"] = frame["time_diff_ms"].astype(float) / (1000.0 * 60.0 * 60.0 * 24.0)
        if "n_images" in frame.columns:
            frame["n_images"] = frame["n_images"].astype(float)
        for col in ["elev", "slope", "s2_ndvi"]:
            if col in frame.columns:
                frame[col] = pd.to_numeric(frame[col], errors="coerce")
        if "soil_texture" in frame.columns:
            frame["soil_texture"] = pd.to_numeric(frame["soil_texture"], errors="coerce")
            frame["soil_texture"] = frame["soil_texture"].round().astype("Int64").astype("string")

    numeric_feature_cols = [
        col
        for col in [
            "VV",
            "VH",
            "VV_lin",
            "VH_lin",
            "angle",
            "angle_centered",
            "VV_VH_ratio",
            "VV_minus_VH",
            "VV_plus_VH",
            "RVI",
            "lin_sum",
            "lin_diff",
            "time_diff_days",
            "n_images",
            "elev",
            "slope",
            "s2_ndvi",
        ]
        if col in df_train_s1.columns
    ]

    for col in numeric_feature_cols:
        df_train_s1[col] = df_train_s1[col].astype(float)
        df_test_s1[col] = df_test_s1[col].astype(float)
        median = df_train_s1[col].median()
        df_train_s1[col] = df_train_s1[col].fillna(median)
        df_test_s1[col] = df_test_s1[col].fillna(median)

    feature_frames_train = [df_train_s1[numeric_feature_cols].copy()]
    feature_frames_test = [df_test_s1[numeric_feature_cols].copy()]

    categorical_feature_cols = []
    if "soil_texture" in df_train_s1.columns:
        df_train_s1["soil_texture"] = df_train_s1["soil_texture"].fillna("missing")
        df_test_s1["soil_texture"] = df_test_s1["soil_texture"].fillna("missing")
        soil_texture_train = pd.get_dummies(df_train_s1["soil_texture"], prefix="soil_texture", dtype=float)
        soil_texture_test = pd.get_dummies(df_test_s1["soil_texture"], prefix="soil_texture", dtype=float)
        soil_texture_test = soil_texture_test.reindex(columns=soil_texture_train.columns, fill_value=0.0)
        if soil_texture_train.shape[1] > 1:
            feature_frames_train.append(soil_texture_train)
            feature_frames_test.append(soil_texture_test)
            categorical_feature_cols = soil_texture_train.columns.tolist()

    Xtr_df = pd.concat(feature_frames_train, axis=1)
    Xte_df = pd.concat(feature_frames_test, axis=1)
    non_constant_cols = [col for col in Xtr_df.columns if Xtr_df[col].nunique(dropna=False) > 1]
    feature_cols = [col for col in (numeric_feature_cols + categorical_feature_cols) if col in non_constant_cols]
    Xtr_df = Xtr_df[feature_cols]
    Xte_df = Xte_df[feature_cols]

    Xtr = Xtr_df.values
    ytr = df_train_s1["sm"].values
    Xte = Xte_df.values
    yte = df_test_s1["sm"].values

    model_searches = [
        (
            "ExtraTrees",
            RandomizedSearchCV(
                estimator=ExtraTreesRegressor(random_state=RANDOM_STATE, n_jobs=-1),
                param_distributions={
                    "n_estimators": [300, 500, 800, 1200],
                    "max_depth": [None, 8, 12, 20, 30],
                    "min_samples_split": [2, 4, 6, 10],
                    "min_samples_leaf": [1, 2, 3, 4],
                    "max_features": ["sqrt", 0.5, 0.8, 1.0],
                    "bootstrap": [False, True],
                },
                n_iter=24,
                cv=5,
                scoring="neg_mean_squared_error",
                random_state=RANDOM_STATE,
                n_jobs=-1,
                verbose=0,
            ),
        ),
        (
            "RandomForest",
            RandomizedSearchCV(
                estimator=RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1),
                param_distributions={
                    "n_estimators": [300, 500, 800, 1200],
                    "max_depth": [None, 8, 12, 20, 30],
                    "min_samples_split": [2, 4, 6, 10],
                    "min_samples_leaf": [1, 2, 3, 4],
                    "max_features": ["sqrt", 0.5, 0.8, 1.0],
                    "bootstrap": [False, True],
                },
                n_iter=24,
                cv=5,
                scoring="neg_mean_squared_error",
                random_state=RANDOM_STATE,
                n_jobs=-1,
                verbose=0,
            ),
        ),
        (
            "HistGradientBoosting",
            RandomizedSearchCV(
                estimator=HistGradientBoostingRegressor(random_state=RANDOM_STATE),
                param_distributions={
                    "learning_rate": [0.01, 0.03, 0.05, 0.1],
                    "max_depth": [None, 3, 5, 8],
                    "max_leaf_nodes": [15, 31, 63],
                    "min_samples_leaf": [5, 10, 15, 20],
                    "l2_regularization": [0.0, 0.01, 0.1, 1.0],
                },
                n_iter=20,
                cv=5,
                scoring="neg_mean_squared_error",
                random_state=RANDOM_STATE,
                n_jobs=-1,
                verbose=0,
            ),
        ),
    ]

    best_name = None
    best_search = None
    best_cv_rmse = None
    for model_name, search in model_searches:
        search.fit(Xtr, ytr)
        cv_rmse = float(np.sqrt(-search.best_score_))
        print(f"{model_name} best CV RMSE = {cv_rmse:.3f}")
        if best_cv_rmse is None or cv_rmse < best_cv_rmse:
            best_name = model_name
            best_search = search
            best_cv_rmse = cv_rmse

    if best_search is None:
        raise RuntimeError("Model search did not produce a fitted estimator.")

    best_model = best_search.best_estimator_
    print(f"Selected model: {best_name}")
    print(f"Selected params: {best_search.best_params_}")
    ypred = best_model.predict(Xte)

    print("Test metrics:")
    print(f"R2   = {r2_score(yte, ypred):.3f}")
    print(f"RMSE = {np.sqrt(mean_squared_error(yte, ypred)):.3f}")
    print(f"MAE  = {mean_absolute_error(yte, ypred):.3f}")
    print(f"Bias = {float(np.mean(ypred - yte)):.3f}")

    out = df_test_s1.copy()
    out["sm_pred"] = ypred
    out["residual"] = out["sm_pred"] - out["sm"]
    out.to_csv(TEST_EVAL_CSV, index=False)
    joblib.dump(best_model, MODEL_PATH)
    FEATURES_PATH.write_text("\n".join(feature_cols), encoding="utf-8")
    print(f"Saved: {TEST_EVAL_CSV}")
    print(f"Saved: {MODEL_PATH}")
    print(f"Saved: {FEATURES_PATH}")


def plot_pred_vs_obs() -> None:
    if not TEST_EVAL_CSV.exists():
        return

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = pd.read_csv(TEST_EVAL_CSV).dropna(subset=["sm", "sm_pred"])
    if len(df) == 0:
        return

    y_true = df["sm"].values
    y_pred = df["sm_pred"].values
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    plt.figure(figsize=(6, 6))
    ax = plt.gca()
    ax.scatter(y_true, y_pred)
    mn = min(y_true.min(), y_pred.min())
    mx = max(y_true.max(), y_pred.max())
    ax.plot([mn, mx], [mn, mx])
    ax.set_xlabel("Observed soil moisture (sm)")
    ax.set_ylabel("Predicted soil moisture (sm_pred)")
    ax.set_title("Predicted vs Observed Soil Moisture\n(ExtraTrees tuned, S1 + NDVI)")
    ax.text(
        0.05,
        0.95,
        f"$R^2$ = {r2:.2f}\nRMSE = {rmse:.2f}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.7},
    )
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=150)
    plt.close()
    print(f"Saved: {PLOT_PATH}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the soil moisture pipeline in WSL/local Python.")
    parser.add_argument("--xlsx-path", type=Path, default=DEFAULT_XLSX, help="Input Excel workbook path.")
    parser.add_argument("--key-path", type=Path, default=DEFAULT_KEY, help="Earth Engine service-account JSON path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    init_earth_engine(args.key_path)

    df_all = load_and_clean_insitu_points(args.xlsx_path)
    df_train, df_test = train_test_split(df_all, test_size=TEST_FRACTION, random_state=RANDOM_STATE)

    aoi = build_aoi_from_points(df_all, AOI_BUFFER_M)
    s1 = get_s1_collection(aoi, S1_ORBIT_PASS)
    s2 = get_s2_collection(aoi)

    start_date = df_all["date"].min()
    end_date = df_all["date"].max()
    start_wide = ee.Date(start_date).advance(-MAX_DAYS_DIFF, "day").format("YYYY-MM-dd").getInfo()
    end_wide = ee.Date(end_date).advance(MAX_DAYS_DIFF, "day").format("YYYY-MM-dd").getInfo()
    print(f"Wide S1/S2 date range: {start_wide} to {end_wide}")

    s1_period = s1.filterDate(start_wide, end_wide)
    s2_period = s2.filterDate(start_wide, end_wide)
    n_s1 = s1_period.size().getInfo()
    n_s2 = s2_period.size().getInfo()
    print(f"S1 images in range: {n_s1}")
    print(f"S2 images in range: {n_s2}")
    if n_s1 == 0:
        raise RuntimeError("S1 image count is 0 in the requested range.")

    comps = make_s1_composites(s1_period, start_wide, end_wide, STEP_DAYS)
    if comps.size().getInfo() == 0:
        raise RuntimeError("No non-empty Sentinel-1 composites were created.")

    df_train_s1 = fc_to_pandas(
        attach_s1_nearest_composite(df_to_fc_points(df_train), comps, MAX_DAYS_DIFF, s2_col=s2_period),
        force_columns=["VV", "VH", "angle", "elev", "slope", "soil_texture", "s2_ndvi"],
    )
    df_test_s1 = fc_to_pandas(
        attach_s1_nearest_composite(df_to_fc_points(df_test), comps, MAX_DAYS_DIFF, s2_col=s2_period),
        force_columns=["VV", "VH", "angle", "elev", "slope", "soil_texture", "s2_ndvi"],
    )

    df_train_s1.to_csv(TRAIN_JOIN_CSV, index=False)
    df_test_s1.to_csv(TEST_JOIN_CSV, index=False)
    print(f"Saved: {TRAIN_JOIN_CSV}")
    print(f"Saved: {TEST_JOIN_CSV}")

    train_eval_save(df_train_s1, df_test_s1)
    plot_pred_vs_obs()


if __name__ == "__main__":
    main()
