# sensor_optimizer/optimizer_core.py
from __future__ import annotations

import numpy as np
import pandas as pd

from .gee_soil_moisture import predict_sm_on_grid


CV_TOLERANCE = float(2.0)  # percentage points


def parse_cell_sizes(cell_sizes_str: str) -> list[int]:
    """
    Robust parser.
    Accepts: "5,10,20,30" or "5 10 20" or "5;10;20" or "5|10|20"
    Returns: [5,10,20,30]
    """
    if not cell_sizes_str or not str(cell_sizes_str).strip():
        raise ValueError("No valid grid cell sizes provided (e.g. 5,10,20,30).")

    # Split on ANY non-digit sequence
    parts = [p for p in __import__("re").split(r"[^\d]+", str(cell_sizes_str).strip()) if p]
    vals = [int(p) for p in parts if int(p) > 0]

    if not vals:
        raise ValueError("No valid grid cell sizes provided (e.g. 5,10,20,30).")

    return vals


def _normalize_cell_sizes(cell_sizes) -> list[int]:
    """
    Accepts either:
      - list[int] / tuple[int]
      - string like "5,10,20"
    Returns list[int]
    """
    if cell_sizes is None:
        raise ValueError("cell_sizes is required (e.g. [5,10,20] or '5,10,20').")

    # If it's already a list/tuple/np array of numbers
    if isinstance(cell_sizes, (list, tuple, np.ndarray)):
        vals = []
        for v in cell_sizes:
            if v is None:
                continue
            try:
                iv = int(v)
            except Exception:
                # handle things like "10 m"
                import re
                m = re.search(r"\d+", str(v))
                if not m:
                    continue
                iv = int(m.group(0))
            if iv > 0:
                vals.append(iv)
        if not vals:
            raise ValueError("No valid grid cell sizes provided (e.g. 5,10,20,30).")
        return vals

    # Otherwise treat as string
    return parse_cell_sizes(str(cell_sizes))


def run_sensor_optimization(date_target: str, geojson_path: str, cell_sizes):
    """
    cell_sizes can be:
      - list[int] e.g. [5,10,20]
      - str       e.g. "5,10,20"

    Returns:
      summary_df     : columns = cell_size_m, n_sensors, cv_percent
      best_row       : dict with chosen configuration using tolerance rule
      centroid_data  : dict[str(cell_size_m)] -> list-of-dicts rows (lon, lat, sm_pred, ...)
    """
    cell_sizes_list = _normalize_cell_sizes(cell_sizes)

    cvs = []
    n_sensors = []
    used_cell_sizes = []
    centroid_data = {}

    for cell_size in cell_sizes_list:
        cv_pct, df_coords, _geom = predict_sm_on_grid(
            date_target=date_target,
            plot_geojson_path=geojson_path,
            cell_size_m=int(cell_size),
        )

        # Ensure df_coords is serializable (session-safe): store list of dicts
        if isinstance(df_coords, pd.DataFrame):
            df_store = df_coords.to_dict(orient="records")
        else:
            # assume already list-of-dicts or similar
            df_store = df_coords

        cvs.append(float(cv_pct))
        n_sensors.append(int(len(df_store)))
        used_cell_sizes.append(int(cell_size))
        centroid_data[str(int(cell_size))] = df_store

    summary_df = (
        pd.DataFrame(
            {
                "cell_size_m": used_cell_sizes,
                "n_sensors": n_sensors,
                "cv_percent": cvs,
            }
        )
        .sort_values("n_sensors")
        .reset_index(drop=True)
    )

    min_cv = float(summary_df["cv_percent"].min())
    candidates = summary_df[summary_df["cv_percent"] <= (min_cv + CV_TOLERANCE)].copy()
    best = candidates.sort_values("n_sensors").iloc[0].to_dict()

    return summary_df, best, centroid_data