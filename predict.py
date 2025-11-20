import os
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.preprocessing import StandardScaler
import joblib

ARTIFACT_DIR = Path(os.environ.get("ARTIFACT_DIR", "data/artifacts"))
ARTIFACT_PATH = ARTIFACT_DIR / "gp_artifacts.joblib"

ZONES = [
    "AECO", "AEPAPT", "AEPIMP", "AEPKPT", "AEPOPT",
    "AP", "BC", "CE", "DAY", "DEOK", "DOM", "DPLCO", "DUQ",
    "EASTON", "EKPC", "JC", "ME", "OE", "OVEC", "PAPWR",
    "PE", "PEPCO", "PLCO", "PN", "PS", "RECO", "SMECO",
    "UGI", "VMEU",
]


# ================================
# Configuration
# ================================

METERED_DIR = Path(os.environ.get("METERED_DIR", "data"))
WEATHER_DIR = Path(os.environ.get("WEATHER_DIR", "data/weather_cities"))

TRAIN_START = date(2016, 1, 1)
TRAIN_END = date(2023, 12, 31)

FORECAST_YEAR = 2025
NOV_START = date(FORECAST_YEAR, 11, 1)
NOV_END = date(FORECAST_YEAR, 11, 29)

# ================================
# Basic utilities
# ================================

def thanksgiving_date(year: int) -> date:
    """US Thanksgiving (4th Thursday in November)."""
    d = date(year, 11, 1)
    w = d.weekday()  # Monday=0 ... Sunday=6
    days_to_thu = (3 - w) % 7
    first_thu = d + timedelta(days=days_to_thu)
    return first_thu + timedelta(weeks=3)


def load_metered_data(metered_dir: Path) -> pd.DataFrame:
    """
    Load all hrl_load_metered_*.csv into one DataFrame.

    We *always* treat the PJM 'load_area' column as the canonical zone name
    (AECO, AEPAPT, ...). Any existing 'zone' column in the raw files is ignored.

    Expected columns in each CSV:
      - datetime_beginning_ept
      - load_area
      - mw
    """
    csvs = sorted(metered_dir.glob("hrl_load_metered_*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No hrl_load_metered_*.csv found in {metered_dir}")

    frames = []
    for f in csvs:
        df = pd.read_csv(f)
        cols = df.columns

        # ---- Enforce use of load_area ----
        if "load_area" not in cols:
            raise ValueError(
                f"{f} does not have 'load_area' column. "
                "All metered files must provide load_area; "
                "do not rely on the short 'zone' codes (AE, AEP, ...)."
            )

        # If a short 'zone' column is present, drop it so we never use it
        if "zone" in cols:
            df = df.drop(columns=["zone"])

        # Rename load_area -> zone for internal consistency
        df = df.rename(columns={"load_area": "zone"})

        # Parse PJM timestamp
        if "timestamp" not in df.columns:
            df["timestamp"] = pd.to_datetime(
                df["datetime_beginning_ept"],
                format="%m/%d/%Y %I:%M:%S %p",
                errors="raise",
            )

        df["date"] = df["timestamp"].dt.date
        df["hour"] = df["timestamp"].dt.hour

        frames.append(df)

    out = pd.concat(frames, ignore_index=True)

    # Drop any duplicated column names that might still exist
    out = out.loc[:, ~out.columns.duplicated()]

    return out




def load_weather_data(weather_dir: Path) -> pd.DataFrame:
    combined_path = weather_dir / "weather_all_zones.csv"

    if combined_path.exists():
        csvs = [combined_path]
    else:
        csvs = sorted(weather_dir.glob("weather_*.csv"))

    if not csvs:
        raise FileNotFoundError(f"No weather CSVs found in {weather_dir}")

    frames = []

    for f in csvs:
        df = pd.read_csv(f)

        # ----- Build/standardize timestamp -----
        if "timestamp" not in df.columns:
            if {"date", "hour"}.issubset(df.columns):
                date_col = pd.to_datetime(df["date"])
                hour_col = pd.to_numeric(df["hour"], errors="coerce").fillna(0).astype(int)
                df["timestamp"] = date_col + pd.to_timedelta(hour_col, unit="h")
            else:
                raise ValueError(
                    f"{f} has no 'timestamp' and no ('date','hour') to construct it."
                )

        # 1) Parse with UTC to handle mixed offsets cleanly
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")

        # 2) Convert to America/New_York (PJM local time)
        df["timestamp"] = (
            df["timestamp"]
            .dt.tz_convert("America/New_York")
            .dt.tz_localize(None)   # make it tz-naive to match load data
        )

        # ----- Standardize date -----
        df["date"] = df["timestamp"].dt.date

        # ----- Standardize zone name -----
        if "zone" not in df.columns and "load_area" in df.columns:
            df = df.rename(columns={"load_area": "zone"})

        # Make sure we have temp_c (in case you later change fetch_weather)
        if "temp_c" not in df.columns and "temp" in df.columns:
            df = df.rename(columns={"temp": "temp_c"})

        frames.append(df[["zone", "timestamp", "date", "temp_c"]])

    out = pd.concat(frames, ignore_index=True)

    # Remove duplicates in case of overlapping files
    out = out.drop_duplicates(subset=["zone", "timestamp"]).reset_index(drop=True)

    return out


# ================================
# Feature engineering helpers
# ================================

def build_regular_week_baseline(train: pd.DataFrame) -> pd.DataFrame:
    """
    Build baseline table: mean mw per (zone, dow, hour, month).
    """
    df = train.copy()
    df["dow"] = df["timestamp"].dt.dayofweek
    df["month"] = df["timestamp"].dt.month
    base = (
        df.groupby(["zone", "dow", "hour", "month"])["mw"]
        .mean()
        .reset_index()
        .rename(columns={"mw": "mw_base"})
    )
    return base


def add_baseline_feature(df: pd.DataFrame, baseline_table: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["dow"] = df["timestamp"].dt.dayofweek
    df["month"] = df["timestamp"].dt.month
    df = df.merge(
        baseline_table,
        on=["zone", "dow", "hour", "month"],
        how="left",
    )
    return df


def add_lags(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["zone", "timestamp"]).copy()
    grp = df.groupby("zone")["mw"]
    df["lag24"] = grp.shift(24)
    df["lag168"] = grp.shift(168)
    return df


def add_thermal_inertia(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add short/long EWMA of temperature per zone.
    """
    df = df.sort_values(["zone", "timestamp"]).copy()
    df["temp_ewm_short"] = (
        df.groupby("zone")["temp_c"]
        .transform(lambda s: s.ewm(span=12, min_periods=1).mean())
    )
    df["temp_ewm_long"] = (
        df.groupby("zone")["temp_c"]
        .transform(lambda s: s.ewm(span=72, min_periods=1).mean())
    )
    return df


def build_thanksgiving_pattern(train_df: pd.DataFrame,
                               actuals_full_df: pd.DataFrame,
                               train_end_year: int) -> dict:
    """
    Build per-zone Thanksgiving-week pattern:
    (zone, hours_from_thu) -> mean mw.
    """
    all_slices = []
    years = list(range(2017, FORECAST_YEAR))  # up through 2024
    for year in years:
        if year <= train_end_year:
            src = train_df[train_df["timestamp"].dt.year == year].copy()
        else:
            src = actuals_full_df[actuals_full_df["timestamp"].dt.year == year].copy()
        if src.empty:
            continue
        tg = thanksgiving_date(year)
        align_start = tg - timedelta(days=7)
        start_ts = datetime.combine(align_start, datetime.min.time())
        end_ts = start_ts + timedelta(days=14)
        slice_df = src[(src["timestamp"] >= start_ts) & (src["timestamp"] <= end_ts)].copy()
        if slice_df.empty:
            continue
        if "zone" not in slice_df.columns:
            slice_df["zone"] = slice_df["load_area"]
        slice_df["hours_from_thu"] = (
            (slice_df["timestamp"] - start_ts).dt.total_seconds() / 3600.0
        ).astype(int)
        all_slices.append(slice_df[["zone", "hours_from_thu", "mw"]])

    if not all_slices:
        raise RuntimeError("No Thanksgiving slices found.")
    cat = pd.concat(all_slices, ignore_index=True)
    pattern = (
        cat.groupby(["zone", "hours_from_thu"])["mw"]
        .mean()
        .to_dict()
    )
    return pattern


def add_thanksgiving_feature(df: pd.DataFrame, pattern: dict,
                             baseline_table: pd.DataFrame) -> pd.DataFrame:
    """
    Add lag_thanksgiving feature, backed off to mw_base when pattern unavailable.
    """
    df = df.copy()
    years = df["timestamp"].dt.year.unique()
    align_starts = {y: thanksgiving_date(y) - timedelta(days=7) for y in years}
    df["align_start"] = df["timestamp"].dt.year.map(align_starts).astype("datetime64[ns]")
    df["hours_from_thu"] = (
        (df["timestamp"] - df["align_start"]).dt.total_seconds() / 3600.0
    ).astype(int)

    pattern_series = pd.Series(pattern)
    idx = list(zip(df["zone"], df["hours_from_thu"]))
    df["lag_thanksgiving_raw"] = pattern_series.reindex(idx).to_numpy()

    if "mw_base" not in df.columns:
        df = add_baseline_feature(df, baseline_table)

    df["lag_thanksgiving"] = df["lag_thanksgiving_raw"]
    missing = df["lag_thanksgiving"].isna()
    df.loc[missing, "lag_thanksgiving"] = df.loc[missing, "mw_base"]

    return df.drop(columns=["align_start", "hours_from_thu", "lag_thanksgiving_raw"])

# ================================
# Task 2: Smoothed peak-hour prediction
# ================================

PEAK_HOUR_SMOOTH_ALPHA: float = 0.75  # tuned smoothing parameter


def _smoothed_scores(loads: np.ndarray,
                     alpha: float = PEAK_HOUR_SMOOTH_ALPHA) -> np.ndarray:
    """
    Compute smoothed scores S_h = L_h + alpha * mean(neighbor loads).

    Parameters
    ----------
    loads : np.ndarray of shape (24,)
        Predicted loads for a single day (hours 0..23).
    alpha : float
        Smoothing parameter. alpha=0 gives plain loads.

    Returns
    -------
    scores : np.ndarray of shape (24,)
        Smoothed scores.
    """
    scores = loads.astype(float).copy()
    if alpha <= 0:
        return scores

    H = len(loads)
    for h in range(H):
        neighbors = []
        if h > 0:
            neighbors.append(loads[h - 1])
        if h < H - 1:
            neighbors.append(loads[h + 1])
        if neighbors:
            scores[h] += alpha * (sum(neighbors) / len(neighbors))
    return scores

# ================================
# GP training (unified across zones)
# ================================

NUM_FEATURES = [
    "hour",
    "dow",
    "month",
    "is_weekend",
    "temp_c",
    "temp_c2",
    "lag24",
    "lag168",
    "mw_base",
    "lag_thanksgiving",
    "temp_ewm_short",
    "temp_ewm_long",
]


def train_unified_gp(global_train: pd.DataFrame) -> tuple[GaussianProcessRegressor, StandardScaler, list[str]]:
    """
    Train unified GP on years 2021–2024 using NUM_FEATURES + zone one-hot.

    - Drops rows with NaNs in NUM_FEATURES or mw.
    - Randomly subsamples to at most n_max rows to keep GP feasible.
    """
    # 1) Filter to years 2021–2024
    mask_21_24 = (global_train["timestamp"].dt.year >= 2021) & (global_train["timestamp"].dt.year <= 2024)
    gp_train_df = global_train.loc[mask_21_24].copy()
    if gp_train_df.empty:
        raise RuntimeError("No GP training rows after filtering 2021–2024.")

    # 2) Keep only needed cols and drop NaNs
    needed_cols = ["zone"] + NUM_FEATURES + ["mw"]
    missing_cols = [c for c in needed_cols if c not in gp_train_df.columns]
    if missing_cols:
        raise RuntimeError(f"Missing expected columns in global_train for GP: {missing_cols}")

    gp_train_small = gp_train_df[needed_cols].copy()
    before = len(gp_train_small)
    gp_train_small = gp_train_small.dropna(subset=NUM_FEATURES + ["mw"])
    after = len(gp_train_small)
    if after == 0:
        raise RuntimeError("All GP training rows were dropped due to NaNs in features/target.")

    print(f"GP training rows (after dropping NaNs): {after} (dropped {before - after})")

    # 3) Subsample to keep GP tractable
    n_max = 8000  # you can tweak this down/up if needed
    if after > n_max:
        rng = np.random.RandomState(0)
        idx_sub = rng.choice(after, size=n_max, replace=False)
        gp_train_small = gp_train_small.iloc[idx_sub].reset_index(drop=True)
        print(f"Subsampled GP training rows to {len(gp_train_small)}")

    # 4) One-hot encode zone
    zone_dummies = pd.get_dummies(gp_train_small["zone"], prefix="zone")
    X_train_full = pd.concat(
        [gp_train_small[NUM_FEATURES].reset_index(drop=True),
         zone_dummies.reset_index(drop=True)],
        axis=1,
    )
    y_train_full = gp_train_small["mw"].to_numpy()
    zone_dummy_cols = list(zone_dummies.columns)

    # 5) Standardize and fit GP
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train_full.to_numpy())

    kernel = (
        ConstantKernel(1.0, (0.1, 10.0))
        * Matern(length_scale=1.0, nu=2.5)
        + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-3, 1e3))
    )

    gpr = GaussianProcessRegressor(
        kernel=kernel,
        normalize_y=True,
        n_restarts_optimizer=1,  # keep this small to avoid long optimize
        random_state=0,
    )

    print("Fitting unified GP on", X_scaled.shape[0], "rows and", X_scaled.shape[1], "features...")
    gpr.fit(X_scaled, y_train_full)
    print("Unified GP kernel:", gpr.kernel_)

    return gpr, scaler, zone_dummy_cols




# ================================
# Sequential forecast helper
# ================================

def make_gp_features_for_timestamp(row, mw_lookup, zone_dummy_cols):
    """
    Build unified GP feature vector for a (zone, timestamp) row, using mw_lookup for lag24/lag168.
    NaNs in lags or temp are safely backed off to mw_base / EWM means.
    """
    z = row["zone"]
    ts = row["timestamp"]

    ts_lag24 = ts - timedelta(hours=24)
    ts_lag168 = ts - timedelta(hours=168)

    base = row["mw_base"]

    # --- lag24 / lag168 with NaN-safe fallback ---
    lag24_val = mw_lookup.get((z, ts_lag24), base)
    lag168_val = mw_lookup.get((z, ts_lag168), base)

    if pd.isna(lag24_val):
        lag24_val = base
    if pd.isna(lag168_val):
        lag168_val = base

    # --- calendar features ---
    dow = ts.dayofweek
    month = ts.month
    is_weekend = 1 if dow in [5, 6] else 0

    # --- temperature: extra safety, though impute_future_features already filled it ---
    temp = row["temp_c"]
    if pd.isna(temp):
        # try short EWM as fallback
        temp = row.get("temp_ewm_short", np.nan)
    if pd.isna(temp):
        # last resort: 0
        temp = 0.0

    num_vals = [
        row["hour"],
        dow,
        month,
        is_weekend,
        float(temp),
        float(temp) ** 2,
        float(lag24_val),
        float(lag168_val),
        float(row["mw_base"]),
        float(row["lag_thanksgiving"]),
        float(row["temp_ewm_short"]),
        float(row["temp_ewm_long"]),
    ]

    # zone one-hot
    zone_dummy = pd.Series(0.0, index=zone_dummy_cols, dtype=float)
    colname = f"zone_{z}"
    if colname in zone_dummy.index:
        zone_dummy[colname] = 1.0

    feat = np.concatenate([np.array(num_vals, dtype=float), zone_dummy.to_numpy()])

    # Debug guard: if anything is still NaN, blow up *here* with context
    if np.isnan(feat).any():
        raise RuntimeError(
            f"NaNs in feature vector for zone {z}, ts {ts}. "
            f"num_vals={num_vals}"
        )

    return feat


def impute_future_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure there are no NaNs in the GP feature columns in future_df.

    - Fill per-zone means for each numeric feature.
    - If an entire zone is missing for that feature, fill with global mean.
    """
    df = df.copy()

    # Features the GP will use directly from the row
    cols_to_fix = [
        "temp_c",
        "mw_base",
        "lag_thanksgiving",
        "temp_ewm_short",
        "temp_ewm_long",
    ]

    for col in cols_to_fix:
        if col not in df.columns:
            continue

        # per-zone mean
        df[col] = df.groupby("zone")[col].transform(
            lambda s: s.fillna(s.mean())
        )

        # global mean fallback
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].mean())

    # temp_c2 should be consistent with temp_c
    if "temp_c" in df.columns:
        df["temp_c2"] = df["temp_c"] ** 2

    return df


# ================================
# Main prediction pipeline
# ================================

def main():
    # ----------------------------------
    # 1) Load metered + weather
    # ----------------------------------
    all_metered = load_metered_data(METERED_DIR)
    weather_df = load_weather_data(WEATHER_DIR)

    # drop RTO if present
    all_metered = all_metered[all_metered["zone"] != "RTO"].copy()

    # ----------------------------------
    # 2) Training and actual frames
    # ----------------------------------
    train_df = all_metered[
        (all_metered["date"] >= TRAIN_START) &
        (all_metered["date"] <= TRAIN_END)
    ].copy()

    # we need full 2024 + 2025 for Thanksgiving / Nov 2025 logic
    actuals_full_df = all_metered[
        all_metered["timestamp"].dt.year.isin([2024, FORECAST_YEAR])
    ].copy()

    # merge weather
    train_merged = train_df.merge(
        weather_df[["zone", "timestamp", "temp_c"]],
        on=["zone", "timestamp"],
        how="left",
    )
    actuals_full_merged = actuals_full_df.merge(
        weather_df[["zone", "timestamp", "temp_c"]],
        on=["zone", "timestamp"],
        how="left",
    )

    # ----------------------------------
    # 3) Baseline + features for training (2016–2023)
    # ----------------------------------
    baseline_table = build_regular_week_baseline(train_merged)

    train_feat = add_baseline_feature(train_merged, baseline_table)
    train_feat = add_lags(train_feat)

    thank_pattern = build_thanksgiving_pattern(train_feat, actuals_full_merged, TRAIN_END.year)

    train_feat = add_thanksgiving_feature(train_feat, thank_pattern, baseline_table)
    train_feat = add_thermal_inertia(train_feat)

    # ----------------------------------
    # 3b) Build 2024 feature frame
    # ----------------------------------
    actuals_2024 = actuals_full_merged[
        actuals_full_merged["timestamp"].dt.year == 2024
    ].copy()

    actuals_2024_feat = add_baseline_feature(actuals_2024, baseline_table)
    actuals_2024_feat = add_lags(actuals_2024_feat)
    actuals_2024_feat = add_thanksgiving_feature(actuals_2024_feat, thank_pattern, baseline_table)
    actuals_2024_feat = add_thermal_inertia(actuals_2024_feat)

    # ----------------------------------
    # 3c) Global training frame = 2016–2023 train_feat + 2024 actuals
    # ----------------------------------
    global_train = pd.concat(
        [train_feat, actuals_2024_feat],
        ignore_index=True,
    )

    # Add calendar + temp^2 for all rows in global_train
    ts = global_train["timestamp"]
    global_train["hour"] = ts.dt.hour
    global_train["dow"] = ts.dt.dayofweek
    global_train["month"] = ts.dt.month
    global_train["is_weekend"] = global_train["dow"].isin([5, 6]).astype(int)
    global_train["temp_c2"] = global_train["temp_c"] ** 2

    # ----------------------------------
    # 4) Train unified GP (filters to 2021–2024 internally)
    # ----------------------------------
        # ----------------------------------
    # 4) Load unified GP artifacts (trained on 2021–2024)
    # ----------------------------------
    if not ARTIFACT_PATH.exists():
        raise RuntimeError(f"Artifact file not found: {ARTIFACT_PATH}. Did you run train.py?")

    art = joblib.load(ARTIFACT_PATH)
    gpr = art["gpr"]
    scaler_gp_forecast = art["scaler"]
    zone_dummy_cols_fore = art["zone_dummy_cols"]
    baseline_table = art["baseline_table"]
    thank_pattern = art["thank_pattern"]


    # ----------------------------------
    # 5) Build Nov 2025 future grid
    # ----------------------------------
    full_hours = pd.date_range(
    datetime(FORECAST_YEAR, 11, 1, 0, 0, 0),
    datetime(FORECAST_YEAR, 11, 29, 23, 0, 0),
    freq="h",
    )

    zones_all = ZONES  # fixed 29 zones in contest order
    future_rows = []
    for z in zones_all:
        for ts_future in full_hours:
            future_rows.append({"zone": z, "timestamp": ts_future})


    future_df = pd.DataFrame(future_rows)
    future_df["date"] = future_df["timestamp"].dt.date
    future_df["hour"] = future_df["timestamp"].dt.hour

    # attach observed mw for Nov 1–13, 2025 (if present)
    nov_obs = all_metered[
        (all_metered["timestamp"].dt.year == FORECAST_YEAR) &
        (all_metered["timestamp"].dt.month == 11) &
        (all_metered["date"] <= date(FORECAST_YEAR, 11, 13))
    ][["zone", "timestamp", "mw"]].copy()

    future_df = future_df.merge(
        nov_obs,
        on=["zone", "timestamp"],
        how="left",
    )

    # attach weather for Nov 2025
    weather_2025 = weather_df[
        (weather_df["date"] >= NOV_START) &
        (weather_df["date"] <= date(FORECAST_YEAR, 11, 30))
    ].copy()

    future_df = future_df.merge(
        weather_2025[["zone", "timestamp", "temp_c"]],
        on=["zone", "timestamp"],
        how="left",
    )

    # baseline + Thanksgiving + thermal inertia for future
    future_df = add_baseline_feature(future_df, baseline_table)
    future_df = add_thanksgiving_feature(future_df, thank_pattern, baseline_table)
    future_df = add_thermal_inertia(future_df)

    # ----------------------------------
    # 6) Fill Nov 14–19 with baseline
    # ----------------------------------
    mask_gap = (
        (future_df["date"] >= date(FORECAST_YEAR, 11, 14)) &
        (future_df["date"] <= date(FORECAST_YEAR, 11, 19)) &
        (future_df["mw"].isna())
    )
    future_df.loc[mask_gap, "mw"] = future_df.loc[mask_gap, "mw_base"]

    # ----------------------------------
    # 7) Initialize mw_lookup with actual / baseline up to 19th
    # ----------------------------------
    mw_lookup = {}
    cutoff_ts = datetime(FORECAST_YEAR, 11, 19, 23, 0, 0)
    for _, row in future_df.iterrows():
        ts_row = row["timestamp"]
        if ts_row <= cutoff_ts:
            mw_lookup[(row["zone"], ts_row)] = row["mw"]

    # also need calendar features + temp_c2 in future_df
    ts_future_all = future_df["timestamp"]
    future_df["hour"] = ts_future_all.dt.hour
    future_df["dow"] = ts_future_all.dt.dayofweek
    future_df["month"] = ts_future_all.dt.month
    future_df["is_weekend"] = future_df["dow"].isin([5, 6]).astype(int)
    future_df["temp_c2"] = future_df["temp_c"] ** 2

    # >>> New: impute any remaining NaNs in features used by the GP <<<
    future_df = impute_future_features(future_df)

    # ----------------------------------
    # 8) Sequential forecast for Nov 20–29
    # ----------------------------------
    forecast_start_ts = datetime(FORECAST_YEAR, 11, 20, 0, 0, 0)
    forecast_end_ts = datetime(FORECAST_YEAR, 11, 29, 23, 0, 0)

    pred_rows = []

    for z in zones_all:
        mask_zone = (
            (future_df["zone"] == z) &
            (future_df["timestamp"] >= forecast_start_ts) &
            (future_df["timestamp"] <= forecast_end_ts)
        )
        zone_future = future_df.loc[mask_zone].sort_values("timestamp").copy()

        for _, row in zone_future.iterrows():
            feat_vec = make_gp_features_for_timestamp(row, mw_lookup, zone_dummy_cols_fore)
            X_scaled = scaler_gp_forecast.transform(feat_vec.reshape(1, -1))
            mw_pred = gpr.predict(X_scaled)[0]
            ts_row = row["timestamp"]
            pred_rows.append({
                "zone": z,
                "timestamp": ts_row,
                "date": ts_row.date(),
                "hour": ts_row.hour,
                "mw_pred_gp_forecast": mw_pred,
            })
            mw_lookup[(z, ts_row)] = mw_pred  # for later lags

    forecast_pred_df = pd.DataFrame(pred_rows)
        # ----------------------------------
    # Task 3: Pre-compute predicted peak days (top-2 daily maxima per zone)
    # ----------------------------------
    # Daily max predicted load per (zone, date)
    daily_max = (
        forecast_pred_df
        .groupby(["zone", "date"], as_index=False)["mw_pred_gp_forecast"]
        .max()
        .rename(columns={"mw_pred_gp_forecast": "mw_daily_max"})
    )

    # Rank days within each zone by descending daily max (1 = highest)
    daily_max["rank"] = daily_max.groupby("zone")["mw_daily_max"].rank(
        method="first", ascending=False
    )

    # Keep top-2 days per zone as predicted peak days
    peak_day_df = daily_max[daily_max["rank"] <= 2].copy()

    # Build a mapping: zone -> set of predicted peak dates
    zone_peak_dates: dict[str, set[date]] = {}
    for z, sub in peak_day_df.groupby("zone"):
        zone_peak_dates[z] = set(sub["date"])


    # ----------------------------------
    # 9) Decide which date to output based on "today"
    # ----------------------------------
    today = datetime.now().date()

    if today <= date(FORECAST_YEAR, 11, 18):
        target_date = date(FORECAST_YEAR, 11, 20)
    elif date(FORECAST_YEAR, 11, 19) <= today <= date(FORECAST_YEAR, 11, 28):
        target_date = today + timedelta(days=1)
    else:
        # Should not happen per your note; cap at 29
        target_date = date(FORECAST_YEAR, 11, 29)

    target_df = forecast_pred_df[forecast_pred_df["date"] == target_date].copy()
    if target_df.empty:
        raise RuntimeError(f"No forecasts found for target date {target_date}.")

    # ----------------------------------
    # 10) Build output in required format
    # ----------------------------------
    # Rename zone -> load_area, and set peak hour & peak day flags
    # 10a) Subset predictions for the target_date
    day_pred = forecast_pred_df[forecast_pred_df["date"] == target_date].copy()
    if day_pred.empty:
        raise RuntimeError(f"No forecasts found for target date {target_date}.")

    # Fixed 29-zone order
    zone_order = ZONES
    n_zones = len(zone_order)
    L_values = []
    PH_values = []
    PD_values = []


    for z in zone_order:
        sub = day_pred[day_pred["zone"] == z].copy()
        loads_by_hour = sub.set_index("hour")["mw_pred_gp_forecast"]

        # Build a dense 24-hour vector of loads with fallback to mean if any hour is missing
        loads_vec = []
        mean_val = float(loads_by_hour.mean()) if not loads_by_hour.empty else 0.0
        for h in range(24):
            val = loads_by_hour.get(h, np.nan)
            if pd.isna(val):
                val = mean_val
            loads_vec.append(float(val))
        loads_vec = np.array(loads_vec, dtype=float)

        # Append rounded loads to L_values (Task 1 output)
        for val in loads_vec:
            L_values.append(int(round(val)))

        # Task 2: smoothed peak hour with alpha = 0.75
        scores = _smoothed_scores(loads_vec, alpha=PEAK_HOUR_SMOOTH_ALPHA)
        ph = int(np.argmax(scores))  # hour in {0,...,23}
        PH_values.append(ph)

        # Task 3 (peak day) placeholder: all zeros for now
        if z in zone_peak_dates and target_date in zone_peak_dates[z]:
            PD_values.append(1)
        else:
            PD_values.append(0)


    # 10d) Current date (today) for the first field
    today_str = datetime.now().strftime("%Y-%m-%d")

    # Build final list: first field quoted, rest plain
    fields = ['"{}"'.format(today_str)]
    fields += [str(v) for v in L_values]
    fields += [str(v) for v in PH_values]
    fields += [str(v) for v in PD_values]

    # Print *one line* of CSV, no header, no extra text
    print(",".join(fields))

if __name__ == "__main__":
    main()
