#!/usr/bin/env python3
import argparse
from datetime import datetime, timedelta, timezone
from dateutil import tz
import pandas as pd
import numpy as np

# Weather (history)
from meteostat import Point, Hourly

# Holidays (no API)
import holidays

# Sunrise/Sunset (no API)
from astral import LocationInfo
from astral.sun import sun

# Forecast (no key)
import requests


def _mk_dt(s: str) -> datetime:
    # Parse YYYY-MM-DD to a timezone-aware midnight UTC
    dt = datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return dt


def fetch_history(lat: float, lon: float, start: str, end: str, tz_name: str) -> pd.DataFrame:
    """
    Historical hourly obs from Meteostat (airport/station blend).
    Returns local-time indexed frame with common power-relevant variables.
    """
    start_dt = _mk_dt(start).replace(tzinfo=None)
    # Meteostat end is exclusive; add one day to include end date through 23:00
    end_dt = (_mk_dt(end) + timedelta(days=1)).replace(tzinfo=None)

    point = Point(lat, lon)
    df = Hourly(point, start_dt, end_dt).fetch()

    if df.empty:
        # Meteostat sometimes delays final station data; fall back to modeled reanalysis
        df = Hourly(point, start_dt, end_dt, model=True).fetch()

    if df.empty:
        raise RuntimeError("Meteostat returned no data for the given period/coords.")

    # Keep and rename common fields if present
    rename_map = {
        "temp": "temp_c",            # degC
        "dwpt": "dewpoint_c",        # degC
        "rhum": "humidity_pct",      # %
        "wspd": "wind_speed_ms",     # m/s
        "pres": "pressure_hpa",      # hPa
        "prcp": "precip_mm",         # mm
        "coco": "weather_code"       # WMO code
    }
    cols_present = {k: v for k, v in rename_map.items() if k in df.columns}
    df = df[list(cols_present.keys())].rename(columns=cols_present)

    # Index returned by Meteostat may be tz-naive UTC; enforce before conversion
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")

    # Index is UTC; convert to local tz
    local = tz.gettz(tz_name)
    df["timestamp_utc"] = df.index.tz_convert("UTC")
    df["timestamp_local"] = df.index.tz_convert(local)
    df = df.reset_index(drop=True)

    # Derived temps (F) and degree-day style pieces
    df["temp_f"] = df["temp_c"] * 9/5 + 32
    # piecewise linear "cooling/heating" style features around 65F
    base_f = 65.0
    df["cooling_deg_f"] = np.clip(df["temp_f"] - base_f, 0, None)
    df["heating_deg_f"] = np.clip(base_f - df["temp_f"], 0, None)

    df["source"] = "history"
    return df


def fetch_forecast(lat: float, lon: float, hours_ahead: int, tz_name: str) -> pd.DataFrame:
    """
    Hourly forecast from Open-Meteo (no key).
    """
    # choose a compact, power-relevant set
    vars_ = [
        "temperature_2m", "dewpoint_2m",
        "cloudcover", "shortwave_radiation",
        "windspeed_10m", "precipitation"
    ]
    url = (
        "https://api.open-meteo.com/v1/forecast?"
        f"latitude={lat}&longitude={lon}"
        f"&hourly={','.join(vars_)}"
        "&timezone=UTC"  # request UTC; convert after
        f"&forecast_days=7"
    )
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    j = r.json()
    H = j.get("hourly", {})
    time = H.get("time", [])
    if not time:
        raise RuntimeError("Open-Meteo forecast: no hourly time series returned.")

    df = pd.DataFrame({"timestamp_utc": pd.to_datetime(time, utc=True)})
    # add variables if present
    def add_col(key, name):
        if key in H:
            df[name] = H[key]
        else:
            df[name] = np.nan

    add_col("temperature_2m", "temp_c")
    add_col("dewpoint_2m", "dewpoint_c")
    add_col("cloudcover", "cloudcover_pct")
    add_col("shortwave_radiation", "shortwave_wm2")
    add_col("windspeed_10m", "wind_speed_ms")
    add_col("precipitation", "precip_mm")

    # keep only the first N hours ahead
    now_utc = datetime.now(timezone.utc)
    df = df[df["timestamp_utc"] >= now_utc].sort_values("timestamp_utc").head(hours_ahead)

    # convert to local tz
    local = tz.gettz(tz_name)
    df["timestamp_local"] = df["timestamp_utc"].dt.tz_convert(local)
    # derived temp (F) and degree pieces
    df["temp_f"] = df["temp_c"] * 9/5 + 32
    base_f = 65.0
    df["cooling_deg_f"] = np.clip(df["temp_f"] - base_f, 0, None)
    df["heating_deg_f"] = np.clip(base_f - df["temp_f"], 0, None)

    df["source"] = "forecast"
    return df


def add_calendar_and_light(df: pd.DataFrame, lat: float, lon: float, tz_name: str, country: str = "US") -> pd.DataFrame:
    """
    Adds: date, hour, DOW, weekend, holiday flag/name, day-of-year,
    sunrise, sunset, daylight_minutes. Works on timestamp_local.
    """
    if "timestamp_local" not in df.columns:
        raise ValueError("timestamp_local column required.")

    df = df.copy()
    df["date_local"] = df["timestamp_local"].dt.date
    df["hour_local"] = df["timestamp_local"].dt.hour
    df["dow"] = df["timestamp_local"].dt.dayofweek  # Mon=0
    df["is_weekend"] = df["dow"].isin([5, 6]).astype(int)
    df["doy"] = df["timestamp_local"].dt.dayofyear

    # Holidays (vectorized by date)
    years = sorted({d.year for d in df["timestamp_local"]})
    hol = holidays.country_holidays(country=country, years=years)
    df["is_holiday"] = df["date_local"].apply(lambda d: int(d in hol))
    df["holiday_name"] = df["date_local"].apply(lambda d: hol.get(d, ""))

    # Sunrise/sunset/daylight (per unique date)
    local = tz.gettz(tz_name)
    loc = LocationInfo(latitude=lat, longitude=lon, timezone=tz_name, name="site", region="")
    uniq_dates = sorted(df["date_local"].unique())
    sr_ss = {}
    for d in uniq_dates:
        # local midnight
        dt_local = datetime(d.year, d.month, d.day, tzinfo=local)
        s = sun(loc.observer, date=dt_local, tzinfo=local)
        sunrise = s["sunrise"]
        sunset = s["sunset"]
        daylight_minutes = int((sunset - sunrise).total_seconds() // 60)
        sr_ss[d] = (sunrise, sunset, daylight_minutes)

    df["sunrise_local"] = df["date_local"].apply(lambda d: sr_ss[d][0])
    df["sunset_local"] = df["date_local"].apply(lambda d: sr_ss[d][1])
    df["daylight_minutes"] = df["date_local"].apply(lambda d: sr_ss[d][2])

    return df


def main():
    p = argparse.ArgumentParser(description="Fetch hourly weather & calendar features for load forecasting.")
    p.add_argument("--lat", type=float, required=True, help="Latitude (e.g., 39.95)")
    p.add_argument("--lon", type=float, required=True, help="Longitude (e.g., -75.16)")
    p.add_argument("--zone", type=str, required=True, help="Short name for zone (used in filename)")
    p.add_argument("--tz", type=str, default="America/Detroit", help="IANA timezone (default America/Detroit)")
    p.add_argument("--history-start", type=str, required=True, help="YYYY-MM-DD inclusive")
    p.add_argument("--history-end", type=str, required=True, help="YYYY-MM-DD inclusive")
    p.add_argument("--forecast-hours", type=int, default=48, help="How many hours ahead to keep from forecast (default 48)")
    p.add_argument("--out", type=str, default="features.csv", help="Output CSV path")
    args = p.parse_args()

    # 1) history
    hist = fetch_history(args.lat, args.lon, args.history_start, args.history_end, args.tz)

    # 2) forecast
    fcst = fetch_forecast(args.lat, args.lon, args.forecast_hours, args.tz)

    # 3) combine & add calendar/light
    df = pd.concat([hist, fcst], ignore_index=True, sort=False)
    df = add_calendar_and_light(df, args.lat, args.lon, args.tz)

    # 4) tidy columns and sort
    keep_cols = [
        "timestamp_local", "timestamp_utc", "source",
        "temp_c", "dewpoint_c", "temp_f", "cooling_deg_f", "heating_deg_f",
        "humidity_pct", "wind_speed_ms", "pressure_hpa",
        "precip_mm", "cloudcover_pct", "shortwave_wm2", "weather_code",
        "date_local", "hour_local", "dow", "is_weekend", "is_holiday", "holiday_name",
        "sunrise_local", "sunset_local", "daylight_minutes"
    ]
    # keep only those that exist
    keep_cols = [c for c in keep_cols if c in df.columns]
    df = df[keep_cols].sort_values(["timestamp_local"]).reset_index(drop=True)

    # 5) write
    df.to_csv(args.out, index=False)
    print(f"wrote {len(df):,} rows to {args.out}")


if __name__ == "__main__":
    main()
