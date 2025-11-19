import argparse
from datetime import datetime, date
from pathlib import Path

import pandas as pd
from meteostat import Hourly, Point
import pgeocode


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fetch hourly weather (temp) for all PJM zones and save as CSVs."
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="data/weather_cities",
        help="Output directory for weather CSVs",
    )
    parser.add_argument(
        "--start",
        type=str,
        default="2021-01-01",
        help="Start date (YYYY-MM-DD) for weather",
    )
    parser.add_argument(
        "--end",
        type=str,
        default="2025-11-30",
        help="End date (YYYY-MM-DD) for weather (inclusive)",
    )
    parser.add_argument(
        "--mapping",
        type=str,
        default="zone_city_zip.csv",
        help="Path to zone-city-zip mapping CSV",
    )
    return parser.parse_args()


def load_zone_mapping(mapping_path: Path) -> pd.DataFrame:
    """
    Expect columns: zone, city, zip
    """
    if not mapping_path.exists():
        raise FileNotFoundError(f"Zone mapping file not found: {mapping_path}")
    df = pd.read_csv(mapping_path, dtype={"zone": str, "city": str, "zip": str})
    if not {"zone", "zip"}.issubset(df.columns):
        raise ValueError("Mapping file must have at least columns: zone, zip")
    return df


def fetch_zone_weather(zone: str, zipcode: str,
                       start: datetime, end: datetime) -> pd.DataFrame:
    """
    Use pgeocode to get lat/lon from ZIP, then Meteostat Hourly to get temp.
    Returns columns: zone, timestamp, temp_c.
    """
    nomi = pgeocode.Nominatim("us")
    rec = nomi.query_postal_code(zipcode)
    if pd.isna(rec.latitude) or pd.isna(rec.longitude):
        print(f"WARNING: Could not geocode ZIP {zipcode} for zone {zone}; skipping this ZIP.")
        return pd.DataFrame(columns=["zone", "timestamp", "temp_c"])

    pt = Point(float(rec.latitude), float(rec.longitude))

    try:
        hourly = Hourly(pt, start, end, timezone="America/New_York")
        df = hourly.fetch()
    except Exception as e:
        print(f"WARNING: Meteostat error for zone {zone}, ZIP {zipcode}: {e}. Skipping this ZIP.")
        return pd.DataFrame(columns=["zone", "timestamp", "temp_c"])

    if df.empty:
        print(f"WARNING: No weather returned for zone {zone}, ZIP {zipcode}; skipping this ZIP.")
        return pd.DataFrame(columns=["zone", "timestamp", "temp_c"])

    df = df.reset_index().rename(columns={"time": "timestamp"})

    # Meteostat calls it 'temp' (°C). Standardize to 'temp_c'.
    if "temp" not in df.columns:
        print(f"WARNING: 'temp' column missing for zone {zone}, ZIP {zipcode}; skipping this ZIP.")
        return pd.DataFrame(columns=["zone", "timestamp", "temp_c"])

    df = df.rename(columns={"temp": "temp_c"})
    df["zone"] = zone

    return df[["zone", "timestamp", "temp_c"]]




def main():
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    start_date = datetime.strptime(args.start, "%Y-%m-%d")
    # Include the full end day up to 23:00
    end_date = datetime.strptime(args.end, "%Y-%m-%d").replace(hour=23)

    mapping_path = Path(args.mapping)
    mapping = load_zone_mapping(mapping_path)

    all_frames = []

    for _, row in mapping.iterrows():
        zone = row["zone"]
        zipcode = row["zip"]
        print(f"Fetching weather for zone {zone} (ZIP {zipcode})...")

        df_zone = fetch_zone_weather(zone, zipcode, start_date, end_date)

        # Skip if this ZIP failed / returned nothing
        if df_zone.empty:
            continue

        all_frames.append(df_zone)

        # Save / append per-zone file (if multiple ZIPs per zone)
        zone_file = outdir / f"weather_{zone}.csv"
        if zone_file.exists():
            # Append to existing data if you want to keep all ZIPs
            existing = pd.read_csv(zone_file)
            df_zone_to_save = pd.concat([existing, df_zone], ignore_index=True)
        else:
            df_zone_to_save = df_zone

        df_zone_to_save.to_csv(zone_file, index=False)

    # Also save a combined file if we actually got anything
    if not all_frames:
        raise RuntimeError("No weather data fetched for any zone; check your mapping or network.")

    combined = pd.concat(all_frames, ignore_index=True)
    combined_file = outdir / "weather_all_zones.csv"
    combined.to_csv(combined_file, index=False)

    print(f"Saved weather CSVs into {outdir}")

if __name__ == "__main__":
    main()
