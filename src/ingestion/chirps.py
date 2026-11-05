"""CHIRPS daily rainfall ingestion, one time series per ~5 km pixel, for every site
in config/sites.yaml.

Save this file as src/ingestion/chirps.py and run from the repo root (venv active):

    # Trial: one site, one month (do this first)
    python -m src.ingestion.chirps --site nakuru --start 2020-04-01 --end 2020-04-30

    # Full run: all sites, full period from config
    python -m src.ingestion.chirps

Output: data/raw/chirps_<site>.parquet in long format
    pixel_id | lat | lon | date | precip_mm
Monthly chunks are cached in data/raw/chirps/<site>/ so an interrupted run resumes
where it stopped.
"""
import argparse
import os
import re
import time
from pathlib import Path

import ee
import pandas as pd
import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "config" / "sites.yaml"
RAW = ROOT / "data" / "raw"
COLLECTION = "UCSB-CHG/CHIRPS/DAILY"
BAND = "precipitation"
BAND_NAME = re.compile(r"^(\d{8})_" + BAND + "$")  # toBands() names bands <date>_<band>


def load_config() -> dict:
    with open(CONFIG, encoding="utf-8") as f:
        return yaml.safe_load(f)


def fetch_chunk(region: ee.Geometry, proj: ee.Projection, start: str, end: str,
                retries: int = 4) -> pd.DataFrame:
    """Fetch every pixel in `region` for [start, end) as a long DataFrame."""
    images = ee.ImageCollection(COLLECTION).select(BAND).filterDate(start, end)
    stacked = images.toBands()  # one band per day
    last_err = None
    for attempt in range(retries):
        try:
            fc = stacked.sample(region=region, projection=proj, geometries=True).getInfo()
            break
        except Exception as err:  # network / quota hiccups
            last_err = err
            wait = 5 * (attempt + 1)
            print(f"    retry in {wait}s ({err})")
            time.sleep(wait)
    else:
        raise RuntimeError(f"Chunk {start}..{end} failed after {retries} tries") from last_err

    rows = []
    for feat in fc["features"]:
        lon, lat = feat["geometry"]["coordinates"]
        lat, lon = round(lat, 3), round(lon, 3)
        pid = f"{lat:.3f}_{lon:.3f}"
        for name, value in feat["properties"].items():
            m = BAND_NAME.match(name)
            if m:
                rows.append((pid, lat, lon, m.group(1), value))

    df = pd.DataFrame(rows, columns=["pixel_id", "lat", "lon", "date", "precip_mm"])
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
    return df


def chunk_complete(df: pd.DataFrame, n_days: int) -> bool:
    """True if every pixel in the chunk has a value for each of the n_days days."""
    if df.empty or df["date"].nunique() != n_days:
        return False
    return len(df) == df["pixel_id"].nunique() * n_days and not df["precip_mm"].isna().any()


def ingest_site(site_key: str, site: dict, start: str, end: str, tag: str) -> None:
    b = site["bbox"]
    region = ee.Geometry.Rectangle([b["lon_min"], b["lat_min"], b["lon_max"], b["lat_max"]])
    proj = ee.Image(ee.ImageCollection(COLLECTION).first()).select(BAND).projection()

    cache_dir = RAW / "chirps" / site_key
    cache_dir.mkdir(parents=True, exist_ok=True)
    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    chunks = []

    print(f"\n=== {site_key}: {site['name']} | {start} to {end} ===")
    for period in pd.period_range(start_ts, end_ts, freq="M"):
        s = max(period.start_time, start_ts)
        e = min(period.end_time.normalize() + pd.Timedelta(days=1),
                end_ts + pd.Timedelta(days=1))  # end is exclusive in filterDate
        cache = cache_dir / f"{s:%Y%m%d}_{e:%Y%m%d}.parquet"
        n_days = (e - s).days  # days in [s, e)
        if cache.exists():
            df = pd.read_parquet(cache)
            # The cache is keyed by dates only. A chunk fetched before CHIRPS had published
            # the whole month would be short; refetch it instead of reusing it.
            if chunk_complete(df, n_days):
                chunks.append(df)
                continue
            print(f"  cached chunk {cache.name} is incomplete, refetching")
        print(f"  fetching {s:%Y-%m-%d} to {(e - pd.Timedelta(days=1)):%Y-%m-%d}")
        df = fetch_chunk(region, proj, f"{s:%Y-%m-%d}", f"{e:%Y-%m-%d}")
        if not chunk_complete(df, n_days):
            raise SystemExit(
                f"CHIRPS returned {df['date'].nunique()} of {n_days} days for "
                f"{s:%Y-%m-%d}..{e:%Y-%m-%d} (not published yet?). Not cached. "
                "Set an earlier period end in config/sites.yaml or wait for CHIRPS.")
        df.to_parquet(cache, index=False)
        chunks.append(df)

    full = (pd.concat(chunks, ignore_index=True)
            .sort_values(["pixel_id", "date"]).reset_index(drop=True))
    out = RAW / f"chirps_{site_key}{tag}.parquet"
    full.to_parquet(out, index=False)

    # Sanity report
    n_pix, n_days = full["pixel_id"].nunique(), full["date"].nunique()
    expected = n_pix * n_days
    years = max(n_days / 365.25, 1e-9)
    print(f"  saved {out}")
    print(f"  pixels: {n_pix} | days: {n_days} | rows: {len(full)} (expected {expected})")
    print(f"  missing values: {int(full['precip_mm'].isna().sum())}")
    print(f"  mean annual rainfall per pixel: {full['precip_mm'].mean() * 365.25:.0f} mm/yr "
          f"(over {years:.2f} yrs)")
    print(f"  wet days (>1 mm): {(full['precip_mm'] > 1).mean() * 100:.1f}%")
    print(f"  lat range {full['lat'].min()}..{full['lat'].max()}, "
          f"lon range {full['lon'].min()}..{full['lon'].max()}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest CHIRPS daily rainfall per pixel.")
    parser.add_argument("--site", help="site key from sites.yaml (default: all sites)")
    parser.add_argument("--start", help="YYYY-MM-DD (default: config period start)")
    parser.add_argument("--end", help="YYYY-MM-DD (default: config period end)")
    args = parser.parse_args()

    load_dotenv(ROOT / ".env")
    project = os.getenv("GEE_PROJECT")
    if not project:
        raise SystemExit("GEE_PROJECT is not set in .env")
    ee.Initialize(project=project)

    cfg = load_config()
    start = args.start or str(cfg["period"]["start"])
    end = args.end or str(cfg["period"]["end"])
    tag = "" if not (args.start or args.end) else f"_{start}_{end}"

    sites = cfg["sites"]
    if args.site:
        if args.site not in sites:
            raise SystemExit(f"Unknown site '{args.site}'. Available: {', '.join(sites)}")
        sites = {args.site: sites[args.site]}

    for key, site in sites.items():
        ingest_site(key, site, start, end, tag)


if __name__ == "__main__":
    main()
