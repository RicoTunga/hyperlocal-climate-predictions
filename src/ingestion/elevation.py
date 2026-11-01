"""Per-pixel elevation for every CHIRPS pixel in each site of config/sites.yaml.

Save this file as src/ingestion/elevation.py and run from the repo root (venv active):

    python -m src.ingestion.elevation                 # all sites
    python -m src.ingestion.elevation --site nakuru   # one site
    python -m src.ingestion.elevation --force         # recompute even if the file is current

Source: SRTM 30 m (Earth Engine asset USGS/SRTMGL1_003, band 'elevation').
Each ~5 km CHIRPS pixel gets summary statistics of the 30 m cells inside its footprint.

Output: data/raw/elevation_<site>.parquet
    pixel_id | lat | lon | elev_mean_m | elev_std_m | elev_min_m | elev_max_m
elev_std_m (and max minus min) describes how rugged the terrain is inside the pixel.
"""
import argparse
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from dotenv import load_dotenv

from src.ingestion.era5 import pixel_grid  # same pixel grid and pixel_id format as CHIRPS/ERA5

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "config" / "sites.yaml"
RAW = ROOT / "data" / "raw"

DEM = "USGS/SRTMGL1_003"
BAND = "elevation"
SCALE_M = 90      # sample the 30 m DEM at 90 m: plenty for a 5 km pixel, and much faster
HALF = 0.025      # half a CHIRPS pixel, in degrees
CHUNK = 150       # pixels per Earth Engine request
COLUMNS = ["elev_mean_m", "elev_std_m", "elev_min_m", "elev_max_m"]


def load_config() -> dict:
    with open(CONFIG, encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_features(fc: dict) -> pd.DataFrame:
    """Turn an Earth Engine reduceRegions result (as a dict) into a DataFrame."""
    rows = []
    for feat in fc["features"]:
        p = feat["properties"]
        rows.append((p["pixel_id"], p.get("mean"), p.get("stdDev"), p.get("min"), p.get("max")))
    df = pd.DataFrame(rows, columns=["pixel_id"] + COLUMNS)
    df[COLUMNS] = df[COLUMNS].astype("float32")
    return df


def fetch_chunk(pids, lats, lons, retries: int = 4) -> pd.DataFrame:
    import ee  # imported here so parse_features can be tested without Earth Engine

    feats = [
        ee.Feature(
            ee.Geometry.Rectangle([lo - HALF, la - HALF, lo + HALF, la + HALF], None, False),
            {"pixel_id": pid},
        )
        for pid, la, lo in zip(pids, lats, lons)
    ]
    reducer = (ee.Reducer.mean()
               .combine(ee.Reducer.stdDev(), None, True)
               .combine(ee.Reducer.minMax(), None, True))
    dem = ee.Image(DEM).select(BAND)
    last_err = None
    for attempt in range(retries):
        try:
            fc = dem.reduceRegions(ee.FeatureCollection(feats), reducer, SCALE_M).getInfo()
            return parse_features(fc)
        except Exception as err:
            last_err = err
            wait = 5 * (attempt + 1)
            print(f"    retry in {wait}s ({err})")
            time.sleep(wait)
    raise RuntimeError("Elevation request failed after retries") from last_err


def ingest_site(site_key: str, site: dict, force: bool) -> None:
    lats, lons = pixel_grid(site["bbox"])
    pids = np.array([f"{la:.3f}_{lo:.3f}" for la, lo in zip(lats, lons)])
    out = RAW / f"elevation_{site_key}.parquet"
    print(f"\n=== {site_key}: {site['name']} | {len(pids)} pixels ===")

    if out.exists() and not force:
        have = pd.read_parquet(out, columns=["pixel_id"])["pixel_id"]
        if set(have) == set(pids):
            print(f"  {out.name} already matches the current pixel grid; nothing to do "
                  "(use --force to recompute)")
            report(pd.read_parquet(out))
            return
        print("  existing file does not match the current grid; recomputing")

    frames = []
    for i in range(0, len(pids), CHUNK):
        print(f"  pixels {i + 1}-{min(i + CHUNK, len(pids))} of {len(pids)}")
        frames.append(fetch_chunk(pids[i:i + CHUNK], lats[i:i + CHUNK], lons[i:i + CHUNK]))
    grid = pd.DataFrame({"pixel_id": pids, "lat": lats, "lon": lons})
    df = grid.merge(pd.concat(frames, ignore_index=True), on="pixel_id",
                    how="left", validate="one_to_one")
    df.to_parquet(out, index=False)
    print(f"  saved {out}")
    report(df)


def report(df: pd.DataFrame) -> None:
    e = df["elev_mean_m"]
    print(f"  pixels: {len(df)} | missing elevation: {int(e.isna().sum())}")
    print(f"  mean elevation: min {e.min():.0f} m, median {e.median():.0f} m, max {e.max():.0f} m "
          f"(10th/90th percentile {e.quantile(.1):.0f} / {e.quantile(.9):.0f} m)")
    print(f"  within-pixel roughness (std): median {df['elev_std_m'].median():.0f} m, "
          f"max {df['elev_std_m'].max():.0f} m")
    low = df[e < 800]
    if len(low):
        print(f"  CHECK: {len(low)} pixels below 800 m (possible water/void issue), e.g. "
              f"{low['pixel_id'].head(5).tolist()}")
    hi = df.nlargest(3, "elev_mean_m")[["pixel_id", "elev_mean_m"]]
    lo = df.nsmallest(3, "elev_mean_m")[["pixel_id", "elev_mean_m"]]
    print("  highest pixels:", [(r.pixel_id, round(float(r.elev_mean_m))) for r in hi.itertuples()])
    print("  lowest pixels: ", [(r.pixel_id, round(float(r.elev_mean_m))) for r in lo.itertuples()])


def main() -> None:
    import ee

    parser = argparse.ArgumentParser(description="Per-pixel SRTM elevation for each site.")
    parser.add_argument("--site", help="site key from sites.yaml (default: all sites)")
    parser.add_argument("--force", action="store_true", help="recompute even if up to date")
    args = parser.parse_args()

    load_dotenv(ROOT / ".env")
    project = os.getenv("GEE_PROJECT")
    if not project:
        raise SystemExit("GEE_PROJECT is not set in .env")
    ee.Initialize(project=project)

    sites = load_config()["sites"]
    if args.site:
        if args.site not in sites:
            raise SystemExit(f"Unknown site '{args.site}'. Available: {', '.join(sites)}")
        sites = {args.site: sites[args.site]}
    for key, site in sites.items():
        ingest_site(key, site, args.force)


if __name__ == "__main__":
    main()
