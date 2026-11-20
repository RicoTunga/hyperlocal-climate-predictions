"""CHIRPS daily rainfall ingestion, one time series per ~5 km pixel, for every site
in config/sites.yaml.

Save this file as src/ingestion/chirps.py and run from the repo root (venv active):

    # Trial: one site, one month (do this first)
    python -m src.ingestion.chirps --site nakuru --start 2020-04-01 --end 2020-04-30

    # Full run: all sites, full period from config
    python -m src.ingestion.chirps

    # Recent months that Earth Engine has not ingested yet: fall back to the CHC server
    # (final product), and allow the PRELIMINARY product for days without a final file:
    python -m src.ingestion.chirps --allow-prelim

Output: data/raw/chirps_<site>.parquet in long format
    pixel_id | lat | lon | date | precip_mm
plus data/raw/chirps_<site>_sources.json: which dates came from the preliminary product.
Monthly chunks are cached in data/raw/chirps/<site>/ so an interrupted run resumes
where it stopped.

Data version: the Earth Engine collection UCSB-CHG/CHIRPS/DAILY is CHIRPS **v2.0** (final).
The CHC server fallback reads the same product, v2.0, as daily GeoTIFFs:
    final:       products/CHIRPS/v2.0/africa_daily/tifs/p05/<year>/chirps-v2.0.<Y>.<M>.<D>.tif.gz
    preliminary: products/CHIRPS/v2.0/prelim/global_daily/tifs/p05/<year>/... (same names)
Final files are cached in data/raw/chirps_chc/final/. Preliminary files are NEVER cached and
preliminary values are NEVER written to the monthly cache: they are replaced by the final product
later, and a cached copy would be reused silently (the cache is keyed by dates only). Months
that contain any preliminary day are therefore refetched on every run until final data exists.
Check before first use: `python -m src.ingestion.chirps_check` (CHC final == Earth Engine).
"""
import argparse
import gzip
import json
import os
import re
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "config" / "sites.yaml"
RAW = ROOT / "data" / "raw"
COLLECTION = "UCSB-CHG/CHIRPS/DAILY"
BAND = "precipitation"
BAND_NAME = re.compile(r"^(\d{8})_" + BAND + "$")  # toBands() names bands <date>_<band>

CHC_BASE = "https://data.chc.ucsb.edu/products/CHIRPS/v2.0"
CHC_URLS = {
    "final": CHC_BASE + "/africa_daily/tifs/p05/{y}/chirps-v2.0.{y}.{m:02d}.{d:02d}.tif.gz",
    "prelim": CHC_BASE + "/prelim/global_daily/tifs/p05/{y}/chirps-v2.0.{y}.{m:02d}.{d:02d}.tif.gz",
}
CHC_CACHE = RAW / "chirps_chc" / "final"
RES = 0.05


def load_config() -> dict:
    with open(CONFIG, encoding="utf-8") as f:
        return yaml.safe_load(f)


def pixel_centres(bbox: dict):
    """CHIRPS pixel centres inside the study box: (lats, lons, pixel_ids), in the format that the
    Earth Engine route produces (identical ids; checked in src.ingestion.chirps_check)."""
    n_lon = round((bbox["lon_max"] - bbox["lon_min"]) / RES)
    n_lat = round((bbox["lat_max"] - bbox["lat_min"]) / RES)
    lons = np.round(bbox["lon_min"] + RES / 2 + RES * np.arange(n_lon), 3)
    lats = np.round(bbox["lat_min"] + RES / 2 + RES * np.arange(n_lat), 3)
    lon2, lat2 = np.meshgrid(lons, lats)
    lat, lon = lat2.ravel(), lon2.ravel()
    return lat, lon, np.array([f"{a:.3f}_{o:.3f}" for a, o in zip(lat, lon)])


def chc_download(kind: str, day: pd.Timestamp, retries: int = 6) -> Path | None:
    """Path to the gzipped GeoTIFF for one day, or None if the CHC has not published it (404).
    Final files are cached; preliminary files go to a temporary file every time."""
    import requests
    url = CHC_URLS[kind].format(y=day.year, m=day.month, d=day.day)
    if kind == "final":
        CHC_CACHE.mkdir(parents=True, exist_ok=True)
        target = CHC_CACHE / url.rsplit("/", 1)[1]
        if target.exists():
            return target
    else:
        target = Path(tempfile.gettempdir()) / f"hl_prelim_{url.rsplit('/', 1)[1]}"
    last = None
    for attempt in range(retries):
        try:
            time.sleep(1.0)  # the CHC server answers 429 (too many requests) to rapid downloads
            r = requests.get(url, timeout=120)
            if r.status_code == 404:
                return None
            r.raise_for_status()
            gzip.decompress(r.content)  # raises if the file is truncated
            tmp = target.with_suffix(".part")
            tmp.write_bytes(r.content)
            tmp.replace(target)
            return target
        except Exception as err:  # network hiccups, truncated files
            last = err
            print(f"    retry {url.rsplit('/', 1)[1]} in {10 * (attempt + 1)}s ({err})")
            time.sleep(10 * (attempt + 1))
    raise RuntimeError(f"Could not download {url}") from last


def read_tif_at(path: Path, lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    """Values of a CHIRPS GeoTIFF (gzipped) at the given pixel centres; negative = no data -> NaN.
    Row/column come from the grid origin and the exact 0.05 degree step: pixel centres lie half a
    cell from every edge, so floor() is safe from rounding in the file's float32 resolution."""
    import rasterio
    with tempfile.TemporaryDirectory() as tmp:
        tif = Path(tmp) / "day.tif"
        with gzip.open(path, "rb") as src, open(tif, "wb") as dst:
            shutil.copyfileobj(src, dst)
        with rasterio.open(tif) as ds:
            left, top = ds.bounds.left, ds.bounds.top
            cols = np.floor((lons - left) / RES).astype(int)
            rows = np.floor((top - lats) / RES).astype(int)
            if rows.min() < 0 or cols.min() < 0 or rows.max() >= ds.height or cols.max() >= ds.width:
                raise SystemExit(f"{path.name} does not cover the study box")
            r0, r1, c0, c1 = rows.min(), rows.max() + 1, cols.min(), cols.max() + 1
            win = ds.read(1, window=((r0, r1), (c0, c1)))
    vals = win[rows - r0, cols - c0].astype("float64")
    vals[vals < 0] = np.nan
    return vals


def fetch_chc(bbox: dict, s: pd.Timestamp, e: pd.Timestamp, allow_prelim: bool):
    """Every pixel for days [s, e) from the CHC server. Returns (DataFrame, list of preliminary
    dates); a day with neither a final nor (if allowed) a preliminary file stops the run."""
    lat, lon, pid = pixel_centres(bbox)
    parts, prelim = [], []
    for day in pd.date_range(s, e - pd.Timedelta(days=1)):
        path, kind = chc_download("final", day), "final"
        if path is None and allow_prelim:
            path, kind = chc_download("prelim", day), "prelim"
        if path is None:
            return None, prelim
        vals = read_tif_at(path, lat, lon)
        if kind == "prelim":
            prelim.append(day)
            path.unlink(missing_ok=True)
        # Earth Engine returns precipitation as float32; match it exactly
        parts.append(pd.DataFrame({"pixel_id": pid, "lat": lat, "lon": lon, "date": day,
                                   "precip_mm": vals.astype("float32").astype("float64")}))
    return pd.concat(parts, ignore_index=True), prelim


def fetch_chunk(region, proj, start: str, end: str,
                retries: int = 4) -> pd.DataFrame:
    """Fetch every pixel in `region` for [start, end) as a long DataFrame."""
    import ee
    images = ee.ImageCollection(COLLECTION).select(BAND).filterDate(start, end)
    if images.size().getInfo() == 0:  # not ingested by Earth Engine yet
        return pd.DataFrame(columns=["pixel_id", "lat", "lon", "date", "precip_mm"])
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


_EE_STATE = {"tried": False, "ok": False}


def ee_ready() -> bool:
    """Initialise Earth Engine the first time a chunk has to be fetched (cached months need no
    account). If it cannot be initialised, say why (without printing the project id) and let the
    caller use the CHC server, which gives identical values (src.ingestion.chirps_check)."""
    if not _EE_STATE["tried"]:
        _EE_STATE["tried"] = True
        import ee
        load_dotenv(ROOT / ".env")
        project = (os.getenv("GEE_PROJECT") or "").strip()
        try:
            if not project:
                raise ValueError("GEE_PROJECT is not set in .env")
            ee.Initialize(project=project)
            _EE_STATE["ok"] = True
        except Exception as err:  # noqa: BLE001 - any failure means: use the CHC route
            print(f"  Earth Engine unavailable ({type(err).__name__}; check GEE_PROJECT in .env); using the CHC server")
    return _EE_STATE["ok"]


def ingest_site(site_key: str, site: dict, start: str, end: str, tag: str,
                allow_prelim: bool = False) -> None:
    b = site["bbox"]
    region = proj = None  # Earth Engine objects, created only when a chunk must be fetched

    cache_dir = RAW / "chirps" / site_key
    cache_dir.mkdir(parents=True, exist_ok=True)
    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    chunks, prelim_dates, chc_dates = [], [], []

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
        if region is None and ee_ready():
            import ee
            region = ee.Geometry.Rectangle([b["lon_min"], b["lat_min"], b["lon_max"], b["lat_max"]])
            proj = ee.Image(ee.ImageCollection(COLLECTION).first()).select(BAND).projection()
        df = fetch_chunk(region, proj, f"{s:%Y-%m-%d}", f"{e:%Y-%m-%d}") if region is not None else None
        prelim = []
        if df is None or not chunk_complete(df, n_days):
            # Earth Engine lags the CHC server (or is unavailable): take the whole chunk from the CHC
            # instead (never a mix of the two routes within one chunk)
            have = "not available" if df is None else f"{df['date'].nunique()} of {n_days} days"
            print(f"    Earth Engine: {have}; using the CHC server")
            df, prelim = fetch_chc(b, s, e, allow_prelim)
            if df is None or not chunk_complete(df, n_days):
                raise SystemExit(
                    f"CHIRPS has no {'final or preliminary' if allow_prelim else 'final'} data for every day of "
                    f"{s:%Y-%m-%d}..{e:%Y-%m-%d} (not published yet?). Not cached. Set an earlier period "
                    "end in config/sites.yaml, wait for CHIRPS" + ("" if allow_prelim else ", or use --allow-prelim") + ".")
            chc_dates += list(pd.date_range(s, e - pd.Timedelta(days=1)))
        if prelim:
            # never cache a chunk with preliminary days: it would be reused after the final arrives
            print(f"    {len(prelim)} PRELIMINARY day(s) ({prelim[0]:%Y-%m-%d}..{prelim[-1]:%Y-%m-%d}); chunk not cached")
            prelim_dates += prelim
        else:
            df.to_parquet(cache, index=False)
        chunks.append(df)

    full = (pd.concat(chunks, ignore_index=True)
            .sort_values(["pixel_id", "date"]).reset_index(drop=True))
    out = RAW / f"chirps_{site_key}{tag}.parquet"
    full.to_parquet(out, index=False)
    sources = {
        "product": "CHIRPS v2.0 daily",
        "written": pd.Timestamp.now().isoformat(timespec="seconds"),
        "period": [start, end],
        "chc_server_dates": [f"{d:%Y-%m-%d}" for d in chc_dates],
        "prelim_dates": [f"{d:%Y-%m-%d}" for d in prelim_dates],
        "final_through": f"{(min(prelim_dates) - pd.Timedelta(days=1)) if prelim_dates else end_ts:%Y-%m-%d}",
    }
    (RAW / f"chirps_{site_key}{tag}_sources.json").write_text(json.dumps(sources, indent=2), encoding="utf-8")
    if prelim_dates:
        print(f"  PRELIMINARY CHIRPS for {len(prelim_dates)} days: {sources['prelim_dates'][0]}..{sources['prelim_dates'][-1]}")

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
    parser.add_argument("--allow-prelim", action="store_true",
                        help="use the CHC PRELIMINARY product for days without final data (never cached)")
    args = parser.parse_args()

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
        ingest_site(key, site, start, end, tag, args.allow_prelim)


if __name__ == "__main__":
    main()
