"""ERA5 ingestion: hourly fields -> daily values at every CHIRPS pixel centre, for each
site in config/sites.yaml.

Save this file as src/ingestion/era5.py and run from the repo root (venv active):

    # Trial: one site, one month (do this first)
    python -m src.ingestion.era5 --site nakuru --start 2020-04-01 --end 2020-04-30

    # Full run: all sites, full period from config
    python -m src.ingestion.era5

    # Process files you downloaded by hand from the CDS web form (no API needed):
    #   put them in data/raw/era5_manual/<site>/ (any file names), then
    python -m src.ingestion.era5 --local --site nakuru
    # --local also picks up any files the API runs already cached in data/raw/era5/<site>/

    # Faster: one terminal per site (they use separate caches), e.g.
    python -m src.ingestion.era5 --site nakuru
    python -m src.ingestion.era5 --site busia

What it does
  1. Downloads hourly ERA5 single-level fields for the site's `era5_area` (CDS API),
     one NetCDF file per chunk (default: one quarter), cached in data/raw/era5/<site>/.
  2. Computes relative humidity from 2 m temperature and 2 m dewpoint (Magnus formula),
     because ERA5 single levels has no direct RH field.
  3. Aggregates hourly -> daily (UTC days, the same convention as CHIRPS).
  4. Bilinearly interpolates the ~25 km ERA5 grid onto each ~5 km CHIRPS pixel centre.

Output: data/raw/era5_<site>.parquet in long format
    pixel_id | lat | lon | date | rh_pct | rh_min_pct | rh_max_pct | t2m_c | ... 
pixel_id uses the same format as the CHIRPS files, so the two tables join directly.

The cache file name includes a hash of the download area and variable list, so changing
either in the config can never silently reuse stale downloads.
"""
import argparse
import hashlib
import json
import tempfile
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "config" / "sites.yaml"
RAW = ROOT / "data" / "raw"

DATASET = "reanalysis-era5-single-levels"
# All instantaneous fields (no accumulated ones), so the download is one simple NetCDF.
# Deliberately left out for now: total precipitation (accumulated, different file stream)
# and soil moisture (undefined over lake grid cells, which both boxes contain).
VARIABLES = [
    "2m_temperature",
    "2m_dewpoint_temperature",
    "surface_pressure",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "total_cloud_cover",
]
HOURS = [f"{h:02d}:00" for h in range(24)]
CHIRPS_RES = 0.05  # pixel size we interpolate onto
DAILY_COLUMNS = [
    "rh_pct", "rh_min_pct", "rh_max_pct",
    "t2m_c", "t2m_min_c", "t2m_max_c", "d2m_c",
    "sp_hpa", "wind_ms", "wind_max_ms", "tcc",
]


def load_config() -> dict:
    with open(CONFIG, encoding="utf-8") as f:
        return yaml.safe_load(f)


def make_chunks(start_ts: pd.Timestamp, end_ts: pd.Timestamp, freq: str):
    """Split [start, end] into (chunk_start, chunk_end) pairs, inclusive (freq Y, Q or M)."""
    chunks = []
    for p in pd.period_range(start_ts, end_ts, freq=freq):
        s = max(p.start_time.normalize(), start_ts)
        e = min(p.end_time.normalize(), end_ts)
        chunks.append((s, e))
    return chunks


def build_request(s: pd.Timestamp, e: pd.Timestamp, area: dict) -> dict:
    months = [f"{m:02d}" for m in range(s.month, e.month + 1)]
    if s.month == e.month:
        days = [f"{d:02d}" for d in range(s.day, e.day + 1)]
    else:
        days = [f"{d:02d}" for d in range(1, 32)]  # extra days are trimmed after download
    return {
        "product_type": ["reanalysis"],
        "variable": VARIABLES,
        "year": [str(s.year)],
        "month": months,
        "day": days,
        "time": HOURS,
        "area": [area["north"], area["west"], area["south"], area["east"]],  # N, W, S, E
        "data_format": "netcdf",
        "download_format": "unarchived",
    }


def download_chunk(client, request: dict, target: Path, retries: int = 3) -> None:
    tmp = target.with_suffix(".part")
    last_err = None
    for attempt in range(retries):
        try:
            client.retrieve(DATASET, request, str(tmp))
            tmp.replace(target)
            return
        except Exception as err:  # queue / network / size problems
            last_err = err
            msg = str(err).lower()
            if "too large" in msg or "cost" in msg or "licence" in msg or "license" in msg:
                raise  # retrying cannot fix these; let the caller split the request or stop
            wait = 20 * (attempt + 1)
            print(f"    attempt {attempt + 1} failed ({err}); retrying in {wait}s")
            time.sleep(wait)
    raise RuntimeError(f"Download failed after {retries} attempts: {target.name}") from last_err


def ensure_chunk(client, site_key, sig, area, s, e, cache_dir, allow_split) -> list:
    """Return the cached/downloaded file(s) covering [s, e]. If a long request fails
    (for example the CDS 'request too large' limit), fall back to month-by-month."""
    target = cache_dir / f"era5_{site_key}_{sig}_{s:%Y%m%d}_{e:%Y%m%d}.nc"
    if target.exists():
        return [target]
    print(f"  requesting {s:%Y-%m-%d} to {e:%Y-%m-%d} (queued on the CDS; may take minutes)")
    t0 = time.time()
    try:
        download_chunk(client, build_request(s, e, area), target)
        print(f"    done in {time.time() - t0:.0f}s ({target.stat().st_size / 1e6:.1f} MB)")
        return [target]
    except Exception as err:
        if allow_split and (e - s).days > 31:
            print(f"    long request failed ({err}); splitting into months")
            files = []
            for ms, me in make_chunks(s, e, "M"):
                files += ensure_chunk(client, site_key, sig, area, ms, me, cache_dir, False)
            return files
        raise


def open_any(path: Path) -> xr.Dataset:
    """Open a downloaded file whether it is a plain NetCDF or a zip of NetCDF files."""
    if zipfile.is_zipfile(path):
        with tempfile.TemporaryDirectory() as tmp:
            with zipfile.ZipFile(path) as z:
                z.extractall(tmp)
            parts = []
            for p in sorted(Path(tmp).glob("*.nc")):
                with xr.open_dataset(p) as d:
                    parts.append(d.load())
            return xr.merge(parts, compat="override")
    with xr.open_dataset(path) as d:
        return d.load()


def relative_humidity(t_c, td_c):
    """Magnus formula (August-Roche-Magnus constants); inputs in deg C, output in %."""
    es = np.exp(17.625 * t_c / (243.04 + t_c))
    e = np.exp(17.625 * td_c / (243.04 + td_c))
    return (100.0 * e / es).clip(0, 100)


def to_daily_grid(ds: xr.Dataset, s: pd.Timestamp, e: pd.Timestamp):
    """Hourly ERA5 grid -> daily grid (UTC days). Returns (daily Dataset, hours-per-day Series)."""
    if "valid_time" in ds.coords:  # the current CDS names the time axis valid_time
        ds = ds.rename({"valid_time": "time"})
    if "expver" in ds.dims:  # recent months can mix ERA5 (1) and ERA5T (5)
        ds = ds.sel(expver=1).combine_first(ds.sel(expver=5))
    ds = ds.drop_vars([v for v in ("number", "expver") if v in ds.coords])
    ds = ds.sortby("latitude").sortby("longitude")
    ds = ds.sel(time=slice(s, e + pd.Timedelta(hours=23)))

    t = ds["t2m"] - 273.15
    td = ds["d2m"] - 273.15
    hourly = xr.Dataset({
        "t2m_c": t,
        "d2m_c": td,
        "rh": relative_humidity(t, td),
        "sp_hpa": ds["sp"] / 100.0,
        "wspd": np.hypot(ds["u10"], ds["v10"]),
        "tcc": ds["tcc"],
    })
    res = hourly.resample(time="1D")
    mean, mn, mx = res.mean(), res.min(), res.max()
    daily = xr.Dataset({
        "rh_pct": mean["rh"], "rh_min_pct": mn["rh"], "rh_max_pct": mx["rh"],
        "t2m_c": mean["t2m_c"], "t2m_min_c": mn["t2m_c"], "t2m_max_c": mx["t2m_c"],
        "d2m_c": mean["d2m_c"], "sp_hpa": mean["sp_hpa"],
        "wind_ms": mean["wspd"], "wind_max_ms": mx["wspd"], "tcc": mean["tcc"],
    })
    hours = (hourly["t2m_c"].isel(latitude=0, longitude=0, drop=True)
             .resample(time="1D").count().to_series())
    return daily, hours


def pixel_grid(bbox: dict):
    """CHIRPS pixel centres inside the study box, in the same order/format as chirps.py."""
    n_lon = round((bbox["lon_max"] - bbox["lon_min"]) / CHIRPS_RES)
    n_lat = round((bbox["lat_max"] - bbox["lat_min"]) / CHIRPS_RES)
    lons = np.round(bbox["lon_min"] + CHIRPS_RES / 2 + CHIRPS_RES * np.arange(n_lon), 3)
    lats = np.round(bbox["lat_min"] + CHIRPS_RES / 2 + CHIRPS_RES * np.arange(n_lat), 3)
    lon2, lat2 = np.meshgrid(lons, lats)
    return lat2.ravel(), lon2.ravel()


def interpolate_to_pixels(daily: xr.Dataset, bbox: dict) -> pd.DataFrame:
    lats, lons = pixel_grid(bbox)
    pids = np.array([f"{la:.3f}_{lo:.3f}" for la, lo in zip(lats, lons)])
    at_px = daily.interp(
        latitude=xr.DataArray(lats, dims="pixel"),
        longitude=xr.DataArray(lons, dims="pixel"),
        method="linear",
    ).assign_coords(pixel_id=("pixel", pids))
    df = at_px.to_dataframe().reset_index()
    df = df.rename(columns={"time": "date", "latitude": "lat", "longitude": "lon"})
    df[DAILY_COLUMNS] = df[DAILY_COLUMNS].astype("float32")
    keep = ["pixel_id", "lat", "lon", "date"] + DAILY_COLUMNS
    return df[keep].sort_values(["pixel_id", "date"]).reset_index(drop=True)


def finish_site(site_key, site, daily, hours, start_ts, end_ts, tag, check_gaps=False) -> None:
    """Interpolate the daily grid onto the CHIRPS pixels, save, and print a sanity report."""
    bbox = site["bbox"]
    lats, lons = pixel_grid(bbox)
    if (float(daily.latitude.min()) > lats.min() or float(daily.latitude.max()) < lats.max()
            or float(daily.longitude.min()) > lons.min() or float(daily.longitude.max()) < lons.max()):
        raise SystemExit(
            f"The ERA5 files for '{site_key}' cover lat {float(daily.latitude.min())}..{float(daily.latitude.max())}, "
            f"lon {float(daily.longitude.min())}..{float(daily.longitude.max())}, which does not contain the study "
            f"box (pixel centres lat {lats.min()}..{lats.max()}, lon {lons.min()}..{lons.max()}). "
            "Check you downloaded the right area for this site and did not mix up Nakuru and Busia files.")
    df = interpolate_to_pixels(daily, bbox)

    out = RAW / f"era5_{site_key}{tag}.parquet"
    df.to_parquet(out, index=False)

    n_pix, n_days = df["pixel_id"].nunique(), df["date"].nunique()
    expected_days = (end_ts - start_ts).days + 1
    print(f"  saved {out}")
    print(f"  pixels: {n_pix} | days: {n_days} (expected {expected_days}) | rows: {len(df)} "
          f"(expected {n_pix * n_days})")
    print(f"  hours per day: min {hours.min()}, max {hours.max()} (should both be 24)")
    if check_gaps:
        expected = pd.date_range(start_ts, end_ts, freq="D")
        missing = expected.difference(pd.DatetimeIndex(hours.index))
        partial = hours[hours != 24]
        if len(missing) == 0 and len(partial) == 0:
            print("  coverage: complete, every day in the range has all 24 hours")
        else:
            print(f"  COVERAGE GAPS: {len(missing)} missing days, {len(partial)} days with fewer than 24 hours")
            by_month = pd.Series(1, index=missing.to_period("M")).groupby(level=0).sum()
            for per, n in by_month.items():
                print(f"    {per}: {n} missing days")
            if len(partial):
                print(f"    incomplete days (first 10): {[d.strftime('%Y-%m-%d') for d in partial.index[:10]]}")
            print("    -> download the missing months, put the files in the same folder, and run again")
    miss = df[DAILY_COLUMNS].isna().sum()
    print(f"  missing values: {int(miss.sum())}" + ("" if miss.sum() == 0 else f"  by column: {miss[miss > 0].to_dict()}"))
    print(f"  relative humidity: mean {df['rh_pct'].mean():.1f}%  "
          f"(min {df['rh_pct'].min():.1f}, max {df['rh_pct'].max():.1f})")
    print(f"  temperature: mean {df['t2m_c'].mean():.1f} C  "
          f"(min {df['t2m_min_c'].min():.1f}, max {df['t2m_max_c'].max():.1f})")
    print(f"  pressure: mean {df['sp_hpa'].mean():.0f} hPa | wind: mean {df['wind_ms'].mean():.1f} m/s")
    print(f"  lat range {df['lat'].min()}..{df['lat'].max()}, lon range {df['lon'].min()}..{df['lon'].max()}")

    chirps_file = RAW / f"chirps_{site_key}{tag}.parquet"
    if chirps_file.exists():
        cp = set(pd.read_parquet(chirps_file, columns=["pixel_id"])["pixel_id"].unique())
        print(f"  pixel ids identical to CHIRPS file: {cp == set(df['pixel_id'].unique())}")


def ingest_site(client, site_key: str, site: dict, start: str, end: str, tag: str,
                chunk: str) -> None:
    area = site["era5_area"]
    sig = hashlib.md5(json.dumps(
        {"area": area, "vars": VARIABLES, "hours": HOURS}, sort_keys=True).encode()
    ).hexdigest()[:8]
    cache_dir = RAW / "era5" / site_key
    cache_dir.mkdir(parents=True, exist_ok=True)
    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)

    print(f"\n=== {site_key}: {site['name']} | {start} to {end} | area N,W,S,E = "
          f"{area['north']}, {area['west']}, {area['south']}, {area['east']} ===")
    dailies, hour_counts = [], []
    freq = {"year": "Y", "quarter": "Q", "month": "M"}[chunk]
    for s, e in make_chunks(start_ts, end_ts, freq):
        files = ensure_chunk(client, site_key, sig, area, s, e, cache_dir, chunk != "month")
        for f in files:
            fs = max(s, pd.Timestamp(f.stem.split("_")[-2]))
            fe = min(e, pd.Timestamp(f.stem.split("_")[-1]))
            d, h = to_daily_grid(open_any(f), fs, fe)
            dailies.append(d)
            hour_counts.append(h)
        print(f"  processed {s:%Y-%m-%d} to {e:%Y-%m-%d}")

    daily = xr.concat(dailies, dim="time").sortby("time")
    hours = pd.concat(hour_counts).sort_index()
    finish_site(site_key, site, daily, hours, start_ts, end_ts, tag)


def ingest_local(site_key: str, site: dict, start: str, end: str, tag: str) -> None:
    """Process NetCDF/zip files that were downloaded by hand (or cached by earlier API runs)."""
    manual = RAW / "era5_manual" / site_key
    manual.mkdir(parents=True, exist_ok=True)
    files = sorted(list(manual.glob("*.nc")) + list(manual.glob("*.zip")))
    api_cache = RAW / "era5" / site_key
    if api_cache.exists():
        files += sorted(api_cache.glob("*.nc"))
    if not files:
        raise SystemExit(f"No .nc or .zip files found in {manual} (or {api_cache}).")
    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)

    print(f"\n=== {site_key}: {site['name']} | LOCAL FILES | {start} to {end} ===")
    dailies, hour_counts = [], []
    for f in files:
        ds = open_any(f)
        tname = "valid_time" if "valid_time" in ds.coords else "time"
        times = pd.to_datetime(ds[tname].values)
        fs = max(start_ts, times.min().normalize())
        fe = min(end_ts, times.max().normalize())
        print(f"  {f.name}: {times.min():%Y-%m-%d} to {times.max():%Y-%m-%d}")
        if fs > fe:
            print("    outside the requested period, skipped")
            continue
        d, h = to_daily_grid(ds, fs, fe)
        dailies.append(d)
        hour_counts.append(h)
    if not dailies:
        raise SystemExit("None of the files overlap the requested period.")

    daily = xr.concat(dailies, dim="time").sortby("time")
    _, keep = np.unique(daily["time"].values, return_index=True)  # drop days duplicated by overlapping files
    daily = daily.isel(time=np.sort(keep))
    hours = pd.concat(hour_counts)
    hours = hours[~hours.index.duplicated()].sort_index()
    finish_site(site_key, site, daily, hours, start_ts, end_ts, tag, check_gaps=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest ERA5 daily fields per CHIRPS pixel.")
    parser.add_argument("--site", help="site key from sites.yaml (default: all sites)")
    parser.add_argument("--start", help="YYYY-MM-DD (default: config period start)")
    parser.add_argument("--end", help="YYYY-MM-DD (default: config period end)")
    parser.add_argument("--chunk", choices=["year", "quarter", "month"], default="quarter",
                        help="request size. The CDS rejected whole-year requests as too large, so the "
                             "default is 'quarter'; anything too large auto-splits into months")
    parser.add_argument("--local", action="store_true",
                        help="do not call the CDS; process files in data/raw/era5_manual/<site>/ "
                             "(and any already cached in data/raw/era5/<site>/)")
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

    if args.local:
        for key, site in sites.items():
            ingest_local(key, site, start, end, tag)
        return

    import cdsapi  # imported here so the module can be imported (and tested) without it
    client = cdsapi.Client()
    for key, site in sites.items():
        ingest_site(client, key, site, start, end, tag, args.chunk)


if __name__ == "__main__":
    main()
