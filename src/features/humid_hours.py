"""Hours per day with relative humidity >= 90%, per pixel, from the hourly ERA5 files already on disk.

Run from the repo root (no download; reads data/raw/era5_manual/<site>/*.nc):

    python -m src.features.humid_hours --site nakuru

Needed for the Hutton criteria for potato late blight (two consecutive days, each with minimum
temperature >= 10 C and at least 6 hours at RH >= 90%). The main merged table only has daily
mean/min/max humidity, so the count is computed here and saved in a SEPARATE file. It is kept out of
<site>_daily.parquet and the feature tables on purpose: adding a column there would change the
input set of every saved model and break the reproducibility checks of the earlier results.

Method (the same path as src.ingestion.era5 for the existing columns): hourly 2 m temperature and
dewpoint -> hourly RH (Magnus formula) on the ERA5 grid -> count of hours >= 90% per UTC day on the
grid -> bilinear interpolation to each CHIRPS pixel centre (so counts can be fractional).
Check: the daily mean RH computed the same way must match rh_pct in the merged table.

Output: data/processed/<site>_humid_hours.parquet  (pixel_id, date, rh_hours_ge90)
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import yaml

from src.ingestion.era5 import interpolate_to_pixels, open_any, relative_humidity

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "config" / "sites.yaml"
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
RH_THRESHOLD = 90.0


def hourly_clean(ds: xr.Dataset, s: pd.Timestamp, e: pd.Timestamp) -> xr.Dataset:
    """The same cleaning steps as src.ingestion.era5.to_daily_grid."""
    if "valid_time" in ds.coords:
        ds = ds.rename({"valid_time": "time"})
    if "expver" in ds.dims:
        ds = ds.sel(expver=1).combine_first(ds.sel(expver=5))
    ds = ds.drop_vars([v for v in ("number", "expver") if v in ds.coords])
    ds = ds.sortby("latitude").sortby("longitude")
    return ds.sel(time=slice(s, e + pd.Timedelta(hours=23)))


def daily_humid_hours(ds: xr.Dataset) -> xr.Dataset:
    rh = relative_humidity(ds["t2m"] - 273.15, ds["d2m"] - 273.15)
    res = rh.resample(time="1D")
    # the daily mean is returned too, only for the consistency check against rh_pct
    return xr.Dataset({"rh_hours_ge90": (rh >= RH_THRESHOLD).resample(time="1D").sum().astype("float32"),
                       "rh_pct_check": res.mean()})


def main() -> None:
    parser = argparse.ArgumentParser(description="Hours per day with RH >= 90% per pixel (from local ERA5).")
    parser.add_argument("--site", required=True)
    args = parser.parse_args()
    with open(CONFIG, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    site = cfg["sites"][args.site]
    s, e = pd.Timestamp(str(cfg["period"]["start"])), pd.Timestamp(str(cfg["period"]["end"]))
    files = sorted((RAW / "era5_manual" / args.site).glob("*.nc")) + sorted((RAW / "era5_manual" / args.site).glob("*.zip"))
    if not files:
        raise SystemExit(f"No ERA5 files in data/raw/era5_manual/{args.site}/")
    print(f"=== {args.site}: {[f.name for f in files]}, period {s:%Y-%m-%d}..{e:%Y-%m-%d} ===")

    parts = []
    for f in files:
        daily = daily_humid_hours(hourly_clean(open_any(f), s, e))
        parts.append(daily)
    daily = xr.concat(parts, dim="time").sortby("time")
    _, keep = np.unique(daily["time"].values, return_index=True)
    daily = daily.isel(time=np.sort(keep))

    # reuse the pipeline's interpolation (it expects the DAILY_COLUMNS names; pass ours explicitly)
    import src.ingestion.era5 as era5
    cols = era5.DAILY_COLUMNS
    era5.DAILY_COLUMNS = ["rh_hours_ge90", "rh_pct_check"]
    try:
        df = interpolate_to_pixels(daily, site["bbox"])
    finally:
        era5.DAILY_COLUMNS = cols

    merged = pd.read_parquet(PROCESSED / f"{args.site}_daily.parquet", columns=["pixel_id", "date", "rh_pct"])
    chk = merged.merge(df, on=["pixel_id", "date"], how="outer", indicator=True)
    same_rows = (chk["_merge"] == "both").all()
    err = float((chk["rh_pct"] - chk["rh_pct_check"]).abs().max())
    print(f"  {'PASS ' if same_rows else 'CHECK'} same pixels and days as the merged table ({len(df):,} rows)")
    print(f"  {'PASS ' if err < 1e-3 else 'CHECK'} daily mean RH recomputed here matches rh_pct (max difference {err:.2g})")
    if not same_rows or err >= 1e-3:
        raise SystemExit("Consistency check failed; nothing saved.")

    out = PROCESSED / f"{args.site}_humid_hours.parquet"
    df[["pixel_id", "date", "rh_hours_ge90"]].to_parquet(out, index=False)
    h = df["rh_hours_ge90"]
    print(f"  saved {out}: hours >= 90% per day: mean {h.mean():.1f}, median {h.median():.1f}, "
          f"days with >= 6 h: {(h >= 6).mean():.1%}, range {h.min():.1f}-{h.max():.1f}")


if __name__ == "__main__":
    main()
