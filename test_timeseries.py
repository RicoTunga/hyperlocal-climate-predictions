"""Diagnostic for the CDS dataset reanalysis-era5-single-levels-timeseries.
Downloads one site's area for a short date range, then prints the file's structure so we
can write the real ingestion code around it. Run from the repo root (venv active):

    python test_timeseries.py                                   # Nakuru, April 2020 (quick)
    python test_timeseries.py --site busia
    python test_timeseries.py --start 2016-01-01 --end 2025-12-31   # full period, to test size limits

Paste the whole printout back to Claude.
"""
import argparse
import tempfile
import time
import zipfile
from pathlib import Path

import cdsapi
import xarray as xr
import yaml

ROOT = Path(__file__).resolve().parent
DATASET = "reanalysis-era5-single-levels-timeseries"
VARIABLES = [
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "2m_dewpoint_temperature",
    "2m_temperature",
    "surface_pressure",
    "total_cloud_cover",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", default="nakuru")
    ap.add_argument("--start", default="2020-04-01")
    ap.add_argument("--end", default="2020-04-30")
    args = ap.parse_args()

    with open(ROOT / "config" / "sites.yaml", encoding="utf-8") as f:
        a = yaml.safe_load(f)["sites"][args.site]["era5_area"]
    request = {
        "variable": VARIABLES,
        "date": [f"{args.start}/{args.end}"],
        "data_format": "netcdf",
        "area": [a["north"], a["west"], a["south"], a["east"]],  # N, W, S, E
    }
    print("Request:", request)

    out_dir = ROOT / "data" / "raw" / "era5_ts_test"
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / f"ts_{args.site}_{args.start}_{args.end}.nc"

    t0 = time.time()
    cdsapi.Client().retrieve(DATASET, request, str(target))
    print(f"\nDownloaded in {time.time() - t0:.0f}s -> {target} ({target.stat().st_size / 1e6:.2f} MB)")

    path = target
    if zipfile.is_zipfile(target):
        print("File is a ZIP; contents:")
        tmp = Path(tempfile.mkdtemp())
        with zipfile.ZipFile(target) as z:
            print("  ", z.namelist())
            z.extractall(tmp)
        path = sorted(tmp.glob("*.nc"))[0]
        print("Inspecting first file:", path.name)

    with xr.open_dataset(path) as ds:
        print("\n=== Dataset structure ===")
        print(ds)
        print("\nDimensions:", dict(ds.sizes))
        for name in ds.data_vars:
            print(f"  {name}: dims={ds[name].dims}, NaNs={int(ds[name].isnull().sum())}")
        for c in ("latitude", "longitude", "lat", "lon"):
            if c in ds.coords:
                v = ds[c].values
                print(f"  {c}: {v.size} values, {v.min()} to {v.max()}")
        for c in ("valid_time", "time"):
            if c in ds.coords:
                v = ds[c].values
                print(f"  {c}: {v.size} steps, {v.min()} to {v.max()}")


if __name__ == "__main__":
    main()
