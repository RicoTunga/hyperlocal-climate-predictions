"""Smoke test for data access. Run from the repo root: python test_access.py
Checks (1) Google Earth Engine / CHIRPS and (2) Copernicus CDS / ERA5.
Needs GEE_PROJECT set in your .env file (copy .env.example to .env first)."""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Nakuru bounding box from the proposal
LON_MIN, LON_MAX, LAT_MIN, LAT_MAX = 35.50, 36.65, -1.10, 0.30


def test_earth_engine() -> None:
    import ee

    project = os.getenv("GEE_PROJECT")
    if not project:
        raise SystemExit("GEE_PROJECT is not set in .env")
    ee.Initialize(project=project)
    box = ee.Geometry.Rectangle([LON_MIN, LAT_MIN, LON_MAX, LAT_MAX])
    img = (
        ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY")
        .filterDate("2020-04-01", "2020-04-02")
        .first()
    )
    mean = img.reduceRegion(ee.Reducer.mean(), box, 5566).getInfo()
    print("Earth Engine OK. CHIRPS mean precip over Nakuru box, 2020-04-01:", mean)


def test_cds() -> None:
    import cdsapi

    out = Path("data/raw/cds_test.nc")
    out.parent.mkdir(parents=True, exist_ok=True)
    cdsapi.Client().retrieve(
        "reanalysis-era5-single-levels",
        {
            "product_type": "reanalysis",
            "variable": ["2m_temperature"],
            "year": "2020",
            "month": "04",
            "day": "01",
            "time": ["12:00"],
            "area": [LAT_MAX, LON_MIN, LAT_MIN, LON_MAX],  # N, W, S, E
            "data_format": "netcdf",
            "download_format": "unarchived",
        },
        str(out),
    )
    print(f"CDS OK. Downloaded {out} ({out.stat().st_size} bytes)")


if __name__ == "__main__":
    test_earth_engine()
    test_cds()
