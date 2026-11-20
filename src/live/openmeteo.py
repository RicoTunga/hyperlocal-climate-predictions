"""Open-Meteo client for the live branch: hourly ECMWF IFS 0.25 degree weather at ERA5's own grid nodes.

Why this source and model (docs/NEXT_STEPS.md section 1):
- `ecmwf_ifs025` answers at exactly the 0.25 degree grid points ERA5 uses (checked 20 Nov 2026: a request
  for -0.25, 36.0 returns -0.25, 36.0), so the existing pipeline is reused unchanged: hourly ->
  daily UTC values (src.ingestion.era5.to_daily_grid) -> bilinear interpolation onto the CHIRPS pixels.
- `elevation=nan` switches off Open-Meteo's statistical downscaling to a 90 m terrain model, so values
  are grid-cell values like ERA5's (temperature and pressure would otherwise be shifted to the point's
  height).
- the same model is served live (forecast API, `past_days` up to 92) and archived (historical-forecast
  API, from 2024-02-03; dew point complete from March 2024), so the backtest and the live job read the
  same product.

Licence and limits: free for non-commercial use, data CC BY 4.0 ("Weather data by Open-Meteo.com"
must be shown next to the data); under 10,000 calls a day, 5,000 an hour, 600 a minute, where a request
counts as one call per location, per 14 days and per 10 variables. No API key is used.

Output of fetch_grid(): an xarray Dataset (time, latitude, longitude) with ERA5's names and units:
t2m and d2m in K, sp in Pa, u10 = wind speed (m/s) and v10 = 0 (to_daily_grid only uses their
hypotenuse), tcc as a fraction 0-1, and tp = precipitation (mm per hour).
"""
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import xarray as xr

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "data" / "raw" / "openmeteo"
MODEL = "ecmwf_ifs025"
ENDPOINTS = {"forecast": "https://api.open-meteo.com/v1/forecast",
             "archive": "https://historical-forecast-api.open-meteo.com/v1/forecast"}
HOURLY = ["temperature_2m", "dew_point_2m", "surface_pressure", "wind_speed_10m", "cloud_cover", "precipitation"]
CALLS_PER_MINUTE = 450  # stay below the free limit of 600
ATTRIBUTION = "Weather data by Open-Meteo.com (ECMWF IFS 0.25°, CC BY 4.0)"


def grid_nodes(area: dict, step: float = 0.25) -> tuple[np.ndarray, np.ndarray]:
    """ERA5 grid nodes of a site's era5_area (config/sites.yaml), south to north and west to east."""
    lats = np.round(np.arange(area["south"], area["north"] + step / 2, step), 4)
    lons = np.round(np.arange(area["west"], area["east"] + step / 2, step), 4)
    return lats, lons


def call_weight(n_locations: int, n_days: int, n_vars: int = len(HOURLY)) -> int:
    """How many calls Open-Meteo counts for one request (its stated rule)."""
    return n_locations * math.ceil(n_days / 14) * math.ceil(n_vars / 10)


def _request(endpoint: str, lats, lons, params: dict, retries: int = 4) -> list:
    q = {"latitude": ",".join(f"{a:g}" for a in lats), "longitude": ",".join(f"{o:g}" for o in lons),
         "hourly": ",".join(HOURLY), "models": MODEL, "timezone": "GMT", "wind_speed_unit": "ms",
         "elevation": ",".join(["nan"] * len(lats)), **params}
    last = None
    for attempt in range(retries):
        try:
            r = requests.get(ENDPOINTS[endpoint], params=q, timeout=180)
            if r.status_code == 429:
                raise RuntimeError("rate limited (429)")
            j = r.json()
            if isinstance(j, dict) and j.get("error"):
                raise ValueError(j.get("reason"))
            return j if isinstance(j, list) else [j]
        except ValueError:
            raise  # a bad request does not get better by retrying
        except Exception as err:  # network, 429, server errors
            last = err
            wait = 30 * (attempt + 1)
            print(f"    Open-Meteo request failed ({err}); retrying in {wait}s")
            time.sleep(wait)
    raise RuntimeError("Open-Meteo request failed") from last


def to_dataset(responses: list, lats, lons) -> xr.Dataset:
    """Responses for every (lat, lon) node -> Dataset in ERA5 names and units."""
    times = pd.to_datetime(responses[0]["hourly"]["time"])
    shape = (len(times), len(lats), len(lons))
    arr = {v: np.full(shape, np.nan, dtype="float32") for v in HOURLY}
    k = 0
    for i, la in enumerate(lats):
        for j, lo in enumerate(lons):
            r = responses[k]
            k += 1
            if abs(r["latitude"] - la) > 0.01 or abs(r["longitude"] - lo) > 0.01:
                raise SystemExit(f"Open-Meteo answered for {r['latitude']}, {r['longitude']} instead of {la}, {lo}: "
                                 "the model grid is not ERA5's; do not mix them")
            for v in HOURLY:
                arr[v][:, i, j] = np.array(r["hourly"][v], dtype="float32")  # None -> nan
    dims = ("time", "latitude", "longitude")
    return xr.Dataset({
        "t2m": (dims, arr["temperature_2m"] + np.float32(273.15)),
        "d2m": (dims, arr["dew_point_2m"] + np.float32(273.15)),
        "sp": (dims, arr["surface_pressure"] * np.float32(100.0)),
        "u10": (dims, arr["wind_speed_10m"]), "v10": (dims, np.zeros(shape, dtype="float32")),
        "tcc": (dims, arr["cloud_cover"] / np.float32(100.0)),
        "tp": (dims, arr["precipitation"]),
    }, coords={"time": times, "latitude": np.asarray(lats, float), "longitude": np.asarray(lons, float)})


def fetch_grid(area: dict, start=None, end=None, endpoint: str = "archive", past_days: int | None = None,
               cache_key: str | None = None) -> xr.Dataset:
    """Hourly IFS 0.25 data for every ERA5 node of `area`.

    Archive: dates [start, end], fetched month by month and cached in data/raw/openmeteo/<cache_key>/.
    Live: endpoint="forecast" with past_days (the last days up to now plus today), never cached."""
    lats, lons = grid_nodes(area)
    la2, lo2 = np.meshgrid(lats, lons, indexing="ij")
    flat_la, flat_lo = la2.ravel(), lo2.ravel()
    if endpoint == "forecast":
        w = call_weight(len(flat_la), past_days + 1)
        print(f"  Open-Meteo live request: {len(flat_la)} nodes, last {past_days} days (~{w} calls)")
        return to_dataset(_request("forecast", flat_la, flat_lo, {"past_days": past_days, "forecast_days": 1}), lats, lons)
    parts = []
    folder = CACHE / (cache_key or "grid")
    folder.mkdir(parents=True, exist_ok=True)
    for p in pd.period_range(pd.Timestamp(start), pd.Timestamp(end), freq="M"):
        s, e = max(p.start_time, pd.Timestamp(start)), min(p.end_time.normalize(), pd.Timestamp(end))
        f = folder / f"{MODEL}_{s:%Y%m%d}_{e:%Y%m%d}.nc"
        if f.exists():
            with xr.open_dataset(f) as d:
                parts.append(d.load())
            continue
        w = call_weight(len(flat_la), (e - s).days + 1)
        print(f"  Open-Meteo archive {s:%Y-%m-%d}..{e:%Y-%m-%d}: {len(flat_la)} nodes (~{w} calls)")
        ds = to_dataset(_request("archive", flat_la, flat_lo, {"start_date": f"{s:%Y-%m-%d}", "end_date": f"{e:%Y-%m-%d}"}),
                        lats, lons)
        ds.attrs.update(source=ATTRIBUTION, fetched=pd.Timestamp.now().isoformat(timespec="seconds"))
        ds.to_netcdf(f)
        parts.append(ds)
        time.sleep(60 * w / CALLS_PER_MINUTE)  # keep under the per-minute limit
    return xr.concat(parts, dim="time")


def write_meta(path: Path, info: dict) -> None:
    path.write_text(json.dumps(info, indent=2, default=str), encoding="utf-8")
