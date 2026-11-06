"""Per-pixel time-series features for forecasting: one row per pixel per issue day t.

Run from the repo root (venv active):

    python -m src.features.build                 # all sites
    python -m src.features.build --site busia

Input:  data/processed/<site>_daily.parquet   (from src.features.merge)
Output: data/processed/features_<site>.parquet
        the 21 merged columns (the values on day t itself) + the features below.

Time rule (AGENTS.md rule 1): every feature on row (pixel, t) uses only data from day t and
earlier. A model for horizon h is trained to predict the target on day t + h from row t, so no
feature contains the target day. Rolling windows END on day t (precip_mm_mean7 = mean of days
t-6..t). Early 2016 rows are NaN where a window is not yet full (warm-up); tree models accept
NaN, or the modelling step can drop those rows.

Features (settings in config/experiment.yaml, section `features`):
- rainfall: lags, rolling mean/std/min/max and wet-day count, 30-day mean, days since last wet day
- ERA5 (selected variables): lags, rolling mean/std, change since the previous day (`_d1`)
- calendar: sine/cosine of the day of year of day t (the target day is t + h)
- regional context: box-wide mean rain, share of wet pixels, mean humidity, and recent history
- neighbourhood: mean rain over 3x3 and 5x5 pixel squares around each pixel (edge pixels use
  only the neighbours that exist), plus its 7-day mean

Regional and neighbourhood features use the observed values of OTHER pixels on day t. That is
fine for the main time-split evaluation. The spatial hold-out (src/models/spatial_holdout.py) keeps
them as built: CHIRPS covers every pixel, so a location the model was not trained on still has
these observations (decided November 2026; a stricter variant would rebuild them without the
held-out pixels).

Memory: per-pixel features are built and written in batches of pixels, so the full table never
sits in memory at once (the laptop has about 8 GB RAM).
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml
from scipy.ndimage import uniform_filter

from src.models.evaluation import doy365

ROOT = Path(__file__).resolve().parents[2]
SITES = ROOT / "config" / "sites.yaml"
EXPERIMENT = ROOT / "config" / "experiment.yaml"
PROCESSED = ROOT / "data" / "processed"


# ---------------------------------------------------------------- time-axis helpers
# All arrays below are "wide": shape (n_days, n_pixels), rows in date order, one column per
# pixel. Every operation works down the date axis inside one column, so one pixel's values can
# never mix with another's.

def lag(a: np.ndarray, k: int) -> np.ndarray:
    """Value k days earlier (NaN for the first k days)."""
    out = np.full(a.shape, np.nan, dtype=np.float32)
    out[k:] = a[:-k]
    return out


def rolling(a: np.ndarray, window: int, stat: str) -> np.ndarray:
    """Trailing window ending on (and including) each day; NaN until the window is full."""
    r = pd.DataFrame(a).rolling(window, min_periods=window)
    return getattr(r, stat)().to_numpy(dtype=np.float32)


def days_since(event: np.ndarray) -> np.ndarray:
    """Days since the most recent True on or before each day (0 if today); NaN before the first."""
    n = event.shape[0]
    steps = np.arange(n)[:, None]
    last = np.maximum.accumulate(np.where(event, steps, -1), axis=0)
    out = (steps - last).astype(np.float32)
    out[last < 0] = np.nan
    return out


def grid_index(values: np.ndarray) -> np.ndarray:
    """Integer position of each coordinate on a regular grid (0, 1, 2, ...)."""
    u = np.unique(values)
    if len(u) == 1:
        return np.zeros(len(values), dtype=int)
    step = np.min(np.diff(u))
    idx = np.rint((values - u.min()) / step).astype(int)
    if not np.allclose(u.min() + idx * step, values, atol=1e-6):
        raise ValueError("Pixel coordinates are not on a regular grid.")
    return idx


# ---------------------------------------------------------------- feature groups
def context_features(precip: np.ndarray, rh: np.ndarray, lat: np.ndarray, lon: np.ndarray,
                     fcfg: dict, wet_mm: float) -> tuple[dict, dict]:
    """Features that need ALL pixels of the site on day t.

    Returns (region, neighbourhood): region values have shape (n_days,), the same for every
    pixel; neighbourhood values have shape (n_days, n_pixels)."""
    reg_p = precip.mean(axis=1, dtype=np.float64)[:, None]
    reg_rh = rh.mean(axis=1, dtype=np.float64)[:, None]
    region = {
        "region_precip_mean": reg_p,
        "region_wet_frac": (precip >= wet_mm).mean(axis=1)[:, None],
        "region_rh_mean": reg_rh,
        "region_precip_mean_lag1": lag(reg_p, 1),
        "region_precip_mean7": rolling(reg_p, 7, "mean"),
        "region_rh_mean7": rolling(reg_rh, 7, "mean"),
    }
    region = {k: np.asarray(v, dtype=np.float32)[:, 0] for k, v in region.items()}

    ii, jj = grid_index(lat), grid_index(lon)
    n_days = precip.shape[0]
    grid = np.zeros((n_days, ii.max() + 1, jj.max() + 1))
    grid[:, ii, jj] = precip
    mask = np.zeros((ii.max() + 1, jj.max() + 1))
    mask[ii, jj] = 1.0
    nbr = {}
    for k in fcfg["neighbourhood_sizes"]:
        num = uniform_filter(grid, size=(1, k, k), mode="constant", cval=0.0)
        den = uniform_filter(mask, size=(k, k), mode="constant", cval=0.0)
        nbr[f"nbr{k}_precip"] = (num[:, ii, jj] / den[ii, jj]).astype(np.float32)
    k_max = max(fcfg["neighbourhood_sizes"])
    nbr[f"nbr{k_max}_precip_mean7"] = rolling(nbr[f"nbr{k_max}_precip"], 7, "mean")
    return region, nbr


def pixel_features(w: dict, fcfg: dict, wet_mm: float) -> dict:
    """Features computed from each pixel's own history. `w` maps column -> wide array."""
    f = {}
    p = w["precip_mm"]
    wet = (p >= wet_mm).astype(np.float32)
    for k in fcfg["rain_lags"]:
        f[f"precip_mm_lag{k}"] = lag(p, k)
    for win in fcfg["rain_windows"]:
        for stat in ("mean", "std", "min", "max"):
            f[f"precip_mm_{stat}{win}"] = rolling(p, win, stat)
        f[f"precip_wetdays{win}"] = rolling(wet, win, "sum")
    f[f"precip_mm_mean{fcfg['rain_long_window']}"] = rolling(p, fcfg["rain_long_window"], "mean")
    f["precip_days_since_wet"] = days_since(wet > 0)
    for v in fcfg["era5_vars"]:
        for k in fcfg["era5_lags"]:
            f[f"{v}_lag{k}"] = lag(w[v], k)
        for win in fcfg["era5_windows"]:
            f[f"{v}_mean{win}"] = rolling(w[v], win, "mean")
            f[f"{v}_std{win}"] = rolling(w[v], win, "std")
    for v in fcfg["tendency_vars"]:
        f[f"{v}_d1"] = w[v] - lag(w[v], 1)
    return f


# ---------------------------------------------------------------- assembly
def check_and_sort(df: pd.DataFrame) -> tuple[pd.DataFrame, list, pd.DatetimeIndex]:
    """Sort by pixel then date and make sure every pixel has every day exactly once, so the
    table can be reshaped into (n_days, n_pixels) arrays without any misalignment."""
    df = df.sort_values(["pixel_id", "date"]).reset_index(drop=True)
    pixels = list(df["pixel_id"].unique())
    dates = pd.date_range(df["date"].min(), df["date"].max(), freq="D")
    n_pix, n_days = len(pixels), len(dates)
    if len(df) != n_pix * n_days:
        raise ValueError(f"Incomplete grid: {len(df):,} rows, expected {n_pix} x {n_days} "
                         "(missing days or duplicate rows).")
    if not (np.array_equal(df["pixel_id"].to_numpy(), np.repeat(pixels, n_days))
            and np.array_equal(df["date"].to_numpy(), np.tile(dates.to_numpy(), n_pix))):
        raise ValueError("Rows are not one per pixel per day in date order.")
    return df, pixels, dates


def wide(df: pd.DataFrame, col: str, n_pix: int, n_days: int, cols=slice(None)) -> np.ndarray:
    """(n_days, n_pixels) array from a table sorted by pixel then date."""
    return df[col].to_numpy(dtype=np.float32).reshape(n_pix, n_days).T[:, cols]


def iter_feature_batches(df: pd.DataFrame, exp: dict, batch_size: int | None = None):
    """Yield the feature table in pixel batches (rows ordered by pixel, then date)."""
    fcfg, wet_mm = exp["features"], exp["rain"]["wet_threshold_mm"]
    batch_size = batch_size or fcfg["pixel_batch"]
    df, pixels, dates = check_and_sort(df)
    n_pix, n_days = len(pixels), len(dates)
    if df[["precip_mm", "rh_pct"] + fcfg["era5_vars"]].isna().any().any():
        raise ValueError("Input has missing values; features assume a complete table.")

    coords = df.drop_duplicates("pixel_id")[["lat", "lon"]].to_numpy()
    region, nbr = context_features(wide(df, "precip_mm", n_pix, n_days),
                                   wide(df, "rh_pct", n_pix, n_days),
                                   coords[:, 0], coords[:, 1], fcfg, wet_mm)
    doy = doy365(dates)
    calendar = {"doy_sin": np.sin(2 * np.pi * doy / 365).astype(np.float32),
                "doy_cos": np.cos(2 * np.pi * doy / 365).astype(np.float32)}

    own_cols = ["precip_mm"] + list(dict.fromkeys(fcfg["era5_vars"] + fcfg["tendency_vars"]))
    for start in range(0, n_pix, batch_size):
        cols = slice(start, min(start + batch_size, n_pix))
        n_b = cols.stop - cols.start
        w = {c: wide(df, c, n_pix, n_days, cols) for c in own_cols}
        feats = pixel_features(w, fcfg, wet_mm)
        feats.update({k: v[:, cols] for k, v in nbr.items()})
        # wide (n_days, n_b) -> long, pixel-major: .T.ravel() lists pixel 1's days, then pixel 2's ...
        out = {k: v.T.ravel() for k, v in feats.items()}
        out.update({k: np.tile(v, n_b) for k, v in {**region, **calendar}.items()})
        base = df.iloc[cols.start * n_days: cols.stop * n_days].reset_index(drop=True)
        yield pd.concat([base, pd.DataFrame(out)], axis=1)


def build_features(df: pd.DataFrame, exp: dict, batch_size: int | None = None) -> pd.DataFrame:
    """Whole feature table in memory (for tests and small inputs)."""
    return pd.concat(iter_feature_batches(df, exp, batch_size), ignore_index=True)


# ---------------------------------------------------------------- run on real data
def spot_check(feat_path: Path, df: pd.DataFrame, fcfg: dict, n: int = 5) -> None:
    """Recompute a few features independently (pandas groupby / coordinate search) and compare."""
    rng = np.random.default_rng(0)
    pixels = rng.choice(df["pixel_id"].unique(), size=3, replace=False)
    f = pq.read_table(feat_path, filters=[("pixel_id", "in", list(pixels))]).to_pandas()
    d = df.sort_values(["pixel_id", "date"])
    g = d.groupby("pixel_id")
    ref = pd.DataFrame({
        "pixel_id": d["pixel_id"], "date": d["date"],
        "precip_mm_lag7": g["precip_mm"].shift(7),
        "precip_mm_mean7": g["precip_mm"].transform(lambda s: s.rolling(7).mean()),
        "rh_pct_lag3": g["rh_pct"].shift(3),
    })
    m = f.merge(ref, on=["pixel_id", "date"], suffixes=("", "_ref"))
    m = m.dropna(subset=["precip_mm_mean7_ref"])
    ok = all(np.allclose(m[c], m[f"{c}_ref"], atol=1e-4, equal_nan=True)
             for c in ["precip_mm_lag7", "precip_mm_mean7", "rh_pct_lag3"])
    # neighbourhood mean by explicit coordinate search on a few random dates
    step = 0.05 + 1e-6
    errs = []
    for _, row in m.sample(n, random_state=0).iterrows():
        day = df[df["date"] == row["date"]]
        near = day[((day["lat"] - row["lat"]).abs() <= step) & ((day["lon"] - row["lon"]).abs() <= step)]
        errs.append(abs(near["precip_mm"].mean() - row["nbr3_precip"]))
    ok_nbr = max(errs) < 1e-3
    print(f"  {'PASS ' if ok else 'CHECK'} lags and rolling means match an independent groupby "
          f"recomputation ({len(m):,} rows, 3 pixels)")
    print(f"  {'PASS ' if ok_nbr else 'CHECK'} 3x3 neighbourhood rain matches a coordinate search "
          f"(max error {max(errs):.2g} mm on {n} pixel-days)")


def build_site(site: str, exp: dict) -> None:
    src = PROCESSED / f"{site}_daily.parquet"
    out = PROCESSED / f"features_{site}.parquet"
    tmp = out.with_suffix(".tmp.parquet")
    print(f"\n=== {site} ===")
    df = pd.read_parquet(src)

    writer, nan_counts, n_rows, warm_nan = None, None, 0, None
    max_window = max([exp["features"]["rain_long_window"]] + exp["features"]["rain_windows"]
                     + exp["features"]["era5_windows"] + exp["features"]["rain_lags"])
    warm_end = df["date"].min() + pd.Timedelta(days=max_window - 1)
    for batch in iter_feature_batches(df, exp):
        table = pa.Table.from_pandas(batch, preserve_index=False).replace_schema_metadata(None)
        if writer is None:
            writer = pq.ParquetWriter(tmp, table.schema)
            feature_cols = [c for c in batch.columns if c not in df.columns]
            nan_counts = pd.Series(0, index=feature_cols)
            warm_nan = pd.Series(0, index=feature_cols)
        writer.write_table(table)
        n_rows += len(batch)
        nan_counts += batch[feature_cols].isna().sum()
        warm_nan += batch.loc[batch["date"] > warm_end, feature_cols].isna().sum()
        print(f"  pixels done: {batch['pixel_id'].nunique()} more, {n_rows // df['date'].nunique()} total")
    writer.close()
    tmp.replace(out)

    meta = pq.ParquetFile(out).metadata
    print(f"  saved {out} ({out.stat().st_size / 1e6:.0f} MB)")
    print(f"  {'PASS ' if meta.num_rows == len(df) else 'CHECK'} rows: {meta.num_rows:,} "
          f"(merged table {len(df):,}) | columns: {meta.num_columns} "
          f"({len(df.columns)} merged + {len(feature_cols)} features)")
    after = warm_nan[warm_nan > 0]
    print(f"  NaN cells (warm-up, first {max_window - 1} days of 2016): {int(nan_counts.sum()):,}")
    only_dsw = set(after.index) <= {"precip_days_since_wet"}
    print(f"  {'PASS ' if only_dsw else 'CHECK'} after {warm_end:%Y-%m-%d}, NaN only where expected: "
          f"{after.to_dict() if len(after) else 'none'}")
    spot_check(out, df, exp["features"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Build per-pixel forecasting features.")
    parser.add_argument("--site", help="site key from sites.yaml (default: all sites)")
    args = parser.parse_args()
    with open(SITES, encoding="utf-8") as f:
        sites = yaml.safe_load(f)["sites"]
    with open(EXPERIMENT, encoding="utf-8") as f:
        exp = yaml.safe_load(f)
    if args.site:
        if args.site not in sites:
            raise SystemExit(f"Unknown site '{args.site}'. Available: {', '.join(sites)}")
        sites = {args.site: sites[args.site]}
    for key in sites:
        build_site(key, exp)


if __name__ == "__main__":
    main()
