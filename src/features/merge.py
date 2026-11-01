"""Merge CHIRPS rainfall, ERA5 weather and pixel elevation into one table per site.

Save this file as src/features/merge.py and run from the repo root (venv active):

    python -m src.features.merge                 # all sites
    python -m src.features.merge --site nakuru   # one site

Inputs (data/raw/):  chirps_<site>.parquet, era5_<site>.parquet, elevation_<site>.parquet
Output (data/processed/): <site>_daily.parquet
    site | pixel_id | lat | lon | date | precip_mm | <11 ERA5 columns> | <4 elevation columns>
One row per pixel per day. This step only joins and checks; it creates no lag or rolling
features, so nothing here can leak information from the future.

Structural problems (different pixels or dates between sources, duplicates, lost rows)
stop the script with an error. Physical-plausibility checks are printed as PASS or CHECK.
"""
import argparse
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "config" / "sites.yaml"
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"

KEYS = ["pixel_id", "date"]
ELEV_COLUMNS = ["elev_mean_m", "elev_std_m", "elev_min_m", "elev_max_m"]


def load_config() -> dict:
    with open(CONFIG, encoding="utf-8") as f:
        return yaml.safe_load(f)


def read_inputs(site_key: str):
    paths = {
        "CHIRPS": RAW / f"chirps_{site_key}.parquet",
        "ERA5": RAW / f"era5_{site_key}.parquet",
        "elevation": RAW / f"elevation_{site_key}.parquet",
    }
    for name, p in paths.items():
        if not p.exists():
            hint = {"CHIRPS": "python -m src.ingestion.chirps",
                    "ERA5": "python -m src.ingestion.era5 --local",
                    "elevation": "python -m src.ingestion.elevation"}[name]
            raise SystemExit(f"Missing {name} file: {p}\nCreate it with: {hint} --site {site_key}")
    return (pd.read_parquet(paths["CHIRPS"]), pd.read_parquet(paths["ERA5"]),
            pd.read_parquet(paths["elevation"]))


def structural_checks(c: pd.DataFrame, e: pd.DataFrame, z: pd.DataFrame) -> None:
    for name, df in (("CHIRPS", c), ("ERA5", e)):
        dups = int(df.duplicated(KEYS).sum())
        if dups:
            raise SystemExit(f"{name} has {dups} duplicate (pixel_id, date) rows.")
    if z["pixel_id"].duplicated().any():
        raise SystemExit("Elevation file has duplicate pixel_id rows.")

    cp, ep, zp = set(c["pixel_id"]), set(e["pixel_id"]), set(z["pixel_id"])
    if not (cp == ep == zp):
        raise SystemExit(
            "Pixel sets differ between sources: "
            f"CHIRPS {len(cp)}, ERA5 {len(ep)}, elevation {len(zp)}; "
            f"in CHIRPS only vs ERA5: {len(cp - ep)}, in ERA5 only: {len(ep - cp)}, "
            f"missing from elevation: {len(cp - zp)}")
    cd, ed = set(c["date"].unique()), set(e["date"].unique())
    if cd != ed:
        raise SystemExit(f"Date sets differ: CHIRPS {len(cd)} days, ERA5 {len(ed)} days, "
                         f"{len(cd - ed)} only in CHIRPS, {len(ed - cd)} only in ERA5.")
    # the same pixel_id must carry the same coordinates in every source
    cc = c[["pixel_id", "lat", "lon"]].drop_duplicates().set_index("pixel_id").sort_index()
    ec = e[["pixel_id", "lat", "lon"]].drop_duplicates().set_index("pixel_id").sort_index()
    zc = z[["pixel_id", "lat", "lon"]].set_index("pixel_id").sort_index()
    if not (cc.round(3).equals(ec.round(3)) and cc.round(3).equals(zc.round(3))):
        raise SystemExit("lat/lon of the same pixel_id differ between sources.")


def merge_site(site_key: str, site: dict) -> None:
    print(f"\n=== {site_key}: {site['name']} ===")
    c, e, z = read_inputs(site_key)
    print(f"  inputs: CHIRPS {len(c):,} rows | ERA5 {len(e):,} rows | elevation {len(z):,} pixels")
    structural_checks(c, e, z)
    print("  structure: same pixels, same dates, same coordinates, no duplicates")

    m = c.merge(e.drop(columns=["lat", "lon"]), on=KEYS, how="outer",
                validate="one_to_one", indicator=True)
    if not (m["_merge"] == "both").all():
        raise SystemExit("Rows were lost or added when joining CHIRPS and ERA5.")
    m = m.drop(columns="_merge")
    m = m.merge(z.drop(columns=["lat", "lon"]), on="pixel_id", how="left", validate="many_to_one")
    if len(m) != len(c):
        raise SystemExit(f"Row count changed: {len(c):,} -> {len(m):,}")

    m.insert(0, "site", site_key)
    era5_cols = [col for col in e.columns if col not in ("pixel_id", "lat", "lon", "date")]
    m = m[["site", "pixel_id", "lat", "lon", "date", "precip_mm"] + era5_cols + ELEV_COLUMNS]
    m[["precip_mm"] + era5_cols] = m[["precip_mm"] + era5_cols].astype("float32")
    m = m.sort_values(KEYS).reset_index(drop=True)

    PROCESSED.mkdir(parents=True, exist_ok=True)
    out = PROCESSED / f"{site_key}_daily.parquet"
    m.to_parquet(out, index=False)
    print(f"  saved {out}")
    report(m, era5_cols)


def flag(ok: bool) -> str:
    return "PASS " if ok else "CHECK"


def report(m: pd.DataFrame, era5_cols: list) -> None:
    n_pix, n_days = m["pixel_id"].nunique(), m["date"].nunique()
    print(f"  table: {len(m):,} rows x {m.shape[1]} columns | {n_pix} pixels x {n_days} days "
          f"({m['date'].min():%Y-%m-%d} to {m['date'].max():%Y-%m-%d})")
    print(f"  columns: {list(m.columns)}")

    num = m.drop(columns=["site", "pixel_id", "date"]).select_dtypes("number")
    miss = int(num.isna().sum().sum())
    print(f"  {flag(miss == 0)} missing values: {miss}")
    print(f"  {flag(len(m) == n_pix * n_days)} complete grid: rows = pixels x days")
    print(f"  {flag(m['precip_mm'].min() >= 0)} rainfall never negative "
          f"(min {m['precip_mm'].min():.2f} mm, max {m['precip_mm'].max():.0f} mm in a day)")
    print(f"  {flag(m['rh_pct'].between(0, 100).all())} humidity within 0-100% "
          f"(min {m['rh_pct'].min():.1f}, max {m['rh_pct'].max():.1f})")
    print(f"  {flag(m['t2m_c'].between(-5, 45).all())} temperature plausible "
          f"(mean range {m['t2m_c'].min():.1f} to {m['t2m_c'].max():.1f} C)")
    print(f"  {flag(m['sp_hpa'].between(500, 1100).all())} pressure plausible "
          f"({m['sp_hpa'].min():.0f} to {m['sp_hpa'].max():.0f} hPa)")

    sample = m.sample(min(len(m), 500_000), random_state=0)
    corr = sample[["precip_mm", "rh_pct", "tcc"]].corr()
    r_rh, r_tcc = corr.loc["precip_mm", "rh_pct"], corr.loc["precip_mm", "tcc"]
    print(f"  {flag(r_rh > 0 and r_tcc > 0)} wetter days are more humid and cloudier "
          f"(correlation of rainfall with humidity {r_rh:+.2f}, with cloud cover {r_tcc:+.2f})")

    per_pixel = m.groupby("pixel_id").agg(rain_mm_day=("precip_mm", "mean"),
                                          elev=("elev_mean_m", "first"))
    per_pixel["rain_mm_yr"] = per_pixel["rain_mm_day"] * 365.25
    rho = per_pixel["rain_mm_yr"].corr(per_pixel["elev"], method="spearman")
    print(f"  average rainfall by pixel: {per_pixel['rain_mm_yr'].min():.0f} to "
          f"{per_pixel['rain_mm_yr'].max():.0f} mm/yr "
          f"(rank correlation with elevation {rho:+.2f}; for information, not a pass/fail test)")

    print("  value ranges:")
    desc = num.describe().loc[["min", "mean", "max"]].T
    for name, row in desc.iterrows():
        print(f"    {name:<13} min {row['min']:>9.2f}   mean {row['mean']:>9.2f}   max {row['max']:>9.2f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge CHIRPS + ERA5 + elevation per site.")
    parser.add_argument("--site", help="site key from sites.yaml (default: all sites)")
    args = parser.parse_args()
    sites = load_config()["sites"]
    if args.site:
        if args.site not in sites:
            raise SystemExit(f"Unknown site '{args.site}'. Available: {', '.join(sites)}")
        sites = {args.site: sites[args.site]}
    for key, site in sites.items():
        merge_site(key, site)


if __name__ == "__main__":
    main()
