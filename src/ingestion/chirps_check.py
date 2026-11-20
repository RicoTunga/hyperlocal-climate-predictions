"""Checks for the CHC-server route of src/ingestion/chirps.py. Run from the repo root:

    python -m src.ingestion.chirps_check                 # September 2026 + October 2026
    python -m src.ingestion.chirps_check --month 2026-08 --prelim-month 2026-10

1. SAME DATA: the CHC final GeoTIFFs, read by chirps.read_tif_at at chirps.pixel_centres, must give
   the same pixel ids and the same values as the Earth Engine chunk already cached for that month
   (data/raw/chirps/<site>/). This is what makes it safe to append CHC months to Earth Engine months.
2. HOW PRELIMINARY DIFFERS: for a month that has both products on the CHC server, compare the
   preliminary with the final values (the size of the revision a "preliminary" label warns about).
   Written to reports/data_summary/chirps_prelim_vs_final.csv.
Downloads about 0.8 MB (final, cached) and 3 MB (preliminary, not cached) per day.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.ingestion.chirps import (RAW, chc_download, load_config, pixel_centres, read_tif_at)

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports" / "data_summary"


def month_days(month: str) -> pd.DatetimeIndex:
    p = pd.Period(month, freq="M")
    return pd.date_range(p.start_time, p.end_time.normalize())


def chc_month(kind: str, bbox: dict, days) -> pd.DataFrame:
    lat, lon, pid = pixel_centres(bbox)
    parts = []
    for d in days:
        path = chc_download(kind, d)
        if path is None:
            raise SystemExit(f"No {kind} CHC file for {d:%Y-%m-%d}")
        v = read_tif_at(path, lat, lon).astype("float32").astype("float64")
        if kind == "prelim":
            path.unlink(missing_ok=True)
        parts.append(pd.DataFrame({"pixel_id": pid, "date": d, "precip_mm": v}))
    return pd.concat(parts, ignore_index=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--month", default="2026-09", help="month cached from Earth Engine to compare with")
    ap.add_argument("--prelim-month", default="2026-10", help="month with both final and preliminary on CHC")
    args = ap.parse_args()
    cfg = load_config()
    ok_all, rows = True, []
    for site, s in cfg["sites"].items():
        days = month_days(args.month)
        cache = RAW / "chirps" / site / f"{days[0]:%Y%m%d}_{days[-1] + pd.Timedelta(days=1):%Y%m%d}.parquet"
        ee_df = pd.read_parquet(cache)[["pixel_id", "date", "precip_mm"]]
        chc = chc_month("final", s["bbox"], days)
        same_ids = set(ee_df["pixel_id"]) == set(chc["pixel_id"])
        m = ee_df.merge(chc, on=["pixel_id", "date"], suffixes=("_ee", "_chc"), validate="one_to_one")
        diff = (m["precip_mm_ee"] - m["precip_mm_chc"]).abs()
        ok = same_ids and len(m) == len(ee_df) == len(chc) and not chc["precip_mm"].isna().any() and diff.max() < 1e-4
        ok_all &= ok
        print(f"{'PASS ' if ok else 'FAIL '} {site} {args.month}: CHC final vs Earth Engine cache: "
              f"pixel ids identical {same_ids}, rows {len(m):,} of {len(ee_df):,}, max |difference| {diff.max():.2g} mm, "
              f"missing {int(chc['precip_mm'].isna().sum())}")

        pdays = month_days(args.prelim_month)
        fin = chc_month("final", s["bbox"], pdays)
        pre = chc_month("prelim", s["bbox"], pdays)
        x = fin.merge(pre, on=["pixel_id", "date"], suffixes=("_final", "_prelim"), validate="one_to_one")
        f, p = x["precip_mm_final"], x["precip_mm_prelim"]
        d = p - f
        wet_f, wet_p = f >= 1, p >= 1
        r = {"site": site, "month": args.prelim_month, "pixel_days": len(x),
             "mean_final_mm": f.mean(), "mean_prelim_mm": p.mean(), "mean_diff_mm": d.mean(),
             "mae_mm": d.abs().mean(), "max_abs_diff_mm": d.abs().max(),
             "share_identical": float((d.abs() < 1e-4).mean()),
             "wet_share_final": float(wet_f.mean()), "wet_share_prelim": float(wet_p.mean()),
             "wet_dry_agreement": float((wet_f == wet_p).mean()),
             "corr": float(np.corrcoef(f, p)[0, 1])}
        rows.append(r)
        print(f"      {site} {args.prelim_month}: preliminary vs final: mean {r['mean_prelim_mm']:.2f} vs {r['mean_final_mm']:.2f} mm/day, "
              f"MAE {r['mae_mm']:.2f} mm, identical {r['share_identical']:.0%}, wet/dry (>= 1 mm) agree {r['wet_dry_agreement']:.1%}, "
              f"correlation {r['corr']:.3f}")
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT / "chirps_prelim_vs_final.csv", index=False)
    print(f"saved {OUT / 'chirps_prelim_vs_final.csv'}")
    if not ok_all:
        raise SystemExit("CHC final does not reproduce the Earth Engine data: do not use the CHC route.")


if __name__ == "__main__":
    main()
