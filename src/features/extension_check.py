"""After extending the data period, check that nothing up to the old end changed.

    python -m src.features.extension_check --backup data/backup_pre_posttest --until 2026-09-30

Compares, for every row dated on or before --until, the files in the backup folder with the
current ones (same rows, same values; NaN equals NaN): the merged daily tables, the feature tables
(in pixel batches, for memory), the late-blight humid hours and the raw CHIRPS / ERA5 tables.
Any difference means a reported result could change, so the script exits with an error.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
FILES = {  # name -> current location
    "nakuru_daily.parquet": "data/processed", "busia_daily.parquet": "data/processed",
    "nakuru_humid_hours.parquet": "data/processed",
    "features_nakuru.parquet": "data/processed", "features_busia.parquet": "data/processed",
    "chirps_nakuru.parquet": "data/raw", "chirps_busia.parquet": "data/raw",
    "era5_nakuru.parquet": "data/raw", "era5_busia.parquet": "data/raw",
}


def compare(old: pd.DataFrame, new: pd.DataFrame, until: pd.Timestamp) -> tuple[bool, str]:
    old = old[old["date"] <= until].sort_values(["pixel_id", "date"]).reset_index(drop=True)
    new = new[new["date"] <= until].sort_values(["pixel_id", "date"]).reset_index(drop=True)
    if list(old.columns) != list(new.columns):
        return False, f"columns differ: {set(old.columns) ^ set(new.columns)}"
    if len(old) != len(new):
        return False, f"{len(old):,} rows before, {len(new):,} now"
    bad = []
    for c in old.columns:
        a, b = old[c], new[c]
        if pd.api.types.is_numeric_dtype(a) and not pd.api.types.is_bool_dtype(a):
            same = np.array_equal(a.to_numpy(float), b.to_numpy(float), equal_nan=True)
        else:
            same = a.equals(b)
        if not same:
            bad.append(c)
    return (not bad), (f"{len(old):,} rows identical" if not bad else f"values differ in {bad[:8]}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backup", required=True)
    ap.add_argument("--until", required=True)
    args = ap.parse_args()
    until, backup, ok_all = pd.Timestamp(args.until), ROOT / args.backup, True
    for name, folder in FILES.items():
        old_f, new_f = backup / name, ROOT / folder / name
        if not old_f.exists():
            print(f"SKIP  {name}: not in the backup")
            continue
        new_max = pd.read_parquet(new_f, columns=["date"])["date"].max()
        if name.startswith("features_"):
            pixels = sorted(pd.read_parquet(old_f, columns=["pixel_id"])["pixel_id"].unique())
            cols = pq.read_schema(old_f).names
            msgs, ok = [], True
            for i in range(0, len(pixels), 100):
                px = pixels[i:i + 100]
                f = [("pixel_id", "in", px), ("date", "<=", until)]
                o, m = compare(pd.read_parquet(old_f, filters=f), pd.read_parquet(new_f, filters=f, columns=cols), until)
                ok &= o
                if not o:
                    msgs.append(m)
            msg = "all pixel batches identical" if ok else "; ".join(msgs[:3])
        else:
            ok, msg = compare(pd.read_parquet(old_f), pd.read_parquet(new_f), until)
        ok_all &= ok
        print(f"{'PASS ' if ok else 'FAIL '} {name}: up to {until:%Y-%m-%d}: {msg} (now ends {new_max:%Y-%m-%d})")
    if not ok_all:
        raise SystemExit("Data up to the old end changed: stop and investigate before scoring anything.")
    print("All files identical up to", f"{until:%Y-%m-%d}")


if __name__ == "__main__":
    main()
