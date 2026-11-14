"""Small data tables for the report (Chapter 5, Section 5.3).

1. Wet-day share: the share of pixel-days with rain at or above 1, 10 and 25 mm,
   per site and per partition (train, validation, test), from the merged daily data.
   Partitions are the calendar periods in config/experiment.yaml, over all pixel-days
   (not only the scored target days, which start 7 days after each split start).
2. Sample record: one pixel on one training day, before feature building (the merged
   row) and after (the engineered columns of the features file).

Reads data/processed/*.parquet only; writes new CSVs to reports/data_summary/.
Run from the repo root:  python -m scripts.report_data_tables
"""
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
OUT = ROOT / "reports" / "data_summary"

# The sample record: a training-period day (never the validation or test period).
SAMPLE_SITE = "nakuru"
SAMPLE_DATE = "2023-04-15"


def wet_share(exp: dict, sites: list[str]) -> pd.DataFrame:
    thresholds = [exp["rain"]["wet_threshold_mm"]] + exp["rain"]["exceedance_mm"]
    rows = []
    for site in sites:
        d = pd.read_parquet(PROC / f"{site}_daily.parquet", columns=["pixel_id", "date", "precip_mm"])
        d["date"] = pd.to_datetime(d["date"])
        for part, span in exp["splits"].items():
            sub = d[(d["date"] >= pd.Timestamp(span["start"])) & (d["date"] <= pd.Timestamp(span["end"]))]
            row = {"site": site, "partition": part, "start": span["start"], "end": span["end"],
                   "days": sub["date"].nunique(), "pixel_days": len(sub)}
            for t in thresholds:
                row[f"share_ge_{t:g}mm"] = (sub["precip_mm"] >= t).mean()
            rows.append(row)
    return pd.DataFrame(rows)


def sample_record() -> pd.DataFrame:
    day = pd.Timestamp(SAMPLE_DATE)
    raw = pd.read_parquet(PROC / f"{SAMPLE_SITE}_daily.parquet")
    raw["date"] = pd.to_datetime(raw["date"])
    pixel = sorted(raw["pixel_id"].unique())[0]
    before = raw[(raw["pixel_id"] == pixel) & (raw["date"] == day)]
    feats = pd.read_parquet(PROC / f"features_{SAMPLE_SITE}.parquet",
                            filters=[("pixel_id", "==", pixel)])
    feats["date"] = pd.to_datetime(feats["date"])
    after = feats[feats["date"] == day]
    assert len(before) == 1 and len(after) == 1, "expected exactly one row before and after"
    new_cols = [c for c in after.columns if c not in before.columns]
    out = pd.concat([
        pd.DataFrame({"stage": "merged (before)", "column": before.columns, "value": before.iloc[0].astype(str).values}),
        pd.DataFrame({"stage": "engineered (after)", "column": new_cols, "value": after.iloc[0][new_cols].astype(str).values}),
    ], ignore_index=True)
    print(f"sample: {SAMPLE_SITE} pixel {pixel} on {SAMPLE_DATE}: "
          f"{before.shape[1]} merged columns, {len(new_cols)} engineered columns")
    return out


def main() -> None:
    exp = yaml.safe_load((ROOT / "config" / "experiment.yaml").read_text())
    sites = list(yaml.safe_load((ROOT / "config" / "sites.yaml").read_text())["sites"])
    OUT.mkdir(parents=True, exist_ok=True)
    ws = wet_share(exp, sites)
    ws.to_csv(OUT / "wet_share.csv", index=False)
    print(ws.to_string(index=False))
    sample_record().to_csv(OUT / "sample_record.csv", index=False)
    print(f"saved {OUT / 'wet_share.csv'} and {OUT / 'sample_record.csv'}")


if __name__ == "__main__":
    main()
