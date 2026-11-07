"""Build the small data files the dashboard reads (a replay of the final test period).

Run from the repo root after the final test and the late-blight test:

    python -m app.build_data

Inputs (saved forecasts, nothing is recomputed):
    models/<site>/final_train60_score<N>/h<h>/test_predictions.parquet      CatBoost and climatology
    models/nakuru/late_blight_final/<rule>/h<h>/test_predictions.parquet     late-blight risk (Nakuru)
Outputs (git-ignored, app/data/):
    <site>_replay.parquet   one row per pixel x issue day x horizon: issue_date, target_date, horizon,
                            forecast and observed rain/humidity, climatology, late-blight risk
    <site>_pixels.parquet   pixel_id, lat, lon, elev_mean_m
    meta.json               per site: issue dates with all 7 horizons available, pixel count

The rows are sorted by issue date and written in small row groups, so the app can read one issue
day with a parquet filter instead of loading the whole table.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
PROCESSED = ROOT / "data" / "processed"
OUT = ROOT / "app" / "data"
HORIZONS = range(1, 8)
SITES = ["nakuru", "busia"]
BLIGHT_RULES = ["hutton", "humid"]

# source column -> app column
FORECAST_COLS = {
    "catboost__occ1": "p_rain1", "catboost__occ10": "p_rain10", "catboost__occ25": "p_rain25",
    "catboost__two_stage": "rain_mm", "catboost__rh": "rh",
    "climatology__occ1": "clim_rain1", "climatology__occ10": "clim_rain10", "climatology__occ25": "clim_rain25",
    "climatology__direct": "clim_rain_mm", "climatology__rh": "clim_rh",
    "y_rain": "obs_rain_mm", "y_rh": "obs_rh",
}


def final_run_dir(site: str) -> Path:
    dirs = sorted((MODELS / site).glob("final_train*_score*"))
    if len(dirs) != 1:
        raise SystemExit(f"Expected one final run folder for {site}, found {[d.name for d in dirs]}")
    return dirs[0]


def site_replay(site: str) -> pd.DataFrame:
    run = final_run_dir(site)
    parts = []
    for h in HORIZONS:
        src = pd.read_parquet(run / f"h{h}" / "test_predictions.parquet",
                              columns=["pixel_id", "target_date"] + list(FORECAST_COLS))
        df = src.rename(columns=FORECAST_COLS)
        if site == "nakuru":
            for rule in BLIGHT_RULES:
                lb = pd.read_parquet(MODELS / site / "late_blight_final" / rule / f"h{h}" / "test_predictions.parquet",
                                     columns=["pixel_id", "target_date", "catboost", "y_risk", "clim_p_risk"])
                lb = lb.rename(columns={"catboost": f"blight_{rule}", "y_risk": f"obs_blight_{rule}",
                                        "clim_p_risk": f"clim_blight_{rule}"})
                df = df.merge(lb, on=["pixel_id", "target_date"], how="left", validate="one_to_one")
        df.insert(1, "horizon", np.int8(h))
        df.insert(1, "issue_date", df["target_date"] - pd.Timedelta(days=h))
        parts.append(df)
    out = pd.concat(parts, ignore_index=True)
    num = [c for c in out.columns if c not in ("pixel_id", "issue_date", "target_date", "horizon")]
    out[num] = out[num].astype("float32")
    return out.sort_values(["issue_date", "horizon", "pixel_id"]).reset_index(drop=True)


def check(site: str, df: pd.DataFrame) -> None:
    """Row counts, keys and a sample of values against the source files."""
    n_px = df["pixel_id"].nunique()
    assert not df.duplicated(["pixel_id", "issue_date", "horizon"]).any(), "duplicate rows"
    assert ((df["target_date"] - df["issue_date"]).dt.days == df["horizon"]).all(), "issue date mismatch"
    for h in HORIZONS:
        n_targets = df.loc[df["horizon"] == h, "target_date"].nunique()
        assert (df["horizon"] == h).sum() == n_px * n_targets, f"h{h}: incomplete grid"
    run = final_run_dir(site)
    src = pd.read_parquet(run / "h3" / "test_predictions.parquet", columns=["pixel_id", "target_date", "catboost__occ1", "y_rh"])
    m = src.sample(2000, random_state=0).merge(df[df["horizon"] == 3], on=["pixel_id", "target_date"])
    assert np.allclose(m["catboost__occ1"], m["p_rain1"], atol=1e-6) and np.allclose(m["y_rh"], m["obs_rh"], atol=1e-4)
    if site == "nakuru":
        assert df[[f"blight_{r}" for r in BLIGHT_RULES]].notna().all().all(), "late-blight values missing"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    meta = {}
    for site in SITES:
        df = site_replay(site)
        check(site, df)
        full = df.groupby("issue_date")["horizon"].nunique()
        complete = full[full == len(HORIZONS)].index
        meta[site] = {"issue_min": str(complete.min().date()), "issue_max": str(complete.max().date()),
                      "n_pixels": int(df["pixel_id"].nunique()), "has_blight": site == "nakuru"}
        f = OUT / f"{site}_replay.parquet"
        pq.write_table(pa.Table.from_pandas(df, preserve_index=False), f, row_group_size=50_000)
        px = pd.read_parquet(PROCESSED / f"{site}_daily.parquet", columns=["pixel_id", "lat", "lon", "elev_mean_m"])
        px = px.drop_duplicates("pixel_id").sort_values("pixel_id").reset_index(drop=True)
        assert set(px["pixel_id"]) == set(df["pixel_id"])
        px.to_parquet(OUT / f"{site}_pixels.parquet", index=False)
        print(f"  PASS  {site}: {len(df):,} rows ({meta[site]['n_pixels']} pixels x 7 horizons), "
              f"complete issue days {meta[site]['issue_min']}..{meta[site]['issue_max']}, "
              f"{f.stat().st_size / 1e6:.0f} MB")
    (OUT / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
