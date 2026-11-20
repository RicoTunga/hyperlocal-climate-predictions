"""Build the small data files the dashboard reads (a replay of the final test period).

Run from the repo root after the final test and the late-blight test:

    python -m app.build_data

Inputs (saved forecasts, nothing is recomputed):
    models/<site>/final_train60_score<N>/h<h>/test_predictions.parquet      CatBoost and climatology
    models/nakuru/late_blight_final/<rule>/h<h>/test_predictions.parquet     late-blight risk (Nakuru)
    .../posttest_predictions.parquet (if present)                              post-test period (src.models.posttest)
Outputs (git-ignored, app/data/):
    <site>_replay.parquet   one row per pixel x issue day x horizon: issue_date, target_date, horizon,
                            forecast and observed rain/humidity, climatology, late-blight risk, and
                            `period`: 0 = reported test, 1 = post-test (final CHIRPS), 2 = post-test
                            (PRELIMINARY CHIRPS), 3 = not yet observed (outcome NaN)
    <site>_pixels.parquet   pixel_id, lat, lon, elev_mean_m
    meta.json               per site: issue dates with all 7 horizons available, pixel count, and the
                            data-status dates the dashboard uses for its labels (see data_status())

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


PERIOD_CODES = {"test": 0, "posttest_final": 1, "posttest_prelim": 2, "forecast": 3}


def read_predictions(d: Path, kind: str, columns: list) -> pd.DataFrame | None:
    """The test file (period 0) or the post-test file (periods 1-3); None if there is no post-test file."""
    f = d / f"{kind}_predictions.parquet"
    if kind == "posttest" and not f.exists():
        return None
    df = pd.read_parquet(f, columns=columns + (["period"] if kind == "posttest" else []))
    df["period"] = df["period"].map(PERIOD_CODES) if kind == "posttest" else PERIOD_CODES["test"]
    return df


def site_replay(site: str) -> pd.DataFrame:
    run = final_run_dir(site)
    parts = []
    for h in HORIZONS:
        frames = []
        for kind in ("test", "posttest"):
            src = read_predictions(run / f"h{h}", kind, ["pixel_id", "target_date"] + list(FORECAST_COLS))
            if src is None:
                continue
            df = src.rename(columns=FORECAST_COLS)
            if site == "nakuru":
                for rule in BLIGHT_RULES:
                    lb = read_predictions(MODELS / site / "late_blight_final" / rule / f"h{h}", kind,
                                          ["pixel_id", "target_date", "catboost", "y_risk", "clim_p_risk"])
                    if lb is None:
                        raise SystemExit(f"late-blight post-test file missing for {rule} h{h}: run src.models.posttest score")
                    lb = lb.drop(columns="period").rename(columns={"catboost": f"blight_{rule}", "y_risk": f"obs_blight_{rule}",
                                                                   "clim_p_risk": f"clim_blight_{rule}"})
                    df = df.merge(lb, on=["pixel_id", "target_date"], how="left", validate="one_to_one")
            frames.append(df)
        df = pd.concat(frames, ignore_index=True)
        assert not df.duplicated(["pixel_id", "target_date"]).any(), "test and post-test rows overlap"
        df.insert(1, "horizon", np.int8(h))
        df.insert(1, "issue_date", df["target_date"] - pd.Timedelta(days=h))
        parts.append(df)
    out = pd.concat(parts, ignore_index=True)
    out["period"] = out["period"].astype("int8")
    num = [c for c in out.columns if c not in ("pixel_id", "issue_date", "target_date", "horizon", "period")]
    out[num] = out[num].astype("float32")
    return out.sort_values(["issue_date", "horizon", "pixel_id"]).reset_index(drop=True)


def data_status(site: str, df: pd.DataFrame) -> dict:
    """Dates behind the dashboard's data labels.

    test_end / posttest_start: where the reported test stops. chirps_final_through: last day of final
    CHIRPS; later observed days use PRELIMINARY CHIRPS (data/raw/chirps_<site>_sources.json).
    observed_through: last day with any observation. era5t_from: from this day on, the ERA5 values
    were most likely still ERA5T (the near-real-time release) when downloaded: ECMWF overwrites
    ERA5T "about two months after the month in question", so the first day of the month two months
    before each download month (the file's modification date) is used; approximate."""
    import yaml
    exp = yaml.safe_load((ROOT / "config" / "experiment.yaml").read_text(encoding="utf-8"))
    test_end = pd.Timestamp(str(exp["splits"]["test"]["end"]))
    obs = df.loc[df["period"] < PERIOD_CODES["forecast"], "target_date"]
    status = {"test_end": str(test_end.date()), "observed_through": str(obs.max().date()),
              "chirps_final_through": str(obs.max().date()), "posttest_start": None, "era5t_from": None}
    if "posttest" in exp["splits"]:
        status["posttest_start"] = str(exp["splits"]["posttest"]["start"])
    src = ROOT / "data" / "raw" / f"chirps_{site}_sources.json"
    if src.exists():
        status["chirps_final_through"] = json.loads(src.read_text(encoding="utf-8"))["final_through"]
    era5_files = sorted((ROOT / "data" / "raw" / "era5_manual" / site).glob("*.nc"))
    if era5_files:
        starts = []
        for f in era5_files:
            dl = pd.Timestamp(f.stat().st_mtime, unit="s")
            starts.append((dl.to_period("M") - 2).start_time)
        status["era5t_from"] = str(min(starts).date())
    return status


def check(site: str, df: pd.DataFrame) -> None:
    """Row counts, keys and a sample of values against the source files."""
    n_px = df["pixel_id"].nunique()
    assert not df.duplicated(["pixel_id", "issue_date", "horizon"]).any(), "duplicate rows"
    assert ((df["target_date"] - df["issue_date"]).dt.days == df["horizon"]).all(), "issue date mismatch"
    for h in HORIZONS:
        n_targets = df.loc[df["horizon"] == h, "target_date"].nunique()
        assert (df["horizon"] == h).sum() == n_px * n_targets, f"h{h}: incomplete grid"
    run = final_run_dir(site)
    for kind in ("test", "posttest"):
        f = run / "h3" / f"{kind}_predictions.parquet"
        if not f.exists():
            continue
        src = pd.read_parquet(f, columns=["pixel_id", "target_date", "catboost__occ1", "y_rh"])
        m = src.sample(2000, random_state=0).merge(df[df["horizon"] == 3], on=["pixel_id", "target_date"])
        assert len(m) == 2000 and np.allclose(m["catboost__occ1"], m["p_rain1"], atol=1e-6)
        assert np.allclose(m["y_rh"], m["obs_rh"], atol=1e-4, equal_nan=True)
    observed = df["period"] < PERIOD_CODES["forecast"]
    assert df.loc[observed, ["obs_rain_mm", "obs_rh"]].notna().all().all(), "observed rows with missing outcomes"
    assert df.loc[~observed, ["obs_rain_mm", "obs_rh"]].isna().all().all(), "unobserved rows with outcomes"
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
                      "n_pixels": int(df["pixel_id"].nunique()), "has_blight": site == "nakuru",
                      "status": data_status(site, df)}
        gaps = pd.date_range(complete.min(), complete.max()).difference(complete)
        assert len(gaps) == 0, f"issue days without all 7 horizons inside the range: {list(gaps[:5])}"
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
