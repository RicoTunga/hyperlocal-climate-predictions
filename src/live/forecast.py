"""Live forecasts (branch feature/live): the FROZEN final models fed with near-real-time inputs.

    python -m src.live.forecast              # forecasts issued on every day from the replay's end to yesterday
    python -m src.live.forecast --issue 2026-11-19

Nothing is retrained and no reported result changes. Each forecast is issued on day t (the last complete
UTC day, normally yesterday) for t+1..t+7, from these inputs (the backtest, src.live.backtest, measures what
they cost in skill on validation 2024):
- rain: CHIRPS v2.0 where published (the merged table up to its last day, then CHC final or PRELIMINARY
  files), and for the newest days that CHIRPS has not published yet, Open-Meteo ECMWF IFS 0.25 rain
  QUANTILE-MAPPED to CHIRPS (backtest scenario S5; the map is fitted on Mar-Dec 2024);
- weather: ERA5 (ERA5T) wherever the merged table has it, and Open-Meteo ECMWF IFS 0.25 only for the days
  after it, processed exactly like ERA5 (src.live.openmeteo) and shifted by each pixel's mean ERA5-minus-IFS
  offset (fitted on Mar-Dec 2024). Backtest on 2024 with this setup (S5): rain about as good as climatology
  at every lead time (the missing CHIRPS days remove day-1 skill); humidity skilful on days 1-2 at Nakuru;
- late blight (Nakuru): hours at RH >= 90% and minimum temperature from the same IFS hours, with the
  elevation correction of src.models.late_blight (ERA5 terrain from the training years);
- climatology, models and smearing factors exactly as in the final test (src.models.posttest.load_frozen).

Outputs (git-ignored; the replay files in app/data are never modified):
    app/data/live/<site>_live.parquet        the same columns as <site>_replay.parquet, period = 4
    app/data/live/<site>_live_shap.parquet   explanations in the dashboard's groups (as app/build_shap)
    app/data/live/live_meta.json             issue days, when made, what each day's inputs were
"""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from app import components as C
from src.features.build import build_features
from src.ingestion.chirps import chc_download, pixel_centres, read_tif_at
from src.ingestion.era5 import DAILY_COLUMNS, interpolate_to_pixels, relative_humidity, to_daily_grid
from src.live.backtest import apply_rain_map, apply_weather_offsets, daily_rain_at_pixels, fit_live_corrections
from src.live.openmeteo import ATTRIBUTION, fetch_grid
from src.models.baselines import fit_climatology
from src.models.evaluation import PROCESSED, load_configs, split_bounds
from src.models.final_test import final_config
from src.models.posttest import issue_table, load_frozen, risk_issue_table
from src.models.train import ALGOS, MODELS, dev_pixel_sample
from src.models.transfer import predict_all

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "app" / "data" / "live"
PAST_DAYS = 35          # IFS history: ERA5-style features need 21 days (14-day windows + 7-day lags)
WINDOW_DAYS = 200       # daily history passed to the feature builder (checked equal to full history in the backtest)
LIVE_PERIOD = 4         # replay `period` code for live forecasts


def chirps_recent(site_cfg: dict, days) -> tuple[pd.DataFrame, dict]:
    """CHIRPS for days after the merged table: final where published, else preliminary; missing days are
    left out (they are filled with IFS later). Returns (rows, {date: 'final'|'prelim'})."""
    lat, lon, pid = pixel_centres(site_cfg["bbox"])
    parts, kind = [], {}
    for d in days:
        path, k = chc_download("final", d), "final"
        if path is None:
            path, k = chc_download("prelim", d), "prelim"
        if path is None:
            continue
        v = read_tif_at(path, lat, lon).astype("float32").astype("float64")
        if k == "prelim":
            path.unlink(missing_ok=True)
        parts.append(pd.DataFrame({"pixel_id": pid, "date": d, "precip_mm": v}))
        kind[d] = k
    return (pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["pixel_id", "date", "precip_mm"])), kind


def live_daily(site: str, sites_cfg: dict, exp: dict, last_issue: pd.Timestamp):
    """Daily table for the feature builder (merged-table columns) up to last_issue, plus IFS humid hours and
    the source of every recent day's rain."""
    s = sites_cfg["sites"][site]
    std = pd.read_parquet(PROCESSED / f"{site}_daily.parquet", filters=[("date", ">=", last_issue - pd.Timedelta(days=WINDOW_DAYS + 40))])
    obs_end = std["date"].max()
    ds = fetch_grid(s["era5_area"], endpoint="forecast", past_days=PAST_DAYS)
    first = pd.Timestamp(ds["time"].values[0]).normalize()
    daily, hours = to_daily_grid(ds, first, last_issue)
    complete = hours[hours == 24].index
    if last_issue not in complete:
        raise SystemExit(f"IFS hours for {last_issue:%Y-%m-%d} are not complete yet; issue on an earlier day")
    ifs = interpolate_to_pixels(daily.sel(time=complete), s["bbox"])
    rain_ifs = daily_rain_at_pixels(ds["tp"].sel(time=slice(first, last_issue + pd.Timedelta(hours=23)))
                                    .resample(time="1D").sum(min_count=24), s["bbox"])
    rh = relative_humidity(ds["t2m"] - 273.15, ds["d2m"] - 273.15)
    hh = (rh >= 90).sel(time=slice(first, last_issue + pd.Timedelta(hours=23))).resample(time="1D").sum().astype("float32")
    import xarray as xr
    from src.ingestion.era5 import pixel_grid
    la, lo = pixel_grid(s["bbox"])
    at = hh.interp(latitude=xr.DataArray(la, dims="pixel"), longitude=xr.DataArray(lo, dims="pixel"), method="linear")
    humid = pd.DataFrame({"pixel_id": np.tile([f"{a:.3f}_{o:.3f}" for a, o in zip(la, lo)], at.sizes["time"]),
                          "date": np.repeat(pd.to_datetime(at["time"].values), len(la)),
                          "rh_hours_ge90": at.transpose("time", "pixel").values.ravel()})

    # rows for the days after the merged table: static columns from the last known day, rain from CHIRPS / IFS
    new_days = pd.date_range(obs_end + pd.Timedelta(days=1), last_issue)
    static = std[std["date"] == obs_end].drop(columns=["date", "precip_mm"] + DAILY_COLUMNS)
    rows = [static.assign(date=d) for d in new_days]
    table = pd.concat([std] + rows, ignore_index=True) if rows else std.copy()
    rain, kinds = chirps_recent(s, new_days)
    src = {f"{d:%Y-%m-%d}": "chirps_final" for d in pd.date_range(std["date"].min(), obs_end)}
    offsets, qmap = fit_live_corrections(site, sites_cfg, exp)  # fitted on Mar-Dec 2024 (backtest S5)
    if len(new_days):
        m = table.loc[table["date"] > obs_end, ["pixel_id", "date"]].merge(rain, on=["pixel_id", "date"], how="left")
        m = m.merge(rain_ifs, on=["pixel_id", "date"], how="left")
        filled = m["precip_mm"].isna()
        m.loc[filled, "precip_mm"] = apply_rain_map(m.loc[filled, "ifs_precip_mm"].to_numpy(float), qmap)
        table.loc[table["date"] > obs_end, "precip_mm"] = m["precip_mm"].to_numpy().astype(table["precip_mm"].dtype)
        for d in new_days:
            src[f"{d:%Y-%m-%d}"] = {"final": "chirps_final", "prelim": "chirps_prelim"}.get(kinds.get(d), "ifs_rain")
    # the merged table's own preliminary days
    sj = ROOT / "data" / "raw" / f"chirps_{site}_sources.json"
    if sj.exists():
        for d in json.loads(sj.read_text(encoding="utf-8"))["prelim_dates"]:
            if d in src:
                src[d] = "chirps_prelim"
    # weather: ERA5 (ERA5T) wherever the merged table has it; bias-corrected IFS only for the days after it
    # (backtest S4/S5: uncorrected IFS cost Busia its rain skill through a +13.6 hPa surface-pressure offset)
    ifs = apply_weather_offsets(ifs[ifs["date"] > obs_end], offsets)
    w = table[["pixel_id", "date"]].merge(ifs[["pixel_id", "date"] + DAILY_COLUMNS], on=["pixel_id", "date"], how="left")
    has = w[DAILY_COLUMNS[0]].notna().to_numpy()
    for c in DAILY_COLUMNS:
        table.loc[has, c] = w.loc[has, c].to_numpy().astype(table[c].dtype)
    table = table[table["date"] <= last_issue]
    if table[["precip_mm"] + DAILY_COLUMNS].isna().any().any():
        raise SystemExit("live daily table has missing values (IFS window too short or CHIRPS gap?)")
    return table.sort_values(["pixel_id", "date"]).reset_index(drop=True), humid, src, obs_end


def run(site: str, issues, sites_cfg: dict, exp: dict) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    from app.build_shap import GROUPS, explain
    fexp = final_config(exp)
    horizons = sites_cfg["forecast"]["horizons"]
    wet_mm, exceed, window = fexp["rain"]["wet_threshold_mm"], fexp["rain"]["exceedance_mm"], fexp["climatology"]["window_days"]
    train_start, train_end = split_bounds(fexp, "train")
    table, humid, src, obs_end = live_daily(site, sites_cfg, exp, max(issues))
    models, smear, cols, run_dir = load_frozen(site, fexp, horizons)
    all_px = dev_pixel_sample(site, 10 ** 9)
    hist = pd.read_parquet(PROCESSED / f"features_{site}.parquet", columns=["pixel_id", "date", "precip_mm", "rh_pct"],
                           filters=[("date", ">=", train_start), ("date", "<=", train_end)])
    clim = fit_climatology(hist, wet_mm, exceed, window)
    del hist
    shap_meta = json.loads((C.DATA / "shap_meta.json").read_text(encoding="utf-8"))["sites"][site]
    has_blight = site == "nakuru"
    if has_blight:
        from src.models.late_blight import binary_climatology, risk_labels
        d_all = pd.read_parquet(PROCESSED / f"{site}_daily.parquet", columns=["pixel_id", "date", "t2m_min_c", "sp_hpa", "elev_mean_m"])
        h_all = pd.read_parquet(PROCESSED / f"{site}_humid_hours.parquet")
        live_part = table[["pixel_id", "date", "t2m_min_c", "sp_hpa", "elev_mean_m"]]
        d_all = pd.concat([d_all[d_all["date"] < live_part["date"].min()], live_part], ignore_index=True)
        # ERA5 humid hours where the file has them, IFS hours (uncorrected) only for the days after it
        h_new = humid[(humid["date"] > h_all["date"].max()) & (humid["date"] <= max(issues))]
        h_all = pd.concat([h_all, h_new], ignore_index=True)
        labels = risk_labels(d_all, h_all, train_start, train_end)
        tr_lab = labels[(labels["date"] >= train_start) & (labels["date"] <= train_end)]
        clim_risk = {r: binary_climatology(tr_lab, f"met_{r}", window) for r in ("hutton", "humid")}
        labels = labels[labels["date"] >= min(issues) - pd.Timedelta(days=3)]
        del d_all, h_all, tr_lab
        lb_models = {(r, h): joblib.load(MODELS / site / "late_blight_final" / r / f"h{h}" / "catboost_risk.joblib")
                     for r in ("hutton", "humid") for h in horizons}
    frames, shap_parts = [], []
    for t in issues:
        f = build_features(table[(table["date"] <= t) & (table["date"] >= t - pd.Timedelta(days=WINDOW_DAYS))], exp)
        f = f[f["date"] == t]
        assert f["pixel_id"].nunique() == len(all_px)
        for h in horizons:
            v = issue_table(f, clim, h, [t])
            p = predict_all(models[(h, "catboost")], smear[(h, "catboost")], v, cols[h], wet_mm)
            row = v[["pixel_id", "issue_date", "target_date"]].assign(
                horizon=np.int8(h), p_rain1=p["occ1"], p_rain10=p["occ10"], p_rain25=p["occ25"], rain_mm=p["two_stage"], rh=p["rh"],
                clim_rain1=v["clim_p_wet"], clim_rain10=v["clim_p_ge10"], clim_rain25=v["clim_p_ge25"],
                clim_rain_mm=v["clim_precip_mean"], clim_rh=v["clim_rh_mean"], obs_rain_mm=np.nan, obs_rh=np.nan)
            sh = v[["pixel_id", "issue_date"]].assign(horizon=np.int8(h))
            for target, model, kind, pred in (("rain1", models[(h, "catboost")]["occ1"], "prob", p["occ1"]),
                                              ("rh", models[(h, "catboost")]["rh"], "pct", p["rh"])):
                contrib, _, _ = explain(model, v, list(model.feature_names_), kind, pd.Series(pred))
                for g, name in enumerate(GROUPS):
                    sh[f"{target}|{name}"] = np.round(contrib[:, g] * 10).astype(np.int16)
            if has_blight:
                for r in ("hutton", "humid"):
                    m = lb_models[(r, h)]
                    rt = risk_issue_table(f, clim, labels, clim_risk[r], r, h, [t])
                    assert (rt["pixel_id"].to_numpy() == v["pixel_id"].to_numpy()).all()
                    pr = m.predict_proba(rt[list(m.feature_names_)])[:, 1]
                    row[f"blight_{r}"], row[f"obs_blight_{r}"], row[f"clim_blight_{r}"] = pr, np.nan, rt["clim_p_risk"].to_numpy()
                    contrib, _, _ = explain(m, rt, list(m.feature_names_), "prob", pd.Series(pr))
                    for g, name in enumerate(GROUPS):
                        sh[f"blight_{r}|{name}"] = np.round(contrib[:, g] * 10).astype(np.int16)
            frames.append(row)
            shap_parts.append(sh)
        print(f"  {site}: forecast issued {t:%Y-%m-%d} (rain on the issue day: {src.get(f'{t:%Y-%m-%d}')})")
    out = pd.concat(frames, ignore_index=True)
    out["period"] = np.int8(LIVE_PERIOD)
    num = [c for c in out.columns if c not in ("pixel_id", "issue_date", "target_date", "horizon", "period")]
    out[num] = out[num].astype("float32")
    shap = pd.concat(shap_parts, ignore_index=True)
    _ = shap_meta  # typical values: the expected values are the same models' (checked by explain's additivity)
    info = {"issues": [f"{d:%Y-%m-%d}" for d in issues], "observed_through": f"{obs_end:%Y-%m-%d}",
            "rain_source": {d: s for d, s in src.items() if d >= f"{min(issues) - pd.Timedelta(days=30):%Y-%m-%d}"}}
    return out.sort_values(["issue_date", "horizon", "pixel_id"]).reset_index(drop=True), shap, info


def main() -> None:
    ap = argparse.ArgumentParser(description="Live forecasts with the frozen final models.")
    ap.add_argument("--issue", help="issue day YYYY-MM-DD (default: every day from the replay's end to yesterday)")
    args = ap.parse_args()
    sites_cfg, exp = load_configs()
    meta = json.loads((C.DATA / "meta.json").read_text(encoding="utf-8"))
    yesterday = pd.Timestamp.now("UTC").tz_localize(None).normalize() - pd.Timedelta(days=1)
    OUT.mkdir(parents=True, exist_ok=True)
    live = {"made": pd.Timestamp.now("UTC").isoformat(timespec="seconds"), "attribution": ATTRIBUTION,
            "models": "final test models (trained 2016-2024), frozen", "sites": {}}
    for site in sites_cfg["sites"]:
        first = pd.Timestamp(meta[site]["issue_max"]) + pd.Timedelta(days=1)
        issues = [pd.Timestamp(args.issue)] if args.issue else list(pd.date_range(first, yesterday))
        if not issues:
            print(f"{site}: nothing to do (replay already reaches {meta[site]['issue_max']})")
            continue
        print(f"=== {site}: live forecasts issued {issues[0]:%Y-%m-%d}..{issues[-1]:%Y-%m-%d} ===")
        df, shap, info = run(site, issues, sites_cfg, exp)
        df.to_parquet(OUT / f"{site}_live.parquet", index=False, row_group_size=50_000)
        shap.to_parquet(OUT / f"{site}_live_shap.parquet", index=False, compression="zstd")
        live["sites"][site] = {**info, "issue_min": info["issues"][0], "issue_max": info["issues"][-1]}
        print(f"  saved {OUT / f'{site}_live.parquet'} ({len(df):,} rows) and the explanations")
    (OUT / "live_meta.json").write_text(json.dumps(live, indent=2), encoding="utf-8")
    print(f"saved {OUT / 'live_meta.json'}")


if __name__ == "__main__":
    main()
