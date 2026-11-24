"""Input-swap backtest for the live branch (docs/NEXT_STEPS.md section 12, option a), on validation 2024.

    python -m src.live.backtest fetch            # Open-Meteo IFS 0.25 archive, Mar-Dec 2024 (cached)
    python -m src.live.backtest run --site nakuru
    python -m src.live.backtest run --site busia
    python -m src.live.backtest analyse

Question: the final models were trained on CHIRPS + ERA5, which arrive days late. If a live system feeds
them other inputs, how much skill is lost? Answered with the DEVELOPMENT models (models/<site>/
train60_score<N>/: fitted 2016-2022, early stopping 2023), which never saw 2024, on validation 2024 only.
The test period (2025-2026) and the post-test period are not used.

Scenarios (same pixels, same target days, same truth and baselines; paired comparisons):
  S0 standard      CHIRPS + ERA5, as in training. GUARD: must reproduce the saved validation forecasts.
  S1 ifs_weather   ERA5's six variables replaced by Open-Meteo ECMWF IFS 0.25 (from 2024-03-01), processed
                   by the same code (hourly -> daily UTC -> bilinear onto the pixels).
  S2 ifs_live      S1 plus the rain a live system would not have yet: CHIRPS preliminary is published
                   2 days after each pentad (days 1-5, 6-10, 11-15, 16-20, 21-25, 26-end) ends, so on
                   issue day t the days after the last published pentad (2 to 7 days, including t) are
                   filled with IFS rainfall. Features are rebuilt for each issue day.
Fixes tried after the first run (S1 and S2 lost most day-1 skill; docs/PROJECT_CONTEXT.md section 20):
  S3 ifs_weather_bc  S1 with each pixel's IFS weather shifted by its mean difference from ERA5 (one offset per
                   pixel and variable). CROSS-FITTED in time: offsets for March-July come from August-December
                   and the other way round, so no day is corrected with its own data.
  S4 live_v2       the realistic live case, improved: ERA5T arrives about 5 days late, so ERA5 is used up to
                   t-5 and bias-corrected IFS (S3's offsets) only for the last 5 days (t-4..t); rain as in S2
                   (CHIRPS preliminary, the gap filled with raw IFS rain).
  S5 live_v2_qm    S4 with the gap rain QUANTILE-MAPPED to CHIRPS (one mapping per site, cross-fitted the same
                   way), so its wet-day frequency and amounts look like CHIRPS'.
Truth stays CHIRPS (rain) and ERA5 (humidity); baselines (climatology, persistence) come from S0.

Target days 2024-04-14..2024-12-31 for every horizon (issue days from 2024-04-07: IFS dew point is
complete only from March 2024, and ERA5 features need 21 days of history).

Outputs: reports/live/backtest_skill.csv, backtest_differences.csv, input_diagnostics.csv,
backtest_late_blight.csv; predictions in models/<site>/train60_score<N>/h<h>/backtest_predictions.parquet.
"""
import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.features.build import build_features
from src.ingestion.era5 import DAILY_COLUMNS, interpolate_to_pixels, to_daily_grid
from src.live.openmeteo import fetch_grid
from src.models.baselines import fit_climatology
from src.models.evaluation import load_configs, load_site, split_bounds
from src.models.posttest import issue_table
from src.models.train import MODELS, baseline_preds, dev_pixel_sample, horizon_table, load_features, predict
from src.models.transfer import predict_all
from src.models.uncertainty import block_bootstrap_weights, bootstrap_site_horizon, paired_differences

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports" / "live"
IFS_START = pd.Timestamp("2024-03-01")
HISTORY_START = pd.Timestamp("2023-06-01")   # rain history for long windows and "days since wet"
ISSUE_START = pd.Timestamp("2024-04-07")
SCENARIOS = ["standard", "ifs_weather", "ifs_live", "ifs_weather_bc", "live_v2", "live_v2_qm"]
FOLD_SPLIT = pd.Timestamp("2024-08-01")  # cross-fitting: dates before / from this day form the two folds
ERA5T_LAG_DAYS = 5                       # on issue day t, ERA5T covers days up to t-5
TASKS = ["occ1", "occ10", "occ25", "wet", "direct", "rh"]


def dev_run(site: str) -> Path:
    n = len(dev_pixel_sample(site, 10 ** 9))
    return MODELS / site / f"train60_score{n}"


def last_prelim_day(t: pd.Timestamp) -> pd.Timestamp:
    """Last day covered by CHIRPS preliminary on issue day t: the end of the latest pentad that ended at
    least 2 days before t (pentads 1-5, 6-10, 11-15, 16-20, 21-25, 26-end; published 2 days after)."""
    d = t - pd.Timedelta(days=2)
    ends = [5, 10, 15, 20, 25, d.days_in_month]
    past = [e for e in ends if e <= d.day]
    if past:
        return d.replace(day=past[-1])
    prev = d.replace(day=1) - pd.Timedelta(days=1)  # end of the previous month (its last pentad)
    return prev


def daily_rain_at_pixels(pr, bbox: dict) -> pd.DataFrame:
    """Daily IFS rain (time, latitude, longitude) -> bilinear at every CHIRPS pixel centre (like ERA5)."""
    import xarray as xr
    from src.ingestion.era5 import pixel_grid
    lats, lons = pixel_grid(bbox)
    at = pr.sortby("latitude").sortby("longitude").interp(
        latitude=xr.DataArray(lats, dims="pixel"), longitude=xr.DataArray(lons, dims="pixel"), method="linear")
    out = pd.DataFrame({"pixel_id": np.tile([f"{a:.3f}_{o:.3f}" for a, o in zip(lats, lons)], at.sizes["time"]),
                        "date": np.repeat(pd.to_datetime(at["time"].values), len(lats)),
                        "ifs_precip_mm": at.transpose("time", "pixel").values.ravel().astype("float32")})
    out["ifs_precip_mm"] = out["ifs_precip_mm"].clip(lower=0)
    return out


def ifs_daily(site: str, sites_cfg: dict, end: pd.Timestamp) -> pd.DataFrame:
    """IFS daily values on every pixel: the 11 ERA5-style columns plus ifs_precip_mm."""
    s = sites_cfg["sites"][site]
    ds = fetch_grid(s["era5_area"], IFS_START, end, endpoint="archive", cache_key=site)
    daily, hours = to_daily_grid(ds, IFS_START, end)
    if (hours != 24).any():
        raise SystemExit(f"IFS hours per day not all 24 ({int((hours != 24).sum())} days)")
    df = interpolate_to_pixels(daily, s["bbox"])
    pr = ds["tp"].sel(time=slice(IFS_START, end + pd.Timedelta(hours=23))).resample(time="1D").sum(min_count=24)
    df = df.merge(daily_rain_at_pixels(pr, s["bbox"]), on=["pixel_id", "date"], validate="one_to_one")
    # the archive's first hour of rain (an accumulated variable) is missing, so the first day's total is NaN;
    # rain is only used for filling days from about two weeks before the first issue day, so that is allowed
    early = df["date"] < ISSUE_START - pd.Timedelta(days=14)
    if df[DAILY_COLUMNS].isna().any().any() or df.loc[~early, "ifs_precip_mm"].isna().any():
        raise SystemExit("missing values in the IFS daily table")
    return df


def scenario_daily(std: pd.DataFrame, ifs: pd.DataFrame) -> pd.DataFrame:
    """The merged daily table with the 11 ERA5 columns replaced by IFS from IFS_START on."""
    out = std.copy()
    m = out[["pixel_id", "date"]].merge(ifs[["pixel_id", "date"] + DAILY_COLUMNS], on=["pixel_id", "date"], how="left")
    has = m[DAILY_COLUMNS[0]].notna().to_numpy()
    for c in DAILY_COLUMNS:
        out.loc[has, c] = m.loc[has, c].to_numpy().astype(out[c].dtype)
    return out


def bias_corrected(std: pd.DataFrame, ifs: pd.DataFrame) -> pd.DataFrame:
    """IFS daily table with (a) the 11 weather columns shifted by each pixel's mean ERA5-minus-IFS difference
    and (b) ifs_precip_mm quantile-mapped to CHIRPS (one mapping per site). Both are cross-fitted: values
    dated before FOLD_SPLIT are corrected with what was fitted on the dates from FOLD_SPLIT on, and the
    other way round."""
    m = paired_inputs(std, ifs)
    later = (m["date"] >= FOLD_SPLIT).to_numpy()
    out = ifs.copy()
    x, y = m["ifs_precip_mm"].to_numpy(float), m["precip_mm"].to_numpy(float)  # IFS rain vs CHIRPS
    ok = ~np.isnan(x)
    rain = x.copy()
    for fold in (False, True):
        fit, use = later != fold, later == fold
        corrected = apply_weather_offsets(ifs[use], weather_offsets(m[fit]))
        for c in DAILY_COLUMNS:
            out.loc[use, c] = corrected[c].to_numpy()
        rain[use & ok] = apply_rain_map(x[use & ok], rain_map(x[fit & ok], y[fit & ok]))
    out["ifs_precip_mm"] = rain.astype("float32")
    return out


def paired_inputs(std: pd.DataFrame, ifs: pd.DataFrame) -> pd.DataFrame:
    """IFS rows (same order) with ERA5's weather as <column>_ref and CHIRPS as precip_mm, same pixel-days."""
    return ifs.merge(std[["pixel_id", "date", "precip_mm"] + DAILY_COLUMNS], on=["pixel_id", "date"],
                     how="left", suffixes=("", "_ref"), validate="one_to_one")


def weather_offsets(m: pd.DataFrame) -> pd.DataFrame:
    """Mean ERA5-minus-IFS difference per pixel (index) and weather variable (columns)."""
    return pd.DataFrame({c: (m[f"{c}_ref"].astype(float) - m[c].astype(float)).groupby(m["pixel_id"]).mean()
                         for c in DAILY_COLUMNS})


def apply_weather_offsets(df: pd.DataFrame, offsets: pd.DataFrame) -> pd.DataFrame:
    """Shift each pixel's IFS weather by its offset and keep every variable in its physical range."""
    out = df.copy()
    for c in DAILY_COLUMNS:
        new = df[c].astype(float).to_numpy() + df["pixel_id"].map(offsets[c]).to_numpy(float)
        if c == "tcc":
            new = np.clip(new, 0.0, 1.0)
        elif c.startswith("rh"):
            new = np.clip(new, 0.0, 100.0)
        elif c.startswith("wind"):
            new = np.maximum(new, 0.0)
        out[c] = new.astype(df[c].dtype)
    return out


def rain_map(ifs_rain: np.ndarray, chirps_rain: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Empirical quantile map IFS -> CHIRPS (1,001 quantiles, pooled over the site's pixels and days)."""
    q = np.linspace(0, 1, 1001)
    return np.maximum.accumulate(np.quantile(ifs_rain, q)), np.quantile(chirps_rain, q)


def apply_rain_map(x: np.ndarray, qmap: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    return np.interp(x, qmap[0], qmap[1])


def fit_live_corrections(site: str, sites_cfg: dict, exp: dict) -> tuple[pd.DataFrame, tuple]:
    """Weather offsets and rain map for the LIVE job, fitted on all of Mar-Dec 2024 (validation-year inputs;
    no cross-fitting is needed because live days lie outside 2024). Uses the cached Open-Meteo archive."""
    val_end = split_bounds(exp, "validation")[1]
    std = load_site(site, until=val_end)
    std = std[std["date"] >= IFS_START].reset_index(drop=True)
    m = paired_inputs(std, ifs_daily(site, sites_cfg, val_end))
    ok = m["ifs_precip_mm"].notna().to_numpy()
    return weather_offsets(m), rain_map(m.loc[ok, "ifs_precip_mm"].to_numpy(float), m.loc[ok, "precip_mm"].to_numpy(float))


def live_v2_features(std: pd.DataFrame, bc_daily: pd.DataFrame, fills: dict, issue_days, lag: dict, exp: dict) -> dict:
    """S4/S5 features, rebuilt per issue day t: ERA5 up to t-5, bias-corrected IFS weather for t-4..t, CHIRPS
    rain up to the last published pentad and each fill (name -> rain Series by pixel_id, date) after it.
    `bc_daily` must be `std` with the weather columns replaced (same rows, same order)."""
    rows = {name: [] for name in fills}
    for t in issue_days:
        sel = ((std["date"] <= t) & (std["date"] >= t - pd.Timedelta(days=200))).to_numpy()
        d = std[sel].copy()
        recent = (d["date"] > t - pd.Timedelta(days=ERA5T_LAG_DAYS)).to_numpy()
        for c in DAILY_COLUMNS:
            d.loc[recent, c] = bc_daily.loc[d.index[recent], c].to_numpy()
        fill = (d["date"] > lag[t]).to_numpy()
        idx = pd.MultiIndex.from_frame(d.loc[fill, ["pixel_id", "date"]])
        for name, series in fills.items():
            e = d.copy()
            e.loc[fill, "precip_mm"] = series.reindex(idx).to_numpy().astype(e["precip_mm"].dtype)
            f = build_features(e, exp)
            rows[name].append(f[f["date"] == t])
    return {name: pd.concat(r, ignore_index=True) for name, r in rows.items()}


def run_site(site: str, sites_cfg: dict, exp: dict) -> None:
    horizons = sites_cfg["forecast"]["horizons"]
    wet_mm, exceed, window = exp["rain"]["wet_threshold_mm"], exp["rain"]["exceedance_mm"], exp["climatology"]["window_days"]
    train_start, train_end = split_bounds(exp, "train")
    val_start, val_end = split_bounds(exp, "validation")
    es_start = pd.Timestamp(str(exp["modelling"]["early_stop_start"]))
    targets = pd.date_range(ISSUE_START + pd.Timedelta(days=max(horizons)), val_end)
    issue_days = pd.date_range(ISSUE_START, val_end - pd.Timedelta(days=1))
    run = dev_run(site)
    print(f"=== {site}: backtest with the development models ({run.name}); targets {targets[0]:%Y-%m-%d}..{targets[-1]:%Y-%m-%d} ===")

    # 1. data: standard daily table to the end of validation only; IFS for Mar-Dec 2024
    std = load_site(site, until=val_end)
    std = std[std["date"] >= HISTORY_START].reset_index(drop=True)
    ifs = ifs_daily(site, sites_cfg, val_end)
    diag = input_diagnostics(site, std, ifs)
    s1_daily = scenario_daily(std, ifs)

    # 2. features. S0 from the saved feature table; S1 built once; S2 rebuilt per issue day
    all_px = dev_pixel_sample(site, 10 ** 9)
    s0 = load_features(site, all_px, until=val_end, start=HISTORY_START)
    s1 = build_features(s1_daily, exp)
    lag = {t: last_prelim_day(t) for t in issue_days}
    ifs_pr = ifs.set_index(["pixel_id", "date"])["ifs_precip_mm"]
    s2_rows, gaps = [], []
    for k, t in enumerate(issue_days):
        d = s1_daily[s1_daily["date"] <= t].copy()
        fill = d["date"] > lag[t]
        idx = pd.MultiIndex.from_frame(d.loc[fill, ["pixel_id", "date"]])
        d.loc[fill, "precip_mm"] = ifs_pr.reindex(idx).to_numpy().astype(d["precip_mm"].dtype)
        gaps.append((t - lag[t]).days)
        f = build_features(d[d["date"] >= t - pd.Timedelta(days=200)], exp)
        s2_rows.append(f[f["date"] == t])
        if k % 40 == 0:
            print(f"  S2 features: issue day {k + 1} of {len(issue_days)} ({t:%Y-%m-%d}, rain filled for the last {gaps[-1]} days)")
    s2 = pd.concat(s2_rows, ignore_index=True)
    # guards: a 200-day window gives the same features as the full history when nothing is filled, both
    # for the IFS table (S1) and for the standard data against the saved feature table (S0)
    for label, daily_tab, full in (("S1", s1_daily, s1), ("S0", std, s0)):
        t5 = issue_days[5]
        chk = build_features(daily_tab[(daily_tab["date"] <= t5) & (daily_tab["date"] >= t5 - pd.Timedelta(days=200))], exp)
        a = chk[chk["date"] == t5].sort_values("pixel_id").reset_index(drop=True)
        b = full[full["date"] == t5].sort_values("pixel_id").reset_index(drop=True)[a.columns]
        num = [c for c in a.columns if pd.api.types.is_numeric_dtype(a[c])]
        if len(a) != len(b) or not np.allclose(a[num].to_numpy(float), b[num].to_numpy(float), equal_nan=True, atol=1e-4):
            raise SystemExit(f"{label}: the 200-day window changes features even without filling: widen it")
    print(f"  PASS  per-issue-day rebuild equals the full build when nothing is filled; rain gaps {min(gaps)}-{max(gaps)} days "
          f"(mean {np.mean(gaps):.1f})")

    # fixes (S3-S5): cross-fitted bias correction of the weather and quantile mapping of the gap rain
    ifs_bc = bias_corrected(std, ifs)
    diag_bc = input_diagnostics(site, std, ifs_bc, label="bias-corrected IFS (cross-fitted)")
    diag = pd.concat([diag.assign(inputs="ifs"), diag_bc.assign(inputs="ifs_bias_corrected")], ignore_index=True)
    s3_daily = scenario_daily(std, ifs_bc)
    s3 = build_features(s3_daily, exp)
    del s1_daily
    qm_pr = ifs_bc.set_index(["pixel_id", "date"])["ifs_precip_mm"]
    v2 = live_v2_features(std, s3_daily, {"live_v2": ifs_pr, "live_v2_qm": qm_pr}, issue_days, lag, exp)
    s4, s5 = v2["live_v2"], v2["live_v2_qm"]
    del s3_daily, v2
    print(f"  S3-S5 features built (ERA5 up to t-{ERA5T_LAG_DAYS}, corrected IFS after; gap rain raw or quantile-mapped)")

    # 3. models, smearing (as fitted in src.models.train) and climatology fitted on the training years
    hist = load_features(site, all_px, until=train_end, start=train_start, columns=["pixel_id", "date", "precip_mm", "rh_pct"])
    clim = fit_climatology(hist, wet_mm, exceed, window)
    del hist
    dev_px = dev_pixel_sample(site, exp["modelling"]["dev_pixels"])
    feat_tr = load_features(site, dev_px, until=train_end)
    clim_tr = fit_climatology(feat_tr[feat_tr["date"] >= train_start], wet_mm, exceed, window)
    rows_skill, rows_diff = [], []
    for h in horizons:
        models = {t: joblib.load(run / f"h{h}" / f"catboost_{t}.joblib") for t in TASKS}
        cols = list(models["occ1"].feature_names_)
        fit = horizon_table(feat_tr, clim_tr, h, pd.date_range(train_start + pd.Timedelta(days=h), es_start - pd.Timedelta(days=1)))
        wf = fit[fit["y_rain"] >= wet_mm]
        smear = float(np.mean(np.exp(np.log1p(wf["y_rain"]) - predict(models["wet"], wf[cols], False))))
        del fit, wf
        issues = targets - pd.Timedelta(days=h)
        base = issue_table(s0, clim, h, issues)
        truth = base[["pixel_id", "target_date", "y_rain", "y_rh"]]
        out = base[["pixel_id", "issue_date", "target_date", "y_rain", "y_rh"]].copy()
        for name, p in baseline_preds(base, wet_mm, exceed).items():
            for key, arr in p.items():
                out[f"{name}__{key}"] = np.asarray(arr, float)
        for name, feat in zip(SCENARIOS, (s0, s1, s2, s3, s4, s5)):
            t = issue_table(feat, clim, h, issues).drop(columns=["y_rain", "y_rh"]).merge(
                truth, on=["pixel_id", "target_date"], validate="one_to_one")
            t = out[["pixel_id", "target_date"]].merge(t, on=["pixel_id", "target_date"], how="left", validate="one_to_one")
            pr = predict_all(models, smear, t, cols, wet_mm)
            for key, arr in pr.items():
                out[f"{name}__{key}"] = np.asarray(arr, float)
        # GUARD: S0 reproduces the saved validation forecasts of the development models
        ref = pd.read_parquet(run / f"h{h}" / "val_predictions.parquet", columns=["pixel_id", "target_date", "catboost__occ1", "catboost__rh"])
        m = ref.merge(out[["pixel_id", "target_date", "standard__occ1", "standard__rh"]], on=["pixel_id", "target_date"])
        err = max(float((m["catboost__occ1"] - m["standard__occ1"]).abs().max()), float((m["catboost__rh"] - m["standard__rh"]).abs().max()))
        if len(m) != len(out) or err > 1e-9:
            raise SystemExit(f"GUARD FAILED h{h}: standard inputs do not reproduce the saved forecasts ({len(m)} rows, max {err:.3g})")
        print(f"  PASS  h{h}: standard inputs reproduce {len(m):,} saved validation forecasts (max difference {err:.1g})")
        out.drop(columns="issue_date").to_parquet(run / f"h{h}" / "backtest_predictions.parquet", index=False)
    diag.to_csv(OUT / f"input_diagnostics_{site}.csv", index=False)
    print(f"  saved {run.name}/h*/backtest_predictions.parquet and reports/live/input_diagnostics_{site}.csv")


def input_diagnostics(site: str, std: pd.DataFrame, ifs: pd.DataFrame, label: str = "IFS 0.25") -> pd.DataFrame:
    """IFS minus ERA5 (and IFS rain minus CHIRPS) on the same pixel-days, Mar-Dec 2024; plus the share of
    IFS values outside the range the models saw in training (2016-2022 fitting rows)."""
    OUT.mkdir(parents=True, exist_ok=True)
    m = std.merge(ifs, on=["pixel_id", "date"], suffixes=("", "_ifs"))
    tr = load_site(site, until=pd.Timestamp("2022-12-31"), columns=["date"] + DAILY_COLUMNS)
    rows = []
    for c in DAILY_COLUMNS + ["precip_mm"]:
        a = m[c].astype(float)
        b = m["ifs_precip_mm" if c == "precip_mm" else f"{c}_ifs"].astype(float)
        ok = a.notna() & b.notna()
        a, b = a[ok], b[ok]
        d = b - a
        r = {"site": site, "variable": c, "era5_or_chirps_mean": a.mean(), "ifs_mean": b.mean(), "bias": d.mean(),
             "mae": d.abs().mean(), "rmse": float(np.sqrt((d ** 2).mean())), "corr": float(np.corrcoef(a, b)[0, 1])}
        if c in tr:
            lo, hi = tr[c].min(), tr[c].max()
            r.update(train_min=lo, train_max=hi, share_outside_training_range=float(((b < lo) | (b > hi)).mean()))
        if c == "precip_mm":
            r["wet_dry_agreement"] = float(((a >= 1) == (b >= 1)).mean())
        rows.append(r)
    out = pd.DataFrame(rows)
    print(f"  inputs, {label} vs ERA5 (rain: vs CHIRPS), Mar-Dec 2024, all pixels:")
    for r in out.itertuples():
        extra = f", outside training range {r.share_outside_training_range:.1%}" if not pd.isna(getattr(r, "share_outside_training_range", np.nan)) else ""
        print(f"    {r.variable:12} bias {r.bias:+8.2f}  MAE {r.mae:7.2f}  corr {r.corr:5.2f}{extra}")
    return out


def analyse(sites_cfg: dict, exp: dict) -> None:
    ucfg, wet_mm, exceed = exp["uncertainty"], exp["rain"]["wet_threshold_mm"], exp["rain"]["exceedance_mm"]
    skill, diffs = [], []
    for site in sites_cfg["sites"]:
        run = dev_run(site)
        for h in sites_cfg["forecast"]["horizons"]:
            pr = pd.read_parquet(run / f"h{h}" / "backtest_predictions.parquet")
            W = block_bootstrap_weights(pr["target_date"].nunique(), ucfg["block_days"], ucfg["n_boot"], ucfg["seed"])
            rows, draws = bootstrap_site_horizon(pr, W, wet_mm, exceed)
            key = {"site": site, "horizon": h, "n_days": pr["target_date"].nunique()}
            skill += [{**key, **r} for r in rows]
            diffs += [{**key, **d} for d in paired_differences(draws, pd.DataFrame(rows))]
    skill, diffs = pd.DataFrame(skill), pd.DataFrame(diffs)
    OUT.mkdir(parents=True, exist_ok=True)
    skill.to_csv(OUT / "backtest_skill.csv", index=False)
    diffs.to_csv(OUT / "backtest_differences.csv", index=False)
    fmt = lambda e, lo, hi: f"{e:+.3f} [{lo:+.3f},{hi:+.3f}]{'*' if lo > 0 or hi < 0 else ' '}"  # noqa: E731
    sk = skill[skill["metric"].isin(["bss", "rmse_skill"]) & (skill["reference"] == "climatology")
               & skill["target"].isin(["rain>=1mm", "rh_pct"]) & skill["model"].isin(SCENARIOS)]
    print("\nBACKTEST (validation 2024, development models): skill vs climatology [95%], * = excludes 0")
    t = sk.assign(v=sk.apply(lambda r: fmt(r["estimate"], r["ci_low"], r["ci_high"]), axis=1)).pivot_table(
        index="horizon", columns=["site", "target", "model"], values="v", aggfunc="first")
    for site in sites_cfg["sites"]:
        for target in ["rain>=1mm", "rh_pct"]:
            print(f"\n  {site}, {target}:")
            print("    " + t[(site, target)][SCENARIOS].to_string().replace("\n", "\n    "))
    # comparisons are stored as "a - b" with a before b alphabetically; flip the sign where needed
    wanted = [("ifs_weather", "standard"), ("ifs_live", "standard"), ("ifs_weather_bc", "ifs_weather"),
              ("live_v2", "ifs_live"), ("live_v2_qm", "live_v2"), ("live_v2_qm", "standard")]
    d = diffs[diffs["target"].isin(["rain>=1mm", "rh_pct"])].set_index(["site", "horizon", "target", "comparison"])
    print("\n  paired differences in skill (first minus second):")
    for a, b in wanted:
        for site in sites_cfg["sites"]:
            for target in ["rain>=1mm", "rh_pct"]:
                for h in sites_cfg["forecast"]["horizons"]:
                    if (site, h, target, f"{a} - {b}") in d.index:
                        r = d.loc[(site, h, target, f"{a} - {b}")]
                        e, lo, hi = r["estimate"], r["ci_low"], r["ci_high"]
                    else:
                        r = d.loc[(site, h, target, f"{b} - {a}")]
                        e, lo, hi = -r["estimate"], -r["ci_high"], -r["ci_low"]
                    print(f"    {a + ' - ' + b:30} {site:6} h{h} {target:9} {fmt(e, lo, hi)}")
    print(f"\nsaved {OUT}/backtest_skill.csv, backtest_differences.csv")


def main() -> None:
    ap = argparse.ArgumentParser(description="Live-input backtest on validation 2024.")
    ap.add_argument("step", choices=["fetch", "run", "analyse"])
    ap.add_argument("--site")
    args = ap.parse_args()
    sites_cfg, exp = load_configs()
    sites = [args.site] if args.site else list(sites_cfg["sites"])
    if args.step == "fetch":
        _, val_end = split_bounds(exp, "validation")
        for s in sites:
            fetch_grid(sites_cfg["sites"][s]["era5_area"], IFS_START, val_end, endpoint="archive", cache_key=s)
    elif args.step == "run":
        for s in sites:
            run_site(s, sites_cfg, exp)
    else:
        analyse(sites_cfg, exp)


if __name__ == "__main__":
    main()
