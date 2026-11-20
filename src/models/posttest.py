"""Post-test period: the FROZEN final models on data after the reported test (approved 20 Nov 2026).

    python -m src.models.posttest score --site nakuru     # forecasts (all pixels, batches)
    python -m src.models.posttest score --site busia
    python -m src.models.posttest analyse                 # skill with 95% block-bootstrap intervals

What it does (nothing is trained; nothing in the reported test changes):
- Loads the saved final-test models (models/<site>/final_train60_score<N>/h<h>/, all three
  algorithms) and the late-blight models (models/nakuru/late_blight_final/<rule>/h<h>/).
- Inputs exactly as in the final test: the per-pixel climatology (model input and baseline) fitted
  on 2016-2024, the smearing factor recomputed from the same fitting rows (src.models.final_test).
- GUARD: before writing anything, the same code path re-predicts the first and last 7 target days
  of the test period for the first pixel batch and must reproduce the saved test forecasts
  (test_predictions.parquet) to 1e-9. If it does not, the run stops.
- Rows: every issue day from 7 days before the post-test start to the last data day, every horizon,
  keeping targets from the post-test start onward. Targets after the last data day have no
  observation yet: they are kept (the newest forecasts, for the dashboard) with NaN outcomes and
  are never scored.
- Each row gets `period`: posttest_final (target on or before `final_through`: final CHIRPS),
  posttest_prelim (later targets: PRELIMINARY CHIRPS, provisional), or forecast (not yet observed).

Outputs (never the test files):
    models/<site>/final_train60_score<N>/h<h>/posttest_predictions.parquet
    models/nakuru/late_blight_final/<rule>/h<h>/posttest_predictions.parquet
    reports/posttest/skill.csv, differences.csv, calibration.csv, late_blight_skill.csv
"""
import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.models.baselines import add_climatology, fit_climatology
from src.models.calibration import calibration_slope
from src.models.evaluation import PROCESSED, doy365, load_configs, split_bounds
from src.models.final_test import final_config
from src.models.train import (ALGOS, MODELS, baseline_preds, dev_pixel_sample, feature_columns, horizon_table,
                              load_features, predict)
from src.models.transfer import predict_all
from src.models.uncertainty import block_bootstrap_weights, bootstrap_site_horizon, paired_differences

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports" / "posttest"
SCORE_BATCH = 200


def posttest_bounds(exp: dict):
    p = exp["splits"]["posttest"]
    return pd.Timestamp(str(p["start"])), pd.Timestamp(str(p["end"])), pd.Timestamp(str(p["final_through"]))


def final_run(site: str) -> Path:
    runs = sorted((MODELS / site).glob("final_train*_score*"))
    if len(runs) != 1:
        raise SystemExit(f"Expected one final run folder for {site}, found {[r.name for r in runs]}")
    return runs[0]


def issue_table(feat: pd.DataFrame, clim: pd.DataFrame, h: int, issue_dates) -> pd.DataFrame:
    """Like train.horizon_table, but built from ISSUE days, so targets without an observation yet
    are allowed (y_rain / y_rh NaN). Columns are selected by name later, so their order here does
    not matter. Checked against the saved test forecasts by the guard."""
    t = feat[feat["date"].isin(pd.DatetimeIndex(issue_dates))].rename(columns={"date": "issue_date"})
    t = t.assign(target_date=t["issue_date"] + pd.Timedelta(days=h))
    y = feat[["pixel_id", "date", "precip_mm", "rh_pct"]].rename(
        columns={"date": "target_date", "precip_mm": "y_rain", "rh_pct": "y_rh"})
    t = t.merge(y, on=["pixel_id", "target_date"], how="left", validate="one_to_one")
    return add_climatology(t, clim).drop(columns="doy").reset_index(drop=True)


def load_frozen(site: str, fexp: dict, horizons) -> tuple[dict, dict, dict, pd.DataFrame]:
    """Saved models, smearing factors and fitting columns, recomputed exactly as final_test does."""
    mcfg, wet_mm = fexp["modelling"], fexp["rain"]["wet_threshold_mm"]
    exceed, window = fexp["rain"]["exceedance_mm"], fexp["climatology"]["window_days"]
    train_start, train_end = split_bounds(fexp, "train")
    es_start = pd.Timestamp(str(mcfg["early_stop_start"]))
    run = final_run(site)
    pixels = dev_pixel_sample(site, mcfg["dev_pixels"])
    feat = load_features(site, pixels, until=train_end)
    clim = fit_climatology(feat[feat["date"] >= train_start], wet_mm, exceed, window)
    tasks = [f"occ{t:g}" for t in [wet_mm] + exceed] + ["wet", "direct", "rh"]
    models, smear, cols = {}, {}, {}
    for h in horizons:
        fit = horizon_table(feat, clim, h, pd.date_range(train_start + pd.Timedelta(days=h),
                                                         es_start - pd.Timedelta(days=1)))
        cols[h] = feature_columns(fit)
        wet_fit = fit[fit["y_rain"] >= wet_mm]
        for algo in ALGOS:
            models[(h, algo)] = {t: joblib.load(run / f"h{h}" / f"{algo}_{t}.joblib") for t in tasks}
            smear[(h, algo)] = float(np.mean(np.exp(np.log1p(wet_fit["y_rain"])
                                                    - predict(models[(h, algo)]["wet"], wet_fit[cols[h]], False))))
        del fit, wet_fit
    return models, smear, cols, run


def predict_rows(va, models, smear, cols, h, wet_mm, exceed) -> pd.DataFrame:
    out = va[["pixel_id", "issue_date", "target_date", "y_rain", "y_rh"]].copy()
    preds = baseline_preds(va, wet_mm, exceed)
    preds.update({a: predict_all(models[(h, a)], smear[(h, a)], va, cols[h], wet_mm) for a in ALGOS})
    for name, p in preds.items():
        for key, arr in p.items():
            out[f"{name}__{key}"] = np.asarray(arr, dtype=float)
    return out


def label_period(df: pd.DataFrame, last_obs: pd.Timestamp, final_through: pd.Timestamp) -> pd.DataFrame:
    df["period"] = np.where(df["target_date"] > last_obs, "forecast",
                            np.where(df["target_date"] <= final_through, "posttest_final", "posttest_prelim"))
    return df


def score_site(site: str, sites_cfg: dict, exp: dict) -> None:
    fexp = final_config(exp)
    horizons = sites_cfg["forecast"]["horizons"]
    wet_mm, exceed, window = fexp["rain"]["wet_threshold_mm"], fexp["rain"]["exceedance_mm"], fexp["climatology"]["window_days"]
    train_start, train_end = split_bounds(fexp, "train")
    test_start, test_end = split_bounds(fexp, "test")
    ps_start, ps_end, final_through = posttest_bounds(exp)
    issue_days = pd.date_range(ps_start - pd.Timedelta(days=max(horizons)), ps_end)
    print(f"=== {site}: post-test with the FROZEN final models (trained to {train_end:%Y-%m-%d}). "
          f"Issue days {issue_days[0]:%Y-%m-%d}..{issue_days[-1]:%Y-%m-%d}, targets from {ps_start:%Y-%m-%d}; "
          f"observed to {ps_end:%Y-%m-%d} (final CHIRPS to {final_through:%Y-%m-%d}) ===")

    models, smear, cols, run = load_frozen(site, fexp, horizons)
    all_px = dev_pixel_sample(site, 10 ** 9)
    batches = [all_px[i:i + SCORE_BATCH] for i in range(0, len(all_px), SCORE_BATCH)]
    guard_targets = list(pd.date_range(test_start + pd.Timedelta(days=7), periods=7)) + \
        list(pd.date_range(test_end - pd.Timedelta(days=6), test_end))
    parts = {h: [] for h in horizons}
    for k, px in enumerate(batches):
        hist = load_features(site, px, until=train_end, start=train_start, columns=["pixel_id", "date", "precip_mm", "rh_pct"])
        clim = fit_climatology(hist, wet_mm, exceed, window)
        del hist
        if k == 0:  # GUARD: reproduce the saved test forecasts with this exact code path
            g_feat = load_features(site, px, until=test_end, start=test_start)
            for h in horizons:
                g_issue = [d - pd.Timedelta(days=h) for d in guard_targets]
                g = predict_rows(issue_table(g_feat, clim, h, g_issue), models, smear, cols, h, wet_mm, exceed)
                ref = pd.read_parquet(run / f"h{h}" / "test_predictions.parquet",
                                      filters=[("pixel_id", "in", px), ("target_date", "in", guard_targets)])
                m = ref.merge(g, on=["pixel_id", "target_date"], suffixes=("", "_new"), validate="one_to_one")
                pcols = [c for c in ref.columns if "__" in c]
                err = max(float((m[c] - m[f"{c}_new"]).abs().max()) for c in pcols + ["y_rain", "y_rh"])
                if len(m) != len(ref) or len(m) != len(px) * len(guard_targets) or err > 1e-9:
                    raise SystemExit(f"GUARD FAILED h{h}: {len(m)} rows, max difference {err:.3g}. Nothing written.")
                print(f"  PASS  guard h{h}: re-predicted {len(m):,} saved test forecasts, max difference {err:.1g}")
            del g_feat
        feat = load_features(site, px, until=ps_end, start=issue_days[0])
        for h in horizons:
            va = issue_table(feat, clim, h, issue_days)
            va = va[va["target_date"] >= ps_start]
            parts[h].append(predict_rows(va, models, smear, cols, h, wet_mm, exceed))
        del feat, clim
        print(f"  batch {k + 1} of {len(batches)} done ({len(px)} pixels)")
    for h in horizons:
        df = label_period(pd.concat(parts[h], ignore_index=True), ps_end, final_through)
        obs = df["period"] != "forecast"
        assert df.loc[obs, ["y_rain", "y_rh"]].notna().all().all(), "observed targets with missing values"
        assert df.loc[~obs, ["y_rain", "y_rh"]].isna().all().all()
        assert not df.duplicated(["pixel_id", "target_date"]).any() and df["pixel_id"].nunique() == len(all_px)
        df.to_parquet(run / f"h{h}" / "posttest_predictions.parquet", index=False)
        n = df["period"].value_counts().to_dict()
        print(f"  saved h{h}: {len(df):,} rows {n}")
    if site == "nakuru":
        score_late_blight(sites_cfg, exp)


# ---------------------------------------------------------------- late blight (Nakuru)
def risk_issue_table(feat, clim, lab, clim_risk_rule, rule, h, issues) -> pd.DataFrame:
    """late_blight.risk_table built from issue days (targets may be unobserved: y_risk NaN)."""
    from src.models.late_blight import EXTRA
    t = issue_table(feat, clim, h, issues)
    iss = lab[["pixel_id", "date"] + EXTRA].rename(columns={"date": "issue_date", **{c: f"lb_{c}" for c in EXTRA}})
    t = t.merge(iss, on=["pixel_id", "issue_date"], how="left", validate="many_to_one")
    t[[f"lb_{c}" for c in EXTRA]] = t[[f"lb_{c}" for c in EXTRA]].astype("float32")
    tgt = lab[["pixel_id", "date", f"met_{rule}"]].rename(columns={"date": "target_date", f"met_{rule}": "y_risk"})
    t = t.merge(tgt, on=["pixel_id", "target_date"], how="left", validate="many_to_one")
    t["y_risk"] = t["y_risk"].astype(float)
    t["doy"] = doy365(t["target_date"])
    t = t.merge(clim_risk_rule, on=["pixel_id", "doy"], how="left", validate="many_to_one").drop(columns="doy")
    t["persistence"] = t[f"lb_met_{rule}"]
    return t


def score_late_blight(sites_cfg: dict, exp: dict) -> None:
    from src.models.late_blight import RULES, SITE, binary_climatology, risk_columns, risk_labels
    fexp = final_config(exp)
    horizons = sites_cfg["forecast"]["horizons"]
    wet_mm, exceed, window = fexp["rain"]["wet_threshold_mm"], fexp["rain"]["exceedance_mm"], fexp["climatology"]["window_days"]
    train_start, train_end = split_bounds(fexp, "train")
    test_start, test_end = split_bounds(fexp, "test")
    ps_start, ps_end, final_through = posttest_bounds(exp)
    issue_days = pd.date_range(ps_start - pd.Timedelta(days=max(horizons)), ps_end)
    daily = pd.read_parquet(PROCESSED / f"{SITE}_daily.parquet", columns=["pixel_id", "date", "t2m_min_c", "sp_hpa", "elev_mean_m"])
    daily = daily[daily["date"] <= ps_end]
    hours = pd.read_parquet(PROCESSED / f"{SITE}_humid_hours.parquet")
    if hours["date"].max() < ps_end:
        raise SystemExit(f"humid hours end {hours['date'].max():%Y-%m-%d}: run src.features.humid_hours --site nakuru first")
    labels = risk_labels(daily, hours[hours["date"] <= ps_end], train_start, train_end)
    del daily, hours
    tr_lab = labels[(labels["date"] >= train_start) & (labels["date"] <= train_end)]
    clim_risk = {r: binary_climatology(tr_lab, f"met_{r}", window) for r in RULES}
    del tr_lab
    models = {(r, h): joblib.load(MODELS / SITE / "late_blight_final" / r / f"h{h}" / "catboost_risk.joblib")
              for r in RULES for h in horizons}
    all_px = dev_pixel_sample(SITE, 10 ** 9)
    batches = [all_px[i:i + SCORE_BATCH] for i in range(0, len(all_px), SCORE_BATCH)]
    guard_targets = list(pd.date_range(test_start + pd.Timedelta(days=7), periods=7)) + \
        list(pd.date_range(test_end - pd.Timedelta(days=6), test_end))

    def table(feat, clim, lab, rule, h, issues):
        return risk_issue_table(feat, clim, lab, clim_risk[rule], rule, h, issues)

    out = {(r, h): [] for r in RULES for h in horizons}
    for k, px in enumerate(batches):
        hist = load_features(SITE, px, until=train_end, start=train_start, columns=["pixel_id", "date", "precip_mm", "rh_pct"])
        clim = fit_climatology(hist, wet_mm, exceed, window)
        del hist
        lab = labels[labels["pixel_id"].isin(px)]
        if k == 0:
            g_feat = load_features(SITE, px, until=test_end, start=test_start)
            for r in RULES:
                for h in horizons:
                    m = models[(r, h)]
                    t = table(g_feat, clim, lab, r, h, [d - pd.Timedelta(days=h) for d in guard_targets])
                    assert sorted(risk_columns(t)) == sorted(m.feature_names_)
                    p = predict(m, t[list(m.feature_names_)], True)
                    ref = pd.read_parquet(MODELS / SITE / "late_blight_final" / r / f"h{h}" / "test_predictions.parquet",
                                          filters=[("pixel_id", "in", px), ("target_date", "in", guard_targets)])
                    x = ref.merge(t[["pixel_id", "target_date", "y_risk"]].assign(new=p), on=["pixel_id", "target_date"],
                                  suffixes=("", "_new"), validate="one_to_one")
                    err = max(float((x["catboost"] - x["new"]).abs().max()), float((x["y_risk"] - x["y_risk_new"]).abs().max()))
                    if len(x) != len(ref) or len(x) != len(px) * len(guard_targets) or err > 1e-9:
                        raise SystemExit(f"LATE-BLIGHT GUARD FAILED {r} h{h}: max difference {err:.3g}")
            print(f"  PASS  late-blight guard: both rules, 7 horizons reproduce the saved test forecasts")
            del g_feat
        feat = load_features(SITE, px, until=ps_end, start=issue_days[0])
        for r in RULES:
            for h in horizons:
                m = models[(r, h)]
                t = table(feat, clim, lab, r, h, issue_days)
                t = t[t["target_date"] >= ps_start]
                p = predict(m, t[list(m.feature_names_)], True)
                out[(r, h)].append(t[["pixel_id", "issue_date", "target_date", "y_risk", "clim_p_risk", "persistence"]]
                                   .assign(catboost=p))
        del feat, clim, lab
        print(f"  late blight batch {k + 1} of {len(batches)} done")
    for (r, h), parts in out.items():
        df = label_period(pd.concat(parts, ignore_index=True), ps_end, final_through)
        obs = df["period"] != "forecast"
        assert df.loc[obs, "y_risk"].notna().all()
        df.to_parquet(MODELS / SITE / "late_blight_final" / r / f"h{h}" / "posttest_predictions.parquet", index=False)
    print("  saved late-blight post-test predictions (models/nakuru/late_blight_final/<rule>/h<h>/)")


# ---------------------------------------------------------------- analysis
def period_names(exp: dict) -> dict:
    """Report labels for the scored sub-periods, from the dates in config/experiment.yaml."""
    start, end, final_through = posttest_bounds(exp)
    d = lambda t: f"{t.day} {t:%b %Y}"  # noqa: E731
    return {f"post-test final ({d(start)} - {d(final_through)})": ["posttest_final"],
            f"post-test preliminary ({d(final_through + pd.Timedelta(days=1))} - {d(end)})": ["posttest_prelim"],
            "post-test all": ["posttest_final", "posttest_prelim"]}


def analyse(sites_cfg: dict, exp: dict) -> None:
    from src.models.calibration import bootstrap_bias_and_bins
    from src.models.late_blight import binary_skill
    ucfg, wet_mm, exceed = exp["uncertainty"], exp["rain"]["wet_threshold_mm"], exp["rain"]["exceedance_mm"]
    PERIODS = period_names(exp)
    skill, diffs, calib, lb = [], [], [], []
    for site in sites_cfg["sites"]:
        run = final_run(site)
        for h in sites_cfg["forecast"]["horizons"]:
            pr = pd.read_parquet(run / f"h{h}" / "posttest_predictions.parquet")
            for name, keep in PERIODS.items():
                sub = pr[pr["period"].isin(keep)].drop(columns=["period", "issue_date"]).reset_index(drop=True)
                n_days = sub["target_date"].nunique()
                W = block_bootstrap_weights(n_days, ucfg["block_days"], ucfg["n_boot"], ucfg["seed"])
                rows, draws = bootstrap_site_horizon(sub, W, wet_mm, exceed)
                key = {"site": site, "horizon": h, "period": name, "n_pixels": sub["pixel_id"].nunique(), "n_days": n_days}
                skill += [{**key, **r} for r in rows]
                diffs += [{**key, **d} for d in paired_differences(draws, pd.DataFrame(rows))]
                if name == "post-test all":
                    day, uniq = pd.factorize(pd.DatetimeIndex(sub["target_date"]), sort=True)
                    for thr in [wet_mm] + exceed:
                        y = (sub["y_rain"] >= thr).to_numpy(float)
                        for m in ["catboost", "climatology"]:
                            p = sub[f"{m}__occ{thr:g}"].to_numpy(float)
                            bias, lo, hi, _ = bootstrap_bias_and_bins(y, p, day, len(uniq), W)
                            slope, _ = calibration_slope(y, p)
                            calib.append({"site": site, "horizon": h, "threshold_mm": thr, "model": m,
                                          "observed_rate": y.mean(), "mean_forecast": p.mean(), "bias": bias,
                                          "bias_ci_low": lo, "bias_ci_high": hi, "slope": slope})
        if site == "nakuru":
            from src.models.late_blight import RULES
            for r in RULES:
                for h in sites_cfg["forecast"]["horizons"]:
                    pr = pd.read_parquet(MODELS / site / "late_blight_final" / r / f"h{h}" / "posttest_predictions.parquet")
                    for name, keep in PERIODS.items():
                        sub = pr[pr["period"].isin(keep)].reset_index(drop=True)
                        day, uniq = pd.factorize(pd.DatetimeIndex(sub["target_date"]), sort=True)
                        W = block_bootstrap_weights(len(uniq), ucfg["block_days"], ucfg["n_boot"], ucfg["seed"])
                        lb.append({"rule": r, "horizon": h, "period": name, "n_days": len(uniq),
                                   **binary_skill(sub, sub["catboost"].to_numpy(), W, day, len(uniq))})
    OUT.mkdir(parents=True, exist_ok=True)
    skill, diffs, calib, lb = pd.DataFrame(skill), pd.DataFrame(diffs), pd.DataFrame(calib), pd.DataFrame(lb)
    skill.to_csv(OUT / "skill.csv", index=False)
    diffs.to_csv(OUT / "differences.csv", index=False)
    calib.to_csv(OUT / "calibration.csv", index=False)
    lb.to_csv(OUT / "late_blight_skill.csv", index=False)
    print_summary(skill, diffs, calib, lb, PERIODS)
    print(f"\nsaved {OUT}/ skill.csv, differences.csv, calibration.csv, late_blight_skill.csv")


def fmt(e, lo, hi) -> str:
    return f"{e:+.3f} [{lo:+.3f},{hi:+.3f}]{'*' if (lo > 0 or hi < 0) else ' '}"


def print_summary(skill, diffs, calib, lb, PERIODS) -> None:
    print("\nPOST-TEST (frozen final models; NOT part of the reported test). CatBoost skill [95%], * = excludes 0.")
    print("Caution: few target days (a month or less per sub-period, i.e. 2-5 seven-day blocks): intervals are rough.")
    sk = skill[(skill["model"] == "catboost") & skill["metric"].isin(["bss", "rmse_skill"])
               & skill["target"].isin(["rain>=1mm", "rh_pct"])]
    for ref in ["climatology", "persistence"]:
        x = sk[sk["reference"] == ref]
        t = x.assign(v=x.apply(lambda r: fmt(r["estimate"], r["ci_low"], r["ci_high"]), axis=1)).pivot_table(
            index="horizon", columns=["period", "site", "target"], values="v", aggfunc="first")
        for period in PERIODS:
            print(f"\n  vs {ref}, {period}:")
            print("    " + t[period].to_string().replace("\n", "\n    "))
    a = diffs[(diffs["period"] == "post-test all") & ~diffs["comparison"].str.contains("two-stage")]
    sig = a[(a["ci_low"] > 0) | (a["ci_high"] < 0)]
    print(f"\n  algorithm differences (post-test all) whose interval excludes 0: {len(sig)} of {len(a)}")
    if len(sig):
        win = sig.apply(lambda r: r["comparison"].split(" - ")[0 if r["estimate"] > 0 else 1], axis=1)
        print(f"    wins by algorithm: {win.value_counts().to_dict()}")
    c = calib[(calib["model"] == "catboost") & (calib["threshold_mm"] == 1.0)]
    print("\n  calibration, CatBoost rain >= 1 mm, post-test all (bias [95%], slope, observed rate):")
    for r in c.itertuples():
        print(f"    {r.site} h{r.horizon}: {fmt(r.bias, r.bias_ci_low, r.bias_ci_high)} slope {r.slope:.2f} rate {r.observed_rate:.3f}")
    if len(lb):
        x = lb[lb["period"] == "post-test all"]
        print("\n  late blight (Nakuru), post-test all, BSS vs climatology [95%] (base rate):")
        for r in x.itertuples():
            print(f"    {r.rule:6} h{r.horizon}: {fmt(r.bss_vs_climatology, r.bss_vs_climatology_low, r.bss_vs_climatology_high)}"
                  f" ({r.base_rate:.3f}), vs persistence {r.bss_vs_persistence:+.3f}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Post-test period with the frozen final models.")
    ap.add_argument("step", choices=["score", "analyse"])
    ap.add_argument("--site", help="site for the score step (default: all)")
    args = ap.parse_args()
    sites_cfg, exp = load_configs()
    if args.step == "score":
        for site in ([args.site] if args.site else list(sites_cfg["sites"])):
            score_site(site, sites_cfg, exp)
    else:
        analyse(sites_cfg, exp)


if __name__ == "__main__":
    main()
