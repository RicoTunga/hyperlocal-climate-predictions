"""The final, one-off test on 2025-01-08 .. 2026-09-30, following docs/PROJECT_CONTEXT.md section 12.

Two steps, so a mistake in the analysis never forces retraining:

    python -m src.models.final_test train --site nakuru     # retrain and save test forecasts
    python -m src.models.final_test train --site busia
    python -m src.models.final_test analyse                 # score everything the protocol lists

Retraining (protocol): the decided settings, unchanged, with 2024 added to the training data.
The only changes from config/experiment.yaml are: training period ends 2024-12-31 and early
stopping uses 2024 (targets 2024-01-08 .. 2024-12-31). The per-pixel climatology (input and
baseline) is fitted on 2016-2024. Training data are loaded only up to 2024-12-31 (checked).
CatBoost, LightGBM, XGBoost; 60 training pixels; every pixel scored; horizons 1-7; six tasks.
Each model is trained once. Scored pixels are processed in batches of SCORE_BATCH (memory): the
first batch through the normal training code, every batch through the saved models, and the
first batch must match exactly before the batches are combined (Busia, 216 pixels, was scored
before batching was added, in a single pass through the training code).

Analysis: skill vs climatology and persistence with 95% moving-block bootstrap intervals, for the
full test period, 2025, January-September 2026 and each season (JF, MAM, JJAS, OND pooled over
years); paired algorithm differences; two-stage vs direct; calibration of the rain probabilities.
Primary outcomes: CatBoost, full period, horizons 1-3, rain >= 1 mm BSS and humidity RMSE skill.

Output: models/<site>/final_train60_score<N>/h<h>/ (models, test_predictions.parquet) and
reports/test/ (skill.csv, differences.csv, calibration.csv).
"""
import argparse
import copy
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.models.baselines import fit_climatology
from src.models.calibration import bootstrap_bias_and_bins, brier_decomposition, calibration_slope
from src.models.evaluation import eval_target_dates, load_configs, split_bounds
from src.models.train import ALGOS, MODELS, dev_pixel_sample, load_features, run_site_horizon
from src.models.uncertainty import block_bootstrap_weights, bootstrap_site_horizon, paired_differences

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports" / "test"
SEASONS = {"JF": [1, 2], "MAM": [3, 4, 5], "JJAS": [6, 7, 8, 9], "OND": [10, 11, 12]}
SCORE_BATCH = 200  # scored pixels per batch: the full Nakuru test set does not fit in memory at once


def final_config(exp: dict) -> dict:
    """The decided config with the protocol's two changes: train to 2024-12-31, early-stop on 2024."""
    f = copy.deepcopy(exp)
    val_start, val_end = split_bounds(exp, "validation")
    f["splits"]["train"]["end"] = str(val_end.date())
    f["modelling"]["early_stop_start"] = str(val_start.date())
    return f


def run_tag(n_train: int, n_score: int) -> str:
    return f"final_train{n_train}_score{n_score}"


def train(site: str, sites_cfg: dict, exp: dict) -> None:
    fexp = final_config(exp)
    mcfg = fexp["modelling"]
    wet_mm, exceed, window = fexp["rain"]["wet_threshold_mm"], fexp["rain"]["exceedance_mm"], fexp["climatology"]["window_days"]
    train_start, train_end = split_bounds(fexp, "train")      # 2016-01-01 .. 2024-12-31
    test_start, test_end = split_bounds(fexp, "test")
    horizons = sites_cfg["forecast"]["horizons"]
    print(f"=== {site}: FINAL retraining. Training data {train_start:%Y-%m-%d}..{train_end:%Y-%m-%d}, "
          f"early stopping from {mcfg['early_stop_start']}, scoring the test period "
          f"{test_start:%Y-%m-%d}..{test_end:%Y-%m-%d} ===")

    pixels = dev_pixel_sample(site, mcfg["dev_pixels"])
    feat = load_features(site, pixels, until=train_end)          # nothing after 2024-12-31
    assert feat["date"].max() <= train_end
    clim = fit_climatology(feat[feat["date"] >= train_start], wet_mm, exceed, window)
    all_px = dev_pixel_sample(site, 10 ** 9)
    batches = [all_px[i:i + SCORE_BATCH] for i in range(0, len(all_px), SCORE_BATCH)]

    def score_inputs(px):
        """Test-period features and climatology (fitted on 2016-2024) for a batch of scored pixels."""
        hist = load_features(site, px, until=train_end, start=train_start,
                             columns=["pixel_id", "date", "precip_mm", "rh_pct"])
        c = fit_climatology(hist, wet_mm, exceed, window)
        del hist
        return load_features(site, px, until=test_end, start=test_start), c

    # 1. train every model once, scoring the first batch through the normal training code
    tag = run_tag(len(pixels), len(all_px))
    first = score_inputs(batches[0])
    for h in horizons:
        run_site_horizon(site, feat, clim, h, fexp, ALGOS, max(horizons), score=first,
                         out_tag=tag, score_split="test")
    if len(batches) == 1:
        print(f"saved models/{site}/{tag}/ ({len(all_px)} pixels in one batch)")
        return

    # 2. score every batch with the saved models; batch 1 must reproduce step 1 exactly
    from src.models.train import baseline_preds, feature_columns, horizon_table, predict
    from src.models.transfer import predict_all
    es_start = pd.Timestamp(str(mcfg["early_stop_start"]))
    targets = eval_target_dates(fexp, "test", max(horizons))
    task_names = [f"occ{t:g}" for t in [wet_mm] + exceed] + ["wet", "direct", "rh"]
    models, smear, cols = {}, {}, None
    for h in horizons:
        d = MODELS / site / tag / f"h{h}"
        fit = horizon_table(feat, clim, h, pd.date_range(train_start + pd.Timedelta(days=h),
                                                         es_start - pd.Timedelta(days=1)))
        cols = feature_columns(fit)
        wet_fit = fit[fit["y_rain"] >= wet_mm]
        for algo in ALGOS:
            models[(h, algo)] = {t: joblib.load(d / f"{algo}_{t}.joblib") for t in task_names}
            smear[(h, algo)] = float(np.mean(np.exp(np.log1p(wet_fit["y_rain"])
                                                    - predict(models[(h, algo)]["wet"], wet_fit[cols], False))))
        del fit, wet_fit
    for k, px in enumerate(batches):
        fv, cv = first if k == 0 else score_inputs(px)
        for h in horizons:
            va = horizon_table(fv, cv, h, targets)
            out = va[["pixel_id", "target_date", "y_rain", "y_rh"]].copy()
            preds = baseline_preds(va, wet_mm, exceed)
            preds.update({algo: predict_all(models[(h, algo)], smear[(h, algo)], va, cols, wet_mm) for algo in ALGOS})
            for name, p in preds.items():
                for key, arr in p.items():
                    out[f"{name}__{key}"] = np.asarray(arr, dtype=float)
            d = MODELS / site / tag / f"h{h}"
            if k == 0:  # guard: the batch path reproduces the training code's own scoring
                ref = pd.read_parquet(d / "test_predictions.parquet")
                m = ref.merge(out, on=["pixel_id", "target_date"], suffixes=("", "_b"), validate="one_to_one")
                err = max(float((m[c] - m[f"{c}_b"]).abs().max()) for c in out.columns if "__" in c)
                if len(m) != len(ref) or err > 1e-6:
                    raise SystemExit(f"h{h}: batch scoring does not reproduce the training scoring (max {err:.2g})")
                print(f"  PASS  h{h}: batch scoring reproduces the training scoring "
                      f"({len(m):,} rows, max difference {err:.1g})")
            out.to_parquet(d / f"test_predictions_part{k}.parquet", index=False)
            del va, out
        del fv, cv
        if k == 0:
            del first
        print(f"  batch {k + 1} of {len(batches)} scored ({len(px)} pixels)")
    for h in horizons:  # combine the parts into the full file, then remove the parts
        d = MODELS / site / tag / f"h{h}"
        parts = sorted(d.glob("test_predictions_part*.parquet"))
        full = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
        assert full["pixel_id"].nunique() == len(all_px) and not full.duplicated(["pixel_id", "target_date"]).any()
        full.to_parquet(d / "test_predictions.parquet", index=False)
        for p in parts:
            p.unlink()
    print(f"saved models/{site}/{tag}/ ({len(all_px)} pixels scored in {len(batches)} batches)")


def periods(dates: pd.Series) -> dict:
    """name -> boolean mask of target dates, as listed in the protocol."""
    d = pd.DatetimeIndex(dates)
    out = {"full test period": np.ones(len(d), bool), "2025": d.year == 2025, "2026 (Jan-Sep)": d.year == 2026}
    out.update({f"season {k}": d.month.isin(v) for k, v in SEASONS.items()})
    return out


def analyse(sites_cfg: dict, exp: dict) -> None:
    ucfg, wet_mm, exceed = exp["uncertainty"], exp["rain"]["wet_threshold_mm"], exp["rain"]["exceedance_mm"]
    skill, diffs, calib = [], [], []
    for site in sites_cfg["sites"]:
        files = sorted((MODELS / site).glob("final_train*_score*/h*/test_predictions.parquet"))
        if not files:
            raise SystemExit(f"No test predictions for {site}: run `final_test train --site {site}` first.")
        for f in files:
            h = int(f.parent.name[1:])
            pr = pd.read_parquet(f)
            n_px = pr["pixel_id"].nunique()
            for name, mask in periods(pr["target_date"]).items():
                sub = pr[mask].reset_index(drop=True)
                if sub.empty:
                    continue
                W = block_bootstrap_weights(sub["target_date"].nunique(), ucfg["block_days"], ucfg["n_boot"], ucfg["seed"])
                rows, draws = bootstrap_site_horizon(sub, W, wet_mm, exceed)
                key = {"site": site, "horizon": h, "period": name, "n_pixels": n_px,
                       "n_days": sub["target_date"].nunique()}
                skill += [{**key, **r} for r in rows]
                diffs += [{**key, **d} for d in paired_differences(draws, pd.DataFrame(rows))]
            # calibration of the rain probabilities, full test period
            day, uniq = pd.factorize(pd.DatetimeIndex(pr["target_date"]), sort=True)
            W = block_bootstrap_weights(len(uniq), ucfg["block_days"], ucfg["n_boot"], ucfg["seed"])
            for thr in [wet_mm] + exceed:
                y = (pr["y_rain"] >= thr).to_numpy(float)
                for m in ["catboost", "climatology"]:
                    p = pr[f"{m}__occ{thr:g}"].to_numpy(float)
                    bias, lo, hi, _ = bootstrap_bias_and_bins(y, p, day, len(uniq), W)
                    slope, intercept = calibration_slope(y, p)
                    calib.append({"site": site, "horizon": h, "threshold_mm": thr, "model": m,
                                  "observed_rate": y.mean(), "mean_forecast": p.mean(), "bias": bias,
                                  "bias_ci_low": lo, "bias_ci_high": hi, "slope": slope, "intercept": intercept,
                                  **brier_decomposition(y, p)})
    OUT.mkdir(parents=True, exist_ok=True)
    skill, diffs, calib = pd.DataFrame(skill), pd.DataFrame(diffs), pd.DataFrame(calib)
    skill.to_csv(OUT / "skill.csv", index=False)
    diffs.to_csv(OUT / "differences.csv", index=False)
    calib.to_csv(OUT / "calibration.csv", index=False)
    print_summary(skill, diffs, calib)
    print(f"\nsaved {OUT / 'skill.csv'}, {OUT / 'differences.csv'}, {OUT / 'calibration.csv'}")


def fmt(r) -> str:
    star = "*" if (r["ci_low"] > 0 or r["ci_high"] < 0) else " "
    return f"{r['estimate']:+.3f} [{r['ci_low']:+.3f},{r['ci_high']:+.3f}]{star}"


def print_summary(skill: pd.DataFrame, diffs: pd.DataFrame, calib: pd.DataFrame) -> None:
    sk = skill[skill["metric"].isin(["bss", "rmse_skill"])]
    print("\nPRIMARY OUTCOMES (CatBoost, full test period, skill vs climatology, [95%], * = excludes 0)")
    prim = sk[(sk["model"] == "catboost") & (sk["period"] == "full test period") & (sk["reference"] == "climatology")
              & (sk["horizon"] <= 3) & sk["target"].isin(["rain>=1mm", "rh_pct"])]
    t = prim.assign(v=prim.apply(fmt, axis=1)).pivot_table(index="horizon", columns=["site", "target"], values="v", aggfunc="first")
    print("  " + t.to_string().replace("\n", "\n  "))

    show = ["rain>=1mm", "rain>=10mm", "rain>=25mm", "rain_mm_wet", "rh_pct"]
    for ref in ["climatology", "persistence"]:
        x = sk[(sk["model"] == "catboost") & (sk["period"] == "full test period") & (sk["reference"] == ref)
               & sk["target"].isin(show)]
        for site, g in x.groupby("site"):
            t = g.assign(v=g.apply(fmt, axis=1)).pivot(index="horizon", columns="target", values="v")[show]
            print(f"\n{site}: CatBoost skill vs {ref}, full test period")
            print("  " + t.to_string().replace("\n", "\n  "))

    x = sk[(sk["model"] == "catboost") & (sk["reference"] == "climatology") & (sk["horizon"] == 1)
           & sk["target"].isin(["rain>=1mm", "rh_pct"])]
    t = x.assign(v=x.apply(fmt, axis=1)).pivot_table(index="period", columns=["site", "target"], values="v", aggfunc="first")
    print("\nDay 1 by period (CatBoost vs climatology):")
    print("  " + t.to_string().replace("\n", "\n  "))

    a = diffs[(diffs["period"] == "full test period") & ~diffs["comparison"].str.contains("two-stage")]
    sig = a[(a["ci_low"] > 0) | (a["ci_high"] < 0)]
    print(f"\nalgorithm differences (full period) whose interval excludes 0: {len(sig)} of {len(a)}")
    if len(sig):
        win = sig.apply(lambda r: r["comparison"].split(" - ")[0 if r["estimate"] > 0 else 1], axis=1)
        print(f"  wins by algorithm: {win.value_counts().to_dict()}")
    ts = diffs[(diffs["period"] == "full test period") & (diffs["comparison"] == "catboost: two-stage - direct")]
    print("catboost two-stage minus direct (full period): "
          + "; ".join(f"h{r.horizon} {fmt(r._asdict()).strip()}" for r in ts.itertuples()))

    c = calib[(calib["model"] == "catboost") & (calib["threshold_mm"] == 1.0)]
    print("\ncalibration, CatBoost rain >= 1 mm (bias [95%], slope):")
    for r in c.itertuples():
        print(f"  {r.site} h{r.horizon}: bias {r.bias:+.3f} [{r.bias_ci_low:+.3f},{r.bias_ci_high:+.3f}], "
              f"slope {r.slope:.2f}, observed rate {r.observed_rate:.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Final one-off test (protocol: PROJECT_CONTEXT section 12).")
    parser.add_argument("step", choices=["train", "analyse"])
    parser.add_argument("--site", help="site key for the train step (default: all sites)")
    args = parser.parse_args()
    sites_cfg, exp = load_configs()
    if args.step == "train":
        for site in ([args.site] if args.site else list(sites_cfg["sites"])):
            train(site, sites_cfg, exp)
    else:
        analyse(sites_cfg, exp)


if __name__ == "__main__":
    main()
