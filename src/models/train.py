"""Train gradient-boosting models per horizon and score them on the validation year next to
persistence and climatology.

Run from the repo root (venv active):

    python -m src.models.train                          # horizon 1, all sites, all algorithms
    python -m src.models.train --site busia --algos lightgbm
    python -m src.models.train --horizons 1 2 3 4 5 6 7

Design (AGENTS.md and docs/PROJECT_CONTEXT.md section 6):
- One model per site x horizon x task x algorithm. Tasks:
    rain_occ1 / rain_occ10 / rain_occ25   probability of >= 1, 10, 25 mm on the target day
    rain_wet                              amount on wet days, trained on log(1 + mm)  (stage 2)
    rain_direct                           amount on all days, single direct model (comparison)
    rh                                    relative humidity (%)
  The two-stage rainfall forecast is P(wet) x expected wet-day amount.
- Row = (pixel, issue day t): features from src/features/build.py (data up to day t) plus the
  per-pixel climatology OF THE TARGET DAY (fitted on training years only). Target = value on t + h.
- Fit on training targets up to the day before `early_stop_start`; early stopping on the last
  training year (with gap_days); score on validation. The final test period is never loaded.
- Developed on `dev_pixels` pixels per site spread evenly over elevation; the baselines are
  rescored on exactly the same rows.

Output: printed report, reports/models_validation.csv, fitted models and validation predictions
(val_predictions.parquet, used by src/models/uncertainty.py) in models/<site>/h<h>/
(git-ignored).
"""
import argparse
import json
import time
from pathlib import Path

import catboost as cb
import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb

from src.models.baselines import add_climatology, fit_climatology
from src.models.evaluation import (PROCESSED, binary_scores, continuous_scores, eval_target_dates,
                                   horizon_frame, load_configs, split_bounds)

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"
MODELS = ROOT / "models"
KEYS = ["site", "pixel_id", "date"]
ALGOS = ["lightgbm", "xgboost", "catboost"]


# ---------------------------------------------------------------- data
def spread_by_elevation(z: pd.DataFrame, n: int) -> list:
    """n pixels of `z` (columns pixel_id, elev_mean_m) evenly spaced by elevation rank (deterministic)."""
    z = z.drop_duplicates("pixel_id").sort_values(["elev_mean_m", "pixel_id"]).reset_index(drop=True)
    if n >= len(z):
        return sorted(z["pixel_id"])
    idx = np.unique(np.linspace(0, len(z) - 1, n).round().astype(int))
    return sorted(z.loc[idx, "pixel_id"])


def site_pixels(site: str) -> pd.DataFrame:
    """One row per pixel: pixel_id, lat, lon, elev_mean_m."""
    z = pd.read_parquet(PROCESSED / f"{site}_daily.parquet", columns=["pixel_id", "lat", "lon", "elev_mean_m"])
    return z.drop_duplicates("pixel_id").reset_index(drop=True)


def dev_pixel_sample(site: str, n: int) -> list:
    """n pixels evenly spaced in elevation (deterministic), so lowlands and highlands are both in."""
    return spread_by_elevation(site_pixels(site), n)


def load_features(site: str, pixels: list, until: pd.Timestamp, start: pd.Timestamp | None = None,
                  columns: list | None = None) -> pd.DataFrame:
    filters = [("pixel_id", "in", pixels), ("date", "<=", until)]
    if start is not None:
        filters.append(("date", ">=", start))
    df = pd.read_parquet(PROCESSED / f"features_{site}.parquet", columns=columns, filters=filters)
    assert df["date"].max() <= until and df["pixel_id"].nunique() == len(pixels)
    return df.sort_values(["pixel_id", "date"]).reset_index(drop=True)


def run_tag(n_train: int, n_score: int) -> str:
    """Folder name for one run size, so runs with different pixel counts never overwrite each other."""
    return f"train{n_train}_score{n_score}"


def run_dir(site: str, n_train: int, n_score: int, h: int) -> Path:
    return MODELS / site / run_tag(n_train, n_score) / f"h{h}"


def horizon_table(feat: pd.DataFrame, clim: pd.DataFrame, h: int, targets) -> pd.DataFrame:
    """Features on issue day t + climatology of target day + both targets on day t + h."""
    rain = horizon_frame(feat, "precip_mm", h, targets)
    rh = horizon_frame(feat, "rh_pct", h, targets)[["pixel_id", "target_date", "y"]]
    t = rain.rename(columns={"y": "y_rain"}).drop(columns="issue_value")
    t = t.merge(rh.rename(columns={"y": "y_rh"}), on=["pixel_id", "target_date"], validate="one_to_one")
    t = add_climatology(t, clim).drop(columns="doy")
    t = t.merge(feat.rename(columns={"date": "issue_date"}), on=["pixel_id", "issue_date"],
                how="left", validate="one_to_one")
    assert len(t) == len(rain)
    return t


def feature_columns(t: pd.DataFrame) -> list:
    drop = set(KEYS) | {"target_date", "issue_date", "y_rain", "y_rh"}
    return [c for c in t.columns if c not in drop]


# ---------------------------------------------------------------- models
def make_model(algo: str, classify: bool, params: dict, seed: int, es_rounds: int):
    p = dict(params[algo])
    if algo == "lightgbm":
        cls = lgb.LGBMClassifier if classify else lgb.LGBMRegressor
        return cls(random_state=seed, **p)
    if algo == "xgboost":
        cls = xgb.XGBClassifier if classify else xgb.XGBRegressor
        return cls(random_state=seed, early_stopping_rounds=es_rounds,
                   eval_metric="logloss" if classify else "rmse", **p)
    if algo == "catboost":
        cls = cb.CatBoostClassifier if classify else cb.CatBoostRegressor
        return cls(random_seed=seed, thread_count=-1, **p)
    raise ValueError(algo)


def fit(model, algo, X, y, Xe, ye, es_rounds):
    if algo == "lightgbm":
        model.fit(X, y, eval_X=Xe, eval_y=ye, callbacks=[lgb.early_stopping(es_rounds, verbose=False)])
        n = model.best_iteration_
    elif algo == "xgboost":
        model.fit(X, y, eval_set=[(Xe, ye)], verbose=False)
        n = model.best_iteration + 1
    else:
        model.fit(X, y, eval_set=(Xe, ye), early_stopping_rounds=es_rounds, verbose=False)
        n = model.get_best_iteration() + 1
    return model, int(n)


def predict(model, X, classify):
    return model.predict_proba(X)[:, 1] if classify else model.predict(X)


# ---------------------------------------------------------------- scoring
def score_all(v: pd.DataFrame, preds: dict, wet_mm: float, exceed: list) -> list[dict]:
    """preds: name -> dict of prediction arrays for the validation rows."""
    rows = []

    def add(model, target, s):
        rows.extend({"model": model, "target": target, "metric": k, "value": val} for k, val in s.items())

    y, wet = v["y_rain"].to_numpy(), v["y_rain"].to_numpy() >= wet_mm
    for name, p in preds.items():
        for thr in [wet_mm] + exceed:
            key = f"occ{thr:g}"
            if key in p:
                ref = v["clim_p_wet"] if thr == wet_mm else v[f"clim_p_ge{thr}"]
                add(name, f"rain>={thr:g}mm", binary_scores(y >= thr, p[key], ref))
        if "wet" in p:
            add(name, "rain_mm_wet", continuous_scores(y[wet], p["wet"][wet], v["clim_wet_amount"][wet]))
        for key, label in [("direct", "rain_mm_all"), ("two_stage", "rain_mm_all_2stage")]:
            if key in p:
                add(name, label, continuous_scores(y, p[key], v["clim_precip_mean"]))
        if "rh" in p:
            add(name, "rh_pct", continuous_scores(v["y_rh"], p["rh"], v["clim_rh_mean"]))
    return rows


def baseline_preds(v: pd.DataFrame, wet_mm: float, exceed: list) -> dict:
    pr = v["precip_mm"].to_numpy()  # value on the issue day
    pers = {f"occ{t:g}": (pr >= t).astype(float) for t in [wet_mm] + exceed}
    pers.update(wet=pr, direct=pr, rh=v["rh_pct"].to_numpy())
    clim = {f"occ{wet_mm:g}": v["clim_p_wet"].to_numpy(),
            **{f"occ{t:g}": v[f"clim_p_ge{t}"].to_numpy() for t in exceed},
            "wet": v["clim_wet_amount"].to_numpy(), "direct": v["clim_precip_mean"].to_numpy(),
            "rh": v["clim_rh_mean"].to_numpy()}
    return {"persistence": pers, "climatology": clim}


# ---------------------------------------------------------------- run
def task_definitions(wet_mm: float, exceed: list) -> dict:
    """task -> (target column, is classification, target transform, row subset or None)."""
    tasks = {f"occ{t:g}": ("y_rain", True, lambda y, t=t: (y >= t).astype(int), None)
             for t in [wet_mm] + exceed}
    tasks["wet"] = ("y_rain", False, np.log1p, lambda d: d["y_rain"] >= wet_mm)
    tasks["direct"] = ("y_rain", False, lambda y: y, None)
    tasks["rh"] = ("y_rh", False, lambda y: y, None)
    return tasks


def fit_predict(algo, task, spec, tr, es, pred_rows, cols, params, mcfg, wet_mm):
    """Fit one model (early stopping on `es`) and predict `pred_rows` on the reporting scale."""
    ycol, classify, transform, subset = spec
    a, e = (tr, es) if subset is None else (tr[subset(tr)], es[subset(es)])
    model = make_model(algo, classify, params, mcfg["seed"], mcfg["early_stopping_rounds"])
    model, n_iter = fit(model, algo, a[cols], transform(a[ycol]), e[cols], transform(e[ycol]),
                        mcfg["early_stopping_rounds"])
    pred = predict(model, pred_rows[cols], classify)
    if task == "wet":
        # back-transform from log(1 + mm) with Duan's smearing factor from the fit rows,
        # so the result estimates the mean (not the median) wet-day amount
        resid = np.log1p(a[ycol]) - predict(model, a[cols], False)
        pred = np.exp(pred) * np.mean(np.exp(resid)) - 1  # E[1 + y] = exp(pred) * smear
        pred = np.clip(pred, wet_mm, None)
    elif task == "direct":
        pred = np.clip(pred, 0, None)
    return model, n_iter, pred


def run_site_horizon(site, feat, clim, h, exp, algos, max_h, tuned=None, score=None,
                     out_tag=None) -> list[dict]:
    """Train on the pixels in `feat`; score on them, or on `score` = (features, climatology) of
    another pixel set (only the validation period is needed for scoring)."""
    mcfg, wet_mm, exceed = exp["modelling"], exp["rain"]["wet_threshold_mm"], exp["rain"]["exceedance_mm"]
    train_start, train_end = split_bounds(exp, "train")
    es_start = pd.Timestamp(str(mcfg["early_stop_start"]))
    gap = exp["gap_days"]

    fit_targets = pd.date_range(train_start + pd.Timedelta(days=h), es_start - pd.Timedelta(days=1))
    es_targets = pd.date_range(es_start + pd.Timedelta(days=gap), train_end)
    val_targets = eval_target_dates(exp, "validation", max_h)
    tr, es = (horizon_table(feat, clim, h, t) for t in (fit_targets, es_targets))
    va = horizon_table(*(score or (feat, clim)), h, val_targets)
    n_train, n_score = feat["pixel_id"].nunique(), va["pixel_id"].nunique()
    cols = feature_columns(tr)
    print(f"\n  --- horizon {h}: fit {len(tr):,} rows ({fit_targets[0]:%Y-%m-%d}..{fit_targets[-1]:%Y-%m-%d}), "
          f"early stop {len(es):,} ({es_targets[0]:%Y-%m-%d}..), validation {len(va):,} on {n_score} pixels "
          f"({val_targets[0]:%Y-%m-%d}..{val_targets[-1]:%Y-%m-%d}) | {len(cols)} features")

    tasks = task_definitions(wet_mm, exceed)
    preds = baseline_preds(va, wet_mm, exceed)
    # out_tag gives an experiment its own folder (e.g. holdout_SW) instead of train<N>_score<M>
    out_dir = MODELS / site / out_tag / f"h{h}" if out_tag else run_dir(site, n_train, n_score, h)
    out_dir.mkdir(parents=True, exist_ok=True)
    info = []
    for algo in algos:
        name = f"{algo}_tuned" if tuned is not None else algo
        p = {}
        for task, spec in tasks.items():
            params = dict(mcfg["params"])
            if tuned is not None and task in tuned.get(site, {}):
                params[algo] = {**mcfg["params"][algo], **tuned[site][task]}
            t0 = time.time()
            model, n_iter, pred = fit_predict(algo, task, spec, tr, es, va, cols, params, mcfg, wet_mm)
            p[task] = pred
            joblib.dump(model, out_dir / f"{name}_{task}.joblib")
            info.append(f"{name}/{task}: {n_iter} trees, {time.time() - t0:.0f}s")
        p["two_stage"] = p[f"occ{wet_mm:g}"] * p["wet"]
        preds[name] = p
        if algo == algos[0]:
            imp = pd.Series(model.feature_importances_, index=cols)  # last task = rh
            occ_model = joblib.load(out_dir / f"{name}_occ{wet_mm:g}.joblib")
            imp_occ = pd.Series(occ_model.feature_importances_, index=cols)
            print(f"  top features ({name}) rain>=1mm: {list(imp_occ.nlargest(8).index)}")
            print(f"  top features ({name}) humidity:  {list(imp.nlargest(8).index)}")
    print("  " + " | ".join(info))

    # save every validation prediction (models and baselines) for confidence intervals etc.;
    # columns of models not re-run here (e.g. other algorithms) are kept from the existing file
    saved = va[["pixel_id", "target_date", "y_rain", "y_rh"]].copy()
    for name, p in preds.items():
        for key, arr in p.items():
            saved[f"{name}__{key}"] = np.asarray(arr, dtype=float)
    f = out_dir / "val_predictions.parquet"
    if f.exists():
        old = pd.read_parquet(f)
        keep = [c for c in old.columns if "__" in c and c not in saved.columns]
        if keep:
            old = old[["pixel_id", "target_date", "y_rain", "y_rh"] + keep]
            m = saved.merge(old, on=["pixel_id", "target_date"], how="left", suffixes=("", "_old"),
                            validate="one_to_one")
            assert np.allclose(m["y_rain"], m["y_rain_old"]) and np.allclose(m["y_rh"], m["y_rh_old"])
            saved = m.drop(columns=["y_rain_old", "y_rh_old"])
    saved.to_parquet(f, index=False)

    rows = score_all(va, preds, wet_mm, exceed)
    for r in rows:
        r.update(site=site, horizon=h, split="validation", n_train_pixels=n_train, n_score_pixels=n_score)
    return rows


SHOW = [("rain>=1mm", "brier"), ("rain>=1mm", "bss"), ("rain>=1mm", "auc"), ("rain>=10mm", "bss"),
        ("rain>=25mm", "bss"), ("rain_mm_wet", "mae"), ("rain_mm_wet", "rmse"),
        ("rain_mm_all", "rmse"), ("rain_mm_all_2stage", "rmse"), ("rh_pct", "rmse"),
        ("rh_pct", "r2"), ("rh_pct", "rmse_skill")]
SHORT = {"rain>=1mm": "r1", "rain>=10mm": "r10", "rain>=25mm": "r25", "rain_mm_wet": "wet",
         "rain_mm_all": "all", "rain_mm_all_2stage": "all2s", "rh_pct": "rh"}


def print_table(res: pd.DataFrame) -> None:
    for (site, h), g in res.groupby(["site", "horizon"]):
        tab = g.pivot_table(index="model", columns=["target", "metric"], values="value")
        tab = tab.reindex(columns=[c for c in SHOW if c in tab.columns])
        tab.columns = [f"{SHORT[t]}_{m}" for t, m in tab.columns]
        order = [m for m in ["persistence", "climatology"] + ALGOS + [a + "_tuned" for a in ALGOS]
                 if m in tab.index]
        print(f"\n  {site}, horizon {h}, validation (BSS and rmse_skill are vs climatology):")
        print("    " + tab.loc[order].round(3).to_string().replace("\n", "\n    "))


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and validate gradient-boosting models.")
    parser.add_argument("--site", help="site key (default: all sites)")
    parser.add_argument("--horizons", type=int, nargs="+", default=[1])
    parser.add_argument("--algos", nargs="+", default=ALGOS, choices=ALGOS)
    parser.add_argument("--pixels", type=int, help="training pixels per site (default: modelling.dev_pixels)")
    parser.add_argument("--score-pixels", help="pixels to score on: a number (elevation-spread sample) "
                        "or 'all' (default: the training pixels)")
    parser.add_argument("--tuned", action="store_true",
                        help="use tuned parameters from reports/tuning/best_params.json; results are "
                             "saved under <algo>_tuned (tasks without tuned parameters keep the defaults)")
    args = parser.parse_args()

    sites_cfg, exp = load_configs()
    sites = list(sites_cfg["sites"]) if not args.site else [args.site]
    max_h = max(sites_cfg["forecast"]["horizons"])
    train_start, train_end = split_bounds(exp, "train")
    val_start, val_end = split_bounds(exp, "validation")
    n_pix = args.pixels or exp["modelling"]["dev_pixels"]
    tuned = None
    if args.tuned:
        tuned_file = REPORTS / "tuning" / "best_params.json"
        tuned = json.loads(tuned_file.read_text(encoding="utf-8"))
        print(f"tuned parameters from {tuned_file}: " + "; ".join(f"{s}: {sorted(v)}" for s, v in tuned.items()))

    all_rows = []
    for site in sites:
        wet_mm, exceed = exp["rain"]["wet_threshold_mm"], exp["rain"]["exceedance_mm"]
        window = exp["climatology"]["window_days"]
        pixels = dev_pixel_sample(site, n_pix)
        feat = load_features(site, pixels, until=val_end)
        train = feat[(feat["date"] >= train_start) & (feat["date"] <= train_end)]
        clim = fit_climatology(train, wet_mm, exceed, window)
        score = None
        if args.score_pixels:
            n_score = 10 ** 9 if args.score_pixels == "all" else int(args.score_pixels)
            score_pixels = dev_pixel_sample(site, n_score)
            if set(score_pixels) != set(pixels):
                # scoring needs only the validation period (lags are already inside the features),
                # plus the training years of rain and humidity to fit those pixels' climatology
                score_train = load_features(site, score_pixels, until=train_end, start=train_start,
                                            columns=["pixel_id", "date", "precip_mm", "rh_pct"])
                score = (load_features(site, score_pixels, until=val_end, start=val_start),
                         fit_climatology(score_train, wet_mm, exceed, window))
                del score_train
        n_scored = len(pixels) if score is None else score[0]["pixel_id"].nunique()
        print(f"\n=== {site}: training on {len(pixels)} pixels (elevation {feat['elev_mean_m'].min():.0f} "
              f"to {feat['elev_mean_m'].max():.0f} m), scoring on {n_scored} pixels, "
              f"data up to {feat['date'].max():%Y-%m-%d} ===")
        for h in args.horizons:
            all_rows += run_site_horizon(site, feat, clim, h, exp, args.algos, max_h, tuned, score)

    res = pd.DataFrame(all_rows)
    key = ["site", "horizon", "model", "n_train_pixels", "n_score_pixels"]
    res = res[["site", "split", "horizon", "n_train_pixels", "n_score_pixels", "model", "target",
               "metric", "value"]]
    print_table(res)
    REPORTS.mkdir(exist_ok=True)
    out = REPORTS / "models_validation.csv"
    if out.exists():  # keep results of other sites/horizons/algorithms from earlier runs
        old = pd.read_csv(out)
        if "n_pixels" in old.columns:  # older file: training and scoring pixels were the same
            old = old.rename(columns={"n_pixels": "n_train_pixels"})
            old.insert(old.columns.get_loc("n_train_pixels") + 1, "n_score_pixels", old["n_train_pixels"])
        old = old[~old.set_index(key).index.isin(res.set_index(key).index)]
        res = pd.concat([old, res], ignore_index=True)
    res.to_csv(out, index=False)
    print(f"\nsaved {out} ({len(res)} rows)")


if __name__ == "__main__":
    main()
