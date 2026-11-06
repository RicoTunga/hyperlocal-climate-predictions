"""Platt recalibration of the CatBoost rain probabilities, fitted on 2023 and tested on 2024.

Run from the repo root after src.models.train (decided setup) and src.models.calibration:

    python -m src.models.recalibrate
    python -m src.models.calibration      # before/after reliability
    python -m src.models.uncertainty      # paired intervals catboost_recal vs catboost

Method
- For each site, horizon and threshold (1, 10, 25 mm), the saved CatBoost occurrence model
  predicts the LAST TRAINING YEAR (targets from early_stop_start + gap_days to the end of the
  training period, 2023) for all scored pixels. Only data up to the end of training is loaded
  for this step.
- A logistic regression of the 2023 outcomes on logit(forecast) gives a and b; the corrected
  forecast is sigmoid(a + b * logit(p)). b < 1 shrinks over-confident forecasts towards the
  middle; a shifts them wetter or drier.
- The correction is applied unchanged to the saved 2024 forecasts and stored as model
  `catboost_recal` in val_predictions.parquet (wet-day amount, direct amount and humidity are
  copied from `catboost`, so only the probabilities, and the two-stage amount through them, differ).
- Guard: before fitting, the 2024 day-1 forecasts are rebuilt from the saved model and must equal
  the saved ones, which proves the model, features and column order line up.

Caveat: 2023 was the early-stopping year of these models, so their 2023 forecasts are slightly
better than truly unseen ones; the correction is fitted on that, and judged only on 2024.

Output: models/<site>/<run>/h<h>/val_predictions.parquet (catboost_recal columns added),
reports/recalibration_params.csv, rows for catboost_recal in reports/models_validation.csv.
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.models.baselines import fit_climatology
from src.models.calibration import decided_run
from src.models.evaluation import eval_target_dates, load_configs, split_bounds
from src.models.train import (REPORTS, dev_pixel_sample, feature_columns, horizon_table, load_features,
                              score_all)

MODEL, NEW = "catboost", "catboost_recal"


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def fit_platt(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    """(a, b) such that sigmoid(a + b * logit(p)) is calibrated on (y, p)."""
    lr = LogisticRegression(C=np.inf, max_iter=1000).fit(logit(p)[:, None], np.asarray(y, int))
    return float(lr.intercept_[0]), float(lr.coef_[0, 0])


def apply_platt(p: np.ndarray, a: float, b: float) -> np.ndarray:
    return 1 / (1 + np.exp(-(a + b * logit(p))))


def predictions_as_frame(pr: pd.DataFrame) -> pd.DataFrame:
    """Rename saved prediction columns to the clim_* names that train.score_all expects."""
    m = {"climatology__occ1": "clim_p_wet", "climatology__wet": "clim_wet_amount",
         "climatology__direct": "clim_precip_mean", "climatology__rh": "clim_rh_mean"}
    m.update({c: c.replace("climatology__occ", "clim_p_ge") for c in pr.columns
              if c.startswith("climatology__occ") and c != "climatology__occ1"})
    return pr.rename(columns=m)


def main() -> None:
    sites_cfg, exp = load_configs()
    mcfg, wet_mm, exceed = exp["modelling"], exp["rain"]["wet_threshold_mm"], exp["rain"]["exceedance_mm"]
    window, gap, max_h = exp["climatology"]["window_days"], exp["gap_days"], max(sites_cfg["forecast"]["horizons"])
    train_start, train_end = split_bounds(exp, "train")
    val_start, val_end = split_bounds(exp, "validation")
    es_start = pd.Timestamp(str(mcfg["early_stop_start"]))
    fit_targets = pd.date_range(es_start + pd.Timedelta(days=gap), train_end)
    thresholds = [wet_mm] + exceed

    params, score_rows = [], []
    for site in sites_cfg["sites"]:
        runs = decided_run(site, mcfg["dev_pixels"])
        run_dir = runs[0][1].parent.parent
        n_score = int(run_dir.name.split("_score")[1])
        pixels = dev_pixel_sample(site, n_score)
        assert len(pixels) == n_score
        hist = load_features(site, pixels, until=train_end, start=train_start,
                             columns=["pixel_id", "date", "precip_mm", "rh_pct"])
        clim = fit_climatology(hist, wet_mm, exceed, window)
        del hist
        feat23 = load_features(site, pixels, until=train_end, start=es_start)  # nothing after 2023
        assert feat23["date"].max() <= train_end
        print(f"\n=== {site}: {run_dir.name}, fitting on {fit_targets[0]:%Y-%m-%d}..{fit_targets[-1]:%Y-%m-%d} "
              f"({len(pixels)} pixels) ===")

        for h, f in runs:
            pr = pd.read_parquet(f)
            t23 = horizon_table(feat23, clim, h, fit_targets)
            cols = feature_columns(t23)
            if h == runs[0][0]:  # guard: the saved model must reproduce the saved 2024 forecasts
                feat24 = load_features(site, pixels, until=val_end, start=val_start)
                t24 = horizon_table(feat24, clim, h, eval_target_dates(exp, "validation", max_h))
                model = joblib.load(f.parent / f"{MODEL}_occ{wet_mm:g}.joblib")
                assert list(model.feature_names_) == cols, "feature columns differ from training"
                re = pd.DataFrame({"pixel_id": t24["pixel_id"], "target_date": t24["target_date"],
                                   "p": model.predict_proba(t24[cols])[:, 1]})
                chk = pr[["pixel_id", "target_date", f"{MODEL}__occ{wet_mm:g}"]].merge(
                    re, on=["pixel_id", "target_date"], validate="one_to_one")
                err = float((chk["p"] - chk[f"{MODEL}__occ{wet_mm:g}"]).abs().max())
                print(f"  {'PASS ' if err < 1e-9 and len(chk) == len(pr) else 'CHECK'} saved model reproduces "
                      f"the saved 2024 forecasts (h{h}, {len(chk):,} rows, max difference {err:.1g})")
                if err >= 1e-9:
                    raise SystemExit("Saved model does not reproduce saved forecasts; stopping.")
                del feat24, t24
            for thr in thresholds:
                model = joblib.load(f.parent / f"{MODEL}_occ{thr:g}.joblib")
                p23 = model.predict_proba(t23[cols])[:, 1]
                y23 = (t23["y_rain"] >= thr).to_numpy()
                a, b = fit_platt(y23, p23)
                params.append({"site": site, "horizon": h, "threshold_mm": thr, "a": a, "b": b,
                               "n_fit": len(y23), "fit_rate": y23.mean(), "fit_mean_forecast": p23.mean()})
                pr[f"{NEW}__occ{thr:g}"] = apply_platt(pr[f"{MODEL}__occ{thr:g}"].to_numpy(), a, b)
            for task in ["wet", "direct", "rh"]:
                pr[f"{NEW}__{task}"] = pr[f"{MODEL}__{task}"]
            pr[f"{NEW}__two_stage"] = pr[f"{NEW}__occ{wet_mm:g}"] * pr[f"{NEW}__wet"]
            pr.to_parquet(f, index=False)

            v = predictions_as_frame(pr)
            p = {k.split("__")[1]: v[k].to_numpy() for k in v.columns if k.startswith(f"{NEW}__")}
            n_train = int(run_dir.name.split("_score")[0].replace("train", ""))
            for r in score_all(v, {NEW: p}, wet_mm, exceed):
                score_rows.append({"site": site, "split": "validation", "horizon": h, "n_train_pixels": n_train,
                                   "n_score_pixels": n_score, **r})
            pp = [x for x in params if x["site"] == site and x["horizon"] == h]
            print(f"  h{h}: " + " | ".join(f">= {x['threshold_mm']:g} mm a {x['a']:+.2f} b {x['b']:.2f}" for x in pp))
        del feat23

    pd.DataFrame(params).to_csv(REPORTS / "recalibration_params.csv", index=False)
    new = pd.DataFrame(score_rows)[["site", "split", "horizon", "n_train_pixels", "n_score_pixels", "model",
                                    "target", "metric", "value"]]
    out = REPORTS / "models_validation.csv"
    old = pd.read_csv(out)
    key = ["site", "horizon", "model", "n_train_pixels", "n_score_pixels"]
    old = old[~old.set_index(key).index.isin(new.set_index(key).index)]
    pd.concat([old, new], ignore_index=True).to_csv(out, index=False)
    print(f"\nsaved {REPORTS / 'recalibration_params.csv'}; {NEW} added to the prediction files and {out}")


if __name__ == "__main__":
    main()
