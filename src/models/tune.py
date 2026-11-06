"""Hyperparameter tuning with Optuna, on the TRAINING years only.

Run from the repo root (venv active):

    python -m src.models.tune                         # settings from config/experiment.yaml
    python -m src.models.tune --site busia --tasks occ1 --trials 3   # quick trial

Method
- Expanding-window cross-validation inside 2016-2023: for each eval year Y in
  `tuning.eval_years`, fit on targets from 2016 up to the day before 1 Jan Y, and score on targets
  in Y starting `gap_days` after 1 Jan. The score of a parameter set is the mean over the folds.
- The per-pixel climatology features are refitted inside each fold on the fold's fitting years
  only, so an eval year never contributes to its own climatology.
- Early stopping in a fold uses that fold's eval year (the same for every trial, so trials are
  compared fairly; the fold scores are slightly optimistic in absolute terms).
- Data are loaded only up to the end of the training period: the validation year 2024 and the
  final test period cannot influence tuning.
- Trial 0 is always the current default parameters, so the summary shows whether tuning beat them.
- Poor trials are stopped after a fold if they are worse than the median (Optuna MedianPruner).
- Progress is stored in models/tuning/optuna.db, so an interrupted run resumes.

Scores: occ1 = Brier score; wet = RMSE (mm) on wet days after back-transform; rh = RMSE (%).

Output: reports/tuning/best_params.json ({site: {task: params}}) used by
`python -m src.models.train --tuned`, and reports/tuning/summary.csv.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import optuna
import pandas as pd

from src.models.baselines import fit_climatology
from src.models.evaluation import load_configs, split_bounds
from src.models.train import (dev_pixel_sample, feature_columns, fit_predict, horizon_table,
                              load_features, task_definitions)

ROOT = Path(__file__).resolve().parents[2]
TUNING_REPORTS = ROOT / "reports" / "tuning"
TUNING_MODELS = ROOT / "models" / "tuning"

DEFAULTS = {"catboost": {"depth": 7, "learning_rate": 0.05, "l2_leaf_reg": 3.0,
                         "random_strength": 1.0, "bagging_temperature": 1.0, "rsm": 1.0}}


def suggest(trial: optuna.Trial, algo: str) -> dict:
    if algo != "catboost":
        raise NotImplementedError("search space defined for catboost only")
    return {
        "depth": trial.suggest_int("depth", 4, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.03, 0.2, log=True),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 50.0, log=True),
        "random_strength": trial.suggest_float("random_strength", 0.1, 10.0, log=True),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 2.0),
        "rsm": trial.suggest_float("rsm", 0.3, 1.0),
    }


def fold_tables(feat: pd.DataFrame, exp: dict, h: int) -> list[tuple]:
    """(year, fit table, eval table) per fold, each with climatology fitted on its fit years."""
    train_start, train_end = split_bounds(exp, "train")
    wet_mm, exceed = exp["rain"]["wet_threshold_mm"], exp["rain"]["exceedance_mm"]
    folds = []
    for year in exp["tuning"]["eval_years"]:
        eval_start = pd.Timestamp(f"{year}-01-01")
        eval_end = min(pd.Timestamp(f"{year}-12-31"), train_end)
        fit_end = eval_start - pd.Timedelta(days=1)
        fit_rows = feat[(feat["date"] >= train_start) & (feat["date"] <= fit_end)]
        clim = fit_climatology(fit_rows, wet_mm, exceed, exp["climatology"]["window_days"])
        fit_t = pd.date_range(train_start + pd.Timedelta(days=h), fit_end)
        eval_t = pd.date_range(eval_start + pd.Timedelta(days=exp["gap_days"]), eval_end)
        folds.append((year, horizon_table(feat, clim, h, fit_t), horizon_table(feat, clim, h, eval_t)))
    return folds


def score(task: str, ev: pd.DataFrame, pred: np.ndarray, wet_mm: float) -> float:
    if task.startswith("occ"):
        thr = float(task[3:])
        return float(np.mean((pred - (ev["y_rain"] >= thr)) ** 2))
    if task == "wet":
        wet = (ev["y_rain"] >= wet_mm).to_numpy()
        return float(np.sqrt(np.mean((pred[wet] - ev["y_rain"].to_numpy()[wet]) ** 2)))
    ycol = "y_rh" if task == "rh" else "y_rain"
    return float(np.sqrt(np.mean((pred - ev[ycol].to_numpy()) ** 2)))


def make_objective(folds, cols, task, spec, algo, mcfg, wet_mm):
    def objective(trial: optuna.Trial) -> float:
        params = {**mcfg["params"], algo: {**mcfg["params"][algo], **suggest(trial, algo)}}
        scores, trees = [], []
        for i, (year, tr, ev) in enumerate(folds):
            _, n_iter, pred = fit_predict(algo, task, spec, tr, ev, ev, cols, params, mcfg, wet_mm)
            scores.append(score(task, ev, pred, wet_mm))
            trees.append(n_iter)
            trial.report(float(np.mean(scores)), step=i)
            if trial.should_prune():
                raise optuna.TrialPruned()
        trial.set_user_attr("fold_scores", scores)
        trial.set_user_attr("trees", trees)
        return float(np.mean(scores))
    return objective


def tune_site_task(site, feat, folds, cols, task, exp, n_trials) -> dict:
    tcfg, mcfg = exp["tuning"], exp["modelling"]
    algo, h, wet_mm = tcfg["algo"], tcfg["horizon"], exp["rain"]["wet_threshold_mm"]
    spec = task_definitions(wet_mm, exp["rain"]["exceedance_mm"])[task]
    TUNING_MODELS.mkdir(parents=True, exist_ok=True)
    study = optuna.create_study(
        study_name=f"{site}_h{h}_{task}_{algo}", direction="minimize", load_if_exists=True,
        storage=f"sqlite:///{(TUNING_MODELS / 'optuna.db').as_posix()}",
        sampler=optuna.samplers.TPESampler(seed=tcfg["seed"]),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=0))
    if not study.trials:
        study.enqueue_trial(DEFAULTS[algo])  # trial 0 = current defaults
    done = len([t for t in study.trials if t.state.is_finished()])
    t0 = time.time()
    if n_trials > done:
        study.optimize(make_objective(folds, cols, task, spec, algo, mcfg, wet_mm),
                       n_trials=n_trials - done, gc_after_trial=True)

    default = study.trials[0]
    best = study.best_trial
    states = pd.Series([t.state.name for t in study.trials]).value_counts().to_dict()
    gain = 1 - best.value / default.value
    print(f"  {site}/{task}: default {default.value:.4f} -> best {best.value:.4f} "
          f"({gain:+.1%}; trial {best.number}) | folds default "
          f"{[round(s, 4) for s in default.user_attrs['fold_scores']]} best "
          f"{[round(s, 4) for s in best.user_attrs['fold_scores']]} | {states} | {time.time() - t0:.0f}s")
    return {"site": site, "horizon": h, "task": task, "algo": algo, "default_score": default.value,
            "best_score": best.value, "improvement": gain, "best_trial": best.number,
            "default_folds": default.user_attrs["fold_scores"], "best_folds": best.user_attrs["fold_scores"],
            "best_trees": best.user_attrs["trees"], "trials": states, "best_params": best.params}


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune hyperparameters on the training years.")
    parser.add_argument("--site", help="site key (default: all sites)")
    parser.add_argument("--tasks", nargs="+", help="default: tuning.tasks")
    parser.add_argument("--trials", type=int, help="default: tuning.n_trials")
    args = parser.parse_args()

    sites_cfg, exp = load_configs()
    tcfg = exp["tuning"]
    sites = [args.site] if args.site else list(sites_cfg["sites"])
    tasks = args.tasks or tcfg["tasks"]
    n_trials = args.trials or tcfg["n_trials"]
    _, train_end = split_bounds(exp, "train")
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    TUNING_REPORTS.mkdir(parents=True, exist_ok=True)
    best_file, summary_file = TUNING_REPORTS / "best_params.json", TUNING_REPORTS / "summary.csv"
    best = json.loads(best_file.read_text(encoding="utf-8")) if best_file.exists() else {}
    summary = pd.read_csv(summary_file) if summary_file.exists() else pd.DataFrame()

    for site in sites:
        pixels = dev_pixel_sample(site, exp["modelling"]["dev_pixels"])
        feat = load_features(site, pixels, until=train_end)  # nothing after the training period
        assert feat["date"].max() <= train_end
        folds = fold_tables(feat, exp, tcfg["horizon"])
        cols = feature_columns(folds[0][1])
        print(f"\n=== {site}: {len(pixels)} pixels, data up to {feat['date'].max():%Y-%m-%d}, "
              f"horizon {tcfg['horizon']}, {tcfg['algo']}, folds "
              + ", ".join(f"{y} ({len(tr):,} fit / {len(ev):,} eval rows)" for y, tr, ev in folds) + " ===")
        for task in tasks:
            r = tune_site_task(site, feat, folds, cols, task, exp, n_trials)
            best.setdefault(site, {})[task] = r["best_params"]
            row = pd.DataFrame([{**r, "best_params": json.dumps(r["best_params"]),
                                 "trials": json.dumps(r["trials"])}])
            if len(summary):
                summary = summary[~((summary["site"] == site) & (summary["task"] == task)
                                    & (summary["horizon"] == r["horizon"]))]
            summary = pd.concat([summary, row], ignore_index=True)
            best_file.write_text(json.dumps(best, indent=2), encoding="utf-8")  # save after each task
            summary.to_csv(summary_file, index=False)
    print(f"\nsaved {best_file} and {summary_file}")


if __name__ == "__main__":
    main()
