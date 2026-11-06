"""Simple forecast fusion: average the three algorithms, and blend models with climatology.

Two steps (the second is a POST-HOC analysis: the test period was already used for the main results):

    python -m src.models.fusion validate   # develop on 2024 only; fit and save the blend weights
    python -m src.models.fusion test       # apply the saved weights unchanged to the final test forecasts

Methods (fixed before any test-period fusion result is computed)
- ens:        equal-weight average of CatBoost, LightGBM and XGBoost, per task.
- blend_cat:  w * CatBoost + (1 - w) * climatology.
- blend_ens:  w * ens + (1 - w) * climatology.
  w is fitted per site, horizon and task by least squares on 2024 (Brier score for probabilities,
  squared error for amounts and humidity) and clipped to [0, 1]. The wet-day amount weight uses wet
  target days only (as the wet-day model). The two-stage amount is recomputed from the fused rain
  probability and wet-day amount.

Validation (2024, 60-pixel scoring, where all three algorithms were scored): `ens` needs no fitting;
for the blends, w is fitted on odd months and applied to even months and vice versa, so every
blended 2024 forecast is out-of-fold. Scored with the moving-block bootstrap, paired against CatBoost.
The weights used for the test are refitted on all of 2024.

Output: reports/fusion/weights.csv, reports/fusion/validation_skill.csv,
reports/fusion/validation_differences.csv, then reports/fusion/test_skill.csv and
reports/fusion/test_differences.csv.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.models.evaluation import load_configs
from src.models.uncertainty import block_bootstrap_weights, bootstrap_site_horizon, paired_differences

ROOT = Path(__file__).resolve().parents[2]
MODELS = ROOT / "models"
OUT = ROOT / "reports" / "fusion"
ALGOS = ["catboost", "lightgbm", "xgboost"]
FUSED = ["ens", "blend_cat", "blend_ens"]
SHOW = ["rain>=1mm", "rain>=10mm", "rain_mm_wet", "rain_mm_all_2stage", "rh_pct"]


def tasks(wet_mm: float, exceed: list) -> list:
    return [f"occ{t:g}" for t in [wet_mm] + exceed] + ["wet", "direct", "rh"]


def target_and_rows(pr: pd.DataFrame, task: str, wet_mm: float):
    """(target values, row mask) that a task is fitted and scored on."""
    if task.startswith("occ"):
        return (pr["y_rain"] >= float(task[3:])).to_numpy(float), np.ones(len(pr), bool)
    if task == "wet":
        return pr["y_rain"].to_numpy(float), (pr["y_rain"] >= wet_mm).to_numpy()
    if task == "rh":
        return pr["y_rh"].to_numpy(float), np.ones(len(pr), bool)
    return pr["y_rain"].to_numpy(float), np.ones(len(pr), bool)


def blend_weight(y: np.ndarray, p: np.ndarray, c: np.ndarray) -> float:
    """Least-squares w for w * p + (1 - w) * c, clipped to [0, 1]."""
    d = p - c
    denom = float(np.sum(d * d))
    return float(np.clip(np.sum(d * (y - c)) / denom, 0.0, 1.0)) if denom > 0 else 1.0


def add_ensemble(pr: pd.DataFrame, task_list: list, wet_mm: float) -> pd.DataFrame:
    pr = pr.copy()
    for t in task_list:
        pr[f"ens__{t}"] = pr[[f"{a}__{t}" for a in ALGOS]].mean(axis=1)
    pr["ens__two_stage"] = pr[f"ens__occ{wet_mm:g}"] * pr["ens__wet"]
    return pr


def add_blends(pr: pd.DataFrame, weights: dict, task_list: list, wet_mm: float) -> pd.DataFrame:
    """weights: (base, task) -> w, or (base, task) -> per-row array of w (for the cross-fitted case)."""
    pr = pr.copy()
    for base, name in [("catboost", "blend_cat"), ("ens", "blend_ens")]:
        for t in task_list:
            w = weights[(base, t)]
            pr[f"{name}__{t}"] = w * pr[f"{base}__{t}"] + (1 - w) * pr[f"climatology__{t}"]
        pr[f"{name}__two_stage"] = pr[f"{name}__occ{wet_mm:g}"] * pr[f"{name}__wet"]
    return pr


def fit_weights(pr: pd.DataFrame, task_list: list, wet_mm: float, rows=None) -> dict:
    rows = np.ones(len(pr), bool) if rows is None else rows
    out = {}
    for base in ["catboost", "ens"]:
        for t in task_list:
            y, m = target_and_rows(pr, t, wet_mm)
            m = m & rows
            out[(base, t)] = blend_weight(y[m], pr[f"{base}__{t}"].to_numpy()[m], pr[f"climatology__{t}"].to_numpy()[m])
    return out


def score(pr: pd.DataFrame, exp: dict, key: dict) -> tuple[list, list]:
    ucfg, wet_mm, exceed = exp["uncertainty"], exp["rain"]["wet_threshold_mm"], exp["rain"]["exceedance_mm"]
    keep = ["pixel_id", "target_date", "y_rain", "y_rh"] + [c for c in pr.columns if "__" in c
                                                             and c.split("__")[0] in ALGOS + FUSED + ["persistence", "climatology"]]
    pr = pr[keep]
    W = block_bootstrap_weights(pr["target_date"].nunique(), ucfg["block_days"], ucfg["n_boot"], ucfg["seed"])
    rows, draws = bootstrap_site_horizon(pr, W, wet_mm, exceed)
    skill = [{**key, **r} for r in rows if r["reference"] == "climatology"]
    diffs = []
    for d in paired_differences(draws, pd.DataFrame(rows)):
        a, b = d["comparison"].split(" - ") if " - " in d["comparison"] else (None, None)
        if "catboost" not in (a, b) or (a in ALGOS and b in ALGOS):
            continue  # keep fused-vs-CatBoost only
        if a == "catboost":  # report fused minus CatBoost
            d = {**d, "comparison": f"{b} - catboost", "estimate": -d["estimate"],
                 "ci_low": -d["ci_high"], "ci_high": -d["ci_low"]}
        diffs.append({**key, **d})
    return skill, diffs


def fmt(r) -> str:
    star = "*" if (r["ci_low"] > 0 or r["ci_high"] < 0) else " "
    return f"{r['estimate']:+.3f} [{r['ci_low']:+.3f},{r['ci_high']:+.3f}]{star}"


def report(skill: pd.DataFrame, diffs: pd.DataFrame, title: str) -> None:
    print(f"\n{title}")
    sk = skill[skill["metric"].isin(["bss", "rmse_skill"]) & skill["target"].isin(SHOW)]
    for site, g in sk[sk["model"] == "catboost"].groupby("site"):
        t = g.assign(v=g.apply(fmt, axis=1)).pivot(index="horizon", columns="target", values="v")[SHOW]
        print(f"\n{site}: CatBoost skill vs climatology")
        print("  " + t.to_string().replace("\n", "\n  "))
    d = diffs[diffs["target"].isin(SHOW)]
    for (site, comp), g in d.groupby(["site", "comparison"]):
        t = g.assign(v=g.apply(fmt, axis=1)).pivot(index="horizon", columns="target", values="v")[SHOW]
        print(f"\n{site}: {comp} (skill difference vs climatology)")
        print("  " + t.to_string().replace("\n", "\n  "))
    sig = d[(d["ci_low"] > 0) | (d["ci_high"] < 0)]
    for comp, g in sig.groupby("comparison"):
        print(f"  {comp}: {int((g['estimate'] > 0).sum())} clearly better, {int((g['estimate'] < 0).sum())} clearly worse "
              f"(of {int((d['comparison'] == comp).sum())})")


def validate(sites_cfg: dict, exp: dict) -> None:
    wet_mm, exceed = exp["rain"]["wet_threshold_mm"], exp["rain"]["exceedance_mm"]
    task_list = tasks(wet_mm, exceed)
    n = exp["modelling"]["dev_pixels"]
    weights, skill, diffs = [], [], []
    for site in sites_cfg["sites"]:
        for h in sites_cfg["forecast"]["horizons"]:
            pr = pd.read_parquet(MODELS / site / f"train{n}_score{n}" / f"h{h}" / "val_predictions.parquet")
            assert pr["target_date"].max() <= pd.Timestamp("2024-12-31")
            pr = add_ensemble(pr, task_list, wet_mm)
            odd = (pd.DatetimeIndex(pr["target_date"]).month % 2 == 1)
            w_odd, w_even = fit_weights(pr, task_list, wet_mm, odd), fit_weights(pr, task_list, wet_mm, ~odd)
            w_all = fit_weights(pr, task_list, wet_mm)
            # out-of-fold: rows in odd months get the weight fitted on even months and vice versa
            cross = {k: np.where(odd, w_even[k], w_odd[k]) for k in w_all}
            scored = add_blends(pr, cross, task_list, wet_mm)
            s, d = score(scored, exp, {"site": site, "horizon": h})
            skill += s
            diffs += d
            weights += [{"site": site, "horizon": h, "base": b, "task": t, "w_all_2024": w_all[(b, t)],
                         "w_fit_odd_months": w_odd[(b, t)], "w_fit_even_months": w_even[(b, t)]} for b, t in w_all]
    OUT.mkdir(parents=True, exist_ok=True)
    weights, skill, diffs = pd.DataFrame(weights), pd.DataFrame(skill), pd.DataFrame(diffs)
    weights.to_csv(OUT / "weights.csv", index=False)
    skill.to_csv(OUT / "validation_skill.csv", index=False)
    diffs.to_csv(OUT / "validation_differences.csv", index=False)
    report(skill, diffs, "VALIDATION 2024 (60 pixels; blends out-of-fold by odd/even month)")
    w = weights[weights["task"].isin(["occ1", "wet", "rh"])].pivot_table(index=["site", "base", "task"],
                                                                          columns="horizon", values="w_all_2024")
    print("\nblend weights on the model (fitted on all of 2024; 1 = model only, 0 = climatology only):")
    print("  " + w.round(2).to_string().replace("\n", "\n  "))
    print(f"\nsaved {OUT / 'weights.csv'}, {OUT / 'validation_skill.csv'}, {OUT / 'validation_differences.csv'}")


def test(sites_cfg: dict, exp: dict) -> None:
    wet_mm, exceed = exp["rain"]["wet_threshold_mm"], exp["rain"]["exceedance_mm"]
    task_list = tasks(wet_mm, exceed)
    wts = pd.read_csv(OUT / "weights.csv")
    skill, diffs = [], []
    for site in sites_cfg["sites"]:
        for f in sorted((MODELS / site).glob("final_train*_score*/h*/test_predictions.parquet")):
            h = int(f.parent.name[1:])
            w = wts[(wts["site"] == site) & (wts["horizon"] == h)]
            weights = {(r.base, r.task): float(r.w_all_2024) for r in w.itertuples()}
            pr = add_blends(add_ensemble(pd.read_parquet(f), task_list, wet_mm), weights, task_list, wet_mm)
            s, d = score(pr, exp, {"site": site, "horizon": h})
            skill += s
            diffs += d
    skill, diffs = pd.DataFrame(skill), pd.DataFrame(diffs)
    skill.to_csv(OUT / "test_skill.csv", index=False)
    diffs.to_csv(OUT / "test_differences.csv", index=False)
    report(skill, diffs, "TEST 2025-01-08..2026-09-30, POST-HOC (all pixels; weights fixed from 2024)")
    print(f"\nsaved {OUT / 'test_skill.csv'}, {OUT / 'test_differences.csv'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Simple forecast fusion (ensemble and climatology blend).")
    parser.add_argument("step", choices=["validate", "test"])
    args = parser.parse_args()
    sites_cfg, exp = load_configs()
    validate(sites_cfg, exp) if args.step == "validate" else test(sites_cfg, exp)


if __name__ == "__main__":
    main()
