"""95% confidence intervals for validation skill scores, by a moving-block bootstrap over days.

Run from the repo root after src.models.train (which saves
models/<site>/train<N>_score<M>/h<h>/val_predictions.parquet; every saved run is processed):

    python -m src.models.uncertainty

Why resample days, in blocks:
- Pixels on the same day are strongly correlated (one storm covers many pixels), so the
  independent unit is the day, not the pixel-day. Each resample keeps all pixels of a day together.
- Consecutive days are also correlated (humidity especially), so days are drawn in circular
  blocks of `block_days` consecutive days.
- Every model and baseline is scored on the SAME resampled days, so differences between models
  (e.g. LightGBM minus CatBoost, two-stage minus direct) get valid paired intervals.

What the intervals do and do not cover: they describe how much the scores would vary over
other sets of days like those in the validation year 2024, for these 60 pixels. They do not
cover year-to-year changes in the weather (2024 is a single year), so they are a lower bound
on the true uncertainty. ROC-AUC is not bootstrapped (it does not split into per-day sums).

Speed: squared and absolute errors are summed per day once; each resample is then a weighted
sum of those daily totals (a matrix product), so thousands of resamples take seconds.

Output: reports/skill_ci_validation.csv (one row per site, horizon, model, target, metric and
reference) and reports/model_differences_ci_validation.csv (paired differences).
"""
import itertools
from pathlib import Path

import numpy as np
import pandas as pd

from src.models.evaluation import load_configs

ROOT = Path(__file__).resolve().parents[2]
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"
ALGOS = ["lightgbm", "xgboost", "catboost"]
BASELINES = ["persistence", "climatology"]


def model_names(columns) -> list:
    """Baselines first, then every model that has predictions saved (e.g. catboost_tuned)."""
    found = sorted({c.split("__")[0] for c in columns if "__" in c} - set(BASELINES))
    order = [a for a in ALGOS if a in found] + [m for m in found if m not in ALGOS]
    return [b for b in BASELINES if f"{b}__rh" in columns] + order


def block_bootstrap_weights(n_days: int, block: int, n_boot: int, seed: int) -> np.ndarray:
    """(n_boot, n_days) array: how many times each day appears in each resample.

    Each resample strings together random circular blocks of `block` consecutive days until it
    has n_days days, so every resample has the same size as the original sample."""
    rng = np.random.default_rng(seed)
    n_blocks = -(-n_days // block)  # ceiling division
    starts = rng.integers(0, n_days, size=(n_boot, n_blocks))
    idx = ((starts[:, :, None] + np.arange(block)) % n_days).reshape(n_boot, -1)[:, :n_days]
    flat = (idx + np.arange(n_boot)[:, None] * n_days).ravel()
    return np.bincount(flat, minlength=n_boot * n_days).reshape(n_boot, n_days).astype(float)


def daily_sums(day: np.ndarray, n_days: int, values: np.ndarray) -> np.ndarray:
    return np.bincount(day, weights=values, minlength=n_days)


def task_specs(wet_mm: float, exceed: list) -> list[dict]:
    """Which prediction column, target and row subset each reported target uses."""
    specs = [{"target": f"rain>={t:g}mm", "pred": f"occ{t:g}", "kind": "binary", "thr": t}
             for t in [wet_mm] + exceed]
    specs += [
        {"target": "rain_mm_wet", "pred": "wet", "kind": "amount", "wet_only": True},
        {"target": "rain_mm_all", "pred": "direct", "kind": "amount"},
        {"target": "rain_mm_all_2stage", "pred": "two_stage", "ref_pred": "direct", "kind": "amount"},
        {"target": "rh_pct", "pred": "rh", "kind": "rh"},
    ]
    return specs


def observed(pr: pd.DataFrame, spec: dict, wet_mm: float) -> tuple[np.ndarray, np.ndarray]:
    """(target values, row mask) for one spec."""
    if spec["kind"] == "binary":
        y = (pr["y_rain"] >= spec["thr"]).astype(float).to_numpy()
    elif spec["kind"] == "rh":
        y = pr["y_rh"].to_numpy()
    else:
        y = pr["y_rain"].to_numpy()
    mask = (pr["y_rain"] >= wet_mm).to_numpy() if spec.get("wet_only") else np.ones(len(pr), bool)
    return y, mask


def bootstrap_site_horizon(pr: pd.DataFrame, W: np.ndarray, wet_mm: float, exceed: list):
    """Point estimates and bootstrap draws for every model, target and reference."""
    days = pd.DatetimeIndex(pr["target_date"])
    day_codes, uniq = pd.factorize(days, sort=True)
    n_days = len(uniq)
    assert W.shape[1] == n_days, "weights do not match the number of target days"
    ones = np.ones(n_days)
    models = model_names(pr.columns)

    rows, draws = [], {}
    for spec in task_specs(wet_mm, exceed):
        y, mask = observed(pr, spec, wet_mm)
        d = day_codes[mask]
        n_d = daily_sums(d, n_days, np.ones(mask.sum()))
        ref_col = spec.get("ref_pred", spec["pred"])
        se_ref = {r: daily_sums(d, n_days, (pr[f"{r}__{ref_col}"].to_numpy()[mask] - y[mask]) ** 2)
                  for r in ["climatology", "persistence"]}
        for m in models:
            col = f"{m}__{spec['pred']}"
            if col not in pr.columns:
                continue
            err = pr[col].to_numpy()[mask] - y[mask]
            se, ae = daily_sums(d, n_days, err ** 2), daily_sums(d, n_days, np.abs(err))
            for w_name, w in (("estimate", ones[None, :]), ("draws", W)):
                n, sse, sae = w @ n_d, w @ se, w @ ae
                out = {"mse": sse / n, "mae": sae / n}
                for r, s_ref in se_ref.items():
                    ratio = sse / (w @ s_ref)
                    skill = 1 - ratio if spec["kind"] == "binary" else 1 - np.sqrt(ratio)
                    out[f"skill_vs_{r}"] = skill
                if w_name == "estimate":
                    point = {k: float(v[0]) for k, v in out.items()}
                else:
                    draws[(m, spec["target"])] = out
            metric_name = "bss" if spec["kind"] == "binary" else "rmse_skill"
            for key, val in point.items():
                if key.startswith("skill_vs_"):
                    metric, ref = metric_name, key.replace("skill_vs_", "")
                else:
                    metric, ref = ("brier" if spec["kind"] == "binary" else "mse") if key == "mse" else key, ""
                dr = draws[(m, spec["target"])][key]
                if key == "mse" and spec["kind"] != "binary":  # report RMSE rather than MSE
                    metric, val, dr = "rmse", np.sqrt(val), np.sqrt(dr)
                lo, hi = np.percentile(dr, [2.5, 97.5])
                rows.append({"model": m, "target": spec["target"], "metric": metric, "reference": ref,
                             "estimate": val, "ci_low": lo, "ci_high": hi})
    return rows, draws


def paired_differences(draws: dict, point: pd.DataFrame) -> list[dict]:
    """Skill differences vs climatology between algorithms, and two-stage minus direct."""
    rows = []
    est = point.set_index(["model", "target", "metric", "reference"])["estimate"]

    def skill_metric(target):
        return "bss" if target.startswith("rain>=") else "rmse_skill"

    targets = sorted({t for _, t in draws})
    names = {m for m, _ in draws} - set(BASELINES)
    algos = [a for a in ALGOS if a in names] + sorted(names - set(ALGOS))
    for t in targets:
        for a, b in itertools.combinations(algos, 2):
            diff = draws[(a, t)]["skill_vs_climatology"] - draws[(b, t)]["skill_vs_climatology"]
            e = est[(a, t, skill_metric(t), "climatology")] - est[(b, t, skill_metric(t), "climatology")]
            rows.append({"comparison": f"{a} - {b}", "target": t, "estimate": e,
                         "ci_low": np.percentile(diff, 2.5), "ci_high": np.percentile(diff, 97.5)})
    for a in algos:
        # both are skill vs the climatological mean amount on all days
        diff = (draws[(a, "rain_mm_all_2stage")]["skill_vs_climatology"]
                - draws[(a, "rain_mm_all")]["skill_vs_climatology"])
        e = (est[(a, "rain_mm_all_2stage", "rmse_skill", "climatology")]
             - est[(a, "rain_mm_all", "rmse_skill", "climatology")])
        rows.append({"comparison": f"{a}: two-stage - direct", "target": "rain_mm_all", "estimate": e,
                     "ci_low": np.percentile(diff, 2.5), "ci_high": np.percentile(diff, 97.5)})
    return rows


def fmt(r) -> str:
    star = "*" if (r["ci_low"] > 0 or r["ci_high"] < 0) else " "
    return f"{r['estimate']:+.3f} [{r['ci_low']:+.3f},{r['ci_high']:+.3f}]{star}"


def prediction_files(site: str) -> list[tuple[int, int, int, Path]]:
    """(n_train_pixels, n_score_pixels, horizon, file) for every saved run of a site."""
    out = []
    for f in sorted((MODELS / site).glob("train*_score*/h*/val_predictions.parquet")):
        n_train, n_score = (int(x) for x in f.parent.parent.name.replace("train", "").split("_score"))
        out.append((n_train, n_score, int(f.parent.name[1:]), f))
    return out


def combine_runs(frames: dict) -> pd.DataFrame:
    """Put the predictions of several runs scored on the SAME rows side by side.

    frames: n_train_pixels -> prediction frame. Model columns are renamed to
    '<model>@train<N>__<task>'; baselines and targets must be identical across runs (same scored
    pixels, same days, same climatology), which is checked."""
    keys = ["pixel_id", "target_date"]
    base_cols = keys + ["y_rain", "y_rh"]
    out = None
    for n_train, pr in sorted(frames.items()):
        pr = pr.sort_values(keys).reset_index(drop=True)
        baselines = [c for c in pr.columns if c.split("__")[0] in BASELINES]
        models = [c for c in pr.columns if "__" in c and c not in baselines]
        renamed = pr[keys + models].rename(
            columns={c: f"{c.split('__')[0]}@train{n_train}__{c.split('__')[1]}" for c in models})
        if out is None:
            out = pr[base_cols + baselines].merge(renamed, on=keys, validate="one_to_one")
            ref = pr[base_cols + baselines]
        else:
            same = pr[base_cols + baselines]
            if len(same) != len(ref) or not (same[keys].equals(ref[keys])
                                             and np.allclose(same[base_cols[2:] + baselines].to_numpy(),
                                                             ref[base_cols[2:] + baselines].to_numpy())):
                raise ValueError("Runs were not scored on identical rows with identical baselines.")
            out = out.merge(renamed, on=keys, validate="one_to_one")
    return out


def training_size_comparisons(sites, ucfg, wet_mm, exceed) -> pd.DataFrame:
    """Paired differences between runs that share the scored pixels but differ in training pixels."""
    rows = []
    for site in sites:
        groups = {}
        for n_train, n_score, h, f in prediction_files(site):
            groups.setdefault((n_score, h), {})[n_train] = f
        for (n_score, h), runs in sorted(groups.items()):
            if len(runs) < 2:
                continue
            pr = combine_runs({n: pd.read_parquet(f) for n, f in runs.items()})
            W = block_bootstrap_weights(pr["target_date"].nunique(), ucfg["block_days"], ucfg["n_boot"], ucfg["seed"])
            r, draws = bootstrap_site_horizon(pr, W, wet_mm, exceed)
            for d in paired_differences(draws, pd.DataFrame(r)):
                if "@" in d["comparison"] and "two-stage" not in d["comparison"]:
                    left, right = d["comparison"].split(" - ")
                    (a, na), (b, nb) = (x.split("@train") for x in (left, right))
                    if a != b:  # only the same algorithm at different training sizes
                        continue
                    if int(na) < int(nb):  # always report "more training pixels minus fewer"
                        d = {**d, "comparison": f"{right} - {left}", "estimate": -d["estimate"],
                             "ci_low": -d["ci_high"], "ci_high": -d["ci_low"]}
                    rows.append({"site": site, "n_score_pixels": n_score, "horizon": h, **d})
    return pd.DataFrame(rows)


def main() -> None:
    sites_cfg, exp = load_configs()
    ucfg, wet_mm, exceed = exp["uncertainty"], exp["rain"]["wet_threshold_mm"], exp["rain"]["exceedance_mm"]
    all_rows, all_diffs = [], []
    for site in sites_cfg["sites"]:
        files = prediction_files(site)
        if not files:
            print(f"  no saved predictions for {site} (run src.models.train first)")
        for n_train, n_score, h, f in files:
            pr = pd.read_parquet(f)
            n_days = pr["target_date"].nunique()
            # same seed everywhere: the same days are resampled for every site, run and horizon
            W = block_bootstrap_weights(n_days, ucfg["block_days"], ucfg["n_boot"], ucfg["seed"])
            rows, draws = bootstrap_site_horizon(pr, W, wet_mm, exceed)
            diffs = paired_differences(draws, pd.DataFrame(rows))
            for r in rows + diffs:
                r.update(site=site, n_train_pixels=n_train, n_score_pixels=n_score, horizon=h)
            all_rows += rows
            all_diffs += diffs

    lead = ["site", "n_train_pixels", "n_score_pixels", "horizon"]
    res, dif = pd.DataFrame(all_rows), pd.DataFrame(all_diffs)
    res = res[lead + [c for c in res.columns if c not in lead]]
    dif = dif[lead + [c for c in dif.columns if c not in lead]]

    # consistency check: point estimates must equal those computed in src.models.train
    mv = pd.read_csv(REPORTS / "models_validation.csv")
    a = res[(res["reference"] == "climatology") & ~res["model"].isin(BASELINES)]
    m = a.merge(mv, on=lead + ["model", "target", "metric"], suffixes=("", "_train"))
    max_diff = (m["estimate"] - m["value"]).abs().max()
    print(f"{'PASS ' if max_diff < 1e-6 else 'CHECK'} point estimates match reports/models_validation.csv "
          f"({len(m)} of {len(a)} values matched, max difference {max_diff:.1g})")
    print(f"bootstrap: {ucfg['n_boot']} resamples of {ucfg['block_days']}-day blocks of target days; "
          "[95% interval], * = interval excludes 0")

    show = [("rain>=1mm", "bss"), ("rain>=10mm", "bss"), ("rain>=25mm", "bss"),
            ("rain_mm_wet", "rmse_skill"), ("rh_pct", "rmse_skill")]
    clim_ref = res[res["reference"] == "climatology"]
    for (site, n_train, n_score), g in clim_ref.groupby(["site", "n_train_pixels", "n_score_pixels"]):
        algo = next((a for a in ["catboost", "lightgbm", "xgboost"] if a in set(g["model"])), None)
        if algo is None:
            continue
        t = g[g["model"] == algo]
        tab = pd.DataFrame({f"{tg} {mt}": t[(t["target"] == tg) & (t["metric"] == mt)]
                            .set_index("horizon").apply(fmt, axis=1) for tg, mt in show})
        print(f"\n{site}, trained on {n_train} / scored on {n_score} pixels, {algo}: skill vs climatology")
        print(tab.to_string())
        dd = dif[(dif["site"] == site) & (dif["n_train_pixels"] == n_train) & (dif["n_score_pixels"] == n_score)]
        pairs = dd[~dd["comparison"].str.contains("two-stage")]
        sig = pairs[(pairs["ci_low"] > 0) | (pairs["ci_high"] < 0)]
        print(f"  model differences whose interval excludes 0: {len(sig)} of {len(pairs)}")
        tuned = pairs[pairs["comparison"].str.contains("_tuned")]
        if len(tuned):
            print("  tuned vs default (skill difference vs climatology):")
            for (h, comp), x in tuned.groupby(["horizon", "comparison"]):
                print(f"    h{h} {comp}: " + "; ".join(f"{r['target']} {fmt(r).strip()}" for _, r in x.iterrows()
                                                    if r["target"] in {"rain>=1mm", "rain_mm_wet", "rh_pct",
                                                                       "rain_mm_all_2stage"}))

    size = training_size_comparisons(sites_cfg["sites"], ucfg, wet_mm, exceed)
    if len(size):
        print("\nmore vs fewer training pixels, same scored pixels and days (skill difference vs climatology):")
        for (site, n_score, comp), g in size.groupby(["site", "n_score_pixels", "comparison"]):
            g = g[g["target"].isin(["rain>=1mm", "rain>=10mm", "rain_mm_wet", "rh_pct"])]
            tab = g.assign(v=g.apply(fmt, axis=1)).pivot(index="horizon", columns="target", values="v")
            print(f"\n{site}, scored on {n_score} pixels: {comp}")
            print(tab.to_string())
        size.to_csv(REPORTS / "training_size_ci_validation.csv", index=False)
        print(f"saved {REPORTS / 'training_size_ci_validation.csv'} ({len(size)} rows)")

    REPORTS.mkdir(exist_ok=True)
    res.to_csv(REPORTS / "skill_ci_validation.csv", index=False)
    dif.to_csv(REPORTS / "model_differences_ci_validation.csv", index=False)
    print(f"\nsaved {REPORTS / 'skill_ci_validation.csv'} ({len(res)} rows) and "
          f"{REPORTS / 'model_differences_ci_validation.csv'} ({len(dif)} rows)")


if __name__ == "__main__":
    main()
