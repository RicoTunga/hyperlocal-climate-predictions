"""SHAP explanations for the decided CatBoost models: which inputs drive the forecasts?

Run from the repo root after src.models.train (decided setup):

    python -m src.explain.shap_analysis

What it computes (validation year 2024 only)
- Rows: `N_DAYS` target days spread evenly over 2024, times EVERY scored pixel of the site.
  For the wet-day amount model only the rows whose target day was wet (it was trained on those).
- SHAP values from CatBoost's exact TreeSHAP (`get_feature_importance(type="ShapValues")`),
  in the model's raw units: log-odds for rain probability, log(1 + mm) for wet-day amount,
  % for humidity. Checked: for every row, SHAP values + expected value = raw model output.
- Two views, per feature and per feature group:
    overall   = mean |SHAP| over all rows: what moves the forecasts in general;
    within-day = mean over days of the spread (std) of SHAP across pixels on that day:
                 what makes one pixel's forecast differ from its neighbours' on the same day.
  Area-wide and calendar inputs are identical for all pixels on a day, so any within-day spread
  they show comes from interactions with pixel-specific inputs.

Caveats: SHAP shows what the model USES, not what causes rain. Correlated inputs (own rain,
neighbourhood rain, regional rain) share credit in ways that depend on the fitted trees.

Output: reports/shap_features_validation.csv, reports/shap_groups_validation.csv,
reports/figures/shap_groups_overall.png, reports/figures/shap_groups_within_day.png.
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import Pool

from src.models.baselines import fit_climatology
from src.models.calibration import decided_run
from src.models.evaluation import eval_target_dates, load_configs, split_bounds
from src.models.train import dev_pixel_sample, feature_columns, horizon_table, load_features

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
N_DAYS = 60
HORIZONS = [1, 3, 7]
TASKS = {"occ1": "rain >= 1 mm (probability)", "wet": "wet-day amount", "rh": "humidity"}
ERA5_PREFIXES = {"rh", "t2m", "d2m", "sp", "wind", "tcc"}
GROUPS = ["own rain history", "neighbourhood rain", "area-wide", "own ERA5 weather",
          "climatology", "calendar", "location & elevation"]


def feature_group(col: str) -> str:
    """Assign one input column to one of GROUPS (raises if it fits none)."""
    if col.startswith("clim_"):
        return "climatology"
    if col.startswith("doy_"):
        return "calendar"
    if col.startswith("region_"):
        return "area-wide"
    if col.startswith("nbr"):
        return "neighbourhood rain"
    if col.startswith("precip_"):
        return "own rain history"
    if col in {"lat", "lon"} or col.startswith("elev_"):
        return "location & elevation"
    if col.split("_")[0] in ERA5_PREFIXES:
        return "own ERA5 weather"
    raise ValueError(f"No feature group for column '{col}'")


def within_day_spread(values: np.ndarray, day: np.ndarray) -> float:
    """Mean over days of the standard deviation of `values` across the rows of that day."""
    s = pd.Series(values).groupby(day).std(ddof=0)
    return float(s.mean())


def shap_values(model, X: pd.DataFrame) -> tuple[np.ndarray, float]:
    """(rows x features SHAP array, expected value), with the additivity check."""
    raw = model.get_feature_importance(Pool(X), type="ShapValues")
    shap, expected = raw[:, :-1], raw[:, -1]
    pred = model.predict(X, prediction_type="RawFormulaVal")
    err = float(np.max(np.abs(shap.sum(axis=1) + expected - pred)))
    if err > 1e-6:
        raise ValueError(f"SHAP values do not add up to the model output (max error {err:.2g})")
    return shap, float(expected[0])


def analyse_site(site: str, exp: dict, max_h: int) -> tuple[list, list]:
    mcfg, wet_mm, exceed = exp["modelling"], exp["rain"]["wet_threshold_mm"], exp["rain"]["exceedance_mm"]
    train_start, train_end = split_bounds(exp, "train")
    val_start, val_end = split_bounds(exp, "validation")
    runs = dict(decided_run(site, mcfg["dev_pixels"]))
    run_dir = next(iter(runs.values())).parent.parent
    n_score = int(run_dir.name.split("_score")[1])
    pixels = dev_pixel_sample(site, n_score)
    hist = load_features(site, pixels, until=train_end, start=train_start,
                         columns=["pixel_id", "date", "precip_mm", "rh_pct"])
    clim = fit_climatology(hist, wet_mm, exceed, exp["climatology"]["window_days"])
    del hist
    feat = load_features(site, pixels, until=val_end, start=val_start)
    targets = eval_target_dates(exp, "validation", max_h)
    days = targets[np.unique(np.linspace(0, len(targets) - 1, N_DAYS).round().astype(int))]
    print(f"\n=== {site}: {run_dir.name}, {len(days)} days x {len(pixels)} pixels ===")

    feat_rows, group_rows = [], []
    for h in HORIZONS:
        t = horizon_table(feat, clim, h, days)
        cols = feature_columns(t)
        groups = pd.Series({c: feature_group(c) for c in cols})
        for task in TASKS:
            model = joblib.load(runs[h].parent / f"catboost_{task}.joblib")
            assert list(model.feature_names_) == cols, "feature columns differ from training"
            rows = t[t["y_rain"] >= wet_mm] if task == "wet" else t
            shap, expected = shap_values(model, rows[cols])
            day = pd.factorize(rows["target_date"])[0]
            key = {"site": site, "horizon": h, "task": task, "n_rows": len(rows)}
            for j, c in enumerate(cols):
                feat_rows.append({**key, "feature": c, "group": groups[c],
                                  "overall": float(np.abs(shap[:, j]).mean()),
                                  "within_day": within_day_spread(shap[:, j], day)})
            for g in GROUPS:
                idx = np.flatnonzero(groups.to_numpy() == g)
                s = shap[:, idx].sum(axis=1)
                group_rows.append({**key, "group": g, "n_features": len(idx),
                                   "overall": float(np.abs(s).mean()), "within_day": within_day_spread(s, day)})
            top = pd.DataFrame(feat_rows[-len(cols):]).nlargest(5, "overall")["feature"].tolist()
            print(f"  h{h} {task:5}: {len(rows):,} rows, SHAP additivity OK | top: {top}")
    return feat_rows, group_rows


# ---------------------------------------------------------------- figure
INK = {"primary": "#0b0b0b", "secondary": "#52514e", "muted": "#898781",
       "grid": "#e1e0d9", "axis": "#c3c2b7", "surface": "#fcfcfb"}
SERIES = [(1, "#2a78d6"), (3, "#eb6834"), (7, "#1baf7a")]  # validated categorical slots 1-3


def group_figure(groups: pd.DataFrame, value: str, title: str, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sites = list(dict.fromkeys(groups["site"]))
    fig, axes = plt.subplots(len(sites), len(TASKS), figsize=(4.4 * len(TASKS), 3.6 * len(sites)),
                             dpi=150, facecolor=INK["surface"], squeeze=False, sharey=True)
    y = np.arange(len(GROUPS))
    height = 0.26
    for i, site in enumerate(sites):
        for j, task in enumerate(TASKS):
            ax = axes[i, j]
            ax.set_facecolor(INK["surface"])
            g = groups[(groups["site"] == site) & (groups["task"] == task)]
            for k, (h, color) in enumerate(SERIES):
                s = g[g["horizon"] == h].set_index("group").reindex(GROUPS)
                share = 100 * s[value] / s[value].sum()
                ax.barh(y + (k - 1) * (height + 0.02), share, height=height, color=color,
                        edgecolor=INK["surface"], linewidth=0.8, label=f"day {h}", zorder=2)
            ax.set_yticks(y)
            ax.set_yticklabels(GROUPS, fontsize=8, color=INK["secondary"])
            ax.grid(axis="x", color=INK["grid"], lw=0.6, zorder=0)
            for side in ("top", "right", "left"):
                ax.spines[side].set_visible(False)
            ax.spines["bottom"].set_color(INK["axis"])
            ax.tick_params(axis="x", colors=INK["muted"], labelsize=8)
            ax.tick_params(axis="y", length=0)
            ax.set_title(f"{site.capitalize()}: {TASKS[task]}", fontsize=9.5, color=INK["primary"], loc="left")
            if i == len(sites) - 1:
                ax.set_xlabel("share of total (%)", fontsize=8.5, color=INK["secondary"])
    axes[0, 0].invert_yaxis()  # y is shared: invert once, so groups read top to bottom, day 1 first
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, fontsize=9,
               labelcolor=INK["secondary"], bbox_to_anchor=(0.5, 1.0))
    fig.suptitle(title, fontsize=10, color=INK["primary"], y=1.05)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight", facecolor=INK["surface"])
    plt.close(fig)


def main() -> None:
    sites_cfg, exp = load_configs()
    max_h = max(sites_cfg["forecast"]["horizons"])
    feats, groups = [], []
    for site in sites_cfg["sites"]:
        f, g = analyse_site(site, exp, max_h)
        feats += f
        groups += g
    feats, groups = pd.DataFrame(feats), pd.DataFrame(groups)
    for v in ("overall", "within_day"):
        groups[f"{v}_share"] = groups[v] / groups.groupby(["site", "horizon", "task"])[v].transform("sum")
    feats.to_csv(REPORTS / "shap_features_validation.csv", index=False)
    groups.to_csv(REPORTS / "shap_groups_validation.csv", index=False)
    group_figure(groups, "overall", "What drives the forecasts: share of mean |SHAP| by input group "
                 "(CatBoost, validation 2024)", FIGURES / "shap_groups_overall.png")
    group_figure(groups, "within_day", "What makes pixels differ on the same day: share of within-day "
                 "SHAP spread by input group", FIGURES / "shap_groups_within_day.png")

    for v in ("overall", "within_day"):
        t = groups.pivot_table(index="group", columns=["site", "task", "horizon"], values=f"{v}_share")
        t = (100 * t.reindex(GROUPS)).round(0).astype(int)
        print(f"\n{v} share (%) by input group:")
        print(t.to_string())
    print(f"\nsaved {REPORTS / 'shap_features_validation.csv'}, {REPORTS / 'shap_groups_validation.csv'} and "
          f"figures in {FIGURES}")


if __name__ == "__main__":
    main()
