"""Probability calibration of the rain-occurrence forecasts on the validation year.

Run from the repo root after src.models.train:

    python -m src.models.calibration

Question: when the model says "30% chance of >= 1 mm", does it rain on about 30% of those
pixel-days? Checked for every rain threshold (1, 10, 25 mm) and horizon, for the decided setup
(CatBoost trained on modelling.dev_pixels pixels, scored on all pixels), with climatology
alongside for reference.

Measures
- bias (calibration-in-the-large): mean forecast probability minus observed frequency.
  > 0 = forecasts too wet on average. 95% interval by the moving-block bootstrap over days.
- calibration slope / intercept: logistic regression of the outcome on logit(forecast).
  slope 1 = ideal; < 1 = too extreme (overconfident); > 1 = too timid (underconfident).
- Brier decomposition (Murphy, forecasts grouped into the bins below):
  Brier ~= reliability - resolution + uncertainty. Reliability is the calibration error
  (0 = perfect), resolution is how well forecasts separate wet from dry days (higher is
  better), uncertainty is the base-rate variance (out of the forecaster's control).
- reliability table: per probability bin, mean forecast vs observed frequency with a 95%
  interval (same day-block bootstrap), and the number of pixel-days.

Output: reports/calibration_validation.csv, reports/calibration_bins_validation.csv,
reports/figures/reliability_validation.png (reliability diagrams; the CSVs are its table view).
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.models.evaluation import load_configs
from src.models.uncertainty import block_bootstrap_weights, daily_sums, prediction_files

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
BIN_EDGES = np.array([0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0 + 1e-9])
MIN_BIN_COUNT = 200  # bins with fewer pixel-days are reported but not drawn


def calibration_slope(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    """(slope, intercept) of a logistic regression of y on logit(p)."""
    p = np.clip(p, 1e-4, 1 - 1e-4)
    x = np.log(p / (1 - p))[:, None]
    lr = LogisticRegression(C=np.inf, max_iter=1000).fit(x, y.astype(int))  # C=inf: unpenalised
    return float(lr.coef_[0, 0]), float(lr.intercept_[0])


def brier_decomposition(y: np.ndarray, p: np.ndarray, edges=BIN_EDGES) -> dict:
    """Murphy decomposition on binned forecasts. Exact when p takes one value per bin."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    k = np.clip(np.digitize(p, edges) - 1, 0, len(edges) - 2)
    n = np.bincount(k, minlength=len(edges) - 1).astype(float)
    sp = np.bincount(k, weights=p, minlength=len(n))
    sy = np.bincount(k, weights=y, minlength=len(n))
    ok = n > 0
    pbar, obar_k, obar = sp[ok] / n[ok], sy[ok] / n[ok], y.mean()
    return {"brier": float(np.mean((p - y) ** 2)),
            "reliability": float(np.sum(n[ok] * (pbar - obar_k) ** 2) / len(y)),
            "resolution": float(np.sum(n[ok] * (obar_k - obar) ** 2) / len(y)),
            "uncertainty": float(obar * (1 - obar))}


def bootstrap_bias_and_bins(y, p, day, n_days, W, edges=BIN_EDGES):
    """Bias (mean p - mean y) and per-bin observed frequency, with block-bootstrap intervals."""
    n_d = daily_sums(day, n_days, np.ones(len(y)))
    bias_d = daily_sums(day, n_days, p - y)
    bias = float(bias_d.sum() / n_d.sum())
    bias_draws = (W @ bias_d) / (W @ n_d)
    k = np.clip(np.digitize(p, edges) - 1, 0, len(edges) - 2)
    bins = []
    for b in range(len(edges) - 1):
        m = k == b
        if not m.any():
            continue
        cnt = daily_sums(day[m], n_days, np.ones(m.sum()))
        sy = daily_sums(day[m], n_days, y[m])
        with np.errstate(invalid="ignore", divide="ignore"):
            draws = (W @ sy) / (W @ cnt)
        draws = draws[np.isfinite(draws)]
        bins.append({"bin_low": edges[b], "bin_high": min(edges[b + 1], 1.0), "n": int(m.sum()),
                     "mean_forecast": float(p[m].mean()), "observed": float(y[m].mean()),
                     "obs_ci_low": float(np.percentile(draws, 2.5)),
                     "obs_ci_high": float(np.percentile(draws, 97.5))})
    lo, hi = np.percentile(bias_draws, [2.5, 97.5])
    return bias, float(lo), float(hi), bins


def decided_run(site: str, n_train: int) -> list[tuple[int, Path]]:
    """(horizon, file) of the decided setup: n_train training pixels, scored on the most pixels."""
    files = [(n_sc, h, f) for n_tr, n_sc, h, f in prediction_files(site) if n_tr == n_train]
    if not files:
        raise SystemExit(f"No runs with {n_train} training pixels for {site}; run src.models.train first.")
    n_score = max(n for n, _, _ in files)
    return sorted((h, f) for n, h, f in files if n == n_score)


def analyse(sites_cfg, exp, models=("catboost", "climatology")):
    ucfg, wet_mm, exceed = exp["uncertainty"], exp["rain"]["wet_threshold_mm"], exp["rain"]["exceedance_mm"]
    summary, bins = [], []
    for site in sites_cfg["sites"]:
        for h, f in decided_run(site, exp["modelling"]["dev_pixels"]):
            pr = pd.read_parquet(f)
            day, uniq = pd.factorize(pd.DatetimeIndex(pr["target_date"]), sort=True)
            W = block_bootstrap_weights(len(uniq), ucfg["block_days"], ucfg["n_boot"], ucfg["seed"])
            n_score = pr["pixel_id"].nunique()
            for thr in [wet_mm] + exceed:
                y = (pr["y_rain"] >= thr).to_numpy(float)
                for m in models:
                    p = pr[f"{m}__occ{thr:g}"].to_numpy(float)
                    bias, lo, hi, b = bootstrap_bias_and_bins(y, p, day, len(uniq), W)
                    slope, intercept = calibration_slope(y, p)
                    key = {"site": site, "n_score_pixels": n_score, "horizon": h,
                           "threshold_mm": thr, "model": m}
                    summary.append({**key, "observed_rate": y.mean(), "mean_forecast": p.mean(),
                                    "bias": bias, "bias_ci_low": lo, "bias_ci_high": hi,
                                    "slope": slope, "intercept": intercept, **brier_decomposition(y, p)})
                    bins += [{**key, **x} for x in b]
    return pd.DataFrame(summary), pd.DataFrame(bins)


# ---------------------------------------------------------------- figure
INK = {"primary": "#0b0b0b", "secondary": "#52514e", "muted": "#898781",
       "grid": "#e1e0d9", "axis": "#c3c2b7", "surface": "#fcfcfb"}
SERIES = [(1, "#2a78d6", "o"), (3, "#eb6834", "s"), (7, "#1baf7a", "^")]  # validated slots 1-3


def reliability_figure(bins: pd.DataFrame, out: Path, model="catboost") -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sites = list(dict.fromkeys(bins["site"]))
    thresholds = sorted(bins["threshold_mm"].unique())
    fig, axes = plt.subplots(len(sites), len(thresholds), figsize=(4.2 * len(thresholds), 4.0 * len(sites)),
                             dpi=150, facecolor=INK["surface"], squeeze=False)
    for i, site in enumerate(sites):
        for j, thr in enumerate(thresholds):
            ax = axes[i, j]
            ax.set_facecolor(INK["surface"])
            g = bins[(bins["site"] == site) & (bins["threshold_mm"] == thr) & (bins["n"] >= MIN_BIN_COUNT)]
            top = max(0.1, float(g[["mean_forecast", "observed"]].max().max()) * 1.1) if len(g) else 1.0
            top = min(1.0, np.ceil(top * 10) / 10)
            ax.plot([0, top], [0, top], color=INK["axis"], lw=1.2, zorder=1)
            c = g[g["model"] == "climatology"]
            for hz in sorted(c["horizon"].unique())[:1]:  # climatology does not depend on the horizon
                cc = c[c["horizon"] == hz].sort_values("mean_forecast")
                ax.plot(cc["mean_forecast"], cc["observed"], color=INK["muted"], lw=1.4, ls=(0, (4, 2)),
                        marker="D", ms=4, zorder=2, label="climatology")
            for hz, color, marker in SERIES:
                s = g[(g["model"] == model) & (g["horizon"] == hz)].sort_values("mean_forecast")
                if s.empty:
                    continue
                ax.errorbar(s["mean_forecast"], s["observed"],
                            yerr=[s["observed"] - s["obs_ci_low"], s["obs_ci_high"] - s["observed"]],
                            color=color, lw=1.6, marker=marker, ms=6, mec=INK["surface"], mew=1.0,
                            elinewidth=0.8, capsize=0, zorder=3, label=f"day {hz}")
                if i == 0 and j == 0:  # selective direct labels in the first panel
                    last = s.iloc[-1]
                    ax.annotate(f"day {hz}", (last["mean_forecast"], last["observed"]), xytext=(6, 0),
                                textcoords="offset points", va="center", fontsize=8, color=INK["secondary"])
            ax.set_xlim(0, top)
            ax.set_ylim(0, top)
            ax.set_aspect("equal")
            ax.grid(color=INK["grid"], lw=0.6)
            ax.set_axisbelow(True)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            for side in ("left", "bottom"):
                ax.spines[side].set_color(INK["axis"])
            ax.tick_params(colors=INK["muted"], labelsize=8)
            ax.set_title(f"{site.capitalize()}: rain >= {thr:g} mm", fontsize=10, color=INK["primary"], loc="left")
            if i == len(sites) - 1:
                ax.set_xlabel("forecast probability", fontsize=9, color=INK["secondary"])
            if j == 0:
                ax.set_ylabel("observed frequency", fontsize=9, color=INK["secondary"])
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(labels), frameon=False, fontsize=9,
               labelcolor=INK["secondary"], bbox_to_anchor=(0.5, 1.0))
    fig.suptitle("Reliability of CatBoost rain probabilities, validation 2024 (diagonal = perfect; bars = 95% interval)",
                 fontsize=10, color=INK["primary"], y=1.04)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight", facecolor=INK["surface"])
    plt.close(fig)


def main() -> None:
    sites_cfg, exp = load_configs()
    summary, bins = analyse(sites_cfg, exp)
    REPORTS.mkdir(exist_ok=True)
    summary.to_csv(REPORTS / "calibration_validation.csv", index=False)
    bins.to_csv(REPORTS / "calibration_bins_validation.csv", index=False)
    fig = FIGURES / "reliability_validation.png"
    reliability_figure(bins, fig)

    def fmt(r):
        return f"{r['bias']:+.3f} [{r['bias_ci_low']:+.3f},{r['bias_ci_high']:+.3f}]"

    for (site, thr), g in summary.groupby(["site", "threshold_mm"], sort=False):
        print(f"\n{site}, rain >= {thr:g} mm (observed rate in 2024: {g['observed_rate'].iloc[0]:.3f})")
        t = pd.DataFrame({
            "cat_bias [95%]": g[g.model == "catboost"].set_index("horizon").apply(fmt, axis=1),
            "cat_slope": g[g.model == "catboost"].set_index("horizon")["slope"].round(2),
            "cat_rel": g[g.model == "catboost"].set_index("horizon")["reliability"].round(4),
            "cat_res": g[g.model == "catboost"].set_index("horizon")["resolution"].round(4),
            "clim_bias [95%]": g[g.model == "climatology"].set_index("horizon").apply(fmt, axis=1),
            "clim_slope": g[g.model == "climatology"].set_index("horizon")["slope"].round(2),
        })
        print("  " + t.to_string().replace("\n", "\n  "))
    print(f"\nsaved {REPORTS / 'calibration_validation.csv'}, {REPORTS / 'calibration_bins_validation.csv'}, {fig}")


if __name__ == "__main__":
    main()
