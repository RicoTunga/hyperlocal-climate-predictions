"""Reference forecasts that every model must beat: persistence and day-of-year climatology.

Run from the repo root (venv active):

    python -m src.models.baselines                 # all sites, validation year
    python -m src.models.baselines --site nakuru

Baselines
- Persistence: the forecast for day t + h is the value observed on the issue day t.
  For rain occurrence this is a yes/no forecast (probability 1 if day t was wet, else 0).
- Climatology: for each pixel and calendar day, the average over the TRAINING years of all
  days within +/- window_days (wrapping round the new year). Gives a wet-day probability,
  exceedance probabilities, mean rainfall, mean wet-day amount and mean humidity.
  Climatology does not depend on the horizon.

Output: a printed report and reports/baselines_validation.csv (one row per site, horizon,
target, baseline and metric). The final test period is never loaded.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.models.evaluation import (binary_scores, continuous_scores, doy365, eval_target_dates,
                                   horizon_frame, load_configs, load_site, split_bounds)

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"
COLUMNS = ["pixel_id", "date", "precip_mm", "rh_pct"]


def fit_climatology(train: pd.DataFrame, wet_mm: float, exceed_mm: list, window: int) -> pd.DataFrame:
    """Per-pixel, per-calendar-day climatology from the rows passed in (training rows only).

    Sums and counts are pooled over the +/- window, then divided, so every day in the
    window counts equally. Returns one row per pixel_id x doy (1..365)."""
    d = train[["pixel_id", "date", "precip_mm", "rh_pct"]].copy()
    d["doy"] = doy365(d["date"])
    p = d["precip_mm"].astype(float)
    d["n"] = 1.0
    d["wet"] = (p >= wet_mm).astype(float)
    d["wet_amount"] = p.where(p >= wet_mm, 0.0)
    for t in exceed_mm:
        d[f"ge{t}"] = (p >= t).astype(float)
    sum_cols = ["n", "precip_mm", "rh_pct", "wet", "wet_amount"] + [f"ge{t}" for t in exceed_mm]

    pixels = sorted(d["pixel_id"].unique())
    full = pd.MultiIndex.from_product([pixels, range(1, 366)], names=["pixel_id", "doy"])
    g = d.groupby(["pixel_id", "doy"])[sum_cols].sum().reindex(full, fill_value=0.0)
    arr = g.to_numpy(dtype=float).reshape(len(pixels), 365, len(sum_cols))
    # circular moving sum over the day-of-year axis: day 1's window includes days 351..365
    win = sum(np.roll(arr, k, axis=1) for k in range(-window, window + 1))
    S = {c: win[:, :, i].ravel() for i, c in enumerate(sum_cols)}

    with np.errstate(invalid="ignore", divide="ignore"):
        clim = pd.DataFrame({
            "clim_precip_mean": S["precip_mm"] / S["n"],
            "clim_rh_mean": S["rh_pct"] / S["n"],
            "clim_p_wet": S["wet"] / S["n"],
            "clim_wet_amount": S["wet_amount"] / S["wet"],  # NaN if no wet day in the window
            **{f"clim_p_ge{t}": S[f"ge{t}"] / S["n"] for t in exceed_mm},
        }, index=full).reset_index()
    if (S["n"] == 0).any():
        raise ValueError("Some pixel/day-of-year windows have no training data.")
    # a window with no wet day at all: fall back to that pixel's overall wet-day mean
    pixel_wet = d.loc[d["wet"] == 1].groupby("pixel_id")["precip_mm"].mean()
    clim["clim_wet_amount"] = clim["clim_wet_amount"].fillna(clim["pixel_id"].map(pixel_wet))
    return clim


def add_climatology(frame: pd.DataFrame, clim: pd.DataFrame) -> pd.DataFrame:
    f = frame.assign(doy=doy365(frame["target_date"]))
    out = f.merge(clim, on=["pixel_id", "doy"], how="left", validate="many_to_one")
    assert len(out) == len(frame) and not out.filter(like="clim_").isna().any().any()
    return out


def score_horizon(df, clim, h, targets, wet_mm, exceed_mm) -> list[dict]:
    rows = []

    def add(target, baseline, scores):
        rows.extend({"horizon": h, "target": target, "baseline": baseline, "metric": k, "value": v}
                    for k, v in scores.items())

    # ---- rainfall
    fr = add_climatology(horizon_frame(df, "precip_mm", h, targets), clim)
    for thr, col in [(wet_mm, "clim_p_wet")] + [(t, f"clim_p_ge{t}") for t in exceed_mm]:
        y = (fr["y"] >= thr).astype(float)
        persist = (fr["issue_value"] >= thr).astype(float)
        name = f"rain>={thr:g}mm"
        add(name, "climatology", binary_scores(y, fr[col], fr[col]))
        add(name, "persistence", binary_scores(y, persist, fr[col]))
    # amount on all days (what a single direct model predicts)
    add("rain_mm_all", "climatology", continuous_scores(fr["y"], fr["clim_precip_mean"], fr["clim_precip_mean"]))
    add("rain_mm_all", "persistence", continuous_scores(fr["y"], fr["issue_value"], fr["clim_precip_mean"]))
    # amount on days that turned out wet (what stage 2 of the two-stage model predicts)
    w = fr[fr["y"] >= wet_mm]
    add("rain_mm_wet", "climatology", continuous_scores(w["y"], w["clim_wet_amount"], w["clim_wet_amount"]))
    add("rain_mm_wet", "persistence", continuous_scores(w["y"], w["issue_value"], w["clim_wet_amount"]))

    # ---- humidity
    fr = add_climatology(horizon_frame(df, "rh_pct", h, targets), clim)
    add("rh_pct", "climatology", continuous_scores(fr["y"], fr["clim_rh_mean"], fr["clim_rh_mean"]))
    add("rh_pct", "persistence", continuous_scores(fr["y"], fr["issue_value"], fr["clim_rh_mean"]))
    return rows


# metrics shown in the printed report (all are saved to the CSV)
SHOW = {"rain>=": ["brier", "bss", "auc"], "rain_mm": ["mae", "rmse", "rmse_skill"],
        "rh_pct": ["mae", "rmse", "r2", "rmse_skill"]}


def print_report(res: pd.DataFrame) -> None:
    for target, t in res.groupby("target", sort=False):
        metrics = next(v for k, v in SHOW.items() if target.startswith(k))
        t = t[t["metric"].isin(metrics)]
        tab = t.pivot_table(index="horizon", columns=["baseline", "metric"], values="value")
        tab = tab.reindex(columns=[(b, m) for b in ["persistence", "climatology"] for m in metrics])
        tab.columns = [f"{'pers' if b == 'persistence' else 'clim'}_{m}" for b, m in tab.columns]
        if "rh" in target or "mm_" in target:
            tab = tab.drop(columns="clim_rmse_skill")  # always 0 by definition
        if target.startswith("rain>="):
            tab = tab.drop(columns="clim_bss")
        print(f"  {target}")
        print("    " + tab.round(3).to_string().replace("\n", "\n    "))


def run_site(site: str, sites_cfg: dict, exp: dict, split: str) -> pd.DataFrame:
    train_start, train_end = split_bounds(exp, "train")
    _, split_end = split_bounds(exp, split)
    horizons = sites_cfg["forecast"]["horizons"]
    wet_mm, exceed_mm = exp["rain"]["wet_threshold_mm"], exp["rain"]["exceedance_mm"]

    df = load_site(site, until=split_end, columns=COLUMNS)
    train = df[(df["date"] >= train_start) & (df["date"] <= train_end)]
    clim = fit_climatology(train, wet_mm, exceed_mm, exp["climatology"]["window_days"])
    targets = eval_target_dates(exp, split, max(horizons))
    tgt = df[df["date"].isin(targets)]

    print(f"\n=== {site} | {split} ===")
    print(f"  data loaded up to {df['date'].max():%Y-%m-%d} (nothing later) | "
          f"{df['pixel_id'].nunique()} pixels")
    print(f"  climatology fitted on {train['date'].min():%Y-%m-%d}..{train['date'].max():%Y-%m-%d} "
          f"({len(train):,} pixel-days), window +/-{exp['climatology']['window_days']} days")
    print(f"  target days {targets[0]:%Y-%m-%d}..{targets[-1]:%Y-%m-%d} ({len(targets)} days, "
          f"{len(tgt):,} pixel-days per horizon); first issue day "
          f"{targets[0] - pd.Timedelta(days=max(horizons)):%Y-%m-%d}")
    print(f"  wet-day share (>= {wet_mm:g} mm): train {(train['precip_mm'] >= wet_mm).mean():.3f}, "
          f"{split} targets {(tgt['precip_mm'] >= wet_mm).mean():.3f} | mean humidity: train "
          f"{train['rh_pct'].mean():.1f}%, {split} {tgt['rh_pct'].mean():.1f}%")

    rows = []
    for h in horizons:
        rows += score_horizon(df, clim, h, targets, wet_mm, exceed_mm)
    res = pd.DataFrame(rows)
    res.insert(0, "split", split)
    res.insert(0, "site", site)
    print_report(res)
    return res


def main() -> None:
    parser = argparse.ArgumentParser(description="Score persistence and climatology baselines.")
    parser.add_argument("--site", help="site key from sites.yaml (default: all sites)")
    args = parser.parse_args()
    split = "validation"  # the final test period is scored only at the very end of the project

    sites_cfg, exp = load_configs()
    sites = sites_cfg["sites"]
    if args.site:
        if args.site not in sites:
            raise SystemExit(f"Unknown site '{args.site}'. Available: {', '.join(sites)}")
        sites = {args.site: sites[args.site]}

    results = pd.concat([run_site(k, sites_cfg, exp, split) for k in sites], ignore_index=True)
    REPORTS.mkdir(exist_ok=True)
    out = REPORTS / f"baselines_{split}.csv"
    results.to_csv(out, index=False)
    print(f"\nsaved {out} ({len(results)} rows)")


if __name__ == "__main__":
    main()
