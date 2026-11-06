"""Potato late-blight risk forecast for Nakuru (optional extra), developed on the validation year.

Run from the repo root after src.features.humid_hours --site nakuru:

    python -m src.models.late_blight validate   # develop on 2024
    python -m src.models.late_blight test       # POST-HOC check on 2025-01-08..2026-09-30 (final-test config)

Risk rules (decided November 2026; see docs/PROJECT_CONTEXT.md)
- hutton (main): the Hutton criteria (James Hutton Institute / AHDB, Great Britain's national
  warning system since 2017): met on day d if days d-1 and d each have minimum temperature >= 10 C
  and at least 6 hours at relative humidity >= 90%. Minimum temperature is ERA5's, corrected to the
  pixel's real elevation: ERA5's terrain height is estimated from its mean surface pressure over
  the training years (standard atmosphere) and a lapse rate of 6.5 C/km is applied.
- humid (sensitivity): the humidity part only (>= 6 h at RH >= 90% on days d-1 and d), because the
  10 C threshold, calibrated in Britain, is rarely reached above 2,600 m at Nakuru.

Forecast: probability that the rule is met on day t + h (h = 1..7), issued on day t. CatBoost
(default settings) with the standard features plus late-blight inputs on day t (humid hours,
corrected minimum temperature, whether the rule is met) and the per-pixel climatology of the rule
for the target day (training years, +/- window days). Trained on the 60 development pixels (fit
2016-2022, early stopping 2023) and scored on every pixel for 2024, next to climatology and
persistence (rule met on day t), with moving-block bootstrap intervals. Models are trained first;
every pixel is then scored in batches of SCORE_BATCH pixels to stay within the laptop's memory.

Output: models/nakuru/late_blight/<rule>/h<h>/, reports/late_blight/validation_skill.csv,
reports/late_blight/validation_by_elevation.csv, reports/late_blight/base_rates.csv.
"""
import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.models.baselines import fit_climatology
from src.models.calibration import calibration_slope
from src.models.evaluation import PROCESSED, doy365, eval_target_dates, load_configs, split_bounds
from src.models.train import (MODELS, dev_pixel_sample, feature_columns, fit, horizon_table, load_features,
                              make_model, predict)
from src.models.uncertainty import block_bootstrap_weights, daily_sums

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports" / "late_blight"
SITE = "nakuru"
RULES = ["hutton", "humid"]
SCORE_BATCH = 200  # scored pixels per batch (memory: the laptop has about 8 GB)
LAPSE_C_PER_M = 0.0065
TMIN_C, HOURS = 10.0, 6.0


# ---------------------------------------------------------------- labels
def era5_terrain_height(sp_hpa: pd.Series) -> pd.Series:
    """Height (m) of ERA5's model surface from its surface pressure, standard atmosphere."""
    return 44330.8 * (1 - (sp_hpa / 1013.25) ** 0.190263)


def risk_labels(daily: pd.DataFrame, hours: pd.DataFrame, train_start, train_end) -> pd.DataFrame:
    """Per pixel and day: corrected minimum temperature, qualifying days and rules met.

    `daily` needs pixel_id, date, t2m_min_c, sp_hpa, elev_mean_m; `hours` pixel_id, date, rh_hours_ge90.
    ERA5 terrain height uses the mean pressure of the training years only (a fixed per-pixel value)."""
    d = daily.merge(hours, on=["pixel_id", "date"], validate="one_to_one").sort_values(["pixel_id", "date"])
    tr = d[(d["date"] >= train_start) & (d["date"] <= train_end)]
    z_era5 = era5_terrain_height(tr.groupby("pixel_id")["sp_hpa"].mean())
    dz = d["elev_mean_m"] - d["pixel_id"].map(z_era5)
    d["tmin_corr_c"] = (d["t2m_min_c"] - LAPSE_C_PER_M * dz).astype("float32")
    d["qual_humid"] = d["rh_hours_ge90"] >= HOURS
    d["qual_hutton"] = d["qual_humid"] & (d["tmin_corr_c"] >= TMIN_C)
    prev_date = d.groupby("pixel_id")["date"].shift(1)
    consecutive = (d["date"] - prev_date).dt.days == 1  # previous row is the previous day of the same pixel
    for r in RULES:
        d[f"met_{r}"] = d[f"qual_{r}"] & d.groupby("pixel_id")[f"qual_{r}"].shift(1, fill_value=False) & consecutive
    return d[["pixel_id", "date", "rh_hours_ge90", "tmin_corr_c", "qual_humid", "qual_hutton",
              "met_humid", "met_hutton"]].reset_index(drop=True)


def binary_climatology(labels: pd.DataFrame, col: str, window: int) -> pd.DataFrame:
    """Per pixel and calendar day: share of days with `col` True within +/- window days (circular)."""
    d = labels[["pixel_id", "date", col]].copy()
    d["doy"] = doy365(d["date"])
    pixels = sorted(d["pixel_id"].unique())
    full = pd.MultiIndex.from_product([pixels, range(1, 366)], names=["pixel_id", "doy"])
    g = d.groupby(["pixel_id", "doy"])[col].agg(["sum", "count"]).reindex(full, fill_value=0)
    s = g["sum"].to_numpy(float).reshape(len(pixels), 365)
    n = g["count"].to_numpy(float).reshape(len(pixels), 365)
    ws = sum(np.roll(s, k, axis=1) for k in range(-window, window + 1))
    wn = sum(np.roll(n, k, axis=1) for k in range(-window, window + 1))
    if (wn == 0).any():
        raise ValueError("Some pixel/day-of-year windows have no data.")
    return pd.DataFrame({"clim_p_risk": (ws / wn).ravel()}, index=full).reset_index()


# ---------------------------------------------------------------- tables
EXTRA = ["rh_hours_ge90", "tmin_corr_c", "qual_humid", "qual_hutton", "met_humid", "met_hutton"]


def risk_table(feat, clim_rain, labels, clim_risk, rule, h, targets) -> pd.DataFrame:
    """Standard horizon table + late-blight inputs on the issue day + rule and its climatology on t + h."""
    t = horizon_table(feat, clim_rain, h, targets)
    n = len(t)
    # joins (not tuple lookups) keep memory low; a left join keeps the row order of t
    issue = labels[["pixel_id", "date"] + EXTRA].rename(columns={"date": "issue_date", **{c: f"lb_{c}" for c in EXTRA}})
    t = t.merge(issue, on=["pixel_id", "issue_date"], how="left", validate="one_to_one")
    t[[f"lb_{c}" for c in EXTRA]] = t[[f"lb_{c}" for c in EXTRA]].astype("float32")
    tgt = labels[["pixel_id", "date", f"met_{rule}"]].rename(columns={"date": "target_date", f"met_{rule}": "y_risk"})
    t = t.merge(tgt, on=["pixel_id", "target_date"], how="left", validate="one_to_one")
    t["y_risk"] = t["y_risk"].astype(float)
    assert len(t) == n
    t["doy"] = doy365(t["target_date"])
    t = t.merge(clim_risk, on=["pixel_id", "doy"], how="left", validate="many_to_one").drop(columns="doy")
    assert t["clim_p_risk"].notna().all() and t["y_risk"].notna().all()
    t["persistence"] = t[f"lb_met_{rule}"]
    return t


def risk_columns(t: pd.DataFrame) -> list:
    return [c for c in feature_columns(t) if c not in ("y_risk", "persistence")]


# ---------------------------------------------------------------- scoring
def binary_skill(t: pd.DataFrame, p: np.ndarray, W: np.ndarray, day: np.ndarray, n_days: int) -> dict:
    y = t["y_risk"].to_numpy(float)
    ones = np.ones(n_days)
    se = {k: daily_sums(day, n_days, (v - y) ** 2) for k, v in
          {"model": p, "climatology": t["clim_p_risk"].to_numpy(), "persistence": t["persistence"].to_numpy()}.items()}
    out = {"brier": float(np.mean((p - y) ** 2)), "base_rate": float(y.mean()), "mean_forecast": float(p.mean()),
           "auc": float(roc_auc_score(y, p)) if 0 < y.mean() < 1 else np.nan}
    for ref in ("climatology", "persistence"):
        est = 1 - (ones @ se["model"]) / (ones @ se[ref])
        draws = 1 - (W @ se["model"]) / (W @ se[ref])
        lo, hi = np.percentile(draws, [2.5, 97.5])
        out.update({f"bss_vs_{ref}": float(est), f"bss_vs_{ref}_low": float(lo), f"bss_vs_{ref}_high": float(hi)})
    slope, intercept = calibration_slope(y, p)
    out.update({"slope": slope, "intercept": intercept, "bias": float(p.mean() - y.mean())})
    return out


def run(sites_cfg: dict, exp: dict, step: str) -> None:
    """step 'validate': train 2016-2023, score 2024. step 'test' (POST-HOC): the final-test config
    (train to 2024-12-31, early stopping on 2024; src.models.final_test.final_config), score the test."""
    split = "validation" if step == "validate" else "test"
    if step == "test":
        from src.models.final_test import final_config
        exp = final_config(exp)
    model_dir = "late_blight" if step == "validate" else "late_blight_final"
    pre = "val" if step == "validate" else "test"
    report = "validation" if step == "validate" else "test"
    mcfg, ucfg = exp["modelling"], exp["uncertainty"]
    wet_mm, exceed, window = exp["rain"]["wet_threshold_mm"], exp["rain"]["exceedance_mm"], exp["climatology"]["window_days"]
    train_start, train_end = split_bounds(exp, "train")
    score_start, score_end = split_bounds(exp, split)
    es_start = pd.Timestamp(str(mcfg["early_stop_start"]))
    horizons = sites_cfg["forecast"]["horizons"]

    daily = pd.read_parquet(PROCESSED / f"{SITE}_daily.parquet",
                            columns=["pixel_id", "date", "t2m_min_c", "sp_hpa", "elev_mean_m"])
    daily = daily[daily["date"] <= score_end]
    hours = pd.read_parquet(PROCESSED / f"{SITE}_humid_hours.parquet")
    labels = risk_labels(daily, hours[hours["date"] <= score_end], train_start, train_end)
    elev = daily.drop_duplicates("pixel_id").set_index("pixel_id")["elev_mean_m"]
    del daily, hours
    tr_lab = labels[(labels["date"] >= train_start) & (labels["date"] <= train_end)]
    clim_risk = {r: binary_climatology(tr_lab, f"met_{r}", window) for r in RULES}
    base = pd.DataFrame({r: tr_lab.assign(band=pd.cut(tr_lab["pixel_id"].map(elev), [0, 1800, 2200, 2600, 4000]))
                        .groupby("band", observed=True)[f"met_{r}"].mean() for r in RULES})
    OUT.mkdir(parents=True, exist_ok=True)
    base.to_csv(OUT / ("base_rates.csv" if step == "validate" else "base_rates_final.csv"))
    print(f"{step}: base rates {train_start.year}-{train_end.year} by elevation band (share of days the rule is met):")
    print("  " + (100 * base).round(1).to_string().replace("\n", "\n  "))

    train_px = dev_pixel_sample(SITE, mcfg["dev_pixels"])
    all_px = dev_pixel_sample(SITE, 10 ** 9)
    targets = eval_target_dates(exp, split, max(horizons))
    fit_t = lambda h: pd.date_range(train_start + pd.Timedelta(days=h), es_start - pd.Timedelta(days=1))  # noqa: E731
    es_t = pd.date_range(es_start + pd.Timedelta(days=exp["gap_days"]), train_end)

    # 1. train every model on the development pixels only (small tables), save, free
    feat_tr = load_features(SITE, train_px, until=train_end)  # nothing after training
    clim_tr = fit_climatology(feat_tr[(feat_tr["date"] >= train_start) & (feat_tr["date"] <= train_end)],
                              wet_mm, exceed, window)
    lab_tr = labels[labels["pixel_id"].isin(train_px)]
    n_trees, cols = {}, None
    for rule in RULES:
        for h in horizons:
            tr = risk_table(feat_tr, clim_tr, lab_tr, clim_risk[rule], rule, h, fit_t(h))
            es = risk_table(feat_tr, clim_tr, lab_tr, clim_risk[rule], rule, h, es_t)
            cols = risk_columns(tr)
            model = make_model("catboost", True, mcfg["params"], mcfg["seed"], mcfg["early_stopping_rounds"])
            model, n_trees[(rule, h)] = fit(model, "catboost", tr[cols], tr["y_risk"].astype(int), es[cols],
                                            es["y_risk"].astype(int), mcfg["early_stopping_rounds"])
            d = MODELS / SITE / model_dir / rule / f"h{h}"
            d.mkdir(parents=True, exist_ok=True)
            joblib.dump(model, d / "catboost_risk.joblib")
            print(f"  trained {rule} h{h}: {n_trees[(rule, h)]} trees")
            del tr, es, model
    del feat_tr, clim_tr, lab_tr

    # 2. score every pixel of 2024 in batches (memory), saving small per-batch prediction files
    batches = [all_px[i:i + SCORE_BATCH] for i in range(0, len(all_px), SCORE_BATCH)]
    models = {(r, h): joblib.load(MODELS / SITE / model_dir / r / f"h{h}" / "catboost_risk.joblib")
              for r in RULES for h in horizons}
    for k, px in enumerate(batches):
        hist = load_features(SITE, px, until=train_end, start=train_start, columns=["pixel_id", "date", "precip_mm", "rh_pct"])
        clim_b = fit_climatology(hist, wet_mm, exceed, window)
        del hist
        feat_b = load_features(SITE, px, until=score_end, start=score_start)
        lab_b = labels[labels["pixel_id"].isin(px)]
        for rule in RULES:
            for h in horizons:
                va = risk_table(feat_b, clim_b, lab_b, clim_risk[rule], rule, h, targets)
                m = models[(rule, h)]
                assert list(m.feature_names_) == risk_columns(va)
                p = predict(m, va[risk_columns(va)], True)
                d = MODELS / SITE / model_dir / rule / f"h{h}"
                va[["pixel_id", "target_date", "y_risk", "clim_p_risk", "persistence"]].assign(catboost=p).to_parquet(
                    d / f"{pre}_predictions_part{k}.parquet", index=False)
                del va
        del feat_b, clim_b, lab_b
        print(f"  scored batch {k + 1} of {len(batches)} ({len(px)} pixels)")

    # 3. combine the parts and score
    rows, by_band = [], []
    for rule in RULES:
        for h in horizons:
            d = MODELS / SITE / model_dir / rule / f"h{h}"
            parts = sorted(d.glob(f"{pre}_predictions_part*.parquet"))
            va = pd.concat([pd.read_parquet(f) for f in parts], ignore_index=True)
            assert va["pixel_id"].nunique() == len(all_px) and not va.duplicated(["pixel_id", "target_date"]).any()
            va.to_parquet(d / f"{pre}_predictions.parquet", index=False)
            for f in parts:
                f.unlink()
            p, n_iter = va["catboost"].to_numpy(), n_trees[(rule, h)]
            day, uniq = pd.factorize(pd.DatetimeIndex(va["target_date"]), sort=True)
            W = block_bootstrap_weights(len(uniq), ucfg["block_days"], ucfg["n_boot"], ucfg["seed"])
            s = binary_skill(va, p, W, day, len(uniq))
            rows.append({"rule": rule, "horizon": h, "n_trees": n_iter, "n_rows": len(va), **s})
            band = pd.cut(va["pixel_id"].map(elev), [0, 1800, 2200, 2600, 4000])
            for b, g in va.assign(p=p, band=band).groupby("band", observed=True):
                y = g["y_risk"].to_numpy()
                bs, bc = np.mean((g["p"] - y) ** 2), np.mean((g["clim_p_risk"] - y) ** 2)
                by_band.append({"rule": rule, "horizon": h, "band": str(b), "base_rate": y.mean(),
                                "bss_vs_climatology": 1 - bs / bc if bc > 0 else np.nan})
            print(f"  {rule:6} h{h}: {n_iter} trees | base rate {s['base_rate']:.3f} | "
                  f"BSS vs clim {s['bss_vs_climatology']:+.3f} [{s['bss_vs_climatology_low']:+.3f},{s['bss_vs_climatology_high']:+.3f}] | "
                  f"vs persistence {s['bss_vs_persistence']:+.3f} | AUC {s['auc']:.3f} | slope {s['slope']:.2f} | bias {s['bias']:+.3f}")
            del va
    pd.DataFrame(rows).to_csv(OUT / f"{report}_skill.csv", index=False)
    bb = pd.DataFrame(by_band)
    bb.to_csv(OUT / f"{report}_by_elevation.csv", index=False)
    print("\nBSS vs climatology by elevation band (point estimates; base rate in 2024 in brackets):")
    for rule, g in bb.groupby("rule"):
        t = g.assign(v=g.apply(lambda r: f"{r['bss_vs_climatology']:+.3f} ({r['base_rate']:.2f})", axis=1)).pivot(
            index="horizon", columns="band", values="v")
        print(f"  {rule}:\n    " + t.to_string().replace("\n", "\n    "))
    print(f"\nsaved {OUT / (report + '_skill.csv')}, {OUT / (report + '_by_elevation.csv')}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Potato late-blight risk forecast (Nakuru).")
    parser.add_argument("step", choices=["validate", "test"])
    args = parser.parse_args()
    sites_cfg, exp = load_configs()
    run(sites_cfg, exp, args.step)


if __name__ == "__main__":
    main()
