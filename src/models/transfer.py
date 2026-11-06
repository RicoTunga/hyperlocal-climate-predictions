"""Transfer between sites: apply one site's models to the other site (Busia -> Nakuru and back).

Run from the repo root after the decided training runs (train60, scored on all pixels):

    python -m src.models.transfer

Design (docs/PROJECT_CONTEXT.md section 6, "train on Busia, test on Nakuru")
- No retraining: the saved decided-setup CatBoost models of the SOURCE site (60 training pixels,
  all six tasks, every horizon) predict the TARGET site's validation year 2024 for every pixel.
- Inputs come from the target site itself: its features, its per-pixel climatology (fitted on
  its own training years) and its own area-wide and neighbourhood context. Only the learned
  model is borrowed.
- The wet-day amount is back-transformed with the source model's smearing factor, recomputed on
  the source's fitting rows exactly as src.models.train does. Guard: the source models must
  reproduce the source site's own saved 2024 forecasts before anything is transferred.
- Compared, paired on identical rows, with the target site's own model (moving-block bootstrap).
- Diagnostic: share of target rows outside the range of each key input seen in the source's
  fitting data (trees cannot extrapolate beyond that range).

Output: reports/transfer_skill_validation.csv, reports/transfer_differences_validation.csv,
reports/transfer_ranges_validation.csv. Columns source_site / target_site name the sites;
`target` is the forecast target (e.g. rain>=1mm), as in the other result tables.
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.models.baselines import fit_climatology
from src.models.calibration import decided_run
from src.models.evaluation import eval_target_dates, load_configs, split_bounds
from src.models.train import dev_pixel_sample, feature_columns, horizon_table, load_features, predict
from src.models.uncertainty import (block_bootstrap_weights, bootstrap_site_horizon, combine_runs,
                                    paired_differences)

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"
ALGO = "catboost"
RANGE_FEATURES = ["elev_mean_m", "sp_hpa", "t2m_c", "rh_pct", "tcc", "region_precip_mean",
                  "region_wet_frac", "clim_p_wet", "clim_rh_mean", "clim_wet_amount"]
SHOW = ["rain>=1mm", "rain>=10mm", "rain_mm_wet", "rh_pct"]


def site_inputs(site: str, exp: dict, n_train: int) -> dict:
    """Decided-run files, validation inputs for all pixels, and the training pixels' fitting inputs."""
    mcfg, wet_mm, exceed = exp["modelling"], exp["rain"]["wet_threshold_mm"], exp["rain"]["exceedance_mm"]
    window = exp["climatology"]["window_days"]
    train_start, train_end = split_bounds(exp, "train")
    val_start, val_end = split_bounds(exp, "validation")
    runs = dict(decided_run(site, n_train))
    n_score = int(next(iter(runs.values())).parent.parent.name.split("_score")[1])
    pixels = dev_pixel_sample(site, n_score)  # every scored pixel, for predicting this site
    hist = load_features(site, pixels, until=train_end, start=train_start,
                         columns=["pixel_id", "date", "precip_mm", "rh_pct"])
    clim_all = fit_climatology(hist, wet_mm, exceed, window)
    del hist
    feat_val = load_features(site, pixels, until=val_end, start=val_start)
    train_px = dev_pixel_sample(site, n_train)  # the training pixels: smearing factor and input ranges
    feat_tr = load_features(site, train_px, until=val_end)
    clim_tr = fit_climatology(feat_tr[(feat_tr["date"] >= train_start) & (feat_tr["date"] <= train_end)],
                              wet_mm, exceed, window)
    es_start = pd.Timestamp(str(mcfg["early_stop_start"]))
    return {"runs": runs, "feat_val": feat_val, "clim_all": clim_all, "feat_tr": feat_tr, "clim_tr": clim_tr,
            "fit_end": es_start - pd.Timedelta(days=1), "train_start": train_start}


def fit_rows(src: dict, h: int) -> pd.DataFrame:
    """The source models' fitting table for horizon h (the same rows as in src.models.train)."""
    t = pd.date_range(src["train_start"] + pd.Timedelta(days=h), src["fit_end"])
    return horizon_table(src["feat_tr"], src["clim_tr"], h, t)


def predict_all(models: dict, smear: float, table: pd.DataFrame, cols: list, wet_mm: float) -> dict:
    """Predictions of every task on `table`, post-processed exactly as in train.fit_predict."""
    out = {}
    for task, model in models.items():
        pred = predict(model, table[cols], task.startswith("occ"))
        if task == "wet":
            pred = np.clip(np.exp(pred) * smear - 1, wet_mm, None)
        elif task == "direct":
            pred = np.clip(pred, 0, None)
        out[task] = pred
    out["two_stage"] = out[f"occ{wet_mm:g}"] * out["wet"]
    return out


def fmt(r) -> str:
    star = "*" if (r["ci_low"] > 0 or r["ci_high"] < 0) else " "
    return f"{r['estimate']:+.3f} [{r['ci_low']:+.3f},{r['ci_high']:+.3f}]{star}"


def table_of(df: pd.DataFrame) -> str:
    x = df[df["target"].isin(SHOW)]
    x = x.assign(v=x.apply(fmt, axis=1)).pivot(index="horizon", columns="target", values="v")[SHOW]
    return "    " + x.to_string().replace("\n", "\n    ")


def main() -> None:
    sites_cfg, exp = load_configs()
    mcfg, ucfg = exp["modelling"], exp["uncertainty"]
    wet_mm, exceed = exp["rain"]["wet_threshold_mm"], exp["rain"]["exceedance_mm"]
    horizons = sites_cfg["forecast"]["horizons"]
    task_names = [f"occ{t:g}" for t in [wet_mm] + exceed] + ["wet", "direct", "rh"]
    sites = list(sites_cfg["sites"])
    inputs = {s: site_inputs(s, exp, mcfg["dev_pixels"]) for s in sites}
    val_targets = eval_target_dates(exp, "validation", max(horizons))

    skill_rows, diff_rows, range_rows = [], [], []
    for h in horizons:
        tables = {s: horizon_table(inputs[s]["feat_val"], inputs[s]["clim_all"], h, val_targets) for s in sites}
        cols = feature_columns(tables[sites[0]])
        assert all(feature_columns(t) == cols for t in tables.values()), "sites have different inputs"
        models, smear = {}, {}
        for s in sites:
            d = inputs[s]["runs"][h].parent
            models[s] = {t: joblib.load(d / f"{ALGO}_{t}.joblib") for t in task_names}
            assert all(list(m.feature_names_) == cols for m in models[s].values())
            fr = fit_rows(inputs[s], h)
            wet_fit = fr[fr["y_rain"] >= wet_mm]
            smear[s] = float(np.mean(np.exp(np.log1p(wet_fit["y_rain"])
                                            - predict(models[s]["wet"], wet_fit[cols], False))))
            # input ranges the source models saw while fitting, and how much of the other site falls outside
            lo, hi = fr[RANGE_FEATURES].min(), fr[RANGE_FEATURES].max()
            for tgt in [x for x in sites if x != s]:
                x = tables[tgt][RANGE_FEATURES]
                for c in RANGE_FEATURES:
                    range_rows.append({"source_site": s, "target_site": tgt, "horizon": h, "feature": c,
                                       "source_min": lo[c], "source_max": hi[c],
                                       "target_min": x[c].min(), "target_max": x[c].max(),
                                       "share_outside": float(((x[c] < lo[c]) | (x[c] > hi[c])).mean())})
            # guard: the source models (with the recomputed smearing factor) reproduce the saved forecasts
            own = pd.read_parquet(inputs[s]["runs"][h])
            p = predict_all(models[s], smear[s], tables[s], cols, wet_mm)
            re = tables[s][["pixel_id", "target_date"]].assign(**{f"re__{k}": v for k, v in p.items()})
            chk = own.merge(re, on=["pixel_id", "target_date"], validate="one_to_one")
            err = max(float((chk[f"re__{k}"] - chk[f"{ALGO}__{k}"]).abs().max()) for k in p)
            if len(chk) != len(own) or err > 1e-6:
                raise SystemExit(f"{s} h{h}: models do not reproduce the saved forecasts (max diff {err:.2g})")
            print(f"  PASS  {s} h{h}: models reproduce the saved 2024 forecasts ({len(chk):,} rows, "
                  f"max difference {err:.1g}); smearing factor {smear[s]:.3f}")

        for src in sites:
            for tgt in [x for x in sites if x != src]:
                own = pd.read_parquet(inputs[tgt]["runs"][h])
                p = predict_all(models[src], smear[src], tables[tgt], cols, wet_mm)
                tr = tables[tgt][["pixel_id", "target_date"]].assign(**{f"{ALGO}__{k}": v for k, v in p.items()})
                base = own[["pixel_id", "target_date", "y_rain", "y_rh"]
                           + [c for c in own.columns if c.split("__")[0] in ("persistence", "climatology")]]
                tr = base.merge(tr, on=["pixel_id", "target_date"], validate="one_to_one")
                pr = combine_runs({"transfer": tr, "own": own}, keep_models=[ALGO])
                W = block_bootstrap_weights(pr["target_date"].nunique(), ucfg["block_days"], ucfg["n_boot"], ucfg["seed"])
                rows, draws = bootstrap_site_horizon(pr, W, wet_mm, exceed)
                key = {"source_site": src, "target_site": tgt, "horizon": h}
                skill_rows += [{**key, **r} for r in rows if r["reference"] == "climatology"]
                for d in paired_differences(draws, pd.DataFrame(rows)):
                    if "two-stage" in d["comparison"]:
                        continue
                    if d["comparison"].startswith(f"{ALGO}@own"):  # report transfer minus own
                        d = {**d, "comparison": f"{ALGO}@transfer - {ALGO}@own", "estimate": -d["estimate"],
                             "ci_low": -d["ci_high"], "ci_high": -d["ci_low"]}
                    diff_rows.append({**key, **d})
        del tables

    skill, diff, ranges = pd.DataFrame(skill_rows), pd.DataFrame(diff_rows), pd.DataFrame(range_rows)
    skill.to_csv(REPORTS / "transfer_skill_validation.csv", index=False)
    diff.to_csv(REPORTS / "transfer_differences_validation.csv", index=False)
    ranges.to_csv(REPORTS / "transfer_ranges_validation.csv", index=False)

    for (src, tgt), s in skill.groupby(["source_site", "target_site"]):
        print(f"\n{src} models -> {tgt}: skill vs climatology")
        print(f"  {src}'s model applied to {tgt}:\n" + table_of(s[s["model"] == f"{ALGO}@transfer"]))
        print(f"  {tgt}'s own model:\n" + table_of(s[s["model"] == f"{ALGO}@own"]))
        d = diff[(diff["source_site"] == src) & (diff["target_site"] == tgt)]
        print("  transfer minus own:\n" + table_of(d))
        r = ranges[(ranges["source_site"] == src) & (ranges["target_site"] == tgt) & (ranges["horizon"] == horizons[0])]
        print(f"  {tgt} rows outside the range {src}'s model saw in training: "
              + ", ".join(f"{x.feature} {x.share_outside:.0%}" for x in r.itertuples()))
    print(f"\nsaved {REPORTS / 'transfer_skill_validation.csv'}, {REPORTS / 'transfer_differences_validation.csv'}, "
          f"{REPORTS / 'transfer_ranges_validation.csv'}")


if __name__ == "__main__":
    main()
