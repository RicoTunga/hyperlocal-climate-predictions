"""Per-forecast SHAP explanations for the dashboard ("Why this forecast?"), in plain-language groups.

Run from the repo root after the final test, the late-blight test and `python -m app.build_data`:

    python -m app.build_shap

For the final test-period CatBoost models (rain >= 1 mm, humidity, and Nakuru late blight with both
rules), every square x issue day x horizon of the test period is explained with exact TreeSHAP:
- the model inputs are rebuilt exactly as for the final test (src.models.final_test /
  src.models.late_blight), pixels in batches to stay within the laptop's memory;
- SHAP values are summed into about 7 plain-language input groups (src.explain.shap_analysis);
- probabilities: SHAP works in log-odds, which mean nothing to most people. Contributions are
  converted to percentage points by the mean slope of the logistic curve between the typical
  forecast p0 and this forecast p: k = (p - p0) / (z - z0) (k = p0(1 - p0) when z = z0). The
  converted contributions keep their signs and add up exactly to (p - p0) x 100;
- humidity contributions are already in % and are used as they are.

Guards (the build stops if one fails): SHAP + expected value = raw model output for every row; the
rebuilt forecasts equal the saved test forecasts; contributions add up to forecast - typical.

Output (git-ignored, app/data/):
    <site>_shap.parquet   pixel_id, issue_date, horizon, then one int16 column per target and group
                          (contribution x 10, i.e. tenths of a point)
    shap_meta.json        per site, target and horizon: the typical forecast and the group order
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from app import components as C
from src.explain.shap_analysis import feature_group, shap_values
from src.models.baselines import fit_climatology
from src.models.evaluation import PROCESSED, eval_target_dates, load_configs, split_bounds
from src.models.final_test import final_config
from src.models.late_blight import binary_climatology, risk_columns, risk_labels, risk_table
from src.models.train import MODELS, dev_pixel_sample, feature_columns, horizon_table, load_features

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "app" / "data"
BATCH = 200
GROUPS, LABELS = C.SHAP_GROUPS, C.SHAP_LABELS  # plain-language groups shown in the dashboard


def group_of(col: str) -> str:
    return "blight conditions today" if col.startswith("lb_") else feature_group(col)


def to_points(shap_groups: np.ndarray, expected: float, kind: str):
    """(contributions in display units, typical forecast, forecast) for one block of rows.

    kind 'prob': log-odds -> percentage points, additive by construction; kind 'pct': unchanged."""
    z = expected + shap_groups.sum(axis=1)
    if kind == "pct":
        return shap_groups, expected, z
    p0 = 1.0 / (1.0 + np.exp(-expected))
    p = 1.0 / (1.0 + np.exp(-z))
    dz = z - expected
    with np.errstate(invalid="ignore", divide="ignore"):
        k = np.where(np.abs(dz) > 1e-9, (p - p0) / dz, p0 * (1 - p0))
    return 100 * shap_groups * k[:, None], 100 * p0, 100 * p


def grouped(shap: np.ndarray, cols: list) -> np.ndarray:
    groups = np.array([GROUPS.index(group_of(c)) for c in cols])
    out = np.zeros((shap.shape[0], len(GROUPS)))
    for g in range(len(GROUPS)):
        m = groups == g
        if m.any():
            out[:, g] = shap[:, m].sum(axis=1)
    return out


def explain(model, table: pd.DataFrame, cols: list, kind: str, saved: pd.Series):
    """Grouped contributions (rows x groups), typical forecast, forecasts; with the guards."""
    assert list(model.feature_names_) == cols, "feature columns differ from training"
    shap, expected = shap_values(model, table[cols])  # raises if SHAP does not add up
    contrib, typical, forecast = to_points(grouped(shap, cols), expected, kind)
    ref = saved.to_numpy() * (100 if kind == "prob" else 1)
    err = float(np.max(np.abs(forecast - ref)))
    if err > 1e-3:
        raise SystemExit(f"rebuilt forecasts differ from the saved ones (max {err:.3g})")
    if float(np.max(np.abs(contrib.sum(axis=1) - (forecast - typical)))) > 1e-6:
        raise SystemExit("contributions do not add up to forecast - typical")
    return contrib, float(np.atleast_1d(typical)[0]), err


def build_site(site: str, sites_cfg: dict, exp: dict) -> dict:
    fexp = final_config(exp)
    wet_mm, exceed, window = fexp["rain"]["wet_threshold_mm"], fexp["rain"]["exceedance_mm"], fexp["climatology"]["window_days"]
    train_start, train_end = split_bounds(fexp, "train")
    test_start, test_end = split_bounds(fexp, "test")
    horizons = sites_cfg["forecast"]["horizons"]
    targets_days = eval_target_dates(fexp, "test", max(horizons))
    run = next((MODELS / site).glob("final_train*_score*"))
    has_blight = site == "nakuru"
    targets = {"rain1": ("catboost_occ1.joblib", "prob"), "rh": ("catboost_rh.joblib", "pct")}
    if has_blight:
        targets.update({"blight_hutton": ("hutton", "prob"), "blight_humid": ("humid", "prob")})

    saved = {h: pd.read_parquet(run / f"h{h}" / "test_predictions.parquet",
                                columns=["pixel_id", "target_date", "catboost__occ1", "catboost__rh"]) for h in horizons}
    models = {(t, h): joblib.load(run / f"h{h}" / f) for t, (f, _) in targets.items() if not t.startswith("blight")
              for h in horizons}
    if has_blight:
        daily = pd.read_parquet(PROCESSED / f"{site}_daily.parquet", columns=["pixel_id", "date", "t2m_min_c", "sp_hpa", "elev_mean_m"])
        hours = pd.read_parquet(PROCESSED / f"{site}_humid_hours.parquet")
        daily, hours = daily[daily["date"] <= test_end], hours[hours["date"] <= test_end]
        labels = risk_labels(daily, hours, train_start, train_end)
        del daily, hours
        tr_lab = labels[(labels["date"] >= train_start) & (labels["date"] <= train_end)]
        clim_risk = {r: binary_climatology(tr_lab, f"met_{r}", window) for r in ("hutton", "humid")}
        del tr_lab
        for r in ("hutton", "humid"):
            for h in horizons:
                d = MODELS / site / "late_blight_final" / r / f"h{h}"
                models[(f"blight_{r}", h)] = joblib.load(d / "catboost_risk.joblib")
                saved[(r, h)] = pd.read_parquet(d / "test_predictions.parquet", columns=["pixel_id", "target_date", "catboost"])

    all_px = dev_pixel_sample(site, 10 ** 9)
    batches = [all_px[i:i + BATCH] for i in range(0, len(all_px), BATCH)]
    typical, parts, worst = {}, [], 0.0
    for k, px in enumerate(batches):
        hist = load_features(site, px, until=train_end, start=train_start, columns=["pixel_id", "date", "precip_mm", "rh_pct"])
        clim = fit_climatology(hist, wet_mm, exceed, window)
        del hist
        feat = load_features(site, px, until=test_end, start=test_start)
        lab_b = labels[labels["pixel_id"].isin(px)] if has_blight else None
        for h in horizons:
            t = horizon_table(feat, clim, h, targets_days)
            cols = feature_columns(t)
            block = t[["pixel_id", "issue_date"]].copy()
            block["horizon"] = np.int8(h)
            s = t[["pixel_id", "target_date"]].merge(saved[h], on=["pixel_id", "target_date"], how="left", validate="one_to_one")
            for target, col in (("rain1", "catboost__occ1"), ("rh", "catboost__rh")):
                contrib, typ, err = explain(models[(target, h)], t, cols, targets[target][1], s[col])
                typical[f"{target}|{h}"], worst = typ, max(worst, err)
                for g, name in enumerate(GROUPS):
                    block[f"{target}|{name}"] = np.round(contrib[:, g] * 10).astype(np.int16)
            if has_blight:
                for r in ("hutton", "humid"):
                    rt = risk_table(feat, clim, lab_b, clim_risk[r], r, h, targets_days)
                    rs = rt[["pixel_id", "target_date"]].merge(saved[(r, h)], on=["pixel_id", "target_date"], how="left",
                                                               validate="one_to_one")
                    assert (rt["pixel_id"].to_numpy() == t["pixel_id"].to_numpy()).all()
                    contrib, typ, err = explain(models[(f"blight_{r}", h)], rt, risk_columns(rt), "prob", rs["catboost"])
                    typical[f"blight_{r}|{h}"], worst = typ, max(worst, err)
                    for g, name in enumerate(GROUPS):
                        block[f"blight_{r}|{name}"] = np.round(contrib[:, g] * 10).astype(np.int16)
                    del rt
            parts.append(block)
            del t
        del feat, clim
        print(f"  {site}: batch {k + 1} of {len(batches)} explained ({len(px)} pixels x 7 horizons)", flush=True)

    out = pd.concat(parts, ignore_index=True).sort_values(["issue_date", "horizon", "pixel_id"]).reset_index(drop=True)
    assert not out.duplicated(["pixel_id", "issue_date", "horizon"]).any()
    f = OUT / f"{site}_shap.parquet"
    pq.write_table(pa.Table.from_pandas(out, preserve_index=False), f, row_group_size=50_000, compression="zstd")
    print(f"  PASS  {site}: {len(out):,} forecasts explained; SHAP adds up; rebuilt forecasts match the saved ones "
          f"(max difference {worst:.1g} points); {f.stat().st_size / 1e6:.0f} MB")
    return {"targets": list(targets), "typical": typical}


def main() -> None:
    sites_cfg, exp = load_configs()
    meta = {"groups": GROUPS, "labels": LABELS, "sites": {}}
    for site in sites_cfg["sites"]:
        meta["sites"][site] = build_site(site, sites_cfg, exp)
    (OUT / "shap_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"saved {OUT / 'shap_meta.json'}")


if __name__ == "__main__":
    main()
