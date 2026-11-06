"""Tests for the moving-block bootstrap in src/models/uncertainty.py (synthetic data)."""
import numpy as np
import pandas as pd
import pytest

from src.models.evaluation import binary_scores, continuous_scores
from src.models.uncertainty import block_bootstrap_weights, bootstrap_site_horizon, paired_differences


def test_weights_have_sample_size_and_contiguous_blocks():
    W = block_bootstrap_weights(n_days=50, block=7, n_boot=200, seed=1)
    assert W.shape == (200, 50) and (W.sum(axis=1) == 50).all()
    # a block as long as the sample is one full circular pass: every day exactly once
    assert (block_bootstrap_weights(30, 30, 10, 0) == 1).all()
    # same seed -> same resamples (needed for paired comparisons across models)
    assert np.array_equal(W, block_bootstrap_weights(50, 7, 200, 1))


def make_predictions(n_days=40, n_pix=5, seed=0):
    rng = np.random.default_rng(seed)
    dates = np.repeat(pd.date_range("2024-01-08", periods=n_days), n_pix)
    n = len(dates)
    y_rain = rng.gamma(0.6, 8, n) * (rng.random(n) < 0.5)
    y_rh = rng.normal(70, 8, n)
    pr = pd.DataFrame({"pixel_id": np.tile([f"p{i}" for i in range(n_pix)], n_days),
                       "target_date": dates, "y_rain": y_rain, "y_rh": y_rh})
    for m, noise in [("persistence", 1.0), ("climatology", 0.6), ("lightgbm", 0.3), ("xgboost", 0.3)]:
        pr[f"{m}__occ1"] = np.clip((y_rain >= 1) + rng.normal(0, noise, n), 0, 1)
        pr[f"{m}__occ10"] = np.clip((y_rain >= 10) + rng.normal(0, noise, n), 0, 1)
        pr[f"{m}__wet"] = np.clip(y_rain + rng.normal(0, 10 * noise, n), 1, None)
        pr[f"{m}__direct"] = np.clip(y_rain + rng.normal(0, 10 * noise, n), 0, None)
        pr[f"{m}__two_stage"] = pr[f"{m}__occ1"] * pr[f"{m}__wet"]
        pr[f"{m}__rh"] = y_rh + rng.normal(0, 10 * noise, n)
    return pr


def test_point_estimates_equal_direct_metrics():
    pr = make_predictions()
    W = block_bootstrap_weights(pr["target_date"].nunique(), 7, 50, 0)
    rows, _ = bootstrap_site_horizon(pr, W, wet_mm=1.0, exceed=[10])
    r = pd.DataFrame(rows).set_index(["model", "target", "metric", "reference"])["estimate"]
    y1 = (pr["y_rain"] >= 1).astype(float)
    b = binary_scores(y1, pr["lightgbm__occ1"], pr["climatology__occ1"])
    assert r[("lightgbm", "rain>=1mm", "bss", "climatology")] == pytest.approx(b["bss"])
    assert r[("lightgbm", "rain>=1mm", "brier", "")] == pytest.approx(b["brier"])
    wet = pr["y_rain"] >= 1
    c = continuous_scores(pr.loc[wet, "y_rain"], pr.loc[wet, "lightgbm__wet"], pr.loc[wet, "climatology__wet"])
    assert r[("lightgbm", "rain_mm_wet", "rmse_skill", "climatology")] == pytest.approx(c["rmse_skill"])
    assert r[("lightgbm", "rain_mm_wet", "rmse", "")] == pytest.approx(c["rmse"])
    assert r[("lightgbm", "rain_mm_wet", "mae", "")] == pytest.approx(c["mae"])
    c = continuous_scores(pr["y_rh"], pr["lightgbm__rh"], pr["persistence__rh"])
    assert r[("lightgbm", "rh_pct", "rmse_skill", "persistence")] == pytest.approx(c["rmse_skill"])


def test_interval_contains_estimate_and_identical_models_differ_by_zero():
    pr = make_predictions()
    for c in [c for c in pr.columns if c.startswith("lightgbm__")]:
        pr[c.replace("lightgbm", "xgboost")] = pr[c]  # make xgboost identical to lightgbm
    W = block_bootstrap_weights(pr["target_date"].nunique(), 7, 300, 0)
    rows, draws = bootstrap_site_horizon(pr, W, wet_mm=1.0, exceed=[10])
    res = pd.DataFrame(rows)
    sk = res[res["reference"] != ""]
    assert ((sk["ci_low"] <= sk["estimate"] + 1e-12) & (sk["estimate"] <= sk["ci_high"] + 1e-12)).all()
    diffs = pd.DataFrame(paired_differences(draws, res))
    same = diffs[diffs["comparison"] == "lightgbm - xgboost"]
    assert np.allclose(same[["estimate", "ci_low", "ci_high"]].to_numpy(), 0)


def test_combine_runs_pairs_identical_rows_and_rejects_different_ones():
    from src.models.uncertainty import combine_runs
    a = make_predictions(seed=0)
    b = a.copy()
    b["lightgbm__occ1"] = np.clip(b["lightgbm__occ1"] + 0.05, 0, 1)  # different model, same rows
    c = combine_runs({60: a, 200: b.sample(frac=1, random_state=0)})  # row order must not matter
    assert {"lightgbm@train60__occ1", "lightgbm@train200__occ1", "climatology__occ1"} <= set(c.columns)
    assert len(c) == len(a)
    W = block_bootstrap_weights(c["target_date"].nunique(), 7, 100, 0)
    rows, draws = bootstrap_site_horizon(c, W, wet_mm=1.0, exceed=[10])
    d = pd.DataFrame(paired_differences(draws, pd.DataFrame(rows)))
    pair = d["comparison"].isin(["lightgbm@train60 - lightgbm@train200", "lightgbm@train200 - lightgbm@train60"])
    same = d[pair & (d["target"] == "rh_pct")]
    assert len(same) == 1  # the comparison exists (the check below is not vacuous)
    assert np.allclose(same[["estimate", "ci_low", "ci_high"]].to_numpy(), 0)  # rh identical in both
    occ = d[pair & (d["target"] == "rain>=1mm")]
    assert len(occ) == 1 and abs(occ["estimate"].item()) > 0  # occ1 differs between the runs
    bad = b.copy()
    bad["climatology__rh"] += 1.0  # baselines differ -> not the same scoring set
    with pytest.raises(ValueError):
        combine_runs({60: a, 200: bad})


def test_training_size_comparison_reads_more_minus_fewer(tmp_path, monkeypatch):
    import src.models.uncertainty as unc
    monkeypatch.setattr(unc, "MODELS", tmp_path)
    a = make_predictions(seed=0)
    b = a.copy()
    b["lightgbm__occ1"] = np.clip(b["lightgbm__occ1"] + 0.05, 0, 1)
    for n, pr in [(60, a), (200, b)]:
        d = tmp_path / "s" / f"train{n}_score300" / "h1"
        d.mkdir(parents=True)
        pr.to_parquet(d / "val_predictions.parquet")
    res = unc.training_size_comparisons(["s"], {"block_days": 7, "n_boot": 100, "seed": 0}, 1.0, [10])
    assert set(res["comparison"]) == {"lightgbm@train200 - lightgbm@train60",
                                      "xgboost@train200 - xgboost@train60"}
    occ = res[(res["comparison"].str.startswith("lightgbm")) & (res["target"] == "rain>=1mm")]
    # b's occ1 forecast was made worse by +0.05 everywhere -> more-pixels run scores lower
    direct = (binary_scores((a["y_rain"] >= 1), b["lightgbm__occ1"], a["climatology__occ1"])["bss"]
              - binary_scores((a["y_rain"] >= 1), a["lightgbm__occ1"], a["climatology__occ1"])["bss"])
    assert occ["estimate"].item() == pytest.approx(direct)
    assert occ["ci_low"].item() <= occ["estimate"].item() <= occ["ci_high"].item()
