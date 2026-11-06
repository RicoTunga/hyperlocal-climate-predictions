"""Tests for src/models/fusion.py (synthetic data)."""
import numpy as np
import pandas as pd
import pytest

from src.models.fusion import ALGOS, add_blends, add_ensemble, blend_weight, fit_weights, tasks


def test_blend_weight_recovers_known_mix_and_clips():
    rng = np.random.default_rng(0)
    c = rng.random(50_000)
    p = rng.random(50_000)
    y = 0.3 * p + 0.7 * c + rng.normal(0, 0.01, 50_000)  # truth is a 30/70 mix
    assert blend_weight(y, p, c) == pytest.approx(0.3, abs=0.01)
    assert blend_weight(p * 2 - c, p, c) == 1.0  # optimum 2 -> clipped to 1
    assert blend_weight(c, c, c) == 1.0  # model identical to climatology -> keep the model


def frame(n=400, seed=0):
    rng = np.random.default_rng(seed)
    dates = np.repeat(pd.date_range("2024-01-08", periods=n // 4), 4)
    pr = pd.DataFrame({"pixel_id": np.tile(list("abcd"), n // 4), "target_date": dates,
                       "y_rain": rng.gamma(0.6, 8, n) * (rng.random(n) < 0.5), "y_rh": rng.normal(70, 8, n)})
    for m in ALGOS + ["climatology"]:
        for t in tasks(1.0, [10, 25]):
            pr[f"{m}__{t}"] = rng.random(n) if t.startswith("occ") else rng.random(n) * 10
    return pr


def test_ensemble_is_the_mean_and_blend_with_w_one_is_the_model():
    pr = add_ensemble(frame(), tasks(1.0, [10, 25]), 1.0)
    assert np.allclose(pr["ens__rh"], pr[[f"{a}__rh" for a in ALGOS]].mean(axis=1))
    assert np.allclose(pr["ens__two_stage"], pr["ens__occ1"] * pr["ens__wet"])
    ones = {(b, t): 1.0 for b in ["catboost", "ens"] for t in tasks(1.0, [10, 25])}
    out = add_blends(pr, ones, tasks(1.0, [10, 25]), 1.0)
    assert np.allclose(out["blend_cat__occ1"], out["catboost__occ1"])
    assert np.allclose(out["blend_ens__rh"], out["ens__rh"])


def test_cross_fitted_weights_use_only_the_other_months():
    pr = add_ensemble(frame(n=1600), tasks(1.0, [10, 25]), 1.0)
    odd = pd.DatetimeIndex(pr["target_date"]).month % 2 == 1
    w_even = fit_weights(pr, tasks(1.0, [10, 25]), 1.0, ~odd)
    # changing every odd-month target must not change the weights fitted on even months
    pr2 = pr.copy()
    pr2.loc[odd, "y_rh"] += 50
    assert fit_weights(pr2, tasks(1.0, [10, 25]), 1.0, ~odd) == w_even
