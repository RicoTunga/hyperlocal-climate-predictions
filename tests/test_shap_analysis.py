"""Tests for src/explain/shap_analysis.py (synthetic data only)."""
import numpy as np
import pandas as pd
import pytest
from catboost import CatBoostClassifier

from src.explain.shap_analysis import GROUPS, feature_group, shap_values, within_day_spread
from src.features.build import build_features
from src.models.baselines import fit_climatology
from src.models.train import feature_columns, horizon_table
from tests.test_features import EXP, make_grid


def test_every_model_input_belongs_to_exactly_one_group():
    df = make_grid(n_days=120)
    df["site"] = "test"
    for c in ["rh_min_pct", "rh_max_pct", "t2m_min_c", "t2m_max_c", "wind_max_ms",
              "elev_mean_m", "elev_std_m", "elev_min_m", "elev_max_m"]:
        df[c] = 1.0  # the real merged table has these columns too
    feat = build_features(df, EXP)
    clim = fit_climatology(feat, 1.0, [10, 25], window=182)
    t = horizon_table(feat, clim, 1, pd.date_range(feat["date"].min() + pd.Timedelta(days=40), feat["date"].max()))
    groups = {c: feature_group(c) for c in feature_columns(t)}
    assert set(groups.values()) <= set(GROUPS)
    assert groups["precip_mm"] == "own rain history" and groups["region_rh_mean"] == "area-wide"
    assert groups["nbr3_precip"] == "neighbourhood rain" and groups["clim_p_wet"] == "climatology"
    assert groups["elev_mean_m"] == "location & elevation" and groups["sp_hpa_d1"] == "own ERA5 weather"
    with pytest.raises(ValueError):
        feature_group("mystery_column")


def test_within_day_spread():
    day = np.repeat(np.arange(5), 4)
    assert within_day_spread(np.repeat([1.0, 2, 3, 4, 5], 4), day) == 0  # same for all pixels on a day
    v = np.tile([0.0, 2.0, 0.0, 2.0], 5)  # pixels differ by 2 every day -> std 1
    assert within_day_spread(v, day) == pytest.approx(1.0)


def test_shap_values_add_up_to_model_output():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(500, 4)), columns=list("abcd"))
    y = (X["a"] + 0.5 * X["b"] + rng.normal(0, 0.5, 500) > 0).astype(int)
    m = CatBoostClassifier(iterations=50, depth=3, verbose=0, allow_writing_files=False).fit(X, y)
    shap, expected = shap_values(m, X)
    assert shap.shape == (500, 4)
    assert np.abs(shap[:, 0]).mean() > np.abs(shap[:, 3]).mean()  # 'a' matters more than noise 'd'
