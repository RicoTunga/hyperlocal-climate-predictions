"""Tests for the per-forecast SHAP explanations (app/build_shap.py), synthetic data."""
import numpy as np
import pytest

from app.build_shap import GROUPS, LABELS, group_of, grouped, to_points


def test_points_add_up_to_forecast_minus_typical_and_keep_signs():
    rng = np.random.default_rng(0)
    phi = rng.normal(0, 0.8, size=(500, len(GROUPS)))  # log-odds contributions
    expected = -1.2
    contrib, typical, forecast = to_points(phi, expected, "prob")
    p0, p = 1 / (1 + np.exp(1.2)), 1 / (1 + np.exp(-(expected + phi.sum(axis=1))))
    assert typical == pytest.approx(100 * p0)
    assert np.allclose(forecast, 100 * p)
    assert np.allclose(contrib.sum(axis=1), forecast - typical)  # exactly additive
    assert np.array_equal(np.sign(contrib), np.sign(phi))  # a push up stays a push up
    assert np.all(np.abs(contrib) <= 25 * np.abs(phi) + 1e-12)  # slope of the logistic curve is at most 1/4


def test_points_when_the_forecast_equals_the_typical_value():
    phi = np.array([[0.5, -0.5, 0, 0, 0, 0, 0, 0]])
    contrib, typical, forecast = to_points(phi, 0.0, "prob")
    assert forecast[0] == pytest.approx(typical) == pytest.approx(50.0)
    assert contrib[0, 0] == pytest.approx(100 * 0.25 * 0.5) and contrib[0, 1] == pytest.approx(-12.5)


def test_humidity_contributions_are_unchanged():
    phi = np.array([[1.5, -2.0, 0, 0, 0, 0, 0, 0]])
    contrib, typical, forecast = to_points(phi, 70.0, "pct")
    assert np.array_equal(contrib, phi) and typical == 70.0 and forecast[0] == pytest.approx(69.5)


def test_every_input_has_a_plain_language_group():
    cols = ["precip_mm", "precip_mm_lag3", "nbr5_precip", "region_wet_frac", "rh_pct_d1", "tcc", "sp_hpa_std14",
            "clim_p_wet", "clim_p_risk", "doy_sin", "elev_mean_m", "lat", "lb_rh_hours_ge90", "lb_met_hutton"]
    for c in cols:
        assert group_of(c) in GROUPS and group_of(c) in LABELS
    shap = np.ones((3, len(cols)))
    g = grouped(shap, cols)
    assert g.sum() == shap.sum() and g[:, GROUPS.index("blight conditions today")].tolist() == [2, 2, 2]
