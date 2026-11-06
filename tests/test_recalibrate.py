"""Tests for the Platt recalibration in src/models/recalibrate.py (synthetic data)."""
import numpy as np
import pytest

from src.models.calibration import calibration_slope
from src.models.recalibrate import apply_platt, fit_platt, logit


def test_platt_recovers_known_distortion_and_fixes_slope():
    rng = np.random.default_rng(0)
    true_p = rng.beta(1.2, 3.0, 200_000)
    y = (rng.random(len(true_p)) < true_p).astype(int)
    # over-confident and too-dry forecast: logit stretched by 1/0.7 and shifted by -0.3
    forecast = 1 / (1 + np.exp(-(logit(true_p) / 0.7 - 0.3)))
    a, b = fit_platt(y, forecast)
    # the inverse map is logit(true) = 0.7 * (logit(f) + 0.3) -> b = 0.7, a = 0.21
    assert b == pytest.approx(0.7, abs=0.03) and a == pytest.approx(0.21, abs=0.03)
    slope_before, _ = calibration_slope(y, forecast)
    slope_after, _ = calibration_slope(y, apply_platt(forecast, a, b))
    assert slope_before < 0.75 and slope_after == pytest.approx(1.0, abs=0.03)


def test_identity_correction_leaves_forecasts_unchanged():
    p = np.linspace(0.01, 0.99, 50)
    assert np.allclose(apply_platt(p, 0.0, 1.0), p)
