"""Tests for src/models/calibration.py on synthetic forecasts with known calibration."""
import numpy as np
import pytest

from src.models.calibration import (BIN_EDGES, bootstrap_bias_and_bins, brier_decomposition,
                                    calibration_slope)
from src.models.uncertainty import block_bootstrap_weights


def synthetic(n=200_000, seed=0):
    rng = np.random.default_rng(seed)
    p = rng.beta(1.2, 3.0, n)            # true probabilities, skewed towards dry like real rain
    y = (rng.random(n) < p).astype(float)  # outcomes drawn from them -> p is perfectly calibrated
    return p, y


def logit(p):
    return np.log(p / (1 - p))


def test_calibrated_forecasts_have_slope_one_and_no_bias():
    p, y = synthetic()
    slope, intercept = calibration_slope(y, p)
    assert slope == pytest.approx(1.0, abs=0.05) and intercept == pytest.approx(0.0, abs=0.05)
    assert abs(p.mean() - y.mean()) < 0.005


def test_overconfident_forecasts_have_slope_below_one():
    p, y = synthetic()
    over = 1 / (1 + np.exp(-2 * logit(np.clip(p, 1e-6, 1 - 1e-6))))  # pushed towards 0 and 1
    slope, _ = calibration_slope(y, over)
    assert slope == pytest.approx(0.5, abs=0.05)


def test_bias_detects_too_wet_forecasts_with_interval():
    p, y = synthetic(n=36_000)
    day = np.repeat(np.arange(360), 100)  # 360 days x 100 pixels
    W = block_bootstrap_weights(360, 7, 300, 0)
    wet = np.clip(p + 0.1, 0, 1)
    bias, lo, hi, bins = bootstrap_bias_and_bins(y, wet, day, 360, W)
    assert bias == pytest.approx(np.mean(wet - y))
    assert lo < bias < hi and lo > 0.05  # clearly too wet
    assert sum(b["n"] for b in bins) == len(y)
    for b in bins:
        assert b["obs_ci_low"] <= b["observed"] <= b["obs_ci_high"]


def test_brier_decomposition_is_exact_for_binned_forecasts():
    rng = np.random.default_rng(1)
    centres = (BIN_EDGES[:-1] + np.minimum(BIN_EDGES[1:], 1.0)) / 2
    p = rng.choice(centres, 50_000)            # one forecast value per bin
    y = (rng.random(50_000) < np.clip(p * 1.2, 0, 1)).astype(float)
    d = brier_decomposition(y, p)
    assert d["brier"] == pytest.approx(d["reliability"] - d["resolution"] + d["uncertainty"], abs=1e-12)
    assert d["reliability"] > 0  # forecasts were deliberately too dry
