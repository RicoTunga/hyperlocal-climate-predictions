"""Tests for src/features/build.py on a small synthetic grid (no real data needed).

Run from the repo root:  python -m pytest -q
"""
import numpy as np
import pandas as pd
import pytest

from src.features.build import build_features, days_since

ERA5 = ["rh_pct", "tcc", "t2m_c", "d2m_c", "sp_hpa", "wind_ms"]
EXP = {
    "rain": {"wet_threshold_mm": 1.0},
    "features": {"rain_lags": [1, 3, 7], "rain_windows": [3, 7], "rain_long_window": 10,
                 "era5_vars": ERA5, "era5_lags": [1, 3], "era5_windows": [3],
                 "tendency_vars": ["rh_pct", "sp_hpa"], "neighbourhood_sizes": [3, 5],
                 "pixel_batch": 5},
}


def make_grid(n_lat=4, n_lon=3, n_days=60, seed=0):
    """Random but realistic-looking data on a regular 0.05 degree grid."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2016-01-01", periods=n_days, freq="D")
    rows = []
    for i in range(n_lat):
        for j in range(n_lon):
            lat, lon = round(-0.5 + 0.05 * i, 3), round(35.0 + 0.05 * j, 3)
            rain = rng.gamma(0.5, 6.0, n_days) * (rng.random(n_days) < 0.4)
            d = {"pixel_id": f"{lat:.3f}_{lon:.3f}", "lat": lat, "lon": lon, "date": dates,
                 "precip_mm": rain}
            d.update({v: rng.normal(50, 10, n_days) for v in ERA5})
            rows.append(pd.DataFrame(d))
    return pd.concat(rows, ignore_index=True)


def feature_cols(f, df):
    return [c for c in f.columns if c not in df.columns]


def test_features_never_use_the_future():
    """Change every value after day T; all features up to and including T must not move."""
    df = make_grid()
    cut = df["date"].min() + pd.Timedelta(days=35)
    a = build_features(df, EXP)
    df2 = df.copy()
    later = df2["date"] > cut
    rng = np.random.default_rng(99)
    for c in ["precip_mm"] + ERA5:
        df2.loc[later, c] = rng.normal(100, 50, later.sum()).clip(0)
    b = build_features(df2, EXP)
    cols = feature_cols(a, df)
    keep = a["date"] <= cut
    pd.testing.assert_frame_equal(a.loc[keep, cols], b.loc[keep, cols])
    # and the change after T does show up (the test would be pointless otherwise)
    assert not a.loc[~keep, cols].equals(b.loc[~keep, cols])


def test_lags_and_rolling_are_per_pixel_and_end_on_day_t():
    df = make_grid()
    f = build_features(df, EXP)
    for pid, g in f.groupby("pixel_id"):
        p = g["precip_mm"].to_numpy()
        assert np.allclose(g["precip_mm_lag3"].to_numpy()[3:], p[:-3])
        assert np.isnan(g["precip_mm_lag3"].to_numpy()[:3]).all()
        # mean3 on day t = mean of days t-2, t-1, t (includes t, nothing after)
        expect = pd.Series(p).rolling(3).mean().to_numpy()
        assert np.allclose(g["precip_mm_mean3"], expect, equal_nan=True, atol=1e-5)
        assert np.isnan(g["precip_mm_mean3"].to_numpy()[:2]).all()
        rh = g["rh_pct"].to_numpy()
        assert np.allclose(g["rh_pct_d1"].to_numpy()[1:], (rh[1:] - rh[:-1]), atol=1e-4)


def test_batch_size_does_not_change_results():
    df = make_grid()
    a = build_features(df, EXP, batch_size=1)
    b = build_features(df, EXP, batch_size=100)
    pd.testing.assert_frame_equal(a, b)


def test_row_order_of_input_does_not_matter():
    df = make_grid()
    a = build_features(df, EXP)
    b = build_features(df.sample(frac=1, random_state=3), EXP)
    pd.testing.assert_frame_equal(a, b)


def test_neighbourhood_mean_at_corner_uses_existing_pixels_only():
    df = make_grid()
    f = build_features(df, EXP)
    day = df["date"].min() + pd.Timedelta(days=20)
    today = df[df["date"] == day]
    corner = today.sort_values(["lat", "lon"]).iloc[0]  # south-west corner pixel
    near = today[(today["lat"] - corner["lat"]).abs().le(0.0501)
                 & (today["lon"] - corner["lon"]).abs().le(0.0501)]
    assert len(near) == 4  # corner of the grid: itself + 3 neighbours
    got = f.loc[(f["pixel_id"] == corner["pixel_id"]) & (f["date"] == day), "nbr3_precip"].item()
    assert got == pytest.approx(near["precip_mm"].mean(), abs=1e-4)


def test_region_features_same_for_all_pixels_and_correct():
    df = make_grid()
    f = build_features(df, EXP)
    per_day = f.groupby("date")["region_precip_mean"].nunique()
    assert (per_day == 1).all()
    expect = df.groupby("date")["precip_mm"].mean()
    got = f.groupby("date")["region_precip_mean"].first()
    assert np.allclose(got, expect, atol=1e-4)


def test_days_since_wet():
    wet = np.array([False, True, False, False, True, False])[:, None]
    out = days_since(wet)[:, 0]
    assert np.isnan(out[0])
    assert list(out[1:]) == [0, 1, 2, 0, 1]


def test_missing_day_raises():
    df = make_grid()
    df = df.drop(df.index[10])
    with pytest.raises(ValueError):
        build_features(df, EXP)
