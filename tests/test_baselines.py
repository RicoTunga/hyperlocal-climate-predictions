"""Tests for the time shifts and climatology used by the baselines (synthetic data only).

Run from the repo root:  python -m pytest -q
"""
import numpy as np
import pandas as pd
import pytest

from src.models.baselines import fit_climatology
from src.models.evaluation import (binary_scores, continuous_scores, doy365, eval_target_dates,
                                   horizon_frame)


def make_df(start="2023-12-01", end="2024-01-31", pixels=("a", "b")):
    """Two pixels whose values encode the date, with a different offset per pixel."""
    dates = pd.date_range(start, end, freq="D")
    rows = []
    for k, pid in enumerate(pixels):
        day_no = np.arange(len(dates), dtype=float)
        rows.append(pd.DataFrame({"pixel_id": pid, "date": dates,
                                  "precip_mm": day_no + 1000 * k, "rh_pct": day_no}))
    return pd.concat(rows, ignore_index=True)


@pytest.mark.parametrize("h", [1, 3, 7])
def test_issue_value_is_exactly_h_days_earlier_same_pixel(h):
    df = make_df()
    targets = pd.date_range("2024-01-08", "2024-01-31")
    fr = horizon_frame(df, "precip_mm", h, targets)
    assert len(fr) == 2 * len(targets)
    assert ((fr["target_date"] - fr["issue_date"]).dt.days == h).all()
    # value encodes day number (+1000 per pixel): issue value = target value - h, same pixel
    assert np.allclose(fr["y"] - fr["issue_value"], h)
    assert fr["target_date"].min() == targets[0] and fr["target_date"].max() == targets[-1]


def test_rows_shuffled_give_same_result():
    df = make_df()
    targets = pd.date_range("2024-01-08", "2024-01-31")
    a = horizon_frame(df, "precip_mm", 2, targets)
    b = horizon_frame(df.sample(frac=1, random_state=1), "precip_mm", 2, targets)
    pd.testing.assert_frame_equal(a, b)


def test_missing_issue_day_raises_instead_of_shifting():
    df = make_df()
    df = df[~((df["pixel_id"] == "a") & (df["date"] == "2024-01-05"))]
    with pytest.raises(ValueError):
        horizon_frame(df, "precip_mm", 3, pd.date_range("2024-01-08", "2024-01-31"))


def test_doy365_aligns_calendar_dates_across_leap_years():
    d = pd.to_datetime(["2023-02-28", "2024-02-28", "2024-02-29", "2023-03-01", "2024-03-01",
                        "2023-12-31", "2024-12-31", "2024-01-01"])
    assert list(doy365(d)) == [59, 59, 59, 60, 60, 365, 365, 1]


def test_eval_targets_and_gap():
    exp = {"splits": {"train": {"start": "2016-01-01", "end": "2023-12-31"},
                      "validation": {"start": "2024-01-01", "end": "2024-12-31"},
                      "test": {"start": "2025-01-01", "end": "2026-09-30"}},
           "gap_days": 7}
    t = eval_target_dates(exp, "validation", max_h=7)
    assert t[0] == pd.Timestamp("2024-01-08") and t[-1] == pd.Timestamp("2024-12-31")
    exp["gap_days"] = 8
    with pytest.raises(ValueError):
        eval_target_dates(exp, "validation", max_h=7)


def test_climatology_window_and_wraparound():
    # rain 10 mm on 1 Jan every year, dry otherwise; one pixel, three years
    dates = pd.date_range("2016-01-01", "2018-12-31")
    df = pd.DataFrame({"pixel_id": "a", "date": dates,
                       "precip_mm": np.where((dates.month == 1) & (dates.day == 1), 10.0, 0.0),
                       "rh_pct": 50.0})
    clim = fit_climatology(df, wet_mm=1.0, exceed_mm=[10], window=2).set_index("doy")
    # window of 5 days: days 364, 365, 1, 2, 3 each contain 1 Jan once per year -> p = 1/5
    for doy in [364, 365, 1, 2, 3]:
        assert clim.loc[doy, "clim_p_wet"] == pytest.approx(0.2)
        assert clim.loc[doy, "clim_wet_amount"] == pytest.approx(10.0)
        assert clim.loc[doy, "clim_p_ge10"] == pytest.approx(0.2)
    assert clim.loc[4, "clim_p_wet"] == 0
    assert clim.loc[4, "clim_wet_amount"] == pytest.approx(10.0)  # falls back to pixel mean
    assert clim["clim_rh_mean"].eq(50.0).all()


def test_scores_basic():
    y = np.array([0, 1, 0, 1])
    s = binary_scores(y, np.array([0, 1, 0, 1]), np.full(4, 0.5))
    assert s["brier"] == 0 and s["bss"] == 1 and s["auc"] == 1
    c = continuous_scores(np.array([1.0, 2, 3]), np.array([1.0, 2, 3]), np.array([2.0, 2, 2]))
    assert c["rmse"] == 0 and c["r2"] == 1 and c["rmse_skill"] == 1
