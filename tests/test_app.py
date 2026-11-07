"""Tests for the dashboard data layer: colour, classes and views (synthetic) and the replay files (real data).

The real-data test needs app/data/ (built by `python -m app.build_data`) and is skipped without it.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app import components as C

DATA_READY = (Path(__file__).resolve().parents[1] / "app" / "data" / "meta.json").exists()


def test_ramps_have_steadily_changing_lightness():
    for ramp in (C.BLUE, C.ORANGE):
        assert all(np.diff(C.lightness(ramp)) < 0)  # light -> dark
    L = C.lightness(C.DIVERGING)
    assert all(np.diff(L[:4]) > 0) and all(np.diff(L[3:]) < 0)  # grey midpoint is the lightest
    assert C.DIVERGING[3] == "#f0efec"


def test_classify_puts_values_in_the_right_class():
    b = C.PROB
    v = np.array([0.0, 0.049, 0.05, 0.25, 0.69, 0.7, 1.0])
    assert list(C.classify(v, b)) == [0, 0, 1, 3, 5, 6, 6]


def test_error_view_for_rain_chance_compares_with_whether_it_rained():
    df = pd.DataFrame({"p_rain1": [0.8, 0.8, 0.1], "obs_rain_mm": [5.0, 0.0, 0.0], "clim_rain1": [0.3, 0.3, 0.3]})
    v, breaks, ramp, unit = C.view_values(df, C.LAYERS["Chance of rain (>= 1 mm)"], "Forecast minus observed")
    assert np.allclose(v, [-0.2, 0.8, 0.1]) and unit == "points" and ramp == tuple(C.DIVERGING)
    v, _, _, unit = C.view_values(df, C.LAYERS["Chance of rain (>= 1 mm)"], "Observed")
    assert np.allclose(v, [5.0, 0.0, 0.0]) and unit == "mm"


def test_squares_are_one_cell_wide_and_centred():
    px = pd.DataFrame({"pixel_id": ["a"], "lat": [0.025], "lon": [35.025], "elev_mean_m": [1500.0]})
    sq = C.squares(pd.DataFrame({"pixel_id": ["a"]}), px)["polygon"].iloc[0]
    assert sq[0] == [35.0, 0.0] and sq[2] == [pytest.approx(35.05), pytest.approx(0.05)]


@pytest.mark.skipif(not DATA_READY, reason="app/data not built")
def test_replay_rows_match_issue_date_plus_horizon():
    m = C.meta()
    d = C.issue_day("busia", m["busia"]["issue_min"])
    assert set(d["horizon"]) == set(range(1, 8))
    assert ((pd.to_datetime(d["target_date"]) - pd.to_datetime(d["issue_date"])).dt.days == d["horizon"]).all()
    assert len(d) == 7 * m["busia"]["n_pixels"]
