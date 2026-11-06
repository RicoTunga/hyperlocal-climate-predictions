"""Tests for the late-blight risk labels and climatology (synthetic data)."""
import numpy as np
import pandas as pd
import pytest

from src.models.late_blight import binary_climatology, era5_terrain_height, risk_labels


def make(pixels=("a", "b"), days=6, hours=None, tmin=None, elev=2000.0, sp=800.0):
    dates = pd.date_range("2020-01-01", periods=days)
    rows = []
    for k, pid in enumerate(pixels):
        rows.append(pd.DataFrame({"pixel_id": pid, "date": dates,
                                  "t2m_min_c": tmin[k] if tmin is not None else 12.0,
                                  "sp_hpa": sp, "elev_mean_m": elev}))
    daily = pd.concat(rows, ignore_index=True)
    h = pd.DataFrame({"pixel_id": daily["pixel_id"], "date": daily["date"],
                      "rh_hours_ge90": np.concatenate(hours) if hours is not None else 8.0})
    return daily, h


def test_rule_needs_two_consecutive_qualifying_days_of_the_same_pixel():
    # pixel a: humid on days 1-2 and 4; pixel b: humid on day 1 only (must not pair with a's last day)
    hours = [np.array([8, 8, 0, 8, 0, 8.0]), np.array([8, 0, 0, 0, 0, 0.0])]
    daily, h = make(hours=hours)
    lab = risk_labels(daily, h, pd.Timestamp("2020-01-01"), pd.Timestamp("2020-12-31")).set_index(["pixel_id", "date"])
    a = lab.loc["a", "met_humid"].to_numpy()
    assert list(a) == [False, True, False, False, False, False]
    assert not lab.loc["b", "met_humid"].any()  # b's day 1 follows a's day 6 in row order: not consecutive


def test_missing_day_breaks_the_pair():
    daily, h = make(pixels=("a",), hours=[np.full(6, 8.0)])
    keep = daily["date"] != "2020-01-03"
    lab = risk_labels(daily[keep], h[keep], pd.Timestamp("2020-01-01"), pd.Timestamp("2020-12-31"))
    met = lab.set_index("date")["met_humid"]
    assert not met[pd.Timestamp("2020-01-04")]  # 01-02 and 01-04 are not consecutive


def test_temperature_correction_cools_pixels_above_era5_terrain():
    z = float(era5_terrain_height(pd.Series([800.0])).iloc[0])
    assert 1900 < z < 2000  # ~1,950 m for 800 hPa in the standard atmosphere
    daily, h = make(pixels=("a",), elev=z + 1000.0)  # pixel 1,000 m above ERA5's terrain
    lab = risk_labels(daily, h, pd.Timestamp("2020-01-01"), pd.Timestamp("2020-12-31"))
    assert lab["tmin_corr_c"].iloc[0] == pytest.approx(12.0 - 6.5, abs=1e-4)
    assert not lab["met_hutton"].any() and lab["met_humid"].iloc[1:].all()  # 5.5 C < 10 C


def test_binary_climatology_window_wraps_round_the_year():
    dates = pd.date_range("2016-01-01", "2018-12-31")
    lab = pd.DataFrame({"pixel_id": "a", "date": dates, "met_x": (dates.month == 1) & (dates.day == 1)})
    c = binary_climatology(lab, "met_x", window=2).set_index("doy")["clim_p_risk"]
    for doy in [364, 365, 1, 2, 3]:
        assert c[doy] == pytest.approx(0.2)
    assert c[4] == 0


def test_risk_table_joins_issue_day_inputs_and_target_day_risk():
    from src.features.build import build_features
    from src.models.baselines import fit_climatology
    from src.models.late_blight import risk_table
    from tests.test_features import EXP, make_grid
    df = make_grid(n_days=120)
    df["site"] = "test"
    df["t2m_min_c"], df["elev_mean_m"] = 12.0, 2000.0
    feat = build_features(df, EXP)
    rng = np.random.default_rng(0)
    hours = df[["pixel_id", "date"]].assign(rh_hours_ge90=rng.choice([0.0, 8.0], len(df)))
    lab = risk_labels(df[["pixel_id", "date", "t2m_min_c", "sp_hpa", "elev_mean_m"]], hours,
                      df["date"].min(), df["date"].max())
    clim_rain = fit_climatology(feat, 1.0, [10], window=182)
    clim_risk = binary_climatology(lab, "met_humid", window=182)
    targets = pd.date_range(df["date"].min() + pd.Timedelta(days=40), df["date"].max())
    t = risk_table(feat, clim_rain, lab, clim_risk, "humid", 3, targets)
    L = lab.set_index(["pixel_id", "date"])
    assert np.array_equal(t["lb_rh_hours_ge90"].to_numpy(),
                          L.loc[list(zip(t["pixel_id"], t["issue_date"])), "rh_hours_ge90"].to_numpy())
    assert np.array_equal(t["y_risk"].to_numpy(),
                          L.loc[list(zip(t["pixel_id"], t["target_date"])), "met_humid"].to_numpy(float))
    assert np.array_equal(t["persistence"].to_numpy(), t["lb_met_humid"].to_numpy())
