"""Tests for the live branch helpers (no network): Open-Meteo grid, units and call counting; the CHIRPS
preliminary schedule used to decide which recent days need filling."""
import numpy as np
import pandas as pd
import pytest

from src.ingestion.era5 import DAILY_COLUMNS
from src.live.backtest import FOLD_SPLIT, bias_corrected, last_prelim_day
from src.live.openmeteo import HOURLY, call_weight, grid_nodes, to_dataset
from src.models.evaluation import load_configs


def test_grid_nodes_match_the_era5_files_on_disk_shape():
    sites, _ = load_configs()
    lats, lons = grid_nodes(sites["sites"]["nakuru"]["era5_area"])
    assert (len(lats), len(lons)) == (9, 9)  # the ERA5 file for Nakuru is 9 x 9
    assert lats[0] == -1.5 and lats[-1] == 0.5 and lons[0] == 35.0 and lons[-1] == 37.0
    lats, lons = grid_nodes(sites["sites"]["busia"]["era5_area"])
    assert (len(lats), len(lons)) == (8, 6)


def test_call_weight_follows_open_meteo_rule():
    assert call_weight(1, 14) == 1 and call_weight(1, 15) == 2
    assert call_weight(81, 31) == 81 * 3
    assert call_weight(10, 7, n_vars=11) == 20


def fake_response(lat, lon, n=24):
    h = {"time": [f"2024-05-01T{k:02d}:00" for k in range(n)]}
    h.update({"temperature_2m": [20.0] * n, "dew_point_2m": [10.0] * n, "surface_pressure": [800.0] * n,
              "wind_speed_10m": [3.0] * n, "cloud_cover": [50] * n, "precipitation": [0.5] * n})
    return {"latitude": lat, "longitude": lon, "hourly": h}


def test_to_dataset_converts_to_era5_names_and_units():
    lats, lons = np.array([0.0, 0.25]), np.array([34.0])
    ds = to_dataset([fake_response(0.0, 34.0), fake_response(0.25, 34.0)], lats, lons)
    assert set(ds.data_vars) == {"t2m", "d2m", "sp", "u10", "v10", "tcc", "tp"}
    assert float(ds["t2m"][0, 0, 0]) == pytest.approx(293.15) and float(ds["sp"][0, 0, 0]) == pytest.approx(80000.0)
    assert float(ds["tcc"][0, 0, 0]) == pytest.approx(0.5) and float(np.hypot(ds["u10"], ds["v10"])[0, 0, 0]) == 3.0
    assert set(HOURLY) >= {"temperature_2m", "dew_point_2m", "surface_pressure", "wind_speed_10m", "cloud_cover"}


def test_to_dataset_refuses_a_different_model_grid():
    with pytest.raises(SystemExit):
        to_dataset([fake_response(-0.246, 35.972)], np.array([-0.25]), np.array([36.0]))


@pytest.mark.parametrize("issue, last", [
    ("2024-05-07", "2024-05-05"),   # pentad 1-5 published on the 7th
    ("2024-05-06", "2024-04-30"),   # not yet: last pentad of April (26-30)
    ("2024-05-12", "2024-05-10"),
    ("2024-05-27", "2024-05-25"),
    ("2024-06-02", "2024-05-31"),   # 31-day month: pentad 26-31 published on the 2nd
    ("2024-03-01", "2024-02-25"),   # leap February: 26-29 published on 2 March
    ("2024-03-02", "2024-02-29"),
])
def test_chirps_preliminary_schedule(issue, last):
    assert last_prelim_day(pd.Timestamp(issue)) == pd.Timestamp(last)


def test_rain_gap_is_two_to_eight_days_all_year():
    gaps = [(t - last_prelim_day(t)).days for t in pd.date_range("2024-01-01", "2024-12-31")]
    assert min(gaps) == 2 and max(gaps) <= 8


def test_live_overlay_is_off_by_default_and_needs_contiguous_days(tmp_path, monkeypatch):
    import json
    from app import live_data
    live = tmp_path / "live"
    live.mkdir()
    (live / "live_meta.json").write_text(json.dumps({"made": "x", "attribution": "Open-Meteo", "sites": {
        "nakuru": {"issue_min": "2026-11-15", "issue_max": "2026-11-19"},
        "busia": {"issue_min": "2026-11-17", "issue_max": "2026-11-19"}}}))
    monkeypatch.setattr(live_data, "LIVE", live)
    live_data.live_meta.cache_clear()
    base = lambda: {"nakuru": {"issue_max": "2026-11-14"}, "busia": {"issue_max": "2026-11-14"}}  # noqa: E731
    monkeypatch.delenv("HL_LIVE", raising=False)
    assert live_data.overlay_meta(base()) == base()          # off unless HL_LIVE=1
    monkeypatch.setenv("HL_LIVE", "1")
    m = live_data.overlay_meta(base())
    assert m["nakuru"]["issue_max"] == "2026-11-19" and "live" in m["nakuru"]
    assert m["busia"]["issue_max"] == "2026-11-14" and "live" not in m["busia"]  # gap: not shown
    assert live_data.is_live("nakuru", "2026-11-16", m) and not live_data.is_live("nakuru", "2026-11-14", m)
    live_data.live_meta.cache_clear()


def test_bias_correction_is_cross_fitted_in_time():
    """Each half of the period is corrected with the offset fitted on the OTHER half only, per pixel."""
    dates = pd.date_range(FOLD_SPLIT - pd.Timedelta(days=10), FOLD_SPLIT + pd.Timedelta(days=9))
    rows = [(p, d) for p in ["a", "b"] for d in dates]
    std = pd.DataFrame(rows, columns=["pixel_id", "date"])
    ifs = std.copy()
    for c in DAILY_COLUMNS:
        std[c], ifs[c] = np.float32(50.0), np.float32(50.0)
    later = (std["date"] >= FOLD_SPLIT).to_numpy()
    # ERA5 is 2 above IFS before the split and 6 above after it; pixel b has no offset at all
    std.loc[(std["pixel_id"] == "a") & ~later, "t2m_c"] = 52.0
    std.loc[(std["pixel_id"] == "a") & later, "t2m_c"] = 56.0
    std["precip_mm"] = np.float32(0.0)
    ifs["ifs_precip_mm"] = np.float32(0.0)
    out = bias_corrected(std, ifs)
    a = out["pixel_id"] == "a"
    assert np.allclose(out.loc[a & ~later, "t2m_c"], 56.0)  # early days get the LATER half's offset (+6)
    assert np.allclose(out.loc[a & later, "t2m_c"], 52.0)   # later days get the EARLY half's offset (+2)
    assert np.allclose(out.loc[~a, "t2m_c"], 50.0)
    assert np.allclose(out.loc[a, "rh_pct"], 50.0) and (out["tcc"] <= 1.0).all()  # tcc clipped to its range
