"""Tests for the post-test extension: CHIRPS CHC reader, the issue-day table and period labels.
Synthetic data only (no downloads, no saved models)."""
import gzip

import numpy as np
import pandas as pd
import pytest

from src.features.build import build_features
from src.ingestion import chirps
from src.ingestion.era5 import pixel_grid
from src.models.baselines import fit_climatology
from src.models.evaluation import load_configs
from src.models.posttest import issue_table, label_period, posttest_bounds
from src.models.train import feature_columns, horizon_table
from tests.test_features import EXP, make_grid


def test_chc_pixel_ids_match_the_era5_grid_and_config_boxes():
    sites, _ = load_configs()
    for s in sites["sites"].values():
        lat, lon, pid = chirps.pixel_centres(s["bbox"])
        la, lo = pixel_grid(s["bbox"])
        assert np.array_equal(lat, la) and np.array_equal(lon, lo)
        assert len(set(pid)) == len(pid) and all(p == f"{a:.3f}_{o:.3f}" for p, a, o in zip(pid, lat, lon))


def write_tif(path, left, top, values):
    """A gzipped GeoTIFF on a 0.05 degree grid (the CHIRPS layout), with float32 resolution like CHC files."""
    rasterio = pytest.importorskip("rasterio")
    from rasterio.transform import from_origin
    tif = path.with_suffix("")
    res = float(np.float32(0.05))
    with rasterio.open(tif, "w", driver="GTiff", height=values.shape[0], width=values.shape[1], count=1,
                       dtype="float32", crs="EPSG:4326", transform=from_origin(left, top, res, res)) as ds:
        ds.write(values.astype("float32"), 1)
    path.write_bytes(gzip.compress(tif.read_bytes()))
    return path


def test_read_tif_at_picks_the_cell_containing_each_centre_and_masks_no_data(tmp_path):
    vals = np.arange(40 * 60, dtype=float).reshape(40, 60)  # rows = latitude from the top
    vals[3, 7] = -9999.0
    f = write_tif(tmp_path / "day.tif.gz", left=33.0, top=1.0, values=vals)
    bbox = {"lon_min": 33.3, "lon_max": 33.6, "lat_min": 0.6, "lat_max": 0.9}
    lat, lon, _ = chirps.pixel_centres(bbox)
    got = chirps.read_tif_at(f, lat, lon)
    rows = np.floor((1.0 - lat) / 0.05).astype(int)
    cols = np.floor((lon - 33.0) / 0.05).astype(int)
    expect = vals[rows, cols]
    expect[expect < 0] = np.nan
    assert np.array_equal(got, expect, equal_nan=True)
    assert np.isnan(got).sum() == 1  # the no-data cell (row 3, col 7 is inside the box)


def test_preliminary_chunks_are_never_cached(tmp_path, monkeypatch):
    """A month that needs preliminary days is returned but not written to the monthly cache."""
    bbox = {"lon_min": 33.9, "lon_max": 34.0, "lat_min": 0.0, "lat_max": 0.1}
    days = pd.date_range("2026-11-01", "2026-11-03")
    lat, lon, pid = chirps.pixel_centres(bbox)

    def fake_fetch_chc(b, s, e, allow_prelim):
        df = pd.DataFrame([(p, a, o, d, 1.0) for d in days for p, a, o in zip(pid, lat, lon)],
                          columns=["pixel_id", "lat", "lon", "date", "precip_mm"])
        return df, list(days[1:])  # days 2 and 3 preliminary

    monkeypatch.setattr(chirps, "RAW", tmp_path)
    monkeypatch.setattr(chirps, "ee_ready", lambda: False)  # no Earth Engine: CHC route
    monkeypatch.setattr(chirps, "fetch_chc", fake_fetch_chc)
    chirps.ingest_site("x", {"name": "x", "bbox": bbox}, "2026-11-01", "2026-11-03", "", allow_prelim=True)
    assert not list((tmp_path / "chirps" / "x").glob("*.parquet")), "preliminary chunk was cached"
    src = (tmp_path / "chirps_x_sources.json").read_text()
    assert '"final_through": "2026-11-01"' in src and "2026-11-03" in src
    assert len(pd.read_parquet(tmp_path / "chirps_x.parquet")) == len(pid) * len(days)


def small_features():
    df = make_grid(n_days=120)
    df["site"] = "test"
    feat = build_features(df, EXP)
    clim = fit_climatology(feat, 1.0, [10], window=182)
    return feat, clim


def test_issue_table_equals_horizon_table_where_targets_are_observed():
    feat, clim = small_features()
    last = feat["date"].max()
    for h in [1, 4, 7]:
        targets = pd.date_range(last - pd.Timedelta(days=20), last)
        ref = horizon_table(feat, clim, h, targets)
        new = issue_table(feat, clim, h, targets - pd.Timedelta(days=h))
        cols = feature_columns(ref)
        assert set(cols) == set(feature_columns(new))
        m = ref.merge(new, on=["pixel_id", "target_date"], suffixes=("", "_n"), validate="one_to_one")
        assert len(m) == len(ref)
        for c in cols + ["y_rain", "y_rh", "issue_date"]:
            a, b = m[c], m[f"{c}_n"]
            assert (a.equals(b) if not pd.api.types.is_numeric_dtype(a)
                    else np.array_equal(a.to_numpy(float), b.to_numpy(float), equal_nan=True)), c


def test_issue_table_keeps_unobserved_targets_with_missing_outcomes():
    feat, clim = small_features()
    last = feat["date"].max()
    t = issue_table(feat, clim, 3, [last - pd.Timedelta(days=1), last])
    future = t["target_date"] > last
    assert future.all() and t["y_rain"].isna().all() and t["y_rh"].isna().all()
    assert t.filter(like="clim_").notna().all().all()  # the normal for the date is still known
    lab = label_period(t.assign(target_date=t["target_date"]), last, last - pd.Timedelta(days=5))
    assert (lab["period"] == "forecast").all()


def test_period_labels_and_config():
    _, exp = load_configs()
    start, end, final_through = posttest_bounds(exp)
    test_end = pd.Timestamp(str(exp["splits"]["test"]["end"]))
    assert start > test_end and start <= final_through <= end  # strictly after the reported test
    d = pd.DataFrame({"target_date": pd.date_range(start, end + pd.Timedelta(days=3))})
    p = label_period(d, end, final_through)["period"]
    assert (p[d["target_date"] <= final_through] == "posttest_final").all()
    assert (p[(d["target_date"] > final_through) & (d["target_date"] <= end)] == "posttest_prelim").all()
    assert (p[d["target_date"] > end] == "forecast").all()
