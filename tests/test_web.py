"""Tests for the dashboard server and colour-field rendering.

Raster tests use synthetic data; server tests need app/data/ (python -m app.build_data) and are
skipped without it. They run through a signed-in Starlette test client with a temporary user database.
"""
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from app import components as C
from web import raster

DATA_READY = (Path(__file__).resolve().parents[1] / "app" / "data" / "meta.json").exists()


# ---------------------------------------------------------------- raster (synthetic)
def tiny_day():
    lats, lons = [0.025, 0.075, 0.125], [35.025, 35.075]
    px = pd.DataFrame([{"pixel_id": f"{la:.3f}_{lo:.3f}", "lat": la, "lon": lo, "elev_mean_m": 1500.0}
                       for la in lats for lo in lons])
    day = pd.DataFrame({"pixel_id": px["pixel_id"], "p_rain1": [0.0, 0.1, 0.3, 0.5, 0.7, 1.0],
                        "clim_rain1": 0.3, "obs_rain_mm": 0.0})
    return day, px


def test_value_grid_puts_north_at_the_top():
    day, px = tiny_day()
    grid, _ = raster.value_grid(day, px, C.LAYERS["Chance of rain (>= 1 mm)"], "forecast")
    assert grid.shape == (3, 2)
    assert grid[0].tolist() == [0.7, 1.0]  # northernmost row (lat 0.125)
    assert grid[-1].tolist() == [0.0, 0.1]


def test_cells_png_has_one_block_per_pixel_and_brighter_for_higher_values():
    day, px = tiny_day()
    img = np.array(Image.open(BytesIO(raster.render_png(day, px, C.LAYERS["Chance of rain (>= 1 mm)"], "forecast", "cells"))))
    s = raster.SCALE
    assert img.shape == (3 * s, 2 * s, 4)
    low, high = img[-1, 0].astype(int), img[0, -1].astype(int)  # p = 0.0 (south-west) and 1.0 (north-east)
    assert high[:3].sum() > low[:3].sum() and high[3] > low[3]  # brighter and more opaque
    assert (img[:s, :s] == img[0, 0]).all()  # a crisp block


def test_colours_do_not_saturate_at_the_top_of_the_scale():
    pos = raster.class_position(np.array([0.0, 0.7, 0.86, 0.99, 1.0]), C.PROB)
    assert pos[0] == 0 and pos[-1] == pytest.approx(6.0, abs=1e-3)  # top edge is 1.0001
    assert pos[2] < pos[3] < pos[4]  # 86%, 99% and 100% all get different colours
    assert pos[1] == pytest.approx(6 * 6 / 7)  # the 70% class edge sits at its legend tick


def test_dark_ramps_brighten_and_diverging_brightens_outward():
    for r in raster.RAMPS_DARK.values():
        assert all(np.diff(C.lightness(r)) > 0)
    L = C.lightness(raster.DIVERGING_DARK)
    assert all(np.diff(L[:4]) < 0) and all(np.diff(L[3:]) > 0)


def test_raster_grid_index_matches_the_feature_code():
    from src.features.build import grid_index as features_grid_index
    v = np.array([0.025, 0.125, 0.075, 0.025, -0.025])
    assert np.array_equal(raster.grid_index(v), features_grid_index(v))


def test_bounds_cover_the_outer_pixel_edges():
    _, px = tiny_day()
    (w, n), _, (e, s), _ = raster.bounds(px)
    assert (w, e) == pytest.approx((35.0, 35.1)) and (s, n) == pytest.approx((0.0, 0.15))


# ---------------------------------------------------------------- server (real data, signed in)
pytestmark_data = pytest.mark.skipif(not DATA_READY, reason="app/data not built")


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    from starlette.testclient import TestClient
    from web.auth import AuthStore
    from web.server import create_app
    from web.settings import Settings
    db = tmp_path_factory.mktemp("auth") / "users.db"
    store = AuthStore(db)
    store.add_user("tester", "correct horse battery", "viewer")
    c = TestClient(create_app(Settings("127.0.0.1", 0, db, False), store))
    r = c.post("/login", data={"username": "tester", "password": "correct horse battery"},
               headers={"Origin": "http://testserver"}, follow_redirects=False)
    assert r.status_code == 303
    yield c
    store.close()


def test_default_settings_listen_on_localhost_only(monkeypatch):
    from web.settings import settings
    monkeypatch.delenv("HL_HOST", raising=False)
    assert settings().host == "127.0.0.1"


@pytestmark_data
def test_meta_day_and_pixel_endpoints(client):
    r = client.get("/api/meta")
    meta = r.json()
    assert r.status_code == 200 and "application/json" in r.headers["content-type"]
    assert set(meta["sites"]) == {"nakuru", "busia"}
    issue = meta["sites"]["busia"]["issue_min"]
    day = client.get(f"/api/day/busia/{issue}").json()
    assert len(day["pixels"]) == meta["sites"]["busia"]["n_pixels"] and set(day["horizons"]) == set("1234567")
    ref = C.issue_day("busia", issue).set_index(["pixel_id", "horizon"])
    k = 7
    pid = day["pixels"][k]
    assert day["horizons"]["3"]["p_rain1"][k] == pytest.approx(float(ref.loc[(pid, 3), "p_rain1"]), abs=1e-4)
    s = client.get(f"/api/pixel/busia/{pid}?h=2").json()
    assert len(s["target"]) == len(s["rh"]) > 500


@pytestmark_data
def test_raster_and_tables(client):
    meta = client.get("/api/meta").json()
    issue = meta["sites"]["nakuru"]["issue_min"]
    r = client.get(f"/api/raster/nakuru/{issue}/1/rain1/forecast.png?mode=cells")
    assert r.status_code == 200 and r.headers["content-type"] == "image/png"
    assert Image.open(BytesIO(r.content)).size[0] > 0
    assert r.headers["cache-control"].startswith("private")  # signed-in content is never cached by proxies
    for name in ("skill", "calibration", "blight_skill"):
        r = client.get(f"/api/{name}")
        assert r.status_code == 200 and len(r.json()) > 0


@pytestmark_data
@pytest.mark.parametrize("path", [
    "/api/day/mars/2025-04-15", "/api/day/busia/1999-01-01", "/api/day/busia/not-a-date",
    "/api/raster/busia/2025-04-15/9/rain1/forecast.png", "/api/raster/busia/2025-04-15/1/blight_hutton/forecast.png",
    "/api/raster/nakuru/2025-04-15/1/rain1/sideways.png", "/api/pixel/busia/nonexistent?h=1",
])
def test_bad_inputs_are_rejected(client, path):
    r = client.get(path)
    assert r.status_code == 400 and "error" in r.json()


@pytestmark_data
@pytest.mark.parametrize("path", ["/../server.py", "/%2e%2e/server.py", "/..%2f..%2fREADME.md", "/nope.js",
                                  "/%2e%2e/%2e%2e/data/auth/users.db"])
def test_static_files_only_from_the_static_folder(client, path):
    r = client.get(path)
    assert r.status_code == 404 and b"def " not in r.content and b"SQLite" not in r.content


@pytestmark_data
def test_front_end_files_are_served(client):
    for path, kind in [("/", "text/html"), ("/app.js", "text/javascript"), ("/panels.js", "text/javascript"),
                       ("/app.css", "text/css")]:
        r = client.get(path)
        assert r.status_code == 200 and kind in r.headers["content-type"], path
