"""Tests for the dashboard server and colour-field rendering.

Raster tests use synthetic data; server tests need app/data/ (python -m app.build_data) and are
skipped without it. The server is started on a free port in a background thread.
"""
import json
import socket
import threading
import urllib.error
import urllib.request
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


def test_dark_ramps_brighten_and_diverging_brightens_outward():
    for r in raster.RAMPS_DARK.values():
        assert all(np.diff(C.lightness(r)) > 0)
    L = C.lightness(raster.DIVERGING_DARK)
    assert all(np.diff(L[:4]) < 0) and all(np.diff(L[3:]) > 0)


def test_bounds_cover_the_outer_pixel_edges():
    _, px = tiny_day()
    (w, n), _, (e, s), _ = raster.bounds(px)
    assert (w, e) == pytest.approx((35.0, 35.1)) and (s, n) == pytest.approx((0.0, 0.15))


# ---------------------------------------------------------------- server (real data)
@pytest.fixture(scope="module")
def base():
    from web.server import make_server
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    httpd = make_server(port)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{port}"
    httpd.shutdown()


def get(url):
    try:
        with urllib.request.urlopen(url, timeout=60) as r:
            return r.status, r.headers.get("Content-Type"), r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.headers.get("Content-Type"), e.read()


pytestmark_data = pytest.mark.skipif(not DATA_READY, reason="app/data not built")


@pytestmark_data
def test_server_binds_to_localhost_only():
    from web.server import make_server
    httpd = make_server(0)
    try:
        assert httpd.server_address[0] == "127.0.0.1"
    finally:
        httpd.server_close()


@pytestmark_data
def test_meta_day_and_pixel_endpoints(base):
    st, ct, body = get(f"{base}/api/meta")
    meta = json.loads(body)
    assert st == 200 and "application/json" in ct and set(meta["sites"]) == {"nakuru", "busia"}
    issue = meta["sites"]["busia"]["issue_min"]
    st, _, body = get(f"{base}/api/day/busia/{issue}")
    day = json.loads(body)
    assert st == 200 and len(day["pixels"]) == meta["sites"]["busia"]["n_pixels"] and set(day["horizons"]) == set("1234567")
    # values equal the replay file for one pixel
    ref = C.issue_day("busia", issue).set_index(["pixel_id", "horizon"])
    k = 7
    pid = day["pixels"][k]
    assert day["horizons"]["3"]["p_rain1"][k] == pytest.approx(float(ref.loc[(pid, 3), "p_rain1"]), abs=1e-4)
    st, _, body = get(f"{base}/api/pixel/busia/{pid}?h=2")
    s = json.loads(body)
    assert st == 200 and len(s["target"]) == len(s["rh"]) > 500


@pytestmark_data
def test_raster_and_tables(base):
    meta = json.loads(get(f"{base}/api/meta")[2])
    issue = meta["sites"]["nakuru"]["issue_min"]
    st, ct, body = get(f"{base}/api/raster/nakuru/{issue}/1/rain1/forecast.png?mode=cells")
    assert st == 200 and ct == "image/png" and Image.open(BytesIO(body)).size[0] > 0
    for name in ("skill", "calibration", "blight_skill"):
        st, _, body = get(f"{base}/api/{name}")
        assert st == 200 and len(json.loads(body)) > 0


@pytestmark_data
@pytest.mark.parametrize("path", [
    "/api/day/mars/2025-04-15", "/api/day/busia/1999-01-01", "/api/day/busia/not-a-date",
    "/api/raster/busia/2025-04-15/9/rain1/forecast.png", "/api/raster/busia/2025-04-15/1/blight_hutton/forecast.png",
    "/api/raster/nakuru/2025-04-15/1/rain1/sideways.png", "/api/pixel/busia/nonexistent?h=1",
])
def test_bad_inputs_are_rejected(base, path):
    st, ct, body = get(base + path)
    assert st == 400 and "error" in json.loads(body)


@pytestmark_data
@pytest.mark.parametrize("path", ["/../server.py", "/%2e%2e/server.py", "/..%2f..%2fREADME.md", "/nope.js"])
def test_static_files_only_from_the_static_folder(base, path):
    st, _, body = get(base + path)
    assert st == 404 and b"def " not in body


@pytestmark_data
def test_front_end_files_are_served(base):
    for path, kind in [("/", "text/html"), ("/app.js", "text/javascript"), ("/panels.js", "text/javascript"),
                       ("/app.css", "text/css")]:
        st, ct, _ = get(base + path)
        assert st == 200 and kind in ct, path
