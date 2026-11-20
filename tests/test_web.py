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


SHAP_READY = (Path(__file__).resolve().parents[1] / "app" / "data" / "shap_meta.json").exists()


@pytest.mark.skipif(not (DATA_READY and SHAP_READY), reason="app/data SHAP files not built")
def test_why_explanations_add_up_and_match_the_forecast(client):
    meta = client.get("/api/meta").json()
    issue = meta["sites"]["nakuru"]["issue_min"]
    pid = client.get("/api/pixels/nakuru").json()["ids"][100]
    why = client.get(f"/api/why/nakuru/{pid}/{issue}").json()
    assert {g["key"] for g in why["groups"]} and set(why["targets"]) == {"rain1", "rh", "blight_hutton", "blight_humid"}
    day = client.get(f"/api/day/nakuru/{issue}").json()
    k = day["pixels"].index(pid)
    for h in ("1", "4", "7"):
        e = why["targets"]["rain1"][h]
        assert sum(e["contributions"].values()) == pytest.approx(e["forecast"] - e["typical"], abs=0.5)  # tenths-of-a-point rounding
        assert e["forecast"] == pytest.approx(100 * day["horizons"][h]["p_rain1"][k], abs=0.5)
        r = why["targets"]["rh"][h]
        assert r["forecast"] == pytest.approx(day["horizons"][h]["rh"][k], abs=0.5)


@pytest.mark.skipif(not (DATA_READY and SHAP_READY), reason="app/data SHAP files not built")
def test_why_rejects_bad_inputs_and_needs_sign_in(client):
    meta = client.get("/api/meta").json()
    issue = meta["sites"]["busia"]["issue_min"]
    assert client.get(f"/api/why/busia/nonexistent/{issue}").status_code == 400
    assert client.get("/api/why/busia/0.025_34.025/1999-01-01").status_code == 400
    from starlette.testclient import TestClient
    anon = TestClient(client.app)
    assert anon.get(f"/api/why/busia/0.025_34.025/{issue}").status_code == 401


@pytestmark_data
def test_shap_summary_has_plain_labels(client):
    s = client.get("/api/shap_summary").json()
    assert s["labels"]["area-wide"] == "Rain and humidity across the area"
    assert {"site", "horizon", "task", "group", "overall_share", "within_day_share"} <= set(s["groups"][0])


# ---------------------------------------------------------------- tomato late-blight layer
def test_tomato_layer_is_the_potato_forecast_in_its_own_colours():
    potato, tomato = C.LAYERS["Late-blight risk (Hutton)"], C.LAYERS["Tomato late-blight risk (potato model)"]
    # same columns: no tomato-specific model exists, so the values must be identical
    assert (tomato.forecast, tomato.observed, tomato.clim) == (potato.forecast, potato.observed, potato.clim)
    assert tomato.crop == "tomato" and tomato.blight
    assert raster.ramp_for(tomato, "forecast") != raster.ramp_for(potato, "forecast")
    assert raster.ramp_for(tomato, "error") == raster.ramp_for(potato, "error")
    lum = C.lightness(raster.RAMPS_DARK["tomato"])
    assert lum == sorted(lum)  # dark-mode ramp brightens with the value, like the others


@pytestmark_data
def test_tomato_layer_served_for_nakuru_only(client):
    meta = client.get("/api/meta").json()
    assert meta["layers"]["blight_tomato"]["crop"] == "tomato"
    assert meta["layers"]["blight_tomato"]["forecast"] == meta["layers"]["blight_hutton"]["forecast"]
    issue = meta["sites"]["nakuru"]["issue_min"]
    t = client.get(f"/api/raster/nakuru/{issue}/2/blight_tomato/forecast.png?mode=cells")
    p = client.get(f"/api/raster/nakuru/{issue}/2/blight_hutton/forecast.png?mode=cells")
    assert t.status_code == 200 and t.content != p.content  # different colours ...
    ta, pa = (np.array(Image.open(BytesIO(r.content)))[..., 3] for r in (t, p))
    assert (ta == pa).all()  # ... over exactly the same values (opacity follows the value)
    assert client.get(f"/api/raster/busia/{issue}/2/blight_tomato/forecast.png").status_code == 400


# ---------------------------------------------------------------- post-test period and data labels
STATUS = {"test_end": "2026-09-30", "observed_through": "2026-11-14", "chirps_final_through": "2026-10-31",
          "posttest_start": "2026-10-01", "era5t_from": "2026-08-01"}


@pytest.mark.parametrize("day, period, rain, era5, note_has", [
    ("2025-04-15", "test", "final", "final", ""),
    ("2026-08-10", "test", "final", "era5t", ""),
    ("2026-10-05", "posttest", "final", "era5t", "not part of the evaluated results"),
    ("2026-11-03", "posttest", "preliminary", "era5t", "preliminary"),
    ("2026-11-17", "forecast", "none", "none", "Not observed yet"),
])
def test_data_label_says_what_kind_of_day_it_is(day, period, rain, era5, note_has):
    lab = C.data_label(day, STATUS)
    assert (lab["period"], lab["rain"], lab["era5"]) == (period, rain, era5)
    assert note_has in lab["note"] and (lab["note"] == "") == (note_has == "")


@pytestmark_data
def test_day_endpoint_labels_post_test_and_unobserved_days(client):
    meta = client.get("/api/meta").json()["sites"]["nakuru"]
    st = meta.get("status")
    if not st or meta["issue_max"] <= st["test_end"]:
        pytest.skip("no post-test period in app/data")
    newest = client.get(f"/api/day/nakuru/{meta['issue_max']}").json()
    for h, H in newest["horizons"].items():  # forecasts from the newest data reach past the last observed day
        assert H["status"]["period"] == "forecast" and all(v is None for v in H["obs_rain_mm"]), h
        assert all(v is not None for v in H["p_rain1"])
    first = client.get(f"/api/day/nakuru/{meta['issue_min']}").json()
    assert all(H["status"]["period"] == "test" for H in first["horizons"].values())
    pid = newest["pixels"][0]
    adv = client.get(f"/api/advice/nakuru/{pid}/{meta['issue_max']}?crops=potato").json()
    assert all(d["status"]["period"] == "forecast" and d["words"] for d in adv["outlook"]["days"])
    s = client.get(f"/api/pixel/nakuru/{pid}?h=1").json()
    assert s["test_end"] == st["test_end"] and s["target"][-1] > st["test_end"]


@pytestmark_data
def test_post_test_skill_is_served_separately_from_the_test(client):
    test_rows = client.get("/api/skill").json()
    post = client.get("/api/posttest_skill").json()
    assert {r["period"] for r in test_rows}.isdisjoint({r["period"] for r in post})
    assert all(r["period"].startswith("post-test") for r in post)
    assert client.get("/api/posttest_blight_skill").status_code == 200


@pytestmark_data
def test_farm_view_files_are_served_and_linked(client):
    page = client.get("/").text
    assert 'id="farm"' in page and "/farm.css" in page
    for path in ("/farm.js", "/farm.css"):
        assert client.get(path).status_code == 200, path
