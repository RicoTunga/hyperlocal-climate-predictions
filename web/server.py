"""Local server for the dashboard: static front end + a small JSON/PNG API over app/data/.

Run from the repo root (after `python -m app.build_data`):

    python -m web.server            # then open the printed address (http://127.0.0.1:8765)

Binds to 127.0.0.1 only, so nothing on the network can reach it. Every input is checked against
a fixed list (sites, layers, views, horizons, dates in range, known pixel ids); static files are
served only from web/static/.

API
    /api/meta                                     sites, date ranges, map bounds, layers (breaks, ramps)
    /api/pixels/<site>                            pixel ids, coordinates, elevation
    /api/day/<site>/<YYYY-MM-DD>                  every pixel x horizon for one issue day (for hover/drawer)
    /api/raster/<site>/<date>/<h>/<layer>/<view>.png?mode=smooth|cells
    /api/pixel/<site>/<pixel_id>?h=<1-7>          one pixel's test-period series at one horizon
    /api/skill, /api/calibration, /api/blight_skill   committed result tables as JSON
"""
import argparse
import json
import math
import mimetypes
import threading
from collections import OrderedDict
from functools import lru_cache
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pandas as pd

from app import components as C
from web import raster

STATIC = Path(__file__).resolve().parent / "static"
SITES = ("nakuru", "busia")
VIEWS = ("forecast", "observed", "error", "normal")
LAYER_IDS = {  # url id -> components.LAYERS key
    "rain1": "Chance of rain (>= 1 mm)", "rain10": "Chance of heavy rain (>= 10 mm)",
    "rain25": "Chance of very heavy rain (>= 25 mm)", "rainmm": "Expected rain (mm)",
    "rh": "Humidity (%)", "blight_hutton": "Late-blight risk (Hutton)",
    "blight_humid": "Late-blight risk (humidity only)",
}
DAY_COLUMNS = ["p_rain1", "p_rain10", "p_rain25", "rain_mm", "rh", "clim_rain1", "clim_rain10", "clim_rain25",
               "clim_rain_mm", "clim_rh", "obs_rain_mm", "obs_rh"]
BLIGHT_COLUMNS = ["blight_hutton", "obs_blight_hutton", "clim_blight_hutton", "blight_humid", "obs_blight_humid",
                  "clim_blight_humid"]


class BadRequest(Exception):
    pass


# ---------------------------------------------------------------- data (cached)
@lru_cache(maxsize=1)
def meta() -> dict:
    return C.meta()


@lru_cache(maxsize=2)
def pixels(site: str) -> pd.DataFrame:
    return C.pixels(site)


@lru_cache(maxsize=8)
def day(site: str, issue: str) -> pd.DataFrame:
    return C.issue_day(site, issue)


_png_cache: "OrderedDict[tuple, bytes]" = OrderedDict()
_png_lock = threading.Lock()


def png(site, issue, h, layer_id, view, mode) -> bytes:
    key = (site, issue, h, layer_id, view, mode)
    with _png_lock:
        if key in _png_cache:
            _png_cache.move_to_end(key)
            return _png_cache[key]
    d = day(site, issue)
    data = raster.render_png(d[d["horizon"] == h].reset_index(drop=True), pixels(site),
                             C.LAYERS[LAYER_IDS[layer_id]], view, mode)
    with _png_lock:
        _png_cache[key] = data
        while len(_png_cache) > 200:
            _png_cache.popitem(last=False)
    return data


# ---------------------------------------------------------------- validation
def check_site(site):
    if site not in SITES:
        raise BadRequest(f"unknown site '{site}'")
    return site


def check_date(site, s):
    try:
        d = pd.Timestamp(s)
    except (ValueError, TypeError):
        raise BadRequest(f"bad date '{s}'") from None
    m = meta()[site]
    if not (pd.Timestamp(m["issue_min"]) <= d <= pd.Timestamp(m["issue_max"])):
        raise BadRequest(f"date outside {m['issue_min']}..{m['issue_max']}")
    return str(d.date())


def check_h(h):
    try:
        h = int(h)
    except (ValueError, TypeError):
        raise BadRequest("horizon must be 1-7") from None
    if not 1 <= h <= 7:
        raise BadRequest("horizon must be 1-7")
    return h


def check_layer(site, layer_id):
    if layer_id not in LAYER_IDS or (layer_id.startswith("blight") and not meta()[site]["has_blight"]):
        raise BadRequest(f"unknown layer '{layer_id}' for {site}")
    return layer_id


def clean(v):
    """JSON-safe number (NaN -> None), rounded to keep responses small."""
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    return round(float(v), 4)


# ---------------------------------------------------------------- API
def api_meta():
    layers = {}
    for lid, key in LAYER_IDS.items():
        layer = C.LAYERS[key]
        views = {}
        for v in VIEWS:
            names = {"forecast": "Forecast", "observed": "Observed", "error": "Forecast minus observed",
                     "normal": "Climatology (normal for the date)"}
            _, breaks, _, unit = C.view_values(pd.DataFrame({c: [0.0] for c in DAY_COLUMNS + BLIGHT_COLUMNS}),
                                               layer, names[v])
            views[v] = {"breaks": [clean(b) for b in raster.finite_breaks(breaks)], "unit": unit,
                        "ramp": raster.ramp_for(layer, v)}
        layers[lid] = {"label": layer.label, "forecast": layer.forecast, "clim": layer.clim,
                       "observed": layer.observed, "unit": layer.unit, "blight": layer.blight,
                       "observed_label": layer.observed_label, "views": views}
    sites = {}
    for s in SITES:
        px = pixels(s)
        sites[s] = {**meta()[s], "bounds": raster.bounds(px), "center": [float(px["lon"].mean()), float(px["lat"].mean())],
                    "lat0": float(px["lat"].min()), "lon0": float(px["lon"].min()), "step": 2 * C.PIXEL_HALF}
    return {"sites": sites, "layers": layers, "views": list(VIEWS)}


def api_pixels(site):
    px = pixels(check_site(site))
    return {"ids": list(px["pixel_id"]), "lat": [clean(x) for x in px["lat"]], "lon": [clean(x) for x in px["lon"]],
            "elev": [round(float(x)) for x in px["elev_mean_m"]]}


def api_day(site, issue):
    site = check_site(site)
    issue = check_date(site, issue)
    d = day(site, issue)
    order = list(pixels(site)["pixel_id"])
    cols = DAY_COLUMNS + (BLIGHT_COLUMNS if meta()[site]["has_blight"] else [])
    out = {"issue": issue, "pixels": order, "horizons": {}}
    for h, g in d.groupby("horizon"):
        g = g.set_index("pixel_id").reindex(order)
        out["horizons"][str(int(h))] = {"target": str(pd.Timestamp(g["target_date"].iloc[0]).date()),
                                        **{c: [clean(v) for v in g[c]] for c in cols}}
    return out


def api_pixel(site, pixel_id, h):
    site, h = check_site(site), check_h(h)
    if pixel_id not in set(pixels(site)["pixel_id"]):
        raise BadRequest(f"unknown pixel '{pixel_id}'")
    cols = ["p_rain1", "obs_rain_mm", "clim_rain1", "rh", "obs_rh", "clim_rh"]
    if meta()[site]["has_blight"]:
        cols += ["blight_hutton", "obs_blight_hutton"]
    s = C.pixel_series(site, pixel_id, h, cols).sort_values("target_date")
    return {"target": [str(pd.Timestamp(t).date()) for t in s["target_date"]], **{c: [clean(v) for v in s[c]] for c in cols}}


def table(path, query=None):
    df = pd.read_csv(C.REPORTS / path)
    if query is not None:
        df = df.query(query)
    return json.loads(df.to_json(orient="records"))


ROUTES_JSON = {
    "meta": lambda parts, q: api_meta(),
    "pixels": lambda parts, q: api_pixels(parts[0]),
    "day": lambda parts, q: api_day(parts[0], parts[1]),
    "pixel": lambda parts, q: api_pixel(parts[0], parts[1], q.get("h", ["1"])[0]),
    "skill": lambda parts, q: table("test/skill.csv", "model == 'catboost' and metric in ['bss', 'rmse_skill']"),
    "calibration": lambda parts, q: table("test/calibration.csv", "model == 'catboost'"),
    "blight_skill": lambda parts, q: table("late_blight/test_skill.csv"),
}


class Handler(BaseHTTPRequestHandler):
    server_version = "HyperlocalDashboard/1.0"

    def log_message(self, fmt, *args):  # keep the console quiet
        pass

    def handle(self):
        try:
            super().handle()
        except (ConnectionAbortedError, BrokenPipeError, ConnectionResetError):
            pass

    def send(self, status, body: bytes, ctype: str, cache: bool = False):
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "max-age=3600" if cache else "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        try:
            self.wfile.write(body)
        except (ConnectionAbortedError, BrokenPipeError, ConnectionResetError):
            pass  # the browser cancelled the request (e.g. the map moved on to a newer image)

    def send_json(self, obj, status=HTTPStatus.OK):
        self.send(status, json.dumps(obj, separators=(",", ":")).encode(), "application/json")

    def do_GET(self):
        url = urlparse(self.path)
        parts = [p for p in url.path.split("/") if p]
        try:
            if parts[:1] == ["api"] and len(parts) >= 2:
                name, args, q = parts[1], parts[2:], parse_qs(url.query)
                if name == "raster" and len(args) == 5 and args[4].endswith(".png"):
                    site = check_site(args[0])
                    issue, h = check_date(site, args[1]), check_h(args[2])
                    layer_id, view = check_layer(site, args[3]), args[4][:-4]
                    if view not in VIEWS:
                        raise BadRequest(f"unknown view '{view}'")
                    mode = q.get("mode", ["smooth"])[0]
                    if mode not in ("smooth", "cells"):
                        raise BadRequest("mode must be smooth or cells")
                    return self.send(HTTPStatus.OK, png(site, issue, h, layer_id, view, mode), "image/png", cache=True)
                if name in ROUTES_JSON:
                    return self.send_json(ROUTES_JSON[name](args, q))
                return self.send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)
            return self.serve_static(url.path)
        except BadRequest as e:
            return self.send_json({"error": str(e)}, HTTPStatus.BAD_REQUEST)
        except (IndexError, KeyError):
            return self.send_json({"error": "bad request"}, HTTPStatus.BAD_REQUEST)

    def serve_static(self, path):
        rel = "index.html" if path in ("", "/") else path.lstrip("/")
        f = (STATIC / rel).resolve()
        if STATIC not in f.parents or not f.is_file():  # only files inside web/static/
            return self.send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)
        ctype = mimetypes.guess_type(f.name)[0] or "application/octet-stream"
        if f.suffix == ".js":
            ctype = "text/javascript"
        return self.send(HTTPStatus.OK, f.read_bytes(), ctype)


def make_server(port: int = 8765) -> ThreadingHTTPServer:
    return ThreadingHTTPServer(("127.0.0.1", port), Handler)


def main() -> None:
    parser = argparse.ArgumentParser(description="Local dashboard server (127.0.0.1 only).")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    httpd = make_server(args.port)
    print(f"Dashboard running at http://127.0.0.1:{args.port}  (Ctrl+C to stop)", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()


if __name__ == "__main__":
    main()
