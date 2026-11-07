"""Dashboard server: front end + JSON/PNG API over app/data/, with sign-in (Starlette on uvicorn).

Run from the repo root (after `python -m app.build_data` and creating a first account with
`python -m web.users add <name> --role admin`):

    python -m web.server                                   # laptop: http://127.0.0.1:8765
    uvicorn web.server:create_app --factory --proxy-headers --host 0.0.0.0 --port 8765   # hosted, behind HTTPS

Security (see web/auth.py for passwords and sessions):
- every page and API route requires a signed-in session, except /login, its two assets and /healthz;
  unauthenticated API calls get 401, pages are redirected to /login;
- POST requests must come from this site (Origin check) and logout needs the session's CSRF token;
- login attempts are rate-limited per IP address; accounts lock after repeated failures;
- the admin-only route /api/admin/users answers 403 for viewers;
- every response carries security headers (Content-Security-Policy limited to this site, the pinned
  CDNs and the CARTO basemap; no framing; no sniffing; HSTS when served over HTTPS);
- every input is checked against a fixed list (sites, layers, views, horizons, dates in range, known
  pixel ids); static files are served only from web/static/.

API
    /api/me                                       signed-in user, role and CSRF token
    /api/meta                                     sites, date ranges, map bounds, layers (breaks, ramps)
    /api/pixels/<site>                            pixel ids, coordinates, elevation
    /api/day/<site>/<YYYY-MM-DD>                  every pixel x horizon for one issue day
    /api/raster/<site>/<date>/<h>/<layer>/<view>.png?mode=smooth|cells
    /api/pixel/<site>/<pixel_id>?h=<1-7>          one pixel's test-period series at one horizon
    /api/why/<site>/<pixel_id>/<YYYY-MM-DD>       SHAP explanation of one square's forecasts (all horizons)
    /api/shap_summary                             what drives the forecasts overall (validation SHAP)
    /api/skill, /api/calibration, /api/blight_skill   committed result tables
    /api/admin/users                              (admin) accounts and recent sign-in events
"""
import hmac
import json
import math
import mimetypes
import threading
from collections import OrderedDict
from functools import lru_cache
from pathlib import Path

import pandas as pd
from starlette.applications import Starlette
from starlette.concurrency import run_in_threadpool
from starlette.middleware import Middleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse, PlainTextResponse, RedirectResponse, Response
from starlette.routing import Route

from app import components as C
from web import raster
from web.auth import ABSOLUTE_SECONDS, AuthError, AuthStore, RateLimiter
from web.settings import Settings, settings

STATIC = Path(__file__).resolve().parent / "static"
PUBLIC = {"/login", "/login.css", "/login.js", "/healthz"}
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
CSP = "; ".join([
    "default-src 'self'",
    "script-src 'self' https://unpkg.com https://cdn.jsdelivr.net",
    "style-src 'self' 'unsafe-inline' https://unpkg.com",  # inline styles: chart and map libraries set them
    "img-src 'self' data: blob: https://basemaps.cartocdn.com https://*.basemaps.cartocdn.com",
    "connect-src 'self' https://basemaps.cartocdn.com https://*.basemaps.cartocdn.com https://cdn.jsdelivr.net https://unpkg.com",
    "worker-src 'self' blob:",
    "font-src 'self'",
    "object-src 'none'",
    "base-uri 'self'",
    "form-action 'self'",
    "frame-ancestors 'none'",
])


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


# ---------------------------------------------------------------- API data
def api_meta():
    names = {"forecast": "Forecast", "observed": "Observed", "error": "Forecast minus observed",
             "normal": "Climatology (normal for the date)"}
    dummy = pd.DataFrame({c: [0.0] for c in DAY_COLUMNS + BLIGHT_COLUMNS})
    layers = {}
    for lid, key in LAYER_IDS.items():
        layer = C.LAYERS[key]
        views = {}
        for v in VIEWS:
            _, breaks, _, unit = C.view_values(dummy, layer, names[v])
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


@lru_cache(maxsize=1)
def shap_meta() -> dict:
    return json.loads((C.DATA / "shap_meta.json").read_text(encoding="utf-8"))


def api_why(site, pixel_id, issue):
    """Plain-language SHAP explanation of every forecast for one square and issue day (all horizons).

    Contributions are percentage points for chances (rain, late blight) and % for humidity; they add
    up to forecast minus the typical forecast (see app/build_shap.py)."""
    site = check_site(site)
    issue = check_date(site, issue)
    if pixel_id not in set(pixels(site)["pixel_id"]):
        raise BadRequest(f"unknown pixel '{pixel_id}'")
    m = shap_meta()
    rows = pd.read_parquet(C.DATA / f"{site}_shap.parquet",
                           filters=[("issue_date", "==", pd.Timestamp(issue)), ("pixel_id", "==", pixel_id)])
    out = {"groups": [{"key": g, "label": m["labels"][g]} for g in m["groups"]], "targets": {}}
    for target in m["sites"][site]["targets"]:
        per_h = {}
        for _, r in rows.iterrows():
            h = int(r["horizon"])
            contrib = {g: float(r[f"{target}|{g}"]) / 10 for g in m["groups"]}  # stored in tenths of a point
            typical = m["sites"][site]["typical"][f"{target}|{h}"]
            per_h[str(h)] = {"typical": round(typical, 2), "forecast": round(typical + sum(contrib.values()), 2),
                             "contributions": contrib}
        out["targets"][target] = per_h
    return out


def api_shap_summary():
    g = pd.read_csv(C.REPORTS / "shap_groups_validation.csv")
    g = g[["site", "horizon", "task", "group", "overall_share", "within_day_share"]]
    return {"labels": C.SHAP_LABELS, "groups": json.loads(g.to_json(orient="records"))}


def table(path, query=None):
    df = pd.read_csv(C.REPORTS / path)
    if query is not None:
        df = df.query(query)
    return json.loads(df.to_json(orient="records"))


# ---------------------------------------------------------------- HTTP helpers
def client_ip(request: Request) -> str:
    return request.client.host if request.client else ""


def same_origin(request: Request) -> bool:
    """True if a state-changing request comes from this site (browsers always send Origin on POST)."""
    origin = request.headers.get("origin")
    if not origin:
        return False
    return origin == f"{request.url.scheme}://{request.url.netloc}"


def json_ok(obj) -> JSONResponse:
    return JSONResponse(obj, headers={"Cache-Control": "no-store"})


def json_error(message: str, status: int) -> JSONResponse:
    return JSONResponse({"error": message}, status_code=status, headers={"Cache-Control": "no-store"})


def guarded(fn):
    """Run an API handler, turning bad input into a 400 response."""
    def endpoint(request: Request):
        try:
            return fn(request)
        except BadRequest as e:
            return json_error(str(e), 400)
        except (KeyError, IndexError):
            return json_error("bad request", 400)
    return endpoint


# ---------------------------------------------------------------- routes: data
@guarded
def r_meta(request):
    return json_ok(api_meta())


@guarded
def r_pixels(request):
    return json_ok(api_pixels(request.path_params["site"]))


@guarded
def r_day(request):
    p = request.path_params
    return json_ok(api_day(p["site"], p["issue"]))


@guarded
def r_pixel(request):
    p = request.path_params
    return json_ok(api_pixel(p["site"], p["pixel_id"], request.query_params.get("h", "1")))


@guarded
def r_raster(request):
    p = request.path_params
    if not p["viewfile"].endswith(".png"):
        raise BadRequest("expected .png")
    site = check_site(p["site"])
    issue, h = check_date(site, p["issue"]), check_h(p["h"])
    layer_id, view = check_layer(site, p["layer"]), p["viewfile"][:-4]
    if view not in VIEWS:
        raise BadRequest(f"unknown view '{view}'")
    mode = request.query_params.get("mode", "smooth")
    if mode not in ("smooth", "cells"):
        raise BadRequest("mode must be smooth or cells")
    return Response(png(site, issue, h, layer_id, view, mode), media_type="image/png",
                    headers={"Cache-Control": "private, max-age=3600"})


@guarded
def r_why(request):
    p = request.path_params
    return json_ok(api_why(p["site"], p["pixel_id"], p["issue"]))


def r_shap_summary(request):
    return json_ok(api_shap_summary())


def r_skill(request):
    return json_ok(table("test/skill.csv", "model == 'catboost' and metric in ['bss', 'rmse_skill']"))


def r_calibration(request):
    return json_ok(table("test/calibration.csv", "model == 'catboost'"))


def r_blight_skill(request):
    return json_ok(table("late_blight/test_skill.csv"))


# ---------------------------------------------------------------- routes: account
def r_me(request):
    s = request.state.session
    return json_ok({"username": s.user.username, "role": s.user.role, "csrf": s.csrf})


def r_admin_users(request):
    s = request.state.session
    if s.user.role != "admin":
        return json_error("admins only", 403)
    store: AuthStore = request.app.state.store
    return json_ok({"users": store.list_users(), "events": store.recent_audit(25)})


def r_login_page(request):
    if request.state.session is not None:
        return RedirectResponse("/", status_code=303)
    return static_file(STATIC / "login.html")


async def r_login(request: Request):
    app = request.app
    if not same_origin(request):
        return PlainTextResponse("cross-site request refused", status_code=403)
    ip = client_ip(request)
    if not app.state.limiter.allow(ip):
        app.state.store.audit("login_rate_limited", None, ip)
        return RedirectResponse("/login?error=rate", status_code=303)
    form = await request.form()
    username, password = str(form.get("username", ""))[:64], str(form.get("password", ""))[:256]
    try:
        user = await run_in_threadpool(app.state.store.authenticate, username, password, ip)
    except AuthError as e:
        return RedirectResponse(f"/login?error={e.code}", status_code=303)
    token, _ = await run_in_threadpool(app.state.store.create_session, user)  # a new token at every login
    cfg: Settings = app.state.settings
    resp = RedirectResponse("/", status_code=303)
    resp.set_cookie(cfg.cookie_name, token, max_age=ABSOLUTE_SECONDS, path="/", httponly=True,
                    samesite="strict", secure=cfg.secure_cookies)
    return resp


async def r_logout(request: Request):
    s = request.state.session
    if not same_origin(request):
        return PlainTextResponse("cross-site request refused", status_code=403)
    if not hmac.compare_digest(request.headers.get("x-csrf-token", ""), s.csrf):
        return json_error("missing or wrong CSRF token", 403)
    cfg: Settings = request.app.state.settings
    await run_in_threadpool(request.app.state.store.delete_session, request.cookies.get(cfg.cookie_name), client_ip(request))
    resp = json_ok({"ok": True})
    resp.delete_cookie(cfg.cookie_name, path="/", httponly=True, samesite="strict", secure=cfg.secure_cookies)
    return resp


def r_health(request):
    return PlainTextResponse("ok")


# ---------------------------------------------------------------- routes: static files
def static_file(f: Path) -> Response:
    ctype = "text/javascript" if f.suffix == ".js" else (mimetypes.guess_type(f.name)[0] or "application/octet-stream")
    return FileResponse(f, media_type=ctype, headers={"Cache-Control": "no-cache"})


def r_static(request):
    rel = request.path_params.get("path") or "index.html"
    f = (STATIC / rel).resolve()
    if STATIC not in f.parents or not f.is_file():  # only files inside web/static/
        return json_error("not found", 404)
    return static_file(f)


# ---------------------------------------------------------------- middleware
class SecurityMiddleware(BaseHTTPMiddleware):
    """Require a session for everything except PUBLIC paths; add security headers to every response."""

    async def dispatch(self, request: Request, call_next):
        app = request.app
        token = request.cookies.get(app.state.settings.cookie_name)
        request.state.session = await run_in_threadpool(app.state.store.get_session, token)
        path = request.url.path
        if request.state.session is None and path not in PUBLIC:
            if path.startswith("/api/") or request.method != "GET":
                resp = json_error("sign in required", 401)
            else:
                resp = RedirectResponse("/login", status_code=303)
        else:
            resp = await call_next(request)
        h = resp.headers
        h["Content-Security-Policy"] = CSP
        h["X-Content-Type-Options"] = "nosniff"
        h["X-Frame-Options"] = "DENY"
        h["Referrer-Policy"] = "same-origin"
        h["Permissions-Policy"] = "geolocation=(), camera=(), microphone=()"
        h["Cross-Origin-Opener-Policy"] = "same-origin"
        if app.state.settings.secure_cookies:
            h["Strict-Transport-Security"] = "max-age=31536000"
        return resp


def create_app(cfg: Settings | None = None, store: AuthStore | None = None) -> Starlette:
    cfg = cfg or settings()
    routes = [
        Route("/healthz", r_health),
        Route("/login", r_login_page, methods=["GET"]),
        Route("/login", r_login, methods=["POST"]),
        Route("/logout", r_logout, methods=["POST"]),
        Route("/api/me", r_me),
        Route("/api/admin/users", r_admin_users),
        Route("/api/meta", r_meta),
        Route("/api/pixels/{site}", r_pixels),
        Route("/api/day/{site}/{issue}", r_day),
        Route("/api/raster/{site}/{issue}/{h}/{layer}/{viewfile}", r_raster),
        Route("/api/pixel/{site}/{pixel_id}", r_pixel),
        Route("/api/why/{site}/{pixel_id}/{issue}", r_why),
        Route("/api/shap_summary", r_shap_summary),
        Route("/api/skill", r_skill),
        Route("/api/calibration", r_calibration),
        Route("/api/blight_skill", r_blight_skill),
        Route("/api/{rest:path}", lambda request: json_error("not found", 404)),
        Route("/", r_static),
        Route("/{path:path}", r_static),
    ]
    app = Starlette(routes=routes, middleware=[Middleware(SecurityMiddleware)])
    app.state.settings = cfg
    app.state.store = store or AuthStore(cfg.auth_db)
    app.state.limiter = RateLimiter(limit=20, window=600)
    return app


def main() -> None:
    import uvicorn
    cfg = settings()
    app = create_app(cfg)
    if not app.state.store.list_users():
        print("No accounts yet. Create one first:  python -m web.users add <name> --role admin")
    print(f"Dashboard running at http://{cfg.host}:{cfg.port}  (Ctrl+C to stop)", flush=True)
    uvicorn.run(app, host=cfg.host, port=cfg.port, log_level="warning", proxy_headers=True)


if __name__ == "__main__":
    main()
