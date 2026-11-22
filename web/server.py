"""Dashboard server: front end + JSON/PNG API over app/data/, with sign-in (Starlette on uvicorn).

Run from the repo root (after `python -m app.build_data` and creating a first account with
`python -m web.users add <name> --role admin`):

    python -m web.server                                   # laptop: http://127.0.0.1:8765
    uvicorn web.server:create_app --factory --proxy-headers --host 0.0.0.0 --port 8765   # hosted, behind HTTPS

Security (see web/auth.py for passwords and sessions):
- every page and API route requires a signed-in session, except the account pages (/login, /signup,
  /forgot, /reset, /verify), their assets and /healthz; unauthenticated API calls get 401, pages are
  redirected to /login;
- POST requests must come from this site (Origin check); every signed-in state-changing request also
  needs the session's CSRF token (X-CSRF-Token header);
- login, sign-up and password-reset requests are rate-limited per IP address; accounts lock after
  repeated failures; the reset page answers the same whether or not an email has an account;
- saved places, crops and settings are always read and written for the signed-in user only (the user id
  comes from the session, never from the request);
- the admin-only route /api/admin/users answers 403 for viewers;
- every response carries security headers (Content-Security-Policy limited to this site, the pinned
  CDNs and the CARTO basemap; no framing; no sniffing; HSTS when served over HTTPS);
- every input is checked against a fixed list (sites, layers, views, horizons, dates in range, known
  pixel ids); static files are served only from web/static/.

API
    /api/me                                       signed-in user, role, CSRF token, account details
    /api/user                                     profile and settings, saved places, crops (GET; POST /api/profile,
                                                  /api/places, /api/places/<id>, DELETE /api/places/<id>, /api/crops)
    /api/account[/password|/sessions|/delete|/verify]   account changes (POST, password needed where it matters)
    /api/crops                                    crop catalogue
    /api/advice/<site>/<pixel_id>/<YYYY-MM-DD>?crops=a,b   7-day outlook and crop advice for one square
    /api/meta                                     sites, date ranges, map bounds, layers (breaks, ramps)
    /api/pixels/<site>                            pixel ids, coordinates, elevation
    /api/day/<site>/<YYYY-MM-DD>                  every pixel x horizon for one issue day
    /api/raster/<site>/<date>/<h>/<layer>/<view>.png?mode=smooth|cells
    /api/pixel/<site>/<pixel_id>?h=<1-7>          one pixel's test-period series at one horizon
    /api/why/<site>/<pixel_id>/<YYYY-MM-DD>       SHAP explanation of one square's forecasts (all horizons)
    /api/shap_summary                             what drives the forecasts overall (validation SHAP)
    /api/skill, /api/calibration, /api/blight_skill   committed result tables (reported test period)
    /api/posttest_skill, /api/posttest_blight_skill   post-test period (after the reported test; [] if not run)
    /api/live_backtest_skill   2024 backtest of live inputs vs normal inputs (only with HL_LIVE=1; else [])
    /api/notifications                            alerts for the saved places from the newest forecast (GET; refreshes
                                                  them first); POST /api/notifications/read {ids|all},
                                                  /api/notifications/prefs {kind: on/off}  (see web/notify.py)
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
from web import crops as crops_mod
from web import raster
from web.auth import ABSOLUTE_SECONDS, AuthError, AuthStore, PolicyError, RateLimiter
from web.mailer import make_mailer, reset_message, verify_message
from web.notify import KIND_NAMES, NotificationStore, NotifyError
from web.notify import detect as detect_alerts
from web.settings import Settings, settings
from web.userdata import USER_TYPES, UserData, UserDataError

STATIC = Path(__file__).resolve().parent / "static"
PUBLIC = {"/login", "/login.css", "/login.js", "/healthz", "/signup", "/signup.js", "/forgot", "/reset",
          "/recover.js", "/verify", "/auth/config", "/favicon.svg"}
MAX_BODY = 16 * 1024
SITES = ("nakuru", "busia")
VIEWS = ("forecast", "observed", "error", "normal")
LAYER_IDS = {  # url id -> components.LAYERS key
    "rain1": "Chance of rain (>= 1 mm)", "rain10": "Chance of heavy rain (>= 10 mm)",
    "rain25": "Chance of very heavy rain (>= 25 mm)", "rainmm": "Expected rain (mm)",
    "rh": "Humidity (%)", "blight_hutton": "Late-blight risk (Hutton)",
    "blight_humid": "Late-blight risk (humidity only)", "blight_tomato": "Tomato late-blight risk (potato model)",
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
                       "observed": layer.observed, "unit": layer.unit, "blight": layer.blight, "crop": layer.crop,
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
    from app import live_data
    out = {"issue": issue, "pixels": order, "horizons": {}, "live": live_data.enabled() and live_data.is_live(site, issue, meta())}
    status = meta()[site].get("status")
    for h, g in d.groupby("horizon"):
        g = g.set_index("pixel_id").reindex(order)
        target = pd.Timestamp(g["target_date"].iloc[0])
        out["horizons"][str(int(h))] = {"target": str(target.date()),
                                        "status": C.data_label(target, status) if status else None,
                                        **{c: [clean(v) for v in g[c]] for c in cols}}
    return out


def api_pixel(site, pixel_id, h):
    site, h = check_site(site), check_h(h)
    if pixel_id not in set(pixels(site)["pixel_id"]):
        raise BadRequest(f"unknown pixel '{pixel_id}'")
    cols = ["p_rain1", "obs_rain_mm", "clim_rain1", "rh", "obs_rh", "clim_rh"]
    if meta()[site]["has_blight"]:
        cols += ["blight_hutton", "obs_blight_hutton"]
    s = C.pixel_series(site, pixel_id, h, cols + ["period"]).sort_values("target_date")
    status = meta()[site].get("status") or {}
    return {"target": [str(pd.Timestamp(t).date()) for t in s["target_date"]],
            "test_end": status.get("test_end"), "observed_through": status.get("observed_through"),
            **{c: [clean(v) for v in s[c]] for c in cols}}


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
    from app import live_data
    if live_data.enabled() and live_data.is_live(site, issue, meta()):
        rows = live_data.shap_rows(site, issue, pixel_id)
    else:
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


def check_pixel(site, pixel_id):
    if pixel_id not in set(pixels(site)["pixel_id"]):
        raise BadRequest(f"unknown pixel '{pixel_id}'")
    return pixel_id


def api_advice(site, pixel_id, issue, crops_param):
    """General 7-day outlook plus advice for each requested crop, for one square and issue day."""
    site = check_site(site)
    issue = check_date(site, issue)
    check_pixel(site, pixel_id)
    wanted = [c for c in (crops_param or "").split(",") if c][:len(crops_mod.CROP_IDS)]
    for c in wanted:
        if c not in crops_mod.CROPS:
            raise BadRequest(f"unknown crop '{c}'")
    days = place_days(site, pixel_id, issue)  # live days: reliability comes from the live backtest, not the final test
    has_blight = meta()[site]["has_blight"]
    return {"issue": issue, "site": site, "pixel_id": pixel_id, "outlook": crops_mod.outlook(days, site),
            "crops": [crops_mod.advise(c, days, site, has_blight) for c in wanted]}


def newest_issue(site) -> str:
    return str(pd.Timestamp(meta()[site]["issue_max"]).date())


def place_days(site, pixel_id, issue) -> list[dict]:
    """The 7 forecast days for one square, with live days marked (as in api_advice)."""
    days = crops_mod.pixel_days(day(site, issue), pixel_id, meta()[site].get("status"))
    from app import live_data
    if live_data.enabled() and live_data.is_live(site, issue, meta()):
        for d in days:
            d["live"] = True
    return days


def refresh_alerts(app, user_id: int) -> None:
    """Check each saved place against the newest forecast of its site and merge the result (web/notify.py)."""
    ud: UserData = app.state.userdata
    store: NotificationStore = app.state.notify
    prefs, crops = store.prefs(user_id), ud.crops(user_id)
    kinds = [k for k, on in prefs.items() if on]
    for p in ud.places(user_id):
        if p["site"] not in SITES or p["pixel_id"] not in set(pixels(p["site"])["pixel_id"]):
            continue
        issue = newest_issue(p["site"])
        days = place_days(p["site"], p["pixel_id"], issue)
        events = detect_alerts(days, p["site"], crops, meta()[p["site"]]["has_blight"], kinds)
        store.sync(user_id, p["id"], issue, days, events)


def api_notifications(app, user_id: int) -> dict:
    refresh_alerts(app, user_id)
    store: NotificationStore = app.state.notify
    items = store.items(user_id, {s: newest_issue(s) for s in SITES})
    return {"items": items, "unread": sum(1 for i in items if not i["read"] and i["state"] == "active" and not i["ended"]),
            "prefs": store.prefs(user_id), "kinds": KIND_NAMES}


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


async def read_json(request: Request) -> dict:
    body = await request.body()
    if len(body) > MAX_BODY:
        raise BadRequest("request too large")
    try:
        data = json.loads(body or b"{}")
    except ValueError:
        raise BadRequest("expected JSON") from None
    if not isinstance(data, dict):
        raise BadRequest("expected a JSON object")
    return data


def write_check(request: Request):
    """None if a signed-in state-changing request is allowed, else the error response."""
    if not same_origin(request):
        return json_error("cross-site request refused", 403)
    s = request.state.session
    if s is None or not hmac.compare_digest(request.headers.get("x-csrf-token", ""), s.csrf):
        return json_error("missing or wrong CSRF token", 403)
    return None


def writer(fn):
    """Signed-in state-changing JSON endpoint: Origin + CSRF checked, bad input -> 400 with a safe message."""
    async def endpoint(request: Request):
        err = write_check(request)
        if err is not None:
            return err
        try:
            return await fn(request)
        except (BadRequest, PolicyError, UserDataError, NotifyError) as e:
            return json_error(str(e), 400)
    return endpoint


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


def r_posttest_skill(request):
    if not (C.REPORTS / "posttest" / "skill.csv").exists():
        return json_ok([])
    return json_ok(table("posttest/skill.csv", "model == 'catboost' and metric in ['bss', 'rmse_skill']"))


def r_live_backtest_skill(request):
    """Validation-2024 backtest (src.live.backtest): the same development models with normal inputs
    ('standard') and with the live inputs actually used ('live_v2_qm'). Shown only when live mode is on."""
    from app import live_data
    if not live_data.enabled() or not (C.REPORTS / "live" / "backtest_skill.csv").exists():
        return json_ok([])
    return json_ok(table("live/backtest_skill.csv", "model in ['standard', 'live_v2_qm'] and metric in ['bss', 'rmse_skill']"))


def r_posttest_blight_skill(request):
    if not (C.REPORTS / "posttest" / "late_blight_skill.csv").exists():
        return json_ok([])
    return json_ok(table("posttest/late_blight_skill.csv"))


@guarded
def r_advice(request):
    p = request.path_params
    return json_ok(api_advice(p["site"], p["pixel_id"], p["issue"], request.query_params.get("crops", "")))


def r_crops(request):
    return json_ok({"crops": crops_mod.catalogue(), "user_types": list(USER_TYPES)})


# ---------------------------------------------------------------- routes: the signed-in user's data
def uid(request) -> int:
    return request.state.session.user.id


def r_user(request):
    return json_ok(request.app.state.userdata.everything(uid(request)))


@writer
async def r_profile(request):
    return json_ok(request.app.state.userdata.update_profile(uid(request), await read_json(request)))


@writer
async def r_place_add(request):
    b = await read_json(request)
    site = check_site(b.get("site"))
    pid = check_pixel(site, str(b.get("pixel_id", "")))
    return json_ok(request.app.state.userdata.add_place(uid(request), site, pid, b.get("name", ""), b.get("note", "")))


def place_id(request) -> int:
    try:
        return int(request.path_params["place_id"])
    except ValueError:
        raise BadRequest("bad place id") from None


@writer
async def r_place_edit(request):
    ud: UserData = request.app.state.userdata
    if request.method == "DELETE":
        ud.delete_place(uid(request), place_id(request))
        return json_ok({"ok": True})
    b = await read_json(request)
    return json_ok(ud.update_place(uid(request), place_id(request), b.get("name"), b.get("note")))


@writer
async def r_crops_set(request):
    b = await read_json(request)
    return json_ok({"crops": request.app.state.userdata.set_crops(uid(request), b.get("crops"))})


async def r_notifications(request):
    return json_ok(await run_in_threadpool(api_notifications, request.app, uid(request)))


@writer
async def r_notifications_read(request):
    b = await read_json(request)
    ids = None if b.get("all") is True else b.get("ids")
    if ids is None and b.get("all") is not True:
        raise BadRequest("give ids or all: true")
    return json_ok({"marked": request.app.state.notify.mark_read(uid(request), ids)})


@writer
async def r_notifications_prefs(request):
    return json_ok({"prefs": request.app.state.notify.set_prefs(uid(request), await read_json(request))})


@writer
async def r_account(request):
    b = await read_json(request)
    store: AuthStore = request.app.state.store
    store.update_account(uid(request), b.get("display_name"))
    return json_ok(store.account(uid(request)))


@writer
async def r_account_password(request):
    b = await read_json(request)
    if b.get("new") != b.get("new2"):
        raise BadRequest("the two new passwords differ")
    token = request.cookies.get(request.app.state.settings.cookie_name)
    await run_in_threadpool(request.app.state.store.change_password, uid(request), str(b.get("current", ""))[:256],
                            str(b.get("new", ""))[:300], token)
    return json_ok({"ok": True})


@writer
async def r_account_sessions(request):
    token = request.cookies.get(request.app.state.settings.cookie_name)
    n = request.app.state.store.end_other_sessions(uid(request), token)
    return json_ok({"ended": n})


@writer
async def r_account_delete(request):
    b = await read_json(request)
    cfg: Settings = request.app.state.settings
    await run_in_threadpool(request.app.state.store.delete_account, uid(request), str(b.get("password", ""))[:256],
                            client_ip(request))
    resp = json_ok({"ok": True})
    resp.delete_cookie(cfg.cookie_name, path="/", httponly=True, samesite="strict", secure=cfg.secure_cookies)
    return resp


@writer
async def r_account_verify(request):
    """Send (again) the email-verification link."""
    store: AuthStore = request.app.state.store
    acc = store.account(uid(request))
    if acc["email_verified"] or not acc["email"]:
        return json_ok({"sent": False})
    if not request.app.state.mail_limiter.allow(f"verify:{uid(request)}"):
        raise BadRequest("a link was sent recently; check your inbox (and spam folder)")
    send_verify(request.app, uid(request), acc["email"], acc["display_name"])
    return json_ok({"sent": True})


def send_verify(app, user_id, email, name):
    cfg: Settings = app.state.settings
    token = app.state.store.create_token(user_id, "verify")
    app.state.mailer.send(email, *verify_message(name, f"{cfg.link_base}/verify?token={token}"))


# ---------------------------------------------------------------- routes: account
def r_me(request):
    s = request.state.session
    acc = request.app.state.store.account(s.user.id)
    return json_ok({"username": s.user.username, "role": s.user.role, "csrf": s.csrf,
                    "display_name": acc["display_name"], "email": acc["email"], "email_verified": acc["email_verified"],
                    "created": acc["created"]})


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
    set_session_cookie(resp, cfg, token)
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


def set_session_cookie(resp, cfg: Settings, token: str):
    resp.set_cookie(cfg.cookie_name, token, max_age=ABSOLUTE_SECONDS, path="/", httponly=True,
                    samesite="strict", secure=cfg.secure_cookies)


def r_auth_config(request):
    return json_ok({"signup_open": request.app.state.settings.signup_open})


def r_signup_page(request):
    if request.state.session is not None:
        return RedirectResponse("/", status_code=303)
    if not request.app.state.settings.signup_open:
        return RedirectResponse("/login?error=closed", status_code=303)
    return static_file(STATIC / "signup.html")


async def r_signup(request: Request):
    """Create a viewer account, sign it in and email a verification link. JSON in, JSON out."""
    app, cfg = request.app, request.app.state.settings
    if not cfg.signup_open:
        return json_error("sign-up is closed; ask an administrator for an account", 403)
    if not same_origin(request):
        return json_error("cross-site request refused", 403)
    ip = client_ip(request)
    if not app.state.signup_limiter.allow(ip):
        app.state.store.audit("signup_rate_limited", None, ip)
        return json_error("too many sign-ups from this connection; try again later", 429)
    try:
        b = await read_json(request)
    except BadRequest as e:
        return json_error(str(e), 400)
    if b.get("website"):  # honeypot field, invisible to people; bots fill it in
        app.state.store.audit("signup_bot", None, ip)
        return json_error("sign-up refused", 400)
    password = str(b.get("password", ""))[:300]
    if password != str(b.get("password2", ""))[:300]:
        return json_error("the two passwords differ", 400)
    if b.get("user_type", "other") not in USER_TYPES:
        return json_error("choose what describes you best", 400)
    if b.get("agree") is not True:
        return json_error("please confirm that you understand this is a research prototype", 400)
    try:
        user = await run_in_threadpool(app.state.store.register, str(b.get("username", ""))[:64],
                                       str(b.get("email", ""))[:254], password, str(b.get("display_name", ""))[:120], ip)
    except PolicyError as e:
        return json_error(str(e), 400)
    app.state.userdata.update_profile(user.id, {"user_type": b.get("user_type", "other")})
    acc = app.state.store.account(user.id)
    send_verify(app, user.id, acc["email"], acc["display_name"])
    token, _ = app.state.store.create_session(user)
    resp = json_ok({"ok": True, "next": "/"})
    set_session_cookie(resp, cfg, token)
    return resp


def r_verify(request):
    ok = request.app.state.store.verify_email(request.query_params.get("token", ""))
    target = "/" if request.state.session is not None else "/login"
    return RedirectResponse(f"{target}?verified={'1' if ok else '0'}", status_code=303)


def r_recover_page(request):
    name = "reset.html" if request.url.path == "/reset" else "forgot.html"
    return static_file(STATIC / name)


async def r_forgot(request: Request):
    """Email a reset link if the address has an account. Same answer either way (no account discovery)."""
    app, cfg = request.app, request.app.state.settings
    if not same_origin(request):
        return json_error("cross-site request refused", 403)
    ip = client_ip(request)
    if not app.state.forgot_limiter.allow(ip):
        return json_error("too many requests from this connection; try again in a few minutes", 429)
    try:
        b = await read_json(request)
    except BadRequest as e:
        return json_error(str(e), 400)
    found = await run_in_threadpool(app.state.store.reset_request, str(b.get("email", ""))[:254], ip)
    if found:
        email, name, token = found
        app.state.mailer.send(email, *reset_message(name, f"{cfg.link_base}/reset?token={token}"))
    return json_ok({"ok": True, "message": "If that email has an account, a reset link is on its way. It works for 30 minutes."})


async def r_reset(request: Request):
    app = request.app
    if not same_origin(request):
        return json_error("cross-site request refused", 403)
    if not app.state.forgot_limiter.allow(client_ip(request)):
        return json_error("too many requests from this connection; try again in a few minutes", 429)
    try:
        b = await read_json(request)
    except BadRequest as e:
        return json_error(str(e), 400)
    if str(b.get("password", "")) != str(b.get("password2", "")):
        return json_error("the two passwords differ", 400)
    try:
        await run_in_threadpool(app.state.store.reset_password, str(b.get("token", ""))[:100], str(b.get("password", ""))[:300])
    except PolicyError as e:
        return json_error(str(e), 400)
    return json_ok({"ok": True, "next": "/login?reset=1"})


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
        h["Permissions-Policy"] = "geolocation=(self), camera=(), microphone=()"  # "use my location" only
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
        Route("/auth/config", r_auth_config),
        Route("/signup", r_signup_page, methods=["GET"]),
        Route("/signup", r_signup, methods=["POST"]),
        Route("/verify", r_verify),
        Route("/forgot", r_recover_page, methods=["GET"]),
        Route("/forgot", r_forgot, methods=["POST"]),
        Route("/reset", r_recover_page, methods=["GET"]),
        Route("/reset", r_reset, methods=["POST"]),
        Route("/api/me", r_me),
        Route("/api/user", r_user),
        Route("/api/profile", r_profile, methods=["POST"]),
        Route("/api/places", r_place_add, methods=["POST"]),
        Route("/api/places/{place_id}", r_place_edit, methods=["POST", "DELETE"]),
        Route("/api/crops", r_crops, methods=["GET"]),
        Route("/api/crops", r_crops_set, methods=["POST"]),
        Route("/api/notifications", r_notifications, methods=["GET"]),
        Route("/api/notifications/read", r_notifications_read, methods=["POST"]),
        Route("/api/notifications/prefs", r_notifications_prefs, methods=["POST"]),
        Route("/api/account", r_account, methods=["POST"]),
        Route("/api/account/password", r_account_password, methods=["POST"]),
        Route("/api/account/sessions", r_account_sessions, methods=["POST"]),
        Route("/api/account/delete", r_account_delete, methods=["POST"]),
        Route("/api/account/verify", r_account_verify, methods=["POST"]),
        Route("/api/advice/{site}/{pixel_id}/{issue}", r_advice),
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
        Route("/api/posttest_skill", r_posttest_skill),
        Route("/api/posttest_blight_skill", r_posttest_blight_skill),
        Route("/api/live_backtest_skill", r_live_backtest_skill),
        Route("/api/{rest:path}", lambda request: json_error("not found", 404)),
        Route("/", r_static),
        Route("/{path:path}", r_static),
    ]
    app = Starlette(routes=routes, middleware=[Middleware(SecurityMiddleware)])
    app.state.settings = cfg
    app.state.store = store or AuthStore(cfg.auth_db)
    app.state.userdata = UserData(app.state.store.path, SITES, crops_mod.CROP_IDS)
    app.state.notify = NotificationStore(app.state.store.path)  # after UserData: its table refers to places
    app.state.mailer = make_mailer(cfg.smtp)
    app.state.limiter = RateLimiter(limit=20, window=600)
    app.state.signup_limiter = RateLimiter(limit=5, window=3600)
    app.state.forgot_limiter = RateLimiter(limit=10, window=900)
    app.state.mail_limiter = RateLimiter(limit=1, window=120)
    return app


def main() -> None:
    import uvicorn
    cfg = settings()
    app = create_app(cfg)
    if not app.state.store.list_users():
        print("No accounts yet. Create one first:  python -m web.users add <name> --role admin")
    print(f"Dashboard running at http://{cfg.host}:{cfg.port}  (Ctrl+C to stop)", flush=True)
    print(f"Sign-up is {'open at /signup' if cfg.signup_open else 'closed (HL_SIGNUP=0)'}; emails "
          f"{'go out by SMTP' if cfg.smtp else 'are printed here (no HL_SMTP_HOST)'}.", flush=True)
    uvicorn.run(app, host=cfg.host, port=cfg.port, log_level="warning", proxy_headers=True)


if __name__ == "__main__":
    main()
