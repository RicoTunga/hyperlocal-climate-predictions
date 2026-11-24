"""Tests for the farmer alerts (web/notify.py): each rule fires on its pattern only, reliability follows the
tested lead times, and the store merges forecasts without duplicates (update, clear, older forecasts ignored,
per-user, cascades). The HTTP test needs app/data and is skipped when it is not built."""
import pytest
from starlette.testclient import TestClient

from web import notify as N
from web.auth import AuthStore
from web.server import create_app
from web.settings import Settings
from web.userdata import UserData
from tests.test_accounts import ORIGIN, PW, Clock, csrf, signup
from tests.test_crops import week
from tests.test_web import DATA_READY


def kinds(events):
    return [e.kind for e in events]


def live(days):
    for d in days:
        d["live"] = True
    return days


# ---------------------------------------------------------------- rules
def test_quiet_week_raises_nothing():
    assert N.detect(week(), "nakuru", ["potato"], True) == []


def test_blight_needs_potato_or_tomato_and_a_blight_model():
    days = week({"blight": {3: 0.7, 4: 0.6}}, p1=0.5)
    assert kinds(N.detect(days, "nakuru", ["maize"], True)) == []
    assert kinds(N.detect(days, "nakuru", ["potato"], False)) == []  # Busia: no tested model
    (e,) = N.detect(days, "nakuru", ["potato"], True)
    assert e.kind == "blight" and e.hs == (3, 4) and e.level == "act" and e.reliability == "good"
    assert "No dry day before it" in e.body
    e = N.detect(week({"blight": {3: 0.7}, "p1": {1: 0.1}}), "nakuru", ["tomato"], True)[0]
    assert "Day1 looks dry" in e.body and "built for potato" in e.body
    assert N.detect(week({"blight": {6: 0.7}}), "nakuru", ["potato"], True)[0].reliability == "low"  # beyond day 5


def test_dry_window_needs_two_dry_days_starting_by_day_3():
    e = N.detect(week({"p1": {2: 0.1, 3: 0.2, 4: 0.1}}), "nakuru")[0]
    assert e.kind == "spray" and e.hs == (2, 3, 4) and e.level == "info" and "Day2 to Day4" in e.body
    assert e.reliability == "partly"  # day 2 is inside Nakuru's 2 tested rain days, days 3-4 are not
    assert N.detect(week({"p1": {1: 0.1, 2: 0.1}}), "nakuru")[0].reliability == "good"
    assert N.detect(week({"p1": {1: 0.1}}), "nakuru") == []           # one dry day only
    assert N.detect(week({"p1": {4: 0.1, 5: 0.1}}), "nakuru") == []   # starts too late
    assert N.detect(week({"p1": {3: 0.1, 4: 0.1}}), "nakuru")[0].reliability == "low"
    assert N.detect(week({"p1": {2: 0.1, 3: 0.1}}), "busia")[0].reliability == "good"  # Busia: 3 tested days


def test_heavy_rain_only_on_days_1_to_3():
    e = N.detect(week({"p10": {2: 0.5}, "p25": {2: 0.2}}), "nakuru")[0]
    assert e.kind == "heavy" and e.hs == (2,) and "50%" in e.body and "20%" in e.body and e.reliability == "good"
    assert N.detect(week({"p10": {5: 0.9}}), "nakuru") == []
    assert N.detect(week({"p25": {3: 0.3}}), "nakuru")[0].reliability == "low"


def test_humid_spell_needs_three_days_and_takes_the_longest():
    assert N.detect(week({"rh": {1: 90, 2: 90}}), "nakuru") == []
    e = N.detect(week({"rh": {1: 90, 2: 90, 3: 90, 5: 88, 6: 88, 7: 88}}), "nakuru")[0]
    assert e.kind == "humid" and e.hs == (1, 2, 3)
    e = N.detect(week({"rh": {2: 90, 3: 90, 4: 90, 5: 90}}), "busia")[0]
    assert e.hs == (2, 3, 4, 5) and e.reliability == "partly"  # Busia humidity: 3 tested days
    assert N.detect(week({"rh": {4: 90, 5: 90, 6: 90}}), "busia")[0].reliability == "low"


def test_live_days_take_reliability_from_the_live_backtest():
    days = live(week({"p1": {1: 0.1, 2: 0.1}, "p10": {1: 0.6}, "rh": {1: 90, 2: 90, 3: 90}, "blight": {1: 0.8}}))
    got = {e.kind: e.reliability for e in N.detect(days, "nakuru", ["potato"], True)}
    # no rain-based or blight day is reliable live; humidity is, on days 1-2 at Nakuru (so a day 1-3 spell: partly)
    assert got == {"blight": "low", "spray": "low", "heavy": "low", "humid": "partly"}
    assert {e.kind: e.reliability for e in N.detect(days, "busia")}["humid"] == "low"


def test_switched_off_kinds_are_not_checked():
    days = week({"p1": {1: 0.1, 2: 0.1}, "p10": {1: 0.6}})
    assert kinds(N.detect(days, "nakuru", kinds=("heavy",))) == ["heavy"]


# ---------------------------------------------------------------- store
@pytest.fixture
def db(tmp_path):
    clock = Clock()
    auth = AuthStore(tmp_path / "users.db", clock=clock)
    ud = UserData(auth.path, crops=("potato",), clock=clock)
    store = N.NotificationStore(auth.path, clock=clock)
    users = [auth.register(n, f"{n}@example.org", PW) for n in ("ann", "ben")]
    places = [ud.add_place(u.id, "nakuru", f"px{k}", "Shamba") for k, u in enumerate(users)]
    yield auth, ud, store, clock, users, places
    store.close(); ud.close(); auth.close()


def dated(start_day, **over):
    """A week of forecast days for the issue day 2026-11-<start_day>, with real dates."""
    days = week(over or None)
    for d in days:
        d["date"] = f"2026-11-{start_day + d['h']:02d}"
        d["label"] = f"D{start_day + d['h']}"
    return days


def sync(store, user, place, start, days):
    store.sync(user.id, place["id"], f"2026-11-{start:02d}", days, N.detect(days, "nakuru", ["potato"], True))


def test_store_merges_overlapping_events_and_keeps_read_state(db):
    auth, ud, store, clock, (ann, ben), (pa, pb) = db
    sync(store, ann, pa, 10, dated(10, p1={1: 0.1, 2: 0.1}))          # dry 11-12 Nov
    sync(store, ann, pa, 10, dated(10, p1={1: 0.1, 2: 0.1}))          # same forecast again: no duplicate
    (a,) = store.items(ann.id)
    assert (a["kind"], a["first_date"], a["last_date"], a["read"]) == ("spray", "2026-11-11", "2026-11-12", False)
    store.mark_read(ann.id, [a["id"]])
    clock.t += 60
    sync(store, ann, pa, 11, dated(11, p1={1: 0.1, 2: 0.1}))          # next day: dry 12-13 Nov overlaps
    (a,) = store.items(ann.id)
    assert (a["first_date"], a["last_date"], a["issue_date"], a["read"]) == ("2026-11-12", "2026-11-13", "2026-11-11", True)
    assert store.items(ben.id) == []  # other users see nothing of it


def test_store_clears_events_a_newer_forecast_drops_and_ignores_older_forecasts(db):
    auth, ud, store, clock, (ann, _), (pa, _) = db
    sync(store, ann, pa, 10, dated(10, p10={2: 0.6}))                 # heavy rain on 12 Nov
    sync(store, ann, pa, 11, dated(11))                               # newer: no heavy rain any more
    (a,) = store.items(ann.id)
    assert a["state"] == "cleared"
    sync(store, ann, pa, 9, dated(9, p10={1: 0.6, 3: 0.6}))           # an older forecast changes nothing
    assert [(i["state"], i["first_date"]) for i in store.items(ann.id)] == [("cleared", "2026-11-12")]
    store.mark_read(ann.id)
    sync(store, ann, pa, 11, dated(11, p10={1: 0.6}))                 # back on 12 Nov: active and unread again
    (a,) = store.items(ann.id)
    assert (a["state"], a["read"]) == ("active", False)


def test_store_prefs_ended_flag_validation_and_cascade(db):
    auth, ud, store, clock, (ann, _), (pa, _) = db
    sync(store, ann, pa, 10, dated(10, p1={1: 0.1, 2: 0.1}, rh={1: 90, 2: 90, 3: 90}))
    assert sorted(i["kind"] for i in store.items(ann.id)) == ["humid", "spray"]
    assert store.set_prefs(ann.id, {"spray": False}) == {"blight": True, "spray": False, "heavy": True, "humid": True}
    assert [i["kind"] for i in store.items(ann.id)] == ["humid"]
    with pytest.raises(N.NotifyError):
        store.set_prefs(ann.id, {"frost": True})
    with pytest.raises(N.NotifyError):
        store.mark_read(ann.id, ["1; DROP TABLE"])
    assert store.items(ann.id, {"nakuru": "2026-11-20"})[0]["ended"] is True
    assert store.items(ann.id, {"nakuru": "2026-11-10"})[0]["ended"] is False
    ud.delete_place(ann.id, pa["id"])
    store.set_prefs(ann.id, {"spray": True})
    assert store.items(ann.id) == []  # deleting the place deleted its alerts


def test_old_alerts_are_pruned(db):
    auth, ud, store, clock, (ann, _), (pa, _) = db
    sync(store, ann, pa, 1, dated(1, p1={1: 0.1, 2: 0.1}))
    days = dated(1)
    for d in days:  # a forecast more than KEEP_DAYS later
        d["date"] = d["date"].replace("2026-11", "2026-12").replace("-12-0", "-12-1")
    store.sync(ann.id, pa["id"], "2026-12-10", days, [])
    assert store.items(ann.id) == []


# ---------------------------------------------------------------- HTTP
@pytest.mark.skipif(not DATA_READY, reason="app/data not built")
def test_alerts_over_http(tmp_path):
    store = AuthStore(tmp_path / "users.db")
    client = TestClient(create_app(Settings("127.0.0.1", 8765, store.path, False), store))
    assert client.get("/api/notifications").status_code == 401
    signup(client)
    empty = client.get("/api/notifications").json()
    assert empty["items"] == [] and empty["unread"] == 0 and set(empty["kinds"]) == set(N.KINDS)
    h = csrf(client)
    ids = client.get("/api/pixels/nakuru").json()["ids"]
    for k in (0, 200, 400):
        client.post("/api/places", json={"site": "nakuru", "pixel_id": ids[k], "name": f"Field {k}"}, headers=h)
    client.post("/api/crops", json={"crops": ["potato"]}, headers=h)
    r = client.get("/api/notifications").json()
    assert r["unread"] == sum(1 for i in r["items"] if not i["read"] and i["state"] == "active" and not i["ended"])
    assert all(i["kind"] in N.KINDS and i["place_name"].startswith("Field") for i in r["items"])
    assert client.get("/api/notifications").json()["items"] == r["items"]  # refreshing again adds nothing
    assert client.post("/api/notifications/read", json={"all": True}, headers=ORIGIN).status_code == 403  # no CSRF
    assert client.post("/api/notifications/read", json={}, headers=h).status_code == 400
    client.post("/api/notifications/read", json={"all": True}, headers=h)
    assert client.get("/api/notifications").json()["unread"] == 0
    assert client.post("/api/notifications/prefs", json={"frost": False}, headers=h).status_code == 400
    off = client.post("/api/notifications/prefs", json={k: False for k in N.KINDS}, headers=h).json()["prefs"]
    assert not any(off.values()) and client.get("/api/notifications").json()["items"] == []
