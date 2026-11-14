"""Tests for self-service accounts and per-user data (sign-up, verification, password reset, account
changes, saved places, crops and settings). Temporary databases; the place tests that check square ids
against the real pixel list are skipped when app/data is not built."""
import re

import pytest
from starlette.testclient import TestClient

from web.auth import AuthStore, PolicyError, RateLimiter
from web.server import create_app
from web.settings import Settings
from web.userdata import UserData, UserDataError
from tests.test_web import DATA_READY

PW = "correct horse battery"
ORIGIN = {"Origin": "http://testserver"}


class Clock:
    def __init__(self, t=1_000_000.0):
        self.t = t

    def __call__(self):
        return self.t


@pytest.fixture
def store(tmp_path):
    clock = Clock()
    s = AuthStore(tmp_path / "users.db", clock=clock)
    s.clock_obj = clock
    yield s
    s.close()


@pytest.fixture
def client(store):
    app = create_app(Settings("127.0.0.1", 8765, store.path, False), store)
    return TestClient(app)


def signup(client, **over):
    body = {"display_name": "Wanjiku", "username": "wanjiku", "email": "wanjiku@example.org", "password": PW,
            "password2": PW, "user_type": "farmer", "agree": True, "website": ""}
    body.update(over)
    return client.post("/signup", json=body, headers=ORIGIN)


def csrf(client):
    return {**ORIGIN, "X-CSRF-Token": client.get("/api/me").json()["csrf"]}


def last_link(client, path):
    body = client.app.state.mailer.sent[-1]["body"]
    return re.search(rf"http://127\.0\.0\.1:8765{path}\?token=([\w-]+)", body).group(1)


# ---------------------------------------------------------------- store
def test_register_rules_and_login_by_email(store):
    store.register("wanjiku", "w@example.org", PW, "Wanjiku")
    with pytest.raises(PolicyError, match="taken"):
        store.register("WANJIKU", "other@example.org", PW)
    with pytest.raises(PolicyError, match="already exists"):
        store.register("other", "W@Example.org", PW)
    for bad_email in ("no-at-sign", "a@b", "x@y.c"):
        with pytest.raises(PolicyError, match="email"):
            store.register("someone", bad_email, PW)
    for weak in ("password1234", "aaaaaaaaaaaaaaa", "short"):
        with pytest.raises(PolicyError):
            store.register("someone", "s@example.org", weak)
    assert store.authenticate("w@example.org", PW).username == "wanjiku"
    assert store.authenticate("wanjiku", PW).role == "viewer"


def test_tokens_are_single_use_hashed_and_expire(store):
    u = store.register("wanjiku", "w@example.org", PW)
    tok = store.create_token(u.id, "verify")
    raw = store._db.execute("SELECT token_hash FROM tokens").fetchone()[0]
    assert tok not in raw and len(raw) == 64
    assert store.verify_email(tok) and store.account(u.id)["email_verified"]
    assert not store.verify_email(tok)  # used
    _, _, reset = store.reset_request("W@example.org")
    store.clock_obj.t += 31 * 60
    with pytest.raises(PolicyError, match="expired"):
        store.reset_password(reset, "a brand new passphrase")
    assert store.reset_request("nobody@example.org") is None


def test_reset_password_ends_sessions_and_weak_password_keeps_link(store):
    u = store.register("wanjiku", "w@example.org", PW)
    token, _ = store.create_session(u)
    _, _, reset = store.reset_request("w@example.org")
    with pytest.raises(PolicyError):
        store.reset_password(reset, "short")
    store.reset_password(reset, "a brand new passphrase")
    assert store.get_session(token) is None
    assert store.authenticate("wanjiku", "a brand new passphrase")
    with pytest.raises(PolicyError):
        store.reset_password(reset, "another new passphrase")  # single use


def test_change_password_keeps_this_session_only(store):
    u = store.register("wanjiku", "w@example.org", PW)
    mine, _ = store.create_session(u)
    other, _ = store.create_session(u)
    with pytest.raises(PolicyError, match="current password"):
        store.change_password(u.id, "wrong", "a brand new passphrase", mine)
    store.change_password(u.id, PW, "a brand new passphrase", mine)
    assert store.get_session(mine) is not None and store.get_session(other) is None


def test_delete_account_cascades_to_user_data(store):
    u = store.register("wanjiku", "w@example.org", PW)
    ud = UserData(store.path, crops=("potato", "maize"))
    ud.add_place(u.id, "nakuru", "p1", "Shamba")
    ud.set_crops(u.id, ["potato"])
    with pytest.raises(PolicyError):
        store.delete_account(u.id, "wrong")
    store.delete_account(u.id, PW)
    assert ud.places(u.id) == [] and ud.crops(u.id) == []
    assert store._db.execute("SELECT COUNT(*) FROM profiles").fetchone()[0] == 0


def test_admins_cannot_delete_themselves_from_the_web(store):
    store.add_user("boss", PW, "admin")
    uid = store.authenticate("boss", PW).id
    with pytest.raises(PolicyError, match="admin"):
        store.delete_account(uid, PW)


def test_old_databases_are_migrated(tmp_path):
    import sqlite3
    db = tmp_path / "old.db"
    con = sqlite3.connect(db)
    con.executescript("""CREATE TABLE users (id INTEGER PRIMARY KEY, username TEXT NOT NULL UNIQUE COLLATE NOCASE,
        role TEXT NOT NULL, pw_salt BLOB NOT NULL, pw_hash BLOB NOT NULL, pw_params TEXT NOT NULL,
        disabled INTEGER NOT NULL DEFAULT 0, failed INTEGER NOT NULL DEFAULT 0, locked_until REAL NOT NULL DEFAULT 0,
        created REAL NOT NULL);""")
    con.close()
    s = AuthStore(db)
    s.add_user("old", PW, "admin")
    assert s.account(s.authenticate("old", PW).id)["display_name"] == "old"
    s.close()


# ---------------------------------------------------------------- user data store
def test_places_are_per_user_validated_and_first_is_home(store):
    a = store.register("anna", "a@example.org", PW)
    b = store.register("bob", "b@example.org", PW)
    ud = UserData(store.path, crops=("potato",))
    p = ud.add_place(a.id, "nakuru", "p1", "  My   farm\x07 ")
    assert p["name"] == "My farm" and ud.profile(a.id)["home_place"] == p["id"]
    with pytest.raises(UserDataError, match="already"):
        ud.add_place(a.id, "nakuru", "p1", "Again")
    with pytest.raises(UserDataError):
        ud.add_place(a.id, "nairobi", "p2", "X")
    with pytest.raises(UserDataError):
        ud.add_place(a.id, "nakuru", "p3", "")
    with pytest.raises(UserDataError):  # bob cannot rename, delete or set anna's place as home
        ud.update_place(b.id, p["id"], name="Mine now")
    with pytest.raises(UserDataError):
        ud.delete_place(b.id, p["id"])
    with pytest.raises(UserDataError):
        ud.update_profile(b.id, {"home_place": p["id"]})
    assert ud.places(b.id) == []
    ud.delete_place(a.id, p["id"])
    assert ud.profile(a.id)["home_place"] is None


def test_profile_settings_and_crops_validated(store):
    a = store.register("anna", "a@example.org", PW)
    ud = UserData(store.path, crops=("potato", "maize"))
    prof = ud.profile(a.id)
    assert prof["tour_on_login"] and not prof["onboarded"] and prof["animations"]
    assert ud.update_profile(a.id, {"tour_on_login": False, "user_type": "farmer"})["tour_on_login"] is False
    for bad in ({"user_type": "king"}, {"role": "admin"}, {"home_site": "mars"}):
        with pytest.raises(UserDataError):
            ud.update_profile(a.id, bad)
    assert ud.set_crops(a.id, ["maize", "potato", "maize"]) == ["potato", "maize"]
    with pytest.raises(UserDataError):
        ud.set_crops(a.id, ["coffee"])


# ---------------------------------------------------------------- HTTP
def test_signup_flow_signs_in_and_sends_verification(client):
    assert client.get("/signup").status_code == 200
    r = signup(client)
    assert r.status_code == 200 and r.json()["next"] == "/"
    me = client.get("/api/me").json()
    assert me["username"] == "wanjiku" and me["role"] == "viewer" and not me["email_verified"]
    assert client.get("/api/user").json()["profile"]["user_type"] == "farmer"
    tok = last_link(client, "/verify")
    r = client.get(f"/verify?token={tok}", follow_redirects=False)
    assert r.headers["location"] == "/?verified=1"
    assert client.get("/api/me").json()["email_verified"]


def test_signup_cannot_make_admins_and_rejects_bad_input(client):
    assert signup(client, role="admin").status_code == 200
    assert client.get("/api/me").json()["role"] == "viewer"
    client.cookies.clear()
    assert signup(client, username="bot1", email="b@example.org", website="http://spam").status_code == 400
    assert signup(client, username="u2", email="u2@example.org", password2="different words").status_code == 400
    assert signup(client, username="u3", email="u3@example.org", agree=False).status_code == 400
    assert signup(client, username="u4", email="u4@example.org", user_type="admin").status_code == 400
    assert client.post("/signup", json={}, headers={"Origin": "https://evil.example"}).status_code == 403


def test_signup_rate_limit_and_closed_setting(client, store):
    client.app.state.signup_limiter = RateLimiter(limit=1, window=3600)
    assert signup(client).status_code == 200
    client.cookies.clear()
    assert signup(client, username="second", email="s@example.org").status_code == 429
    closed = TestClient(create_app(Settings("127.0.0.1", 8765, store.path, False, signup_open=False), store))
    assert closed.get("/signup", follow_redirects=False).headers["location"] == "/login?error=closed"
    assert signup(closed, username="third", email="t@example.org").status_code == 403
    assert closed.get("/auth/config").json() == {"signup_open": False}


def test_forgot_and_reset_over_http_do_not_reveal_accounts(client):
    signup(client)
    client.cookies.clear()
    n = len(client.app.state.mailer.sent)
    r1 = client.post("/forgot", json={"email": "nobody@example.org"}, headers=ORIGIN)
    assert len(client.app.state.mailer.sent) == n
    r2 = client.post("/forgot", json={"email": "wanjiku@example.org"}, headers=ORIGIN)
    assert r1.json() == r2.json() and len(client.app.state.mailer.sent) == n + 1
    tok = last_link(client, "/reset")
    assert client.get(f"/reset?token={tok}").status_code == 200
    r = client.post("/reset", json={"token": tok, "password": "a brand new passphrase",
                                    "password2": "a brand new passphrase"}, headers=ORIGIN)
    assert r.json()["next"] == "/login?reset=1"
    r = client.post("/login", data={"username": "wanjiku@example.org", "password": "a brand new passphrase"},
                    headers=ORIGIN, follow_redirects=False)
    assert r.headers["location"] == "/"


def test_writes_need_csrf_and_same_origin(client):
    signup(client)
    for path, body in (("/api/profile", {"tour_done": True}), ("/api/crops", {"crops": ["maize"]}),
                       ("/api/account", {"display_name": "X"})):
        assert client.post(path, json=body, headers=ORIGIN).status_code == 403
        assert client.post(path, json=body, headers={**csrf(client), "Origin": "https://evil.example"}).status_code == 403
        assert client.post(path, json=body, headers=csrf(client)).status_code == 200
    assert client.get("/api/user").json()["crops"] == ["maize"]


def test_account_changes_over_http(client):
    signup(client)
    h = csrf(client)
    assert client.post("/api/account", json={"display_name": "  Mama   Wanjiku "}, headers=h).json()["display_name"] == "Mama Wanjiku"
    assert client.post("/api/account/password", json={"current": "wrong", "new": "x" * 20, "new2": "x" * 20},
                       headers=h).status_code == 400
    assert client.post("/api/account/password", json={"current": PW, "new": "a brand new passphrase",
                                                       "new2": "a brand new passphrase"}, headers=h).status_code == 200
    assert client.get("/api/me").status_code == 200  # this session survives
    assert client.post("/api/account/delete", json={"password": "a brand new passphrase"}, headers=h).status_code == 200
    assert client.get("/api/me").status_code == 401


def test_geolocation_allowed_for_this_site_only(client):
    assert client.get("/login").headers["permissions-policy"].startswith("geolocation=(self)")


@pytest.mark.skipif(not DATA_READY, reason="app/data not built")
def test_places_and_advice_over_http(client):
    signup(client)
    h = csrf(client)
    pid = client.get("/api/pixels/nakuru").json()["ids"][100]
    r = client.post("/api/places", json={"site": "nakuru", "pixel_id": pid, "name": "Shamba"}, headers=h)
    assert r.status_code == 200
    place = r.json()
    assert client.post("/api/places", json={"site": "nakuru", "pixel_id": "nope", "name": "X"}, headers=h).status_code == 400
    assert client.post(f"/api/places/{place['id']}", json={"name": "Lower field"}, headers=h).json()["name"] == "Lower field"
    data = client.get("/api/user").json()
    assert data["profile"]["home_place"] == place["id"] and len(data["places"]) == 1
    issue = client.get("/api/meta").json()["sites"]["nakuru"]["issue_min"]
    adv = client.get(f"/api/advice/nakuru/{pid}/{issue}?crops=potato,maize,beans,wheat,tomato").json()
    assert [c["crop"] for c in adv["crops"]] == ["potato", "maize", "beans", "wheat", "tomato"]
    assert adv["crops"][0]["basis"] == "tested" and adv["crops"][1]["basis"] == "rule"
    assert len(adv["outlook"]["days"]) == 7 and adv["outlook"]["summary"].endswith(".")
    assert client.get(f"/api/advice/nakuru/{pid}/{issue}?crops=coffee").status_code == 400
    assert client.delete(f"/api/places/{place['id']}", headers=h).status_code == 200
    assert client.get("/api/user").json()["places"] == []
