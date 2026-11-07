"""Tests for sign-in and access control (web/auth.py, web/server.py, web/users.py).

Uses temporary databases and Starlette's test client; none of these tests need app/data.
"""
import sqlite3

import pytest
from starlette.testclient import TestClient

from web import auth
from web.auth import AuthError, AuthStore, PolicyError, RateLimiter
from web.server import create_app
from web.settings import Settings

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
    s.add_user("viewer1", PW, "viewer")
    s.add_user("admin1", PW + "!", "admin")
    yield s
    s.close()


# ---------------------------------------------------------------- passwords and policy
def test_scrypt_hash_verifies_and_salts_are_unique():
    s1, h1, p1 = auth.hash_password(PW)
    s2, h2, _ = auth.hash_password(PW)
    assert p1.startswith("scrypt:n=16384") and len(s1) == 16 and len(h1) == 64
    assert s1 != s2 and h1 != h2  # same password, different salt -> different hash
    assert auth.verify_password(PW, s1, h1, p1) and not auth.verify_password(PW + "x", s1, h1, p1)


@pytest.mark.parametrize("user,pw,role", [("ab", PW, "viewer"), ("bad name", PW, "viewer"), ("valid", "short", "viewer"),
                                          ("samesamesame", "samesamesame", "viewer"), ("valid", PW, "root")])
def test_policy_rejects_bad_usernames_passwords_and_roles(user, pw, role):
    with pytest.raises(PolicyError):
        auth.check_policy(user, pw, role)


def test_passwords_are_never_stored_in_plain_text(store):
    raw = (store.path).read_bytes()
    assert PW.encode() not in raw
    assert all("hash" not in k and "salt" not in k for k in store.list_users()[0])
    with pytest.raises(PolicyError):
        store.add_user("viewer1", PW)  # duplicate


# ---------------------------------------------------------------- login, lockout
def test_login_success_and_generic_failures(store):
    assert store.authenticate("viewer1", PW).role == "viewer"
    for user, pw in [("viewer1", "wrong password!"), ("nobody", PW)]:
        with pytest.raises(AuthError) as e:
            store.authenticate(user, pw)
        assert e.value.code == "invalid" and "Wrong username or password" in str(e.value)


def test_lockout_after_five_failures_and_release_after_fifteen_minutes(store):
    for _ in range(auth.MAX_FAILED):
        with pytest.raises(AuthError):
            store.authenticate("viewer1", "wrong password!")
    with pytest.raises(AuthError) as e:  # even the right password is refused while locked
        store.authenticate("viewer1", PW)
    assert e.value.code == "locked"
    store.clock_obj.t += auth.LOCK_SECONDS + 1
    assert store.authenticate("viewer1", PW).username == "viewer1"
    events = [e["event"] for e in store.recent_audit()]
    assert "lockout" in events and "login_locked" in events


def test_disabled_users_cannot_sign_in_and_lose_their_sessions(store):
    token, _ = store.create_session(store.authenticate("viewer1", PW))
    store.set_disabled("viewer1", True)
    assert store.get_session(token) is None
    with pytest.raises(AuthError):
        store.authenticate("viewer1", PW)


# ---------------------------------------------------------------- sessions
def test_sessions_store_only_token_hashes(store):
    token, csrf = store.create_session(store.authenticate("viewer1", PW))
    rows = sqlite3.connect(store.path).execute("SELECT token_hash FROM sessions").fetchall()
    assert rows == [(auth.token_hash(token),)] and token not in str(rows)
    s = store.get_session(token)
    assert s.user.username == "viewer1" and s.csrf == csrf


def test_idle_and_absolute_expiry(store):
    user = store.authenticate("viewer1", PW)
    t1, _ = store.create_session(user)
    store.clock_obj.t += auth.IDLE_SECONDS + 1
    assert store.get_session(t1) is None  # idle too long
    t2, _ = store.create_session(user)
    for _ in range(auth.ABSOLUTE_SECONDS // 3600 + 1):  # active every hour, but past the absolute limit
        store.clock_obj.t += 3600
        session = store.get_session(t2)
    assert session is None


def test_logout_and_password_reset_end_sessions(store):
    user = store.authenticate("viewer1", PW)
    t1, _ = store.create_session(user)
    store.delete_session(t1)
    assert store.get_session(t1) is None
    t2, _ = store.create_session(user)
    store.set_password("viewer1", "a brand new passphrase")
    assert store.get_session(t2) is None


def test_rate_limiter():
    clock = Clock()
    rl = RateLimiter(limit=3, window=60, clock=clock)
    assert [rl.allow("1.2.3.4") for _ in range(4)] == [True, True, True, False]
    assert rl.allow("5.6.7.8")
    clock.t += 61
    assert rl.allow("1.2.3.4")


# ---------------------------------------------------------------- HTTP: access control
@pytest.fixture
def client(store, tmp_path):
    app = create_app(Settings("127.0.0.1", 0, store.path, False), store)
    return TestClient(app)


def login(client, user="viewer1", pw=PW):
    return client.post("/login", data={"username": user, "password": pw}, headers=ORIGIN, follow_redirects=False)


def test_everything_needs_a_session_except_public_paths(client):
    assert client.get("/api/meta").status_code == 401
    assert client.get("/api/admin/users").status_code == 401
    r = client.get("/", follow_redirects=False)
    assert r.status_code == 303 and r.headers["location"] == "/login"
    for path in ("/login", "/login.css", "/login.js", "/healthz"):
        assert client.get(path).status_code == 200, path
    assert client.get("/app.js", follow_redirects=False).status_code == 303  # the app itself is private


def test_login_sets_a_hardened_cookie_and_wrong_password_redirects_with_error(client):
    r = login(client, pw="wrong password!")
    assert r.status_code == 303 and r.headers["location"] == "/login?error=invalid"
    r = login(client)
    assert r.status_code == 303 and r.headers["location"] == "/"
    cookie = r.headers["set-cookie"].lower()
    assert "hl_session=" in cookie and "httponly" in cookie and "samesite=strict" in cookie and "path=/" in cookie
    me = client.get("/api/me").json()
    assert me["username"] == "viewer1" and me["role"] == "viewer" and len(me["csrf"]) > 20


def test_cross_site_posts_are_refused(client):
    for headers in ({}, {"Origin": "https://evil.example"}):
        r = client.post("/login", data={"username": "viewer1", "password": PW}, headers=headers, follow_redirects=False)
        assert r.status_code == 403
    login(client)
    csrf = client.get("/api/me").json()["csrf"]
    assert client.post("/logout", headers={"Origin": "https://evil.example", "X-CSRF-Token": csrf}).status_code == 403
    assert client.post("/logout", headers=ORIGIN).status_code == 403  # no CSRF token
    assert client.get("/api/me").status_code == 200  # still signed in


def test_logout_ends_the_session(client):
    login(client)
    csrf = client.get("/api/me").json()["csrf"]
    assert client.post("/logout", headers={**ORIGIN, "X-CSRF-Token": csrf}).status_code == 200
    assert client.get("/api/me").status_code == 401


def test_viewers_cannot_use_admin_routes_admins_can(client):
    login(client)
    assert client.get("/api/admin/users").status_code == 403
    client.cookies.clear()
    login(client, "admin1", PW + "!")
    data = client.get("/api/admin/users").json()
    assert {u["username"] for u in data["users"]} == {"viewer1", "admin1"}
    assert all("hash" not in str(u) for u in data["users"])


def test_security_headers_on_every_response(client):
    for path in ("/login", "/healthz", "/api/meta"):
        h = client.get(path).headers
        assert "default-src 'self'" in h["content-security-policy"] and "frame-ancestors 'none'" in h["content-security-policy"]
        assert h["x-frame-options"] == "DENY" and h["x-content-type-options"] == "nosniff"
        assert "strict-transport-security" not in h  # only over HTTPS


def test_https_settings_use_secure_host_cookie_and_hsts(store):
    app = create_app(Settings("0.0.0.0", 0, store.path, True), store)
    c = TestClient(app, base_url="https://testserver")
    r = c.post("/login", data={"username": "viewer1", "password": PW}, headers={"Origin": "https://testserver"},
               follow_redirects=False)
    cookie = r.headers["set-cookie"]
    assert cookie.startswith("__Host-hl_session=") and "Secure" in cookie and "Domain" not in cookie
    assert c.get("/healthz").headers["strict-transport-security"].startswith("max-age=")


def test_rate_limit_on_login(client):
    client.app.state.limiter = RateLimiter(limit=2, window=600)
    assert login(client, pw="wrong password!").headers["location"].endswith("invalid")
    assert login(client, pw="wrong password!").headers["location"].endswith("invalid")
    assert login(client).headers["location"] == "/login?error=rate"


# ---------------------------------------------------------------- command line
def test_cli_add_list_disable(tmp_path, monkeypatch, capsys):
    from web import users
    monkeypatch.setenv("HL_AUTH_DB", str(tmp_path / "cli.db"))
    monkeypatch.setattr(users.getpass, "getpass", lambda prompt="": PW)
    users.main(["add", "erica", "--role", "admin"])
    users.main(["disable", "erica"])
    users.main(["list"])
    out = capsys.readouterr().out
    assert "added erica (admin)" in out and "disabled" in out
    with pytest.raises(SystemExit):
        users.main(["remove", "nobody"])
