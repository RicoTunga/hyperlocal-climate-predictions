"""Authentication for the dashboard: users, password hashing, sessions, lockout, audit log.

Standard library only for everything security-related:
- passwords hashed with scrypt (hashlib.scrypt, memory-hard) and a random 16-byte salt per user,
  compared in constant time (hmac.compare_digest); never stored or logged in plain text;
- session tokens from secrets.token_urlsafe(32); only their SHA-256 hash is stored, so a copy of the
  database does not reveal usable sessions; idle and absolute expiry; server-side logout;
- a per-session CSRF token for state-changing requests;
- 5 failed logins lock an account for 15 minutes; failures and lockouts go to an audit table.

Accounts are invite-only: they are created by an admin with `python -m web.users` (no sign-up).
Storage is a single SQLite file (default data/auth/users.db, git-ignored with the rest of data/).
"""
import hashlib
import hmac
import re
import secrets
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path

SCRYPT = {"n": 2 ** 14, "r": 8, "p": 1, "dklen": 64}  # about 16 MB of memory per hash
MIN_PASSWORD = 12
MAX_FAILED = 5
LOCK_SECONDS = 15 * 60
IDLE_SECONDS = 2 * 3600
ABSOLUTE_SECONDS = 12 * 3600
ROLES = ("admin", "viewer")
USERNAME = re.compile(r"^[A-Za-z0-9_.-]{3,32}$")

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY,
    username TEXT NOT NULL UNIQUE COLLATE NOCASE,
    role TEXT NOT NULL CHECK (role IN ('admin', 'viewer')),
    pw_salt BLOB NOT NULL,
    pw_hash BLOB NOT NULL,
    pw_params TEXT NOT NULL,
    disabled INTEGER NOT NULL DEFAULT 0,
    failed INTEGER NOT NULL DEFAULT 0,
    locked_until REAL NOT NULL DEFAULT 0,
    created REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS sessions (
    token_hash TEXT PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    csrf TEXT NOT NULL,
    created REAL NOT NULL,
    last_seen REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS audit (
    at REAL NOT NULL,
    event TEXT NOT NULL,
    username TEXT,
    ip TEXT
);
"""


class AuthError(Exception):
    """Login refused. The message is safe to show to the user (it never says which part was wrong).
    code: 'invalid' (wrong username or password, or disabled) or 'locked' (too many failures)."""

    def __init__(self, message: str, code: str = "invalid"):
        super().__init__(message)
        self.code = code


class PolicyError(ValueError):
    """A username, password or role that does not meet the rules."""


@dataclass(frozen=True)
class User:
    id: int
    username: str
    role: str


@dataclass(frozen=True)
class Session:
    user: User
    csrf: str


# ---------------------------------------------------------------- passwords
def _params_str() -> str:
    return "scrypt:n={n},r={r},p={p},dklen={dklen}".format(**SCRYPT)


def _parse_params(s: str) -> dict:
    kind, _, rest = s.partition(":")
    if kind != "scrypt":
        raise ValueError(f"unknown password hash '{kind}'")
    return {k: int(v) for k, v in (kv.split("=") for kv in rest.split(","))}


def hash_password(password: str, salt: bytes | None = None) -> tuple[bytes, bytes, str]:
    """(salt, hash, parameters) for a password."""
    salt = salt if salt is not None else secrets.token_bytes(16)
    digest = hashlib.scrypt(password.encode("utf-8"), salt=salt, maxmem=64 * 1024 * 1024, **SCRYPT)
    return salt, digest, _params_str()


def verify_password(password: str, salt: bytes, expected: bytes, params: str) -> bool:
    p = _parse_params(params)
    digest = hashlib.scrypt(password.encode("utf-8"), salt=salt, maxmem=64 * 1024 * 1024, **p)
    return hmac.compare_digest(digest, expected)


def check_policy(username: str, password: str | None, role: str | None = None) -> None:
    if not USERNAME.match(username or ""):
        raise PolicyError("username: 3-32 characters, letters, digits, '.', '_' or '-'")
    if password is not None:
        if len(password) < MIN_PASSWORD:
            raise PolicyError(f"password: at least {MIN_PASSWORD} characters")
        if password.lower() == username.lower():
            raise PolicyError("password must differ from the username")
    if role is not None and role not in ROLES:
        raise PolicyError(f"role must be one of {', '.join(ROLES)}")


def token_hash(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------- store
class AuthStore:
    """Users, sessions and the audit log in one SQLite file. Thread-safe (one lock)."""

    def __init__(self, path: str | Path, clock=time.time):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.clock = clock
        self._lock = threading.Lock()
        self._db = sqlite3.connect(self.path, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        with self._lock, self._db:
            self._db.execute("PRAGMA foreign_keys = ON")
            self._db.executescript(SCHEMA)
        # a fixed dummy hash, so a login for an unknown user costs the same time as a real one
        self._dummy = hash_password("not-a-real-password", salt=b"\x00" * 16)

    def close(self) -> None:
        self._db.close()

    # ------------------------------------------------ user management (CLI)
    def add_user(self, username: str, password: str, role: str = "viewer") -> None:
        check_policy(username, password, role)
        salt, digest, params = hash_password(password)
        with self._lock, self._db:
            try:
                self._db.execute("INSERT INTO users (username, role, pw_salt, pw_hash, pw_params, created) "
                                 "VALUES (?, ?, ?, ?, ?, ?)", (username, role, salt, digest, params, self.clock()))
            except sqlite3.IntegrityError:
                raise PolicyError(f"user '{username}' already exists") from None
        self.audit("user_added", username)

    def set_password(self, username: str, password: str) -> None:
        check_policy(username, password)
        salt, digest, params = hash_password(password)
        with self._lock, self._db:
            n = self._db.execute("UPDATE users SET pw_salt=?, pw_hash=?, pw_params=?, failed=0, locked_until=0 "
                                 "WHERE username=?", (salt, digest, params, username)).rowcount
            self._db.execute("DELETE FROM sessions WHERE user_id IN (SELECT id FROM users WHERE username=?)", (username,))
        if not n:
            raise PolicyError(f"no user '{username}'")
        self.audit("password_reset", username)

    def set_disabled(self, username: str, disabled: bool) -> None:
        with self._lock, self._db:
            n = self._db.execute("UPDATE users SET disabled=? WHERE username=?", (int(disabled), username)).rowcount
            if disabled:
                self._db.execute("DELETE FROM sessions WHERE user_id IN (SELECT id FROM users WHERE username=?)", (username,))
        if not n:
            raise PolicyError(f"no user '{username}'")
        self.audit("user_disabled" if disabled else "user_enabled", username)

    def remove_user(self, username: str) -> None:
        with self._lock, self._db:
            n = self._db.execute("DELETE FROM users WHERE username=?", (username,)).rowcount
        if not n:
            raise PolicyError(f"no user '{username}'")
        self.audit("user_removed", username)

    def list_users(self) -> list[dict]:
        """Users without any password material."""
        with self._lock:
            rows = self._db.execute("SELECT username, role, disabled, failed, locked_until, created "
                                    "FROM users ORDER BY username").fetchall()
        now = self.clock()
        return [{"username": r["username"], "role": r["role"], "disabled": bool(r["disabled"]),
                 "locked": r["locked_until"] > now, "created": r["created"]} for r in rows]

    # ------------------------------------------------ login and sessions
    def authenticate(self, username: str, password: str, ip: str = "") -> User:
        now = self.clock()
        with self._lock:
            row = self._db.execute("SELECT * FROM users WHERE username=?", (username or "",)).fetchone()
        if row is None:
            verify_password(password or "", *self._dummy)  # same cost as a real check
            self.audit("login_failed", username, ip)
            raise AuthError("Wrong username or password.")
        if row["locked_until"] > now:
            self.audit("login_locked", row["username"], ip)
            raise AuthError("Too many failed attempts. Try again in a few minutes.", "locked")
        ok = verify_password(password or "", row["pw_salt"], row["pw_hash"], row["pw_params"])
        if not ok or row["disabled"]:
            failed = row["failed"] + 1 if not ok else row["failed"]
            locked = now + LOCK_SECONDS if failed >= MAX_FAILED else 0
            with self._lock, self._db:
                self._db.execute("UPDATE users SET failed=?, locked_until=? WHERE id=?",
                                 (0 if locked else failed, locked, row["id"]))
            self.audit("lockout" if locked else "login_failed", row["username"], ip)
            raise AuthError("Wrong username or password.")
        with self._lock, self._db:
            self._db.execute("UPDATE users SET failed=0, locked_until=0 WHERE id=?", (row["id"],))
        self.audit("login", row["username"], ip)
        return User(row["id"], row["username"], row["role"])

    def create_session(self, user: User) -> tuple[str, str]:
        """(session token for the cookie, CSRF token). Only the token's hash is stored."""
        token, csrf, now = secrets.token_urlsafe(32), secrets.token_urlsafe(32), self.clock()
        with self._lock, self._db:
            self._db.execute("INSERT INTO sessions (token_hash, user_id, csrf, created, last_seen) VALUES (?, ?, ?, ?, ?)",
                             (token_hash(token), user.id, csrf, now, now))
        return token, csrf

    def get_session(self, token: str | None) -> Session | None:
        if not token:
            return None
        now, th = self.clock(), token_hash(token)
        with self._lock, self._db:
            row = self._db.execute(
                "SELECT s.csrf, s.created, s.last_seen, u.id, u.username, u.role, u.disabled "
                "FROM sessions s JOIN users u ON u.id = s.user_id WHERE s.token_hash=?", (th,)).fetchone()
            if row is None:
                return None
            expired = (now - row["last_seen"] > IDLE_SECONDS or now - row["created"] > ABSOLUTE_SECONDS
                       or row["disabled"])
            if expired:
                self._db.execute("DELETE FROM sessions WHERE token_hash=?", (th,))
                return None
            if now - row["last_seen"] > 60:  # refresh the idle timer at most once a minute
                self._db.execute("UPDATE sessions SET last_seen=? WHERE token_hash=?", (now, th))
        return Session(User(row["id"], row["username"], row["role"]), row["csrf"])

    def delete_session(self, token: str | None, ip: str = "") -> None:
        if not token:
            return
        with self._lock, self._db:
            row = self._db.execute("SELECT u.username FROM sessions s JOIN users u ON u.id = s.user_id "
                                   "WHERE s.token_hash=?", (token_hash(token),)).fetchone()
            self._db.execute("DELETE FROM sessions WHERE token_hash=?", (token_hash(token),))
        if row:
            self.audit("logout", row["username"], ip)

    def purge_expired(self) -> int:
        now = self.clock()
        with self._lock, self._db:
            return self._db.execute("DELETE FROM sessions WHERE ? - last_seen > ? OR ? - created > ?",
                                    (now, IDLE_SECONDS, now, ABSOLUTE_SECONDS)).rowcount

    def audit(self, event: str, username: str | None = None, ip: str = "") -> None:
        with self._lock, self._db:
            self._db.execute("INSERT INTO audit (at, event, username, ip) VALUES (?, ?, ?, ?)",
                             (self.clock(), event, username, ip))

    def recent_audit(self, limit: int = 50) -> list[dict]:
        with self._lock:
            rows = self._db.execute("SELECT at, event, username, ip FROM audit ORDER BY at DESC LIMIT ?",
                                    (limit,)).fetchall()
        return [dict(r) for r in rows]


class RateLimiter:
    """At most `limit` events per `window` seconds per key (e.g. login attempts per IP address)."""

    def __init__(self, limit: int = 20, window: float = 600, clock=time.time):
        self.limit, self.window, self.clock = limit, window, clock
        self._hits: dict[str, list[float]] = {}
        self._lock = threading.Lock()

    def allow(self, key: str) -> bool:
        now = self.clock()
        with self._lock:
            hits = [t for t in self._hits.get(key, []) if now - t < self.window]
            if len(hits) >= self.limit:
                self._hits[key] = hits
                return False
            hits.append(now)
            self._hits[key] = hits
            return True
