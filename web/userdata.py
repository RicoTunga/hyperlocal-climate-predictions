"""Per-user data: profile and settings, saved places (5 km squares) and crops of interest.

Kept in the same SQLite file as the accounts (web/auth.py), so deleting an account removes its rows
through ON DELETE CASCADE. Every method takes the signed-in user's id from the server, never from the
request body, so one user can never read or change another user's places.

A saved place is one ~5 km forecast square (site + pixel_id) with a name the user chose. Coordinates are
not stored: they come from the square, which keeps the data minimal (a farm is personal information).
"""
import sqlite3
import threading
import time
from pathlib import Path

USER_TYPES = ("farmer", "extension", "researcher", "other")
MAX_PLACES = 50
SCHEMA = """
CREATE TABLE IF NOT EXISTS profiles (
    user_id INTEGER PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
    user_type TEXT NOT NULL DEFAULT 'other',
    home_site TEXT,
    home_place INTEGER,
    onboarded INTEGER NOT NULL DEFAULT 0,
    tour_on_login INTEGER NOT NULL DEFAULT 1,
    tour_done INTEGER NOT NULL DEFAULT 0,
    animations INTEGER NOT NULL DEFAULT 1,
    hud_open INTEGER NOT NULL DEFAULT 1,
    updated REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS places (
    id INTEGER PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    site TEXT NOT NULL,
    pixel_id TEXT NOT NULL,
    name TEXT NOT NULL,
    note TEXT NOT NULL DEFAULT '',
    created REAL NOT NULL,
    UNIQUE (user_id, site, pixel_id)
);
CREATE TABLE IF NOT EXISTS user_crops (
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    crop TEXT NOT NULL,
    PRIMARY KEY (user_id, crop)
);
"""
BOOL_FIELDS = ("onboarded", "tour_on_login", "tour_done", "animations", "hud_open")


class UserDataError(ValueError):
    """Bad input from the user; the message is safe to show."""


def clean_text(s, limit: int, what: str, required: bool = True) -> str:
    """One line of plain text: control characters removed, spaces collapsed, length limited."""
    s = "".join(ch for ch in str(s or "") if ch.isprintable())
    s = " ".join(s.split())
    if required and not s:
        raise UserDataError(f"{what} cannot be empty")
    if len(s) > limit:
        raise UserDataError(f"{what}: at most {limit} characters")
    return s


class UserData:
    def __init__(self, path: str | Path, sites: tuple = ("nakuru", "busia"), crops: tuple = (), clock=time.time):
        self.sites, self.crops_known, self.clock = tuple(sites), tuple(crops), clock
        self._lock = threading.Lock()
        self._db = sqlite3.connect(Path(path), check_same_thread=False, timeout=10)
        self._db.row_factory = sqlite3.Row
        with self._lock, self._db:
            self._db.execute("PRAGMA foreign_keys = ON")
            self._db.executescript(SCHEMA)

    def close(self) -> None:
        self._db.close()

    # ------------------------------------------------ profile
    def profile(self, user_id: int) -> dict:
        with self._lock, self._db:
            self._db.execute("INSERT OR IGNORE INTO profiles (user_id, updated) VALUES (?, ?)", (user_id, self.clock()))
            r = self._db.execute("SELECT * FROM profiles WHERE user_id=?", (user_id,)).fetchone()
        out = {k: r[k] for k in r.keys() if k not in ("user_id", "updated")}
        for k in BOOL_FIELDS:
            out[k] = bool(out[k])
        return out

    def update_profile(self, user_id: int, patch: dict) -> dict:
        cur = self.profile(user_id)
        sets = {}
        for k, v in patch.items():
            if k in BOOL_FIELDS:
                sets[k] = int(bool(v))
            elif k == "user_type":
                if v not in USER_TYPES:
                    raise UserDataError(f"user type must be one of {', '.join(USER_TYPES)}")
                sets[k] = v
            elif k == "home_site":
                if v is not None and v not in self.sites:
                    raise UserDataError("unknown site")
                sets[k] = v
            elif k == "home_place":
                if v is not None and not any(p["id"] == v for p in self.places(user_id)):
                    raise UserDataError("that place is not one of yours")
                sets[k] = v
            else:
                raise UserDataError(f"unknown setting '{k}'")
        if sets:
            cols = ", ".join(f"{k}=?" for k in sets)
            with self._lock, self._db:
                self._db.execute(f"UPDATE profiles SET {cols}, updated=? WHERE user_id=?",  # keys from the fixed list above
                                 (*sets.values(), self.clock(), user_id))
        return {**cur, **{k: (bool(v) if k in BOOL_FIELDS else v) for k, v in sets.items()}}

    # ------------------------------------------------ places
    def places(self, user_id: int) -> list[dict]:
        with self._lock:
            rows = self._db.execute("SELECT id, site, pixel_id, name, note, created FROM places WHERE user_id=? "
                                    "ORDER BY created", (user_id,)).fetchall()
        return [dict(r) for r in rows]

    def add_place(self, user_id: int, site: str, pixel_id: str, name: str, note: str = "") -> dict:
        if site not in self.sites:
            raise UserDataError("unknown site")
        name, note = clean_text(name, 60, "name"), clean_text(note, 200, "note", required=False)
        with self._lock, self._db:
            n = self._db.execute("SELECT COUNT(*) FROM places WHERE user_id=?", (user_id,)).fetchone()[0]
            if n >= MAX_PLACES:
                raise UserDataError(f"you can save up to {MAX_PLACES} places; delete one first")
            try:
                cur = self._db.execute("INSERT INTO places (user_id, site, pixel_id, name, note, created) VALUES (?, ?, ?, ?, ?, ?)",
                                       (user_id, site, pixel_id, name, note, self.clock()))
            except sqlite3.IntegrityError:
                raise UserDataError("this square is already one of your places") from None
            first = n == 0
        if first:  # the first place becomes home, so the at-a-glance panel has something to show
            self.update_profile(user_id, {"home_place": cur.lastrowid, "home_site": site})
        return next(p for p in self.places(user_id) if p["id"] == cur.lastrowid)

    def update_place(self, user_id: int, place_id: int, name: str | None = None, note: str | None = None) -> dict:
        sets = {}
        if name is not None:
            sets["name"] = clean_text(name, 60, "name")
        if note is not None:
            sets["note"] = clean_text(note, 200, "note", required=False)
        if sets:
            cols = ", ".join(f"{k}=?" for k in sets)
            with self._lock, self._db:
                n = self._db.execute(f"UPDATE places SET {cols} WHERE id=? AND user_id=?",
                                     (*sets.values(), place_id, user_id)).rowcount
            if not n:
                raise UserDataError("no such place")
        p = next((p for p in self.places(user_id) if p["id"] == place_id), None)
        if p is None:
            raise UserDataError("no such place")
        return p

    def delete_place(self, user_id: int, place_id: int) -> None:
        with self._lock, self._db:
            n = self._db.execute("DELETE FROM places WHERE id=? AND user_id=?", (place_id, user_id)).rowcount
            if n:
                self._db.execute("UPDATE profiles SET home_place=NULL WHERE user_id=? AND home_place=?", (user_id, place_id))
        if not n:
            raise UserDataError("no such place")

    # ------------------------------------------------ crops
    def crops(self, user_id: int) -> list[str]:
        with self._lock:
            rows = self._db.execute("SELECT crop FROM user_crops WHERE user_id=?", (user_id,)).fetchall()
        mine = {r["crop"] for r in rows}
        return [c for c in self.crops_known if c in mine]  # catalogue order

    def set_crops(self, user_id: int, crops: list) -> list[str]:
        if not isinstance(crops, list) or len(crops) > 50:
            raise UserDataError("crops must be a list")
        crops = list(dict.fromkeys(crops))  # drop repeats, keep order
        bad = [c for c in crops if c not in self.crops_known]
        if bad:
            raise UserDataError(f"unknown crop '{bad[0]}'")
        with self._lock, self._db:
            self._db.execute("DELETE FROM user_crops WHERE user_id=?", (user_id,))
            self._db.executemany("INSERT INTO user_crops (user_id, crop) VALUES (?, ?)", [(user_id, c) for c in set(crops)])
        return self.crops(user_id)

    def everything(self, user_id: int) -> dict:
        return {"profile": self.profile(user_id), "places": self.places(user_id), "crops": self.crops(user_id)}
