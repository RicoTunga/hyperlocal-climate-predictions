"""In-app alerts for farmers: the newest forecast for each saved place, checked against four rules.

Alert types (`KINDS`):
- "blight": high late-blight risk (the tested model, Nakuru only), for users who grow potato or tomato;
- "spray":  two or more dry days in a row starting within 3 days, a window for spraying, harvesting or drying;
- "heavy":  heavy rain likely in the next 3 days (the same "heavy" rule as the crop advice, web/crops.py);
- "humid":  3 or more days in a row with humidity at or above 85% (long leaf wetness; fungal diseases).

How it runs: whenever the app asks for alerts (GET /api/notifications), each of the user's saved places is
checked against the newest issue day of its site (live forecasts too, when HL_LIVE=1), and the result is
merged into the `notifications` table:
- a new event that overlaps the dates of an existing alert of the same type and place UPDATES that alert
  instead of adding another one (a dry spell seen on five issue days is one alert, not five). It becomes
  unread again only if it got more serious (see LEVEL_RANK) or had been cleared;
- an alert that a newer forecast no longer shows is marked "cleared" (kept, so the farmer sees it eased);
- a forecast older than the newest one already used for that place changes nothing (for example when live
  mode is switched off again and the newest issue day goes back to the end of the replay).

Reliability: every alert says whether its days lie within the lead times where that kind of forecast
beat the normal for the date on the final test (reports/test/skill.csv, 95% interval above 0): all of them
('good'), only the first ('partly') or none ('low'). Live
forecasts take theirs from the 2024 live backtest (reports/live/backtest_skill.csv): no rain-based day is
reliable there; humidity only on days 1-2 at Nakuru. The rules themselves (thresholds) were not checked
against farm outcomes, the same as the crop guidance rules.

Pure functions (`detect`) decide what fires; `NotificationStore` keeps the alerts in the accounts database,
next to the saved places, so deleting a place or an account deletes its alerts (ON DELETE CASCADE).
"""
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from web import crops as C

KINDS = ("blight", "spray", "heavy", "humid")
KIND_NAMES = {"blight": "Late-blight risk", "spray": "Spray and harvest windows", "heavy": "Heavy rain",
              "humid": "Humid spells"}
LEVEL_RANK = {"info": 0, "watch": 1, "act": 2}
BLIGHT_CROPS = ("potato", "tomato")
SPRAY_RUN = 2          # dry days in a row for a spray window ...
SPRAY_START_BY = 3     # ... starting on day 1, 2 or 3
HEAVY_WITHIN = 3       # heavy rain looked for on days 1-3
HUMID_RUN = 3          # humid days in a row for a humid spell
KEEP_DAYS = 30         # alerts that ended more than this many days before the newest forecast are deleted
MAX_LIST = 40
# lead times whose test skill beat the normal for the date (95% interval above 0), reports/test/skill.csv
HEAVY_SKILL_DAYS = {"nakuru": 2, "busia": 2}       # rain >= 10 mm
RH_SKILL_DAYS = {"nakuru": 7, "busia": 3}          # humidity
LIVE_RH_SKILL_DAYS = {"nakuru": 2, "busia": 0}     # live backtest 2024, scenario live_v2_qm

SCHEMA = """
CREATE TABLE IF NOT EXISTS notifications (
    id INTEGER PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    place_id INTEGER NOT NULL REFERENCES places(id) ON DELETE CASCADE,
    kind TEXT NOT NULL,
    issue_date TEXT NOT NULL,
    first_date TEXT NOT NULL,
    last_date TEXT NOT NULL,
    title TEXT NOT NULL,
    body TEXT NOT NULL,
    level TEXT NOT NULL,
    reliability TEXT NOT NULL,
    live INTEGER NOT NULL DEFAULT 0,
    state TEXT NOT NULL DEFAULT 'active',
    read INTEGER NOT NULL DEFAULT 0,
    created REAL NOT NULL,
    updated REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS notifications_user ON notifications (user_id, place_id, kind);
CREATE TABLE IF NOT EXISTS notify_off (
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    kind TEXT NOT NULL,
    PRIMARY KEY (user_id, kind)
);
"""


@dataclass(frozen=True)
class Event:
    kind: str
    hs: tuple          # horizons (days ahead) the event covers
    title: str
    body: str
    level: str         # info | watch | act
    reliability: str   # good | partly | low


# ---------------------------------------------------------------- reliability
def reliable(kind: str, site: str, h: int, live: bool) -> str:
    """'good' if this kind of forecast beat the normal for the date at lead time h, else 'low'."""
    if kind == "blight":
        return C.confidence(site, h, "blight", live)
    if kind == "humid":
        limit = (LIVE_RH_SKILL_DAYS if live else RH_SKILL_DAYS).get(site, 0)
    elif kind == "heavy":
        limit = 0 if live else HEAVY_SKILL_DAYS.get(site, 0)
    else:  # spray: dry days, i.e. the rain >= 1 mm forecast
        limit = C.rain_skill_days(site, live)
    return "good" if h <= limit else "low"


def reliable_span(kind: str, site: str, hs, live: bool) -> str:
    """'good' if every day of the event is within the tested lead times, 'partly' if only the first is, else 'low'."""
    first, last = reliable(kind, site, hs[0], live), reliable(kind, site, hs[-1], live)
    return "good" if last == "good" else "partly" if first == "good" else "low"


# ---------------------------------------------------------------- rules
def runs(days, test, n) -> list[tuple[int, int]]:
    """(first, last) horizons of every run of at least n consecutive days where test(day) is true."""
    out, start, last = [], None, None
    for d in days:
        if test(d):
            start = d["h"] if start is None else start
            last = d["h"]
        else:
            if start is not None and last - start + 1 >= n:
                out.append((start, last))
            start = None
    if start is not None and last - start + 1 >= n:
        out.append((start, last))
    return out


def pct(p) -> str:
    return f"{round(100 * (p or 0))}%"


def blight_event(days, site, live, crops):
    grown = [c for c in BLIGHT_CROPS if c in crops]
    high = [d["h"] for d in days if (d.get("blight") or 0) >= C.BLIGHT_HIGH]
    if not grown or not high:
        return None
    first = high[0]
    dry_before = [d["h"] for d in days if d["h"] <= first and C.dry(d)]
    crop_names = " and ".join(C.CROPS[c].name.lower() for c in grown).capitalize()
    body = f"{crop_names}: blight weather is likely on {C.names(days, high)}. "
    body += (f"{days[dry_before[0] - 1]['label']} looks dry, a chance to spray a protectant fungicide before it."
             if dry_before else "No dry day before it: spray at the first break in the rain.")
    if "tomato" in grown:
        body += " (The model was built for potato; tomato gets the same disease.)"
    return Event("blight", tuple(high), "High late-blight risk", body, "act", reliable_span("blight", site, high, live))


def spray_event(days, site, live, crops):
    window = next(((a, b) for a, b in runs(days, C.dry, SPRAY_RUN) if a <= SPRAY_START_BY), None)
    if window is None:
        return None
    a, b = window
    worst = max(days[h - 1]["p1"] for h in range(a, b + 1))
    body = (f"{C.span(days, a, b)} looks dry (chance of rain at most {pct(worst)} each day): a window for spraying, "
            "harvesting or drying grain.")
    hs = tuple(range(a, b + 1))
    return Event("spray", hs, "Dry window", body, "info", reliable_span("spray", site, hs, live))


def heavy_event(days, site, live, crops):
    hs = [d["h"] for d in days[:HEAVY_WITHIN] if C.heavy(d)]
    if not hs:
        return None
    p10 = max(days[h - 1]["p10"] or 0 for h in hs)
    p25 = max(days[h - 1]["p25"] or 0 for h in hs)
    body = (f"Heavy rain is likely on {C.names(days, hs)} (chance of 10 mm or more up to {pct(p10)}, 25 mm or more up "
            f"to {pct(p25)}). Clear drainage furrows, and hold off fertiliser and spraying until it passes.")
    return Event("heavy", tuple(hs), "Heavy rain likely", body, "watch", reliable_span("heavy", site, hs, live))


def humid_event(days, site, live, crops):
    spells = runs(days, C.humid, HUMID_RUN)
    if not spells:
        return None
    a, b = max(spells, key=lambda r: r[1] - r[0])  # the longest spell
    body = (f"Humidity at or above {C.HUMID_RH:.0f}% on {C.span(days, a, b)}: leaves stay wet longer, which favours "
            "fungal diseases, and crops dry slowly. Scout for leaf spots after the spell.")
    hs = tuple(range(a, b + 1))
    return Event("humid", hs, "Humid spell", body, "watch", reliable_span("humid", site, hs, live))


RULES = {"blight": blight_event, "spray": spray_event, "heavy": heavy_event, "humid": humid_event}


def detect(days: list[dict], site: str, crops=(), has_blight: bool = False, kinds=KINDS) -> list[Event]:
    """The alerts one square's 7 forecast days raise (days from web.crops.pixel_days, live days marked)."""
    live = C.is_live(days)
    out = []
    for kind in kinds:
        if kind == "blight" and not has_blight:
            continue  # no tested late-blight model for this site
        e = RULES[kind](days, site, live, tuple(crops))
        if e is not None:
            out.append(e)
    return out


# ---------------------------------------------------------------- store
class NotifyError(ValueError):
    """Bad input from the user; the message is safe to show."""


class NotificationStore:
    def __init__(self, path: str | Path, clock=time.time):
        self.clock = clock
        self._lock = threading.Lock()
        self._db = sqlite3.connect(Path(path), check_same_thread=False, timeout=10)
        self._db.row_factory = sqlite3.Row
        with self._lock, self._db:
            self._db.execute("PRAGMA foreign_keys = ON")
            self._db.executescript(SCHEMA)

    def close(self) -> None:
        self._db.close()

    # ------------------------------------------------ settings
    def prefs(self, user_id: int) -> dict:
        with self._lock:
            off = {r["kind"] for r in self._db.execute("SELECT kind FROM notify_off WHERE user_id=?", (user_id,))}
        return {k: k not in off for k in KINDS}

    def set_prefs(self, user_id: int, patch: dict) -> dict:
        if not isinstance(patch, dict) or any(k not in KINDS for k in patch):
            raise NotifyError(f"alert types are {', '.join(KINDS)}")
        with self._lock, self._db:
            for k, on in patch.items():
                if on:
                    self._db.execute("DELETE FROM notify_off WHERE user_id=? AND kind=?", (user_id, k))
                else:
                    self._db.execute("INSERT OR IGNORE INTO notify_off (user_id, kind) VALUES (?, ?)", (user_id, k))
        return self.prefs(user_id)

    # ------------------------------------------------ merging a forecast into the alerts
    def sync(self, user_id: int, place_id: int, issue: str, days: list[dict], events: list[Event]) -> None:
        """Merge one place's events from the forecast issued on `issue` (see the module docstring)."""
        issue = str(pd.Timestamp(issue).date())
        live, now = int(C.is_live(days)), self.clock()
        window = (days[0]["date"], days[-1]["date"])
        with self._lock, self._db:
            old = [dict(r) for r in self._db.execute(
                "SELECT * FROM notifications WHERE user_id=? AND place_id=?", (user_id, place_id))]
            if old and issue < max(o["issue_date"] for o in old):
                return  # an older forecast (e.g. live mode switched off again) never changes newer alerts
            matched = set()
            for e in events:
                first, last = days[e.hs[0] - 1]["date"], days[e.hs[-1] - 1]["date"]
                hit = next((o for o in old if o["kind"] == e.kind and o["id"] not in matched
                            and o["first_date"] <= last and o["last_date"] >= first), None)
                if hit is None:
                    self._db.execute(
                        "INSERT INTO notifications (user_id, place_id, kind, issue_date, first_date, last_date, title, "
                        "body, level, reliability, live, state, read, created, updated) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'active', 0, ?, ?)",
                        (user_id, place_id, e.kind, issue, first, last, e.title, e.body, e.level, e.reliability, live,
                         now, now))
                    continue
                matched.add(hit["id"])
                new = {"issue_date": issue, "first_date": first, "last_date": last, "title": e.title, "body": e.body,
                       "level": e.level, "reliability": e.reliability, "live": live, "state": "active"}
                if all(hit[k] == v for k, v in new.items()):
                    continue
                worse = LEVEL_RANK[e.level] > LEVEL_RANK.get(hit["level"], 0) or hit["state"] == "cleared"
                self._db.execute(
                    "UPDATE notifications SET issue_date=?, first_date=?, last_date=?, title=?, body=?, level=?, "
                    "reliability=?, live=?, state='active', read=?, updated=? WHERE id=?",
                    (issue, first, last, e.title, e.body, e.level, e.reliability, live,
                     0 if worse else hit["read"], now, hit["id"]))
            # alerts a newer forecast no longer shows, for days that forecast covers
            for o in old:
                if (o["id"] not in matched and o["state"] == "active" and issue > o["issue_date"]
                        and o["last_date"] >= window[0]):
                    self._db.execute("UPDATE notifications SET state='cleared', issue_date=?, updated=? WHERE id=?",
                                     (issue, now, o["id"]))
            cutoff = str((pd.Timestamp(issue) - pd.Timedelta(days=KEEP_DAYS)).date())
            self._db.execute("DELETE FROM notifications WHERE user_id=? AND place_id=? AND last_date < ?",
                             (user_id, place_id, cutoff))

    # ------------------------------------------------ reading
    def items(self, user_id: int, newest: dict | None = None) -> list[dict]:
        """The user's alerts, newest first, for the kinds they have switched on. `newest` maps site -> the newest
        issue day, to mark alerts whose days are over."""
        on = [k for k, v in self.prefs(user_id).items() if v]
        if not on:
            return []
        q = ",".join("?" * len(on))
        with self._lock:
            rows = self._db.execute(
                "SELECT n.*, p.name AS place_name, p.site, p.pixel_id FROM notifications n "
                f"JOIN places p ON p.id = n.place_id WHERE n.user_id=? AND n.kind IN ({q}) "  # q: placeholders only
                "ORDER BY n.updated DESC, n.id DESC LIMIT ?", (user_id, *on, MAX_LIST)).fetchall()
        out = []
        for r in rows:
            d = {k: r[k] for k in ("id", "place_id", "place_name", "site", "pixel_id", "kind", "issue_date", "first_date",
                                   "last_date", "title", "body", "level", "reliability", "state", "created", "updated")}
            d["live"], d["read"] = bool(r["live"]), bool(r["read"])
            start = (newest or {}).get(r["site"])
            first_target = str((pd.Timestamp(start) + pd.Timedelta(days=1)).date()) if start else None
            d["ended"] = bool(first_target and r["last_date"] < first_target)
            out.append(d)
        return out

    def mark_read(self, user_id: int, ids=None) -> int:
        """Mark the given alerts (or all, ids=None) as read; only the user's own alerts are touched."""
        with self._lock, self._db:
            if ids is None:
                return self._db.execute("UPDATE notifications SET read=1 WHERE user_id=? AND read=0", (user_id,)).rowcount
            if not isinstance(ids, list) or len(ids) > 200 or not all(isinstance(i, int) for i in ids):
                raise NotifyError("ids must be a list of alert ids")
            n = 0
            for i in ids:
                n += self._db.execute("UPDATE notifications SET read=1 WHERE id=? AND user_id=?", (i, user_id)).rowcount
            return n
