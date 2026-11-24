"""Live mode for the dashboard (branch feature/live). Off unless HL_LIVE=1, so the replay is unchanged.

When on, and `python -m src.live.forecast` has written app/data/live/, the dashboard's issue days run on
past the replay's last day to the newest live forecast (yesterday). The replay files are never modified:
live rows are read from their own files and joined in memory.
"""
import json
import os
from functools import lru_cache
from pathlib import Path

import pandas as pd

DATA = Path(__file__).resolve().parents[1] / "app" / "data"
LIVE = DATA / "live"


def enabled() -> bool:
    return os.environ.get("HL_LIVE", "0") == "1" and (LIVE / "live_meta.json").exists()


@lru_cache(maxsize=1)
def live_meta() -> dict:
    return json.loads((LIVE / "live_meta.json").read_text(encoding="utf-8"))


def overlay_meta(meta: dict) -> dict:
    """Extend each site's issue range with the live forecasts, if they follow the replay without a gap."""
    if not enabled():
        return meta
    lm = live_meta()
    for site, m in meta.items():
        info = lm["sites"].get(site)
        m["replay_issue_max"] = m["issue_max"]
        if not info:
            continue
        if pd.Timestamp(info["issue_min"]) != pd.Timestamp(m["issue_max"]) + pd.Timedelta(days=1):
            continue  # a gap: run `python -m src.live.forecast` again to backfill
        m["issue_max"] = info["issue_max"]
        m["live"] = {"issue_min": info["issue_min"], "issue_max": info["issue_max"], "made": lm["made"],
                     "attribution": lm["attribution"], "rain_source": info.get("rain_source", {})}
    return meta


def is_live(site: str, issue, meta: dict) -> bool:
    m = meta.get(site, {})
    return "live" in m and pd.Timestamp(issue) > pd.Timestamp(m["replay_issue_max"])


def issue_day(site: str, issue) -> pd.DataFrame:
    return pd.read_parquet(LIVE / f"{site}_live.parquet", filters=[("issue_date", "==", pd.Timestamp(issue))])


def pixel_rows(site: str, pixel_id: str, horizon: int, columns: list) -> pd.DataFrame:
    return pd.read_parquet(LIVE / f"{site}_live.parquet", columns=["issue_date", "target_date"] + columns,
                           filters=[("pixel_id", "==", pixel_id), ("horizon", "==", horizon)])


def shap_rows(site: str, issue, pixel_id: str) -> pd.DataFrame:
    return pd.read_parquet(LIVE / f"{site}_live_shap.parquet",
                           filters=[("issue_date", "==", pd.Timestamp(issue)), ("pixel_id", "==", pixel_id)])
