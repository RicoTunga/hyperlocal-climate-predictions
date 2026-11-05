"""Evaluation helpers shared by the baselines and (later) the models.

Forecast convention (AGENTS.md rules 1-3):
- A forecast is *issued* on day t, using data up to and including day t, and is for the
  *target* day t + h, with h = 1..7 days.
- Every horizon is scored on the same target days: [split start + longest horizon, split end].
  So the earliest issue day equals the split start, and the first target day lies
  `gap_days` clear days after the previous split's last target day.
- Values are matched by (pixel_id, date), never by row position, so one pixel's history can
  never leak into another's and a missing day raises an error instead of shifting silently.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
SITES = ROOT / "config" / "sites.yaml"
EXPERIMENT = ROOT / "config" / "experiment.yaml"
PROCESSED = ROOT / "data" / "processed"

SPLIT_ORDER = ["train", "validation", "test"]


def load_configs() -> tuple[dict, dict]:
    with open(SITES, encoding="utf-8") as f:
        sites = yaml.safe_load(f)
    with open(EXPERIMENT, encoding="utf-8") as f:
        exp = yaml.safe_load(f)
    return sites, exp


def split_bounds(exp: dict, name: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    s = exp["splits"][name]
    return pd.Timestamp(str(s["start"])), pd.Timestamp(str(s["end"]))


def load_site(site: str, until: pd.Timestamp, columns: list | None = None) -> pd.DataFrame:
    """Read the merged table, keeping only rows dated on or before `until`.

    Filtering at load time means nothing downstream can see later data (such as the final
    test period) by accident."""
    df = pd.read_parquet(PROCESSED / f"{site}_daily.parquet", columns=columns)
    df = df[df["date"] <= until].reset_index(drop=True)
    assert df["date"].max() <= until
    return df


def doy365(dates) -> np.ndarray:
    """Calendar day of year 1..365. In leap years 29 Feb shares day 59 with 28 Feb, so that
    1 March is always day 60 and the same calendar date always gets the same number."""
    d = pd.DatetimeIndex(dates)
    doy = d.day_of_year.to_numpy()
    return doy - (d.is_leap_year & (doy >= 60)).astype(int)


def eval_target_dates(exp: dict, split: str, max_h: int) -> pd.DatetimeIndex:
    """Common target days for all horizons, with the gap to the previous split checked."""
    start, end = split_bounds(exp, split)
    first = start + pd.Timedelta(days=max_h)
    i = SPLIT_ORDER.index(split)
    if i > 0:
        _, prev_end = split_bounds(exp, SPLIT_ORDER[i - 1])
        clear_days = (first - prev_end).days - 1
        if clear_days < exp["gap_days"]:
            raise ValueError(f"Only {clear_days} clear days between {SPLIT_ORDER[i - 1]} and "
                             f"{split} targets; need {exp['gap_days']}.")
    return pd.date_range(first, end, freq="D")


def horizon_frame(df: pd.DataFrame, var: str, h: int, target_dates) -> pd.DataFrame:
    """One row per pixel per target day:
        pixel_id | target_date | y (value on target day) | issue_date (target - h) | issue_value
    """
    target_dates = pd.DatetimeIndex(target_dates)
    tgt = (df.loc[df["date"].isin(target_dates), ["pixel_id", "date", var]]
           .rename(columns={"date": "target_date", var: "y"}))
    iss = df[["pixel_id", "date", var]].rename(columns={"date": "issue_date", var: "issue_value"})
    iss["target_date"] = iss["issue_date"] + pd.Timedelta(days=h)
    iss = iss[iss["target_date"].isin(target_dates)]
    out = tgt.merge(iss, on=["pixel_id", "target_date"], how="left", validate="one_to_one")

    expected = df["pixel_id"].nunique() * len(target_dates)
    if len(out) != expected:
        raise ValueError(f"{var} h={h}: {len(out)} target rows, expected {expected} (missing days?)")
    if out["issue_value"].isna().any() or out["y"].isna().any():
        raise ValueError(f"{var} h={h}: missing issue-day or target-day values")
    assert ((out["target_date"] - out["issue_date"]).dt.days == h).all()
    return out.sort_values(["pixel_id", "target_date"]).reset_index(drop=True)


# ---------------------------------------------------------------- metrics
def brier(y, p) -> float:
    y, p = np.asarray(y, float), np.asarray(p, float)
    return float(np.mean((p - y) ** 2))


def binary_scores(y, p, p_ref) -> dict:
    """Brier score, Brier skill score against a reference forecast, ROC-AUC.
    BSS > 0 means better than the reference; 0 = same; < 0 = worse."""
    y = np.asarray(y, float)
    b, b_ref = brier(y, p), brier(y, p_ref)
    auc = roc_auc_score(y, p) if 0 < y.mean() < 1 else np.nan
    return {"brier": b, "bss": 1 - b / b_ref if b_ref > 0 else np.nan, "auc": float(auc)}


def continuous_scores(y, yhat, yhat_ref) -> dict:
    """MAE, RMSE, R2 and RMSE skill (1 - RMSE / RMSE of the reference forecast).

    R2 is pooled over all pixel-days, so it includes differences *between* pixels; it is
    not the same as the average R2 of each pixel's own time series."""
    y, yhat, yhat_ref = (np.asarray(a, float) for a in (y, yhat, yhat_ref))
    err = yhat - y
    rmse = float(np.sqrt(np.mean(err ** 2)))
    rmse_ref = float(np.sqrt(np.mean((yhat_ref - y) ** 2)))
    sst = float(np.sum((y - y.mean()) ** 2))
    return {"mae": float(np.mean(np.abs(err))), "rmse": rmse,
            "r2": 1 - float(np.sum(err ** 2)) / sst if sst > 0 else np.nan,
            "rmse_skill": 1 - rmse / rmse_ref if rmse_ref > 0 else np.nan}
