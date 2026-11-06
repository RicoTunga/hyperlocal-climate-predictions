"""Tests for the training-table logic in src/models/train.py (synthetic data only)."""
import numpy as np
import pandas as pd

from src.features.build import build_features
from src.models.baselines import fit_climatology
from src.models.train import feature_columns, horizon_table
from tests.test_features import EXP, make_grid


def make_table(h):
    df = make_grid(n_days=120)
    df["site"] = "test"
    feat = build_features(df, EXP)
    # 120 synthetic days cannot fill every calendar day; a +/-182-day window covers the year
    clim = fit_climatology(feat, 1.0, [10], window=182)
    targets = pd.date_range(feat["date"].min() + pd.Timedelta(days=40), feat["date"].max())
    return df, horizon_table(feat, clim, h, targets), targets


def test_features_from_issue_day_and_target_from_t_plus_h():
    for h in [1, 3, 7]:
        df, t, targets = make_table(h)
        assert ((t["target_date"] - t["issue_date"]).dt.days == h).all()
        raw = df.set_index(["pixel_id", "date"])
        # target = rain on the target day; the "precip_mm" feature = rain on the issue day
        y_ref = raw.loc[list(zip(t["pixel_id"], t["target_date"])), "precip_mm"].to_numpy()
        x_ref = raw.loc[list(zip(t["pixel_id"], t["issue_date"])), "precip_mm"].to_numpy()
        assert np.allclose(t["y_rain"], y_ref) and np.allclose(t["precip_mm"], x_ref)
        assert len(t) == df["pixel_id"].nunique() * len(targets)


def test_targets_are_not_features():
    _, t, _ = make_table(1)
    cols = feature_columns(t)
    assert not {"y_rain", "y_rh", "target_date", "issue_date", "pixel_id", "date"} & set(cols)
    # no feature column is identical to the target (a sign of leakage)
    for c in cols:
        assert not np.allclose(t[c].fillna(-999).to_numpy(), t["y_rain"].to_numpy()), c


def test_runs_with_different_pixel_counts_use_different_folders(tmp_path, monkeypatch):
    import src.models.train as train
    import src.models.uncertainty as unc
    monkeypatch.setattr(train, "MODELS", tmp_path)
    monkeypatch.setattr(unc, "MODELS", tmp_path)
    a, b, c = (train.run_dir("busia", 60, 60, 1), train.run_dir("busia", 60, 216, 1),
               train.run_dir("busia", 216, 216, 1))
    assert len({a, b, c}) == 3
    for d in (a, b, c):
        d.mkdir(parents=True)
        (d / "val_predictions.parquet").write_bytes(b"")
    found = {(n_tr, n_sc, h) for n_tr, n_sc, h, _ in unc.prediction_files("busia")}
    assert found == {(60, 60, 1), (60, 216, 1), (216, 216, 1)}
