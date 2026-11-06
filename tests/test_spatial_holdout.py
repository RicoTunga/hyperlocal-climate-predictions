"""Tests for the spatial hold-out design in src/models/spatial_holdout.py (synthetic grids)."""
import numpy as np
import pandas as pd

from src.models.spatial_holdout import BLOCKS, assign_blocks, training_pool
from src.models.uncertainty import combine_runs
from tests.test_uncertainty import make_predictions


def grid(n_lat=9, n_lon=7):
    rows = [{"pixel_id": f"{i}_{j}", "lat": round(-1 + 0.05 * i, 3), "lon": round(35 + 0.05 * j, 3),
             "elev_mean_m": 1000 + 10 * i + j} for i in range(n_lat) for j in range(n_lon)]
    return pd.DataFrame(rows)


def test_blocks_partition_the_grid_into_contiguous_quadrants():
    z = assign_blocks(grid())
    assert set(z["block"]) == set(BLOCKS) and z["block"].notna().all()
    assert len(z) == z["pixel_id"].nunique()  # each pixel in exactly one block
    sw, ne = z[z["block"] == "SW"], z[z["block"] == "NE"]
    assert sw["gi"].max() < ne["gi"].min() and sw["gj"].max() < ne["gj"].min()


def test_training_pool_keeps_the_buffer():
    z = assign_blocks(grid())
    for block in BLOCKS:
        held = z[z["block"] == block]
        pool = training_pool(z, block, buffer=2)
        assert not set(pool["pixel_id"]) & set(held["pixel_id"])
        d = np.max(np.abs(pool[["gi", "gj"]].to_numpy()[:, None] - held[["gi", "gj"]].to_numpy()[None]), axis=2)
        assert d.min() > 2  # every training candidate is more than 2 cells from the held-out block
        assert len(pool) > 0


def test_combine_runs_with_text_labels_and_model_filter():
    a = make_predictions(seed=0)
    b = a.copy()
    b["lightgbm__occ1"] = np.clip(b["lightgbm__occ1"] + 0.05, 0, 1)
    c = combine_runs({"holdout": a, "standard": b}, keep_models=["lightgbm"])
    assert {"lightgbm@holdout__occ1", "lightgbm@standard__occ1"} <= set(c.columns)
    assert not any(col.startswith("xgboost") for col in c.columns)  # filtered out
