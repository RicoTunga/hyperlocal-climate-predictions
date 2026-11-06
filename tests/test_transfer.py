"""Test that the transfer script's prediction path matches training exactly (synthetic data)."""
import numpy as np
import pandas as pd

from src.models.train import feature_columns, fit_predict, predict, task_definitions
from src.models.transfer import predict_all
from tests.test_train import make_table


def test_predict_all_reproduces_fit_predict_including_smearing():
    _, t, targets = make_table(1)
    t = t.reset_index(drop=True)
    cols = feature_columns(t)
    days = sorted(t["target_date"].unique())
    tr, es = t[t["target_date"].isin(days[:50])], t[t["target_date"].isin(days[50:])]
    mcfg = {"seed": 0, "early_stopping_rounds": 10,
            "params": {"catboost": {"iterations": 40, "depth": 3, "learning_rate": 0.1, "verbose": 0,
                                    "allow_writing_files": False}}}
    wet_mm = 1.0
    tasks = task_definitions(wet_mm, [10])
    models, expected = {}, {}
    for task in ["occ1", "wet", "direct", "rh"]:
        models[task], _, expected[task] = fit_predict("catboost", task, tasks[task], tr, es, t, cols,
                                                      mcfg["params"], mcfg, wet_mm)
    wet_fit = tr[tr["y_rain"] >= wet_mm]
    smear = float(np.mean(np.exp(np.log1p(wet_fit["y_rain"]) - predict(models["wet"], wet_fit[cols], False))))
    got = predict_all(models, smear, t, cols, wet_mm)
    for task in expected:
        assert np.allclose(got[task], expected[task], atol=1e-9), task
    assert np.allclose(got["two_stage"], expected["occ1"] * expected["wet"])
