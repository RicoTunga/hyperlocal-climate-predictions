"""Tests for the final-test helpers in src/models/final_test.py."""
import numpy as np
import pandas as pd

from src.models.evaluation import load_configs
from src.models.final_test import SEASONS, final_config, periods


def test_final_config_changes_only_the_two_protocol_fields():
    _, exp = load_configs()
    f = final_config(exp)
    assert str(f["splits"]["train"]["end"]) == "2024-12-31"
    assert str(f["modelling"]["early_stop_start"]) == "2024-01-01"
    f2 = {**f, "splits": {**f["splits"], "train": exp["splits"]["train"]},
          "modelling": {**f["modelling"], "early_stop_start": exp["modelling"]["early_stop_start"]}}
    assert f2 == exp  # nothing else differs
    assert str(exp["splits"]["train"]["end"]) == "2023-12-31"  # the original config is not modified


def test_seasons_cover_every_month_once_and_periods_split_years():
    months = sorted(m for ms in SEASONS.values() for m in ms)
    assert months == list(range(1, 13))
    d = pd.Series(pd.date_range("2025-01-08", "2026-09-30"))
    p = periods(d)
    assert p["full test period"].all()
    assert np.array_equal(p["2025"] | p["2026 (Jan-Sep)"], p["full test period"])
    assert sum(p[f"season {k}"].sum() for k in SEASONS) == len(d)
