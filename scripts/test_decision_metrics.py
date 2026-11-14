"""Decision-style scores for the rain probabilities on the final test period (post-hoc).

For each site, lead time and rain threshold (1, 10, 25 mm), from the saved test forecasts:
  - ROC-AUC of CatBoost and of climatology (how well each ranks wet days above dry days);
  - with a warning issued when the probability is at least 0.5:
      hit rate          = warned wet days / all wet days           (probability of detection)
      false-alarm ratio = warned dry days / all warnings
      false-alarm rate  = warned dry days / all dry days

These were not in the pre-registered test protocol, so they are post-hoc descriptions.
Reads models/<site>/final_train*_score*/h*/test_predictions.parquet (never changes them)
and writes the new file reports/test/decision_metrics.csv.
Run from the repo root:  python -m scripts.test_decision_metrics
"""
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
OUT = ROOT / "reports" / "test" / "decision_metrics.csv"
WARN_AT = 0.5  # probability at or above which a warning is issued


def scores(y: np.ndarray, p: np.ndarray) -> dict:
    warn = p >= WARN_AT
    wet, dry = y == 1, y == 0
    hits, fa = int((warn & wet).sum()), int((warn & dry).sum())
    return {"auc": roc_auc_score(y, p),
            "hit_rate": hits / wet.sum() if wet.any() else np.nan,
            "false_alarm_ratio": fa / warn.sum() if warn.any() else np.nan,
            "false_alarm_rate": fa / dry.sum() if dry.any() else np.nan,
            "warnings": int(warn.sum())}


def main() -> None:
    exp = yaml.safe_load((ROOT / "config" / "experiment.yaml").read_text())
    sites = list(yaml.safe_load((ROOT / "config" / "sites.yaml").read_text())["sites"])
    thresholds = [exp["rain"]["wet_threshold_mm"]] + exp["rain"]["exceedance_mm"]
    rows = []
    for site in sites:
        for f in sorted((MODELS / site).glob("final_train*_score*/h*/test_predictions.parquet")):
            h = int(f.parent.name[1:])
            pr = pd.read_parquet(f, columns=["y_rain"] + [f"{m}__occ{t:g}" for m in ("catboost", "climatology") for t in thresholds])
            for t in thresholds:
                y = (pr["y_rain"] >= t).to_numpy(int)
                for m in ("catboost", "climatology"):
                    rows.append({"site": site, "horizon": h, "threshold_mm": t, "model": m,
                                 "n": len(y), "events": int(y.sum()), "warn_at": WARN_AT,
                                 **scores(y, pr[f"{m}__occ{t:g}"].to_numpy(float))})
    out = pd.DataFrame(rows).sort_values(["site", "threshold_mm", "horizon", "model"])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)
    with pd.option_context("display.width", 200, "display.float_format", "{:.3f}".format):
        print(out.to_string(index=False))
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
