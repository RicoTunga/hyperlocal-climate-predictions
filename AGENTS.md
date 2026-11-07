# AGENTS.md

Instructions for any AI coding agent working in this repository. Read this file first, then `docs/PROJECT_CONTEXT.md` for the full background, decisions and plan.

## What this project is

Final-year BSc project (Strathmore University): forecast **daily rainfall (mm)** and **relative humidity (%)**, **1 to 7 days ahead**, for each ~5 km grid cell ("pixel") in Nakuru County and a Busia (Kenya) study area, using gradient boosting (XGBoost, LightGBM, CatBoost) with SHAP explanations, then a Streamlit dashboard. Supervisor priority: model accuracy and honest evaluation over UI polish.

The user is a student. Explain what you do and why, keep code readable and commented, and do not silently change decisions that are recorded in `docs/PROJECT_CONTEXT.md`.

## Environment

- Windows, PowerShell. Python virtual environment in `.venv` (activate with `.venv\Scripts\activate`).
- Run scripts as modules from the repo root: `python -m src.features.merge` (not `python src/...`).
- Dependencies are in `requirements.txt`. Laptop has no dedicated server; keep training CPU-friendly.
- Secrets: Earth Engine project ID is in `.env` (`GEE_PROJECT`), Copernicus CDS key is in `~/.cdsapirc`. Never print, copy or commit either.

## Repo map

```
config/sites.yaml        study sites, boxes, ERA5 areas, period 2016-01-01..2026-09-30, horizons, targets
config/experiment.yaml   train/validation/test dates, gap_days, rain thresholds, climatology window
src/ingestion/           chirps.py, era5.py, elevation.py   (data acquisition)
src/features/            merge.py (joins sources), build.py (per-pixel features -> data/processed/features_<site>.parquet), humid_hours.py (hours at RH >= 90%, separate file)
src/models/              evaluation.py (splits, horizon frames, metrics), baselines.py, train.py (models per horizon/task/algorithm), uncertainty.py (bootstrap intervals), tune.py (Optuna), calibration.py (reliability of rain probabilities), recalibrate.py (Platt test, not used), spatial_holdout.py (2 x 2 block hold-out), transfer.py (one site's models on the other), final_test.py (protocol test), fusion.py (ensemble and climatology blends, no gain), late_blight.py (Nakuru late-blight risk)
src/explain/             shap_analysis.py (SHAP by input group: overall and within-day)
app/                     build_data.py (dashboard data from the saved test forecasts -> app/data/, git-ignored), components.py (layers, class breaks, OKLab colour ramps)
web/                     dashboard: server.py (local server, 127.0.0.1 only, JSON/PNG API), raster.py (colour fields), static/ (HTML/CSS/JS front end), screenshot.py (Playwright dev tool)
notebooks/               01_data_tour.ipynb
tests/                   pytest tests (run: python -m pytest -q)
reports/                 small result tables (committed), e.g. baselines_validation.csv
data/raw, data/processed, models/   git-ignored, rebuilt by the scripts
```

## Current data state (all built and checked)

`data/processed/nakuru_daily.parquet` (2,732,496 rows, 696 pixels) and `busia_daily.parquet` (848,016 rows, 216 pixels). One row per pixel per day, 2016-01-01 to 2026-09-30 (3,926 days), 21 columns: `site, pixel_id, lat, lon, date, precip_mm`, 11 ERA5 daily columns (`rh_pct, rh_min_pct, rh_max_pct, t2m_c, t2m_min_c, t2m_max_c, d2m_c, sp_hpa, wind_ms, wind_max_ms, tcc`), 4 elevation columns (`elev_mean_m, elev_std_m, elev_min_m, elev_max_m`). No missing values.

To rebuild from scratch, in order:
1. `python -m src.ingestion.chirps`
2. ERA5: `python test_timeseries.py --site <site> --start 2016-01-01 --end 2026-10-21`, move the downloaded file into `data/raw/era5_manual/<site>/`, then `python -m src.ingestion.era5 --local --site <site>`
3. `python -m src.ingestion.elevation`
4. `python -m src.features.merge`
5. `python -m src.features.build`
6. Dashboard (after the final test and late-blight test): `python -m app.build_data`, then `python -m web.server` and open http://127.0.0.1:8765. Screenshots for review: `python -m web.screenshot` (needs Playwright; browsers live in the git-ignored `.playwright/`). (72 features, settings in `config/experiment.yaml`; about 1 minute, builds in pixel batches because the laptop has ~8 GB RAM)

## Evaluation harness (built)

`src/models/evaluation.py` defines the convention every model must use: a forecast issued on day t (data up to t) is for target day t + h. All horizons are scored on the same target days, `[split start + 7 days, split end]` (validation: 2024-01-08..2024-12-31), which leaves 7 clear days after the last training target. `load_site(site, until=...)` drops later rows on load, so pass the split end, never the test period while developing. `python -m src.models.baselines` scores persistence and per-pixel day-of-year climatology (training years only, +/- 15 days) on validation and writes `reports/baselines_validation.csv`. `python -m src.models.train --horizons 1 ...` trains LightGBM, XGBoost and CatBoost on `modelling.dev_pixels` pixels per site (evenly spread over elevation), fits on train targets before 2023, early-stops on 2023 (never on validation), scores on validation next to both baselines on the same rows, and appends to `reports/models_validation.csv`; fitted models and `val_predictions.parquet` go to `models/<site>/train<N>_score<M>/h<h>/` (N training pixels, M scored pixels; `--pixels N --score-pixels M|all`), so runs of different sizes never overwrite each other; `models_validation.csv` is keyed by site, horizon, model, n_train_pixels and n_score_pixels. `python -m src.models.uncertainty` then gives 95% intervals by a moving-block bootstrap over target days (7-day blocks, all pixels of a day kept together, the same resamples for every model so differences are paired), written to `reports/skill_ci_validation.csv` and `reports/model_differences_ci_validation.csv`. Report skill with these intervals; they cover day-to-day variation within 2024 only, not year-to-year. `python -m src.models.tune` tunes CatBoost with Optuna on training years only (expanding-window folds scored on 2021, 2022, 2023; climatology refitted per fold; data loaded only to 2023-12-31); `train --tuned` uses `reports/tuning/best_params.json` and saves results as `catboost_tuned`.

## Rules (these protect the validity of the results)

1. **Time-series integrity.** Never use random train/test splits or random K-fold. Use time-ordered splits (`TimeSeriesSplit` with a gap of at least the forecast horizon). Features for a prediction made on day t may use only data from day t and earlier. Rolling and lag features must be computed per pixel and must not include the target day. Fit any scaler or imputer inside the training fold only.
2. **Keep a final test period untouched** until the very end. Do not tune on it or look at it while developing. The test protocol is fixed in `docs/PROJECT_CONTEXT.md` section 12; follow it exactly.
3. **Pixels on the same day are strongly correlated** (storms cover many pixels), so split by time, never by pixel or row, for the main evaluation. Spatial hold-out (unseen pixels) is a separate, additional test.
4. **Baselines first.** Every model result must be reported next to persistence and day-of-year climatology, per horizon and per target.
5. **The unit of analysis is the pixel-day.** Do not average over the whole box.
6. **No hard-coded coordinates or dates.** Read them from `config/sites.yaml` (sites, period) and `config/experiment.yaml` (splits).
7. **Never commit** anything in `data/`, `models/`, `.env`, or credentials. Run `git status` before committing and check no `.parquet`, `.nc` or key files appear.
8. **Ask before** large downloads, changing `config/sites.yaml`, or changing a decision listed in `docs/PROJECT_CONTEXT.md`.
9. **Verify, don't assume.** Run what you write on the real data where possible, report the actual output, and say plainly when something was not tested.
10. Prefer small, runnable scripts with a printed sanity report (the existing scripts are the model to follow). Add tests for feature code, especially anything involving time shifts.

## Rainfall modelling decisions (already made)

- Wet day = **at least 1 mm**; also report 10 mm and 25 mm exceedances.
- Rainfall is modelled in **two stages**: probability of rain (>= 1 mm), then amount on wet days (log scale); compare with a single direct model. Metrics: Brier score / ROC-AUC and Brier skill vs climatology for occurrence; MAE and RMSE for amount; RMSE, MAE, R2 for humidity.
- Extreme rainfall days are real widespread events. **Keep them**; do not delete or cap.

## Modelling decisions (already made)

- Main model: **CatBoost with default settings** (tuning gave no measurable gain). LightGBM and XGBoost are reported as comparisons.
- Train on **60 pixels per site** (elevation-spread sample) and **score on all pixels**: `python -m src.models.train --algos catboost --pixels 60 --score-pixels all --horizons 1 2 3 4 5 6 7`. Larger training samples gave no consistent gain and nearly ran out of memory on the 8 GB laptop.
- Runs scored on the same pixels can be compared as paired differences (`uncertainty.py` writes `reports/training_size_ci_validation.csv`).
- `python -m src.models.calibration` checks the rain probabilities of the decided setup (bias with intervals, calibration slope, Brier decomposition, reliability diagrams in `reports/figures/reliability_validation.png`).
- No recalibration: a 2023-fitted Platt correction did not improve 2024 (`recalibrate.py`, kept as the record; its forecasts are saved as `catboost_recal`).
- Memory: runs that load all Nakuru pixels for two periods came within 0.07 GB of the 8 GB limit; close other programs first or work in batches.

## Data split (decided, do not change without asking)
Train 2016-01-01 to 2023-12-31. Validation 2024. Final test 2025-01-01 to 2026-09-30 (about 21 months), touched only for final reporting. Use time-ordered splits with a gap of at least 7 days; never shuffle.
