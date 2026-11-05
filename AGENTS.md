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
src/features/            merge.py                            (joins sources; feature code goes here)
src/models/              evaluation.py (splits, horizon frames, metrics), baselines.py; training and tuning go here
src/explain/             (empty) SHAP
app/                     (empty) Streamlit dashboard, built last
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

## Evaluation harness (built)

`src/models/evaluation.py` defines the convention every model must use: a forecast issued on day t (data up to t) is for target day t + h. All horizons are scored on the same target days, `[split start + 7 days, split end]` (validation: 2024-01-08..2024-12-31), which leaves 7 clear days after the last training target. `load_site(site, until=...)` drops later rows on load, so pass the split end, never the test period while developing. `python -m src.models.baselines` scores persistence and per-pixel day-of-year climatology (training years only, +/- 15 days) on validation and writes `reports/baselines_validation.csv`.

## Rules (these protect the validity of the results)

1. **Time-series integrity.** Never use random train/test splits or random K-fold. Use time-ordered splits (`TimeSeriesSplit` with a gap of at least the forecast horizon). Features for a prediction made on day t may use only data from day t and earlier. Rolling and lag features must be computed per pixel and must not include the target day. Fit any scaler or imputer inside the training fold only.
2. **Keep a final test period untouched** until the very end. Do not tune on it or look at it while developing.
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

## Data split (decided, do not change without asking)
Train 2016-01-01 to 2023-12-31. Validation 2024. Final test 2025-01-01 to 2026-09-30 (about 21 months), touched only for final reporting. Use time-ordered splits with a gap of at least 7 days; never shuffle.
