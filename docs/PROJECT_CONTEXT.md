# Project context and decision log

Save as `docs/PROJECT_CONTEXT.md`. Written at the end of the data-preparation stage (late October / early November 2026) so that any assistant or collaborator can continue the work. Where something is a recommendation rather than a settled decision, it says so.

## 1. Goal

**Title:** Predictive Agro-Meteorological Analytics for Kenyan Smallholders Utilizing Time Series Gradient Boosting Architectures (Strathmore University, ICS Project II).

Produce multi-step daily forecasts (1 to 7 days ahead) of **precipitation (mm)** and **relative humidity (%)** per ~5 km pixel, using XGBoost, LightGBM and CatBoost on engineered time-series features, explain predictions with SHAP, and present results in a Streamlit dashboard. The original proposal (Chapters 1 to 3) is the starting plan, open to change as the project develops.

## 2. Constraints and guidance from the supervisor meeting

- Supervisor prioritises **model output and accuracy over UI**. Delivery channel (SMS, voice, web) is left to the student.
- Wants insight into **downstream effects** of weather patterns (for example disease risk) and suggested looking at **forecast fusion** (combining several forecasts).
- Precipitation and relative humidity were confirmed as sufficient target metrics.
- Told to look at what AfriClimate Africa is doing and **recalibrate scope downward**; told to check the GitHub repo "piggycast" as a reference.
- Kenya Met is the overall authority on forecasting; user testing cannot happen before Kenya Met sign-off. Currently an internal Strathmore iLab research project.
- The "16 GB GPU" is a gaming laptop, not a server. Single researcher, one academic cycle.
- **Mr Solomon** will provide weather-station data from Busia; **nighttime readings may be unreliable**. TAHMO data will not be available.

## 3. Study design

- **Sites:** Nakuru County (pilot, 696 pixels) and a Busia (Kenya) area (216 pixels, includes some Ugandan land).
- **Period:** 2016-01-01 to 2026-09-30 (3,926 days) once the extension is run (CHIRPS ends 2026-09-30; ERA5 files extend to 2026-10-21).
- **Pixel grid:** CHIRPS 0.05 degree cells; `pixel_id` = `"<lat>_<lon>"` of the cell centre with 3 decimals (for example `-0.275_36.025`), identical in every file.
- **Boxes** (lon, lat; snapped outward to the 0.05 degree grid), in `config/sites.yaml`:
  - Nakuru: 35.40 to 36.60 E, -1.20 to 0.25 N. Source: official county extent in the Nakuru County CIDP 2023-2027 (35.41 to 36.60 E, 1.16 S to 0.23 N). The original proposal box (35.50 to 36.65 E, 1.10 S to 0.30 N) clipped about 0.09 degrees off the west edge and 0.06 off the south.
  - Busia: 33.90 to 34.50 E, -0.10 to 0.80 N. **Weak source**: Busia County's own plan gives latitude 0 to 0.75 N and a single longitude (34.42 E); the western edge is inferred from settlement coordinates near the border. Treat as an inferred envelope; revisit when the station location is known.
  - ERA5 download areas (padded one extra 0.25 degree cell so edge pixels have neighbours): Nakuru N 0.5, W 35.0, S -1.5, E 37.0; Busia N 1.25, W 33.5, S -0.5, E 34.75.

## 4. Data sources and processing

| Item | Source | Notes |
|---|---|---|
| Rainfall | CHIRPS v3 daily via Earth Engine, collection `UCSB-CHG/CHIRPS/DAILY` (the proposal text says `UCSB-CHC`; that is a typo) | 0.05 degree, satellite plus gauge; extracted per pixel, monthly chunks cached in `data/raw/chirps/<site>/` |
| Weather | ERA5 via the CDS time-series dataset `reanalysis-era5-single-levels-timeseries` (labelled experimental), one request per site | Hourly, 0.25 degree grid. Variables: 2 m temperature, 2 m dewpoint, surface pressure, 10 m wind u and v, total cloud cover. Download is a zip containing one NetCDF with `valid_time`, `latitude`, `longitude`. The gridded dataset `reanalysis-era5-single-levels` was too slow and rejected year-long requests as too large |
| Humidity | Computed from temperature and dewpoint (Magnus formula), hourly, then aggregated | ERA5 single levels has no direct RH field |
| Daily aggregation | UTC days (same convention as CHIRPS) | Means, plus min and max for RH and temperature, wind max |
| Spatial step | Bilinear interpolation of the ERA5 daily grid onto each CHIRPS pixel centre | ERA5 columns are therefore smooth across neighbouring pixels |
| Elevation | SRTM 30 m, Earth Engine `USGS/SRTMGL1_003`, band `elevation`, sampled at 90 m | Mean, std, min, max per pixel footprint |

NASA POWER and TAHMO are not used (TAHMO unavailable; NASA POWER not needed so far). Soil moisture and ERA5 total precipitation were deliberately left out for now (lake cells; accumulated-variable file stream).

## 5. Findings so far

| | Nakuru | Busia |
|---|---|---|
| Average rainfall per pixel | about 1,293 mm/yr (pixels 695 to 2,294) | about 1,853 mm/yr (pixels 1,062 to 2,398) |
| Wet days (> 1 mm) | 20.9% | 49.6% |
| Days > 25 mm / > 50 mm | 4.1% / 0.7% | 2.0% / 0.1% |
| Mean humidity / temperature | 70.7% / 17.2 C | 75.5% / 22.6 C |
| Elevation range | 1,135 to 3,440 m | 1,080 to 1,518 m |
| Rain vs elevation (rank correlation across pixels) | +0.63 | +0.44 |
| Correlation of rain with humidity / cloud | +0.29 / +0.25 | +0.38 / +0.33 |

Predictability from the past alone (correlation of anomalies, pooled over pixels):

| Lag | Nakuru rain | Nakuru humidity | Busia rain | Busia humidity |
|---|---|---|---|---|
| 1 day | 0.16 | 0.83 | 0.27 | 0.72 |
| 3 days | 0.04 | 0.55 | 0.13 | 0.45 |
| 7 days | 0.00 | 0.37 | 0.05 | 0.28 |

Extreme days are real, widespread events: Nakuru 2018-06-16 (214 pixels above 50 mm, region average 41.6 mm, maximum 240 mm in a pixel); Busia 2020-11-28 (152 of 216 pixels above 50 mm).

## 6. Decisions

Made:
1. Per-pixel extraction (not a box-wide average); both sites ingested.
2. Wet day = at least 1 mm (0.1 and 1 mm barely differ; report 10 and 25 mm too).
3. Rainfall modelled in two stages (occurrence, then amount on wet days, log scale) with a direct single model for comparison.
4. Keep extreme days; model amounts on a log scale; report errors on the raw scale.
5. Expectations: humidity useful to about day 3; rainfall little better than climatology beyond day 1 to 2. Success means beating persistence and climatology on a held-out period, not reaching a fixed accuracy.
6. DECIDED (November 2026): main model = **CatBoost with default settings** (`modelling.params.catboost` in `config/experiment.yaml`). Evidence: in paired comparisons on validation 2024 CatBoost won 23 of 33 clear algorithm differences and lost none (margins at most 0.038 skill); tuning gave no measurable gain (1 of 24 comparisons). LightGBM and XGBoost are kept as reported comparisons and LightGBM for quick experiments.
7. DECIDED (November 2026): train on **60 pixels per site** (evenly spaced by elevation rank, `dev_pixel_sample`) and **score on all pixels** (`--pixels 60 --score-pixels all`). Evidence: training on 200 (Nakuru) or 216 (Busia) pixels gave no consistent gain (7 of 56 paired comparisons excluded 0, 3 better and 4 worse) and nearly exhausted the laptop's memory (0.13 GB free at peak).
8. DECIDED (November 2026): **no recalibration** of the rain probabilities. A Platt correction fitted on 2023 (`src/models/recalibrate.py`) and tested once on 2024 was clearly worse in 4 of 42 paired comparisons and better in none (Nakuru >= 1 mm days 4-6 about -0.025 BSS, because 2023 suggested drier forecasts and 2024 was wetter; Busia's 2024 overconfidence was absent in 2023, fitted slope 1.02-1.22). The 2024 miscalibration is year-to-year variation, not a stable model defect. Calibration is re-checked on the test period.

Provisional, now checked with SHAP (November 2026, `src/explain/shap_analysis.py`): "local skill comes mainly from CHIRPS history and elevation, not ERA5" is only partly right. Day-to-day skill comes mainly from area-wide conditions and ERA5 weather (day-1 rain probability: area-wide 31% / 46% of mean |SHAP| at Nakuru / Busia, own ERA5 25% / 19%, own rain history 13% / 5%; day-1 humidity: own ERA5 73% / 57%). Differences between pixels on the same day come mainly from the per-pixel climatology (which carries elevation and location), the pixel's own rain history and ERA5 gradients (Nakuru day-1 rain: ERA5 31%, climatology 22%, own rain 21%, neighbourhood 10%, elevation/location 9% of within-day SHAP spread). Elevation acts mostly through climatology. By day 7 climatology and season dominate (44-57%), which is how the skill fades. Original provisional text: local skill from CHIRPS history and elevation (needed the between-pixel spread table, notebook section 9); seasonality (two rainy seasons at Nakuru, weaker at Busia; needs notebook section 3).

DECIDED (Option B): train 2016-2023, validation 2024, final test Jan 2025 to 2026-09-30 (about 21 months, untouched until final reporting). Validate with `TimeSeriesSplit` plus a gap of at least 7 days; develop on a sample of roughly 50 to 60 pixels spread across elevations, then scale to all pixels; one model per horizon per target per algorithm; a separate spatial hold-out (unseen pixels) experiment; a "train on Busia, test on Nakuru" transfer experiment.

## 7. Deviations from the proposal (to document in the write-up)

1. Per-pixel instead of box-averaged data. 2. Nakuru box taken from the official county extent. 3. ERA5 from the CDS time-series dataset with six variables. 4. Humidity computed from temperature and dewpoint. 5. No TAHMO; no NASA POWER so far. 6. Busia added as a second site. 7. Rainfall evaluated with occurrence metrics (Brier, ROC-AUC) in addition to RMSE, MAE and R2. 8. Stage-one rainfall classifier plus amount model.

## 8. Plan from here

1. DONE (November 2026): baselines (persistence, day-of-year climatology) and the evaluation harness (time-ordered splits, metrics, per horizon); see `src/models/evaluation.py`, `src/models/baselines.py`, `reports/baselines_validation.csv`. Validation 2024 results: for rainfall, climatology beats persistence at every horizon and threshold (Brier at 1 mm: Nakuru 0.177 vs 0.24 to 0.35; Busia 0.237 vs 0.36 to 0.48), so climatology is the bar for rain. For humidity, persistence beats climatology up to day 3 at Nakuru (RMSE 5.4 vs 8.4 % on day 1) and up to day 2 at Busia (5.1 vs 6.3 %).
2. DONE (November 2026): `src/features/build.py`, 72 features per pixel-day (settings in `config/experiment.yaml`): rainfall lags 1/3/7/14, 7 and 14-day mean/std/min/max and wet-day counts, 30-day mean, days since last wet day; six ERA5 variables (rh, tcc, t2m, d2m, sp, wind) with lags 1/3/7, 7 and 14-day mean/std and day-to-day change for rh, sp, t2m (ERA5 rolling min/max left out to save memory; daily min/max columns exist); regional mean rain, wet-pixel share, humidity and their recent history; 3x3 and 5x5 neighbourhood mean rain; sine/cosine day of year. Regional and neighbourhood features use other pixels on day t. DECIDED (November 2026): the spatial hold-out keeps them as built, because CHIRPS covers every pixel, so a location the model was not trained on still has its own rain history and its area-wide and neighbourhood context; a stricter variant (rebuilding them without the held-out block) was considered and not run. Original plan: per-pixel lags (1, 3, 7, 14 days) and rolling mean/std/min/max (7 and 14 days) of rainfall and ERA5 variables, sine/cosine day of year, pixel latitude, longitude and elevation, regional context (lagged region-wide mean rainfall and humidity). All shifted so day t uses only data up to t.
3. First models (XGBoost, LightGBM, CatBoost) for horizon 1, then all horizons; then Optuna tuning. IN PROGRESS (November 2026): `src/models/train.py`, 60 dev pixels per site, six tasks (rain >= 1/10/25 mm probability, wet-day amount on log scale with smearing back-transform, direct amount, humidity). Horizon 1, validation 2024: rain >= 1 mm BSS vs climatology about +0.20 (Nakuru) and +0.09 to +0.11 (Busia), AUC 0.80 / 0.72; humidity RMSE 4.7 vs 5.4 % persistence (Nakuru) and 4.2 vs 5.1 % (Busia). The three algorithms differ by about 0.005 BSS (not meaningful on one year); two-stage and direct amounts tie on RMSE. Split-count importance is dominated by regional rain/humidity, season and pressure tendency (check with SHAP).
   All horizons 1-7 (validation 2024, best of 3 algorithms, slightly optimistic): rain >= 1 mm BSS vs climatology Nakuru 0.20, 0.10, 0.05, 0.03, 0.02, 0.02, 0.02; Busia 0.11, 0.035, 0.03, 0.03, 0.02, 0.01, 0.00. Humidity RMSE (%) Nakuru 4.7, 6.4, 7.2, 7.6, 7.9, 8.1, 8.2 vs climatology 8.4; Busia 4.2, 5.1, 5.4, 5.5, 5.6, 5.6, 5.7 vs climatology 6.4. So rainfall is skilful on days 1-2 (marginal on day 3) and humidity to about day 3, matching the expectations in section 6; >= 25 mm skill is small (about +0.07 on day 1) and slightly negative from day 4 at Busia. Beyond day 3 the spread between algorithms (0.005-0.025 BSS) is as large as the skill, so confidence intervals are needed before claiming skill there. Beyond day 3 compare with climatology, not persistence (persistence is then worse than climatology).
   Confidence intervals (`src/models/uncertainty.py`, block bootstrap over days, LightGBM): skill vs climatology whose 95% interval excludes 0 - Nakuru rain >= 1 mm days 1-3 (day 1 +0.20 [+0.12, +0.28]), >= 10 and >= 25 mm days 1-2, wet-day amount days 1-3 and 5-7 (small), humidity days 1-4; Busia rain >= 1 mm day 1 only (+0.09 [+0.01, +0.17]), heavy rain never, and >= 10 / >= 25 mm on day 7 clearly WORSE than climatology (check calibration), humidity days 1-6. Two-stage and direct amounts tie at every horizon. Algorithms: 33 of 294 pairwise differences exclude 0 (about 15 expected by chance); CatBoost won 23 and lost none, so it is the preferred final algorithm (differences at most 0.038). The intervals cover day-to-day variation within 2024 for 60 pixels, not year-to-year variation, so they are a lower bound on uncertainty.
   Tuning (`src/models/tune.py`, Optuna, CatBoost, horizon 1, tasks occ1 / wet / rh, 20 trials each, cross-validation on 2021-2023 within the training years, defaults as trial 0): cross-validation gains over the defaults were small (Nakuru +0.6 / +1.4 / +0.2 %, Busia +0.5 / +1.0 / +0.3 % for rain probability / wet-day amount / humidity), much smaller than the differences between years. On validation 2024 (horizons 1-3, paired intervals) only 1 of 24 tuned-vs-default comparisons excluded 0 (Busia day-1 wet-day amount, +0.012 skill; reversed on day 2), about what chance gives. Decided: keep default CatBoost as the main model (section 6, decision 6); report tuning as evidence that results are not sensitive to hyperparameters. Tuned parameters are kept in `reports/tuning/`.
   Scale-up (CatBoost, horizons 1-7, scored on ALL pixels: 696 Nakuru, 216 Busia): models trained on 60 pixels score almost exactly as well on all pixels as on their own 60 (Nakuru rain >= 1 mm day 1 +0.201 [+0.12, +0.28] vs +0.203; humidity +0.441 vs +0.438; Busia rain +0.110 [+0.04, +0.19], humidity +0.327), so skill holds across both boxes (not a strict unseen-pixel test: regional and neighbourhood features still use other pixels). More training pixels did not help (decision 7); comparisons in `reports/training_size_ci_validation.csv`. Run cost for 60 training pixels: 6-7 minutes per site for 7 horizons, peak memory 1.1-2.2 GB.
   Calibration (`src/models/calibration.py`, CatBoost, validation 2024, all pixels): Nakuru is well calibrated (rain >= 1 mm bias -0.011 [-0.034, +0.010] on day 1, -0.02 to -0.03 later with intervals including 0, slope 0.98 on day 1 and 1.0-1.2 later; climatology is clearly too dry in wet 2024, -0.046 [-0.097, -0.000], and CatBoost halves that). Busia shows no significant bias but consistent overconfidence: slope 0.87 (day 1) falling to 0.65 (day 7) for >= 1 mm, 0.68-0.83 for >= 10 mm. Climatology has a similar slope at Busia (0.68), so part of this is 2024's weaker seasonal cycle. Calibration error is small next to resolution at both sites. The negative day-7 heavy-rain skill at Busia comes from zero resolution (no information that far ahead) plus slight overconfidence, not from a systematic error. A Platt recalibration fitted on 2023 did not help on 2024 and is not used (section 6, decision 8).
4. DONE (November 2026): SHAP on the decided CatBoost models (findings in section 6, provisional item).
   DONE (November 2026): spatial hold-out (`src/models/spatial_holdout.py`): 2 x 2 blocks per site, each held out once, training pixels more than 2 grid cells (about 11 km) from the held-out block, 60 training pixels by elevation rank, CatBoost defaults, horizons 1-3, validation 2024. Pooled over the four blocks, every pixel is forecast by a model that never trained near it; paired against the standard model on the same rows: Nakuru rain >= 1 mm day 1 +0.176 [+0.10, +0.26] vs +0.201 (difference -0.025 [-0.035, -0.016]), humidity +0.432 vs +0.441 (-0.009 [-0.014, -0.004]), wet-day amount unchanged; Busia rain >= 1 mm +0.097 [+0.02, +0.17] vs +0.110 (-0.013 [-0.024, -0.003]), humidity unchanged. About 88% of day-1 rain skill and 98-99% of humidity skill carry over to unseen areas; the cost of no nearby training pixel is concentrated in rain occurrence on days 1-2 and is about zero by day 3. All clear differences were negative (11 of 48 per-block comparisons at Nakuru, 7 of 48 at Busia; largest Nakuru SE day-1 rain -0.042); no block fails.
   DONE (November 2026): site transfer (`src/models/transfer.py`): the saved decided-setup CatBoost models of one site applied to the other site's 2024 data (inputs, climatology and area-wide context from the target site), paired against the target's own model; the source models reproduced their own saved forecasts exactly first. Busia -> Nakuru, day 1: rain >= 1 mm +0.105 [+0.03, +0.17] vs own +0.201 (about half), humidity +0.360 vs +0.441 (82%). Nakuru -> Busia, day 1: rain +0.042 [-0.06, +0.14] (not significant) vs own +0.110, humidity +0.289 vs +0.327 (88%). Beyond day 1 transferred rain skill is about zero in both directions (Nakuru's model is worse than climatology at Busia on days 5-7, about -0.06); humidity stays useful to day 2 (Nakuru) and day 3 (Busia). Cause: the sites occupy different input ranges and trees cannot extrapolate (Nakuru rows outside Busia's training range: elevation 98%, surface pressure 96%, wet-day climatology 34%; Busia rows outside Nakuru's: pressure 98%, rain-chance climatology 18%). Conclusion: models generalise within a study area (spatial hold-out) but not well between climatically different areas; a new region needs local training data. Possible future work: relative inputs (anomalies from pixel climatology, pressure change instead of absolute pressure) or pooled training on both sites.
   Then: spatial hold-out and transfer experiments; compare against Busia station data when available.
5. Forecast fusion and downstream (disease-risk) indicators if time allows; Streamlit dashboard last.

## 9. Known issues and caveats

- The "truth" is itself gridded: CHIRPS rainfall estimate and ERA5 reanalysis humidity, not gauge readings. Busia station data is the only independent check (pending).
- ERA5 temperature and pressure come from ERA5's smooth terrain and do not reflect each pixel's true elevation (for example about 747 hPa at a 3,440 m pixel). A lapse-rate correction is a possible later refinement.
- Busia shows 50% wet days and rainfall up to about 2,398 mm/yr at some pixels (the county plan says 760 to 2,000 mm). Possibly a CHIRPS characteristic near Lake Victoria; check against station data.
- Both boxes include land outside the county (and Uganda, for Busia) and lake pixels.
- The CDS time-series dataset is labelled experimental and ERA5 has a reporting delay, so live inference for a deployed dashboard needs a different input (for example an NWP forecast feed). Not solved yet.
- `notebooks/01_data_tour.ipynb` was tested on synthetic data and run by the user; sections 3 and 9 outputs were not yet reported back.
- Evaluation years are wetter than training. Nakuru wet-day share: train 2016-2023 20.1%, validation 2024 24.8%. During the extension data check (area-average monthly totals only, no model results), Jan-Sep 2026 had several months outside the 2016-2025 range (very wet Feb-Mar, dry Jan and Jun-Sep), and ERA5 humidity/cloud agree. Consequences: climatology is a weaker baseline in those years, and probability forecasts calibrated to the training base rate may be miscalibrated. Plan: report test results for 2025 and 2026 separately and by season, and show skill against both baselines. No feature or tuning choice may be based on this test-period check.

## 10. Gotchas

- The CHIRPS monthly cache is keyed by dates only. If a site box in `config/sites.yaml` changes, delete `data/raw/chirps/<site>/` and the `chirps_<site>.parquet` file before re-running. Since November 2026 a chunk is only cached (or reused) if every pixel has every day, so a month fetched before CHIRPS published it in full is refetched. The ERA5 cache includes a hash of area and variables, so it cannot go stale this way.
- `era5 --local` reads every `.nc` or `.zip` in `data/raw/era5_manual/<site>/` and `data/raw/era5/<site>/`. Do not mix download routes for one site. A zip file may carry a `.nc` extension (the code checks content).
- Commands run as modules from the repo root. PowerShell quoting: wrap `python -c` code in double quotes with single quotes inside.
- CDS and Earth Engine licences/accounts are the user's own; the ERA5 licence acceptance and Earth Engine API enablement were already done.

## 11. What was tested, and how

- Run on the user's real data: `chirps.py`, `era5.py --local`, `elevation.py`, `merge.py`, `test_access.py`, `test_timeseries.py`. Reports matched expected pixel, day and row counts with zero missing values.
- Tested only offline on synthetic data: the data-tour notebook, the ERA5 download/split/fallback logic, merge failure detection.
- Data extended to 2026-09-30 (November 2026) and checked on the real data: expected row counts, identical pixel and date sets across sources, and 2016-2025 values identical to before the extension.
- Baselines run on the real data (validation 2024). `tests/test_baselines.py` (9 tests, synthetic data) covers the time shifts, leap-year day numbering, split gap, climatology window wrap-around and metrics.
- Features built on the real data for both sites (row counts match, NaN only in the 29-day warm-up, spot checks against independent recalculation PASS). `tests/test_features.py` (8 tests) includes a future-invariance test: changing all data after day T leaves every feature up to T unchanged.
- Horizon-1 models trained and scored on the real data (60 pixels per site). `tests/test_train.py` (2 tests) checks that features come from the issue day and targets from t + h, and that no feature equals the target.
- Horizons 1-7 trained and scored on the real data (60 pixels per site, three algorithms, about 25 minutes).
- Confidence intervals computed on the real validation predictions; point estimates match `train.py` to 1e-15, and a re-run of training reproduced every score exactly. `tests/test_uncertainty.py` (3 tests) checks the resampling weights, that per-day sums reproduce the direct metrics, and that identical models differ by exactly zero.
- Tuning run on the real data (about 1 hour for 120 trials); tuned CatBoost scored once on validation for horizons 1-3. Outputs moved to `models/<site>/train60_score60/`; a re-run after this refactor reproduced every score exactly. `tests/test_train.py` now also checks that runs with different pixel counts use different folders (23 tests in total).
- Scale-up runs on the real data: Busia 60 and 216 training pixels, Nakuru 60 and 200, all scored on every pixel; `uncertainty.py` compares runs scored on identical rows (checked) as paired differences. 25 tests in total.
- Calibration check run on the real validation predictions; `tests/test_calibration.py` (4 tests with known answers: calibrated, overconfident, too-wet forecasts, exact Brier decomposition). 29 tests in total.
- Recalibration test run on the real data; the saved models reproduced the saved 2024 forecasts exactly (guard in the script). `tests/test_recalibrate.py` (2 tests). 31 tests in total.
- SHAP run on the real validation data (60 evenly spaced days x all pixels, horizons 1/3/7, tasks rain >= 1 mm, wet-day amount, humidity); SHAP values add up to the model output for every row. `tests/test_shap_analysis.py` (3 tests). 34 tests in total.
- Spatial hold-out run on the real data (24 minutes, peak memory 2.0 GB). `tests/test_spatial_holdout.py` (3 tests: blocks partition the grid, the buffer holds, pairing by text labels). 37 tests in total.
- Transfer test run on the real data (31 seconds). `tests/test_transfer.py` (1 test: the transfer prediction path, including the smearing factor, matches training exactly). 38 tests in total.
- Not yet built: final test scoring, dashboard.

## Confirmed findings to keep in mind
- Both sites are bimodal: long rains Mar-May are about 37% of annual rain at each site; Nakuru also has a wet Jun-Sep (~31%) and dry Dec-Feb; Busia has no dry month.
- Pixel-vs-region rainfall correlation is 0.62 (Nakuru) and 0.83 (Busia), so add neighbourhood features (lagged nearby-pixel and regional mean rain).
- Humidity and temperature correlate 0.93-0.98 with the region because ERA5 is smoothly interpolated; do not claim measured microclimate.

## 12. Final test protocol (fixed before the test is run, November 2026)

Written and committed before any test-period forecast was made or scored. The test is run once. If a bug is found afterwards, it is fixed, the test is re-run, and both results are reported.

**Data.** Test targets 2025-01-08 to 2026-09-30 (the first 7 days of 2025 are skipped so the gap after the last early-stopping target, 2024-12-31, is 7 clear days; every horizon is scored on the same days). Issue days from 2025-01-01. The only earlier look at the test period was the area-average monthly rainfall check during the data extension (section 9).

**Models (decided, no changes).** CatBoost (main model), LightGBM and XGBoost with the default settings in `config/experiment.yaml`; no tuning, no recalibration, no feature changes. Same 60 training pixels per site (`dev_pixel_sample`); scored on all pixels. Retrained to include 2024: fit on targets 2016-01-02 to 2023-12-31, early stopping on targets 2024-01-08 to 2024-12-31, per-pixel climatology (as an input and as the baseline) fitted on 2016-2024. All six tasks, horizons 1-7.

**Primary outcomes (declared in advance).** CatBoost, full test period, horizons 1-3, both sites: Brier skill vs climatology for rain >= 1 mm, and RMSE skill vs climatology for humidity, with 95% moving-block bootstrap intervals (7-day blocks over target days).

**Also reported (secondary).** All targets (rain >= 1 / 10 / 25 mm, wet-day amount, direct and two-stage amount, humidity) and horizons 1-7; skill vs persistence too; 2025 and January-September 2026 separately; by season (JF, MAM, JJAS, OND, pooled over years); paired algorithm differences; two-stage vs direct; calibration (bias, slope, reliability) on the test period.

**Expectations stated in advance** (from validation): rain skill clearly above climatology on days 1-2 (Nakuru also day 3), about zero beyond; humidity skilful to about day 3 (Nakuru) and day 6 (Busia). Because 2026 had several months outside the 2016-2025 range (wet February-March, dry June-September), climatology is expected to be a weaker baseline in 2026, which inflates skill vs climatology; this is why persistence is reported too and 2025 and 2026 are shown separately.
