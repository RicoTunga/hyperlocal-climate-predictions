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

Provisional (confirm with data): local skill comes mainly from CHIRPS history and elevation, not ERA5 (needs the between-pixel spread table, notebook section 9); seasonality (two rainy seasons at Nakuru, weaker at Busia; needs notebook section 3).

DECIDED (Option B): train 2016-2023, validation 2024, final test Jan 2025 to 2026-09-30 (about 21 months, untouched until final reporting). Validate with `TimeSeriesSplit` plus a gap of at least 7 days; develop on a sample of roughly 50 to 60 pixels spread across elevations, then scale to all pixels; one model per horizon per target per algorithm; a separate spatial hold-out (unseen pixels) experiment; a "train on Busia, test on Nakuru" transfer experiment.

## 7. Deviations from the proposal (to document in the write-up)

1. Per-pixel instead of box-averaged data. 2. Nakuru box taken from the official county extent. 3. ERA5 from the CDS time-series dataset with six variables. 4. Humidity computed from temperature and dewpoint. 5. No TAHMO; no NASA POWER so far. 6. Busia added as a second site. 7. Rainfall evaluated with occurrence metrics (Brier, ROC-AUC) in addition to RMSE, MAE and R2. 8. Stage-one rainfall classifier plus amount model.

## 8. Plan from here

1. DONE (November 2026): baselines (persistence, day-of-year climatology) and the evaluation harness (time-ordered splits, metrics, per horizon); see `src/models/evaluation.py`, `src/models/baselines.py`, `reports/baselines_validation.csv`. Validation 2024 results: for rainfall, climatology beats persistence at every horizon and threshold (Brier at 1 mm: Nakuru 0.177 vs 0.24 to 0.35; Busia 0.237 vs 0.36 to 0.48), so climatology is the bar for rain. For humidity, persistence beats climatology up to day 3 at Nakuru (RMSE 5.4 vs 8.4 % on day 1) and up to day 2 at Busia (5.1 vs 6.3 %).
2. Feature engineering: per-pixel lags (1, 3, 7, 14 days) and rolling mean/std/min/max (7 and 14 days) of rainfall and ERA5 variables, sine/cosine day of year, pixel latitude, longitude and elevation, regional context (lagged region-wide mean rainfall and humidity). All shifted so day t uses only data up to t.
3. First models (XGBoost, LightGBM, CatBoost) for horizon 1, then all horizons; then Optuna tuning.
4. SHAP on the best models; spatial hold-out and transfer experiments; compare against Busia station data when available.
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
- Not yet built: features, models, SHAP, dashboard.

## Confirmed findings to keep in mind
- Both sites are bimodal: long rains Mar-May are about 37% of annual rain at each site; Nakuru also has a wet Jun-Sep (~31%) and dry Dec-Feb; Busia has no dry month.
- Pixel-vs-region rainfall correlation is 0.62 (Nakuru) and 0.83 (Busia), so add neighbourhood features (lagged nearby-pixel and regional mean rain).
- Humidity and temperature correlate 0.93-0.98 with the region because ERA5 is smoothly interpolated; do not claim measured microclimate.
