# Hyperlocal rainfall and humidity forecasts for Kenyan smallholders

Daily forecasts of **rainfall** and **relative humidity**, **1 to 7 days ahead**, for every ~5 km square of
**Nakuru County** (696 squares) and a **Busia** area (216 squares), made with gradient boosting (CatBoost,
LightGBM, XGBoost), explained with SHAP, and shown in an interactive web dashboard with a potato
late-blight risk layer for Nakuru.

**Mutunga Eric Musyimi**, final-year BSc project (ICS Project II), Strathmore University, 2026.

> **Research prototype, not an official forecast.** The Kenya Meteorological Department is the forecasting
> authority. The dashboard is a *replay* of forecasts for 2025-2026, which the models never saw during
> development, so every forecast can be compared with what actually happened.

## Results (final test, January 2025 to September 2026, run once)

Skill compared with the normal for the date (climatology): 0 = no better, 1 = perfect, with 95% intervals
from resampling blocks of days. CatBoost, every square of both sites.

| Days ahead | Nakuru rain (>= 1 mm) | Nakuru humidity | Busia rain (>= 1 mm) | Busia humidity |
|---|---|---|---|---|
| 1 | **+0.20** [0.15, 0.26] | **+0.52** [0.47, 0.56] | **+0.17** [0.12, 0.22] | **+0.39** [0.30, 0.45] |
| 2 | **+0.08** [0.04, 0.12] | **+0.31** [0.25, 0.37] | **+0.08** [0.04, 0.11] | **+0.23** [0.15, 0.29] |
| 3 | +0.02 [-0.00, 0.05] | **+0.23** [0.16, 0.29] | **+0.05** [0.02, 0.07] | **+0.14** [0.07, 0.19] |

- Rain forecasts beat climatology on days 1-2 (Busia to day 3); humidity forecasts for the whole week;
  both beat "same as today" (persistence) at every horizon. Probabilities are well calibrated.
- The test protocol, primary outcomes and expectations were fixed and committed **before** the test was run.
- Also tested: tuning (no measurable gain), more training pixels (no gain), an unseen-area hold-out (about
  88% of day-1 rain skill carries over), transfer between sites (poor: models are site-specific), forecast
  fusion (no gain), and a late-blight risk forecast (skilful to day 6).

Full methods, decisions and numbers: [`docs/PROJECT_CONTEXT.md`](docs/PROJECT_CONTEXT.md).

## Data

| Source | Use |
|---|---|
| CHIRPS daily rainfall (UCSB Climate Hazards Center), via Google Earth Engine | rainfall, 0.05 degree (~5 km) |
| ERA5 reanalysis (Copernicus Climate Change Service), CDS time-series dataset | temperature, dewpoint, pressure, wind, cloud; humidity computed from them |
| SRTM 30 m elevation (NASA / USGS), via Google Earth Engine | elevation per square |

Data are not in this repository (`data/` and `models/` are git-ignored); the scripts below rebuild them.
Earth Engine and Copernicus accounts are needed for the downloads.

## Reproduce

```bash
python -m venv .venv && .venv\Scripts\activate      # Windows; on Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

Credentials: `GEE_PROJECT` in `.env` (see `.env.example`) and a Copernicus key in `~/.cdsapirc`; never commit either.

| Step | Command |
|---|---|
| Rainfall | `python -m src.ingestion.chirps` |
| Weather | `python test_timeseries.py --site <site> --start 2016-01-01 --end 2026-10-21`, move the file to `data/raw/era5_manual/<site>/`, then `python -m src.ingestion.era5 --local --site <site>` |
| Elevation | `python -m src.ingestion.elevation` |
| Merge and features | `python -m src.features.merge`, then `python -m src.features.build` |
| Baselines | `python -m src.models.baselines` |
| Models (decided setup) | `python -m src.models.train --algos catboost --pixels 60 --score-pixels all --horizons 1 2 3 4 5 6 7` |
| Intervals, calibration, SHAP | `python -m src.models.uncertainty`, `python -m src.models.calibration`, `python -m src.explain.shap_analysis` |
| Final test (once) | `python -m src.models.final_test train`, then `python -m src.models.final_test analyse` |
| Late blight | `python -m src.features.humid_hours --site nakuru`, then `python -m src.models.late_blight validate` / `test` |

Splits: train 2016-2023, validation 2024, test 2025-01-08 to 2026-09-30 (settings in `config/experiment.yaml`;
sites in `config/sites.yaml`). Tests: `python -m pytest -q` (105 tests).

## Dashboard

A Windy-style map: forecast layers (rain chances, expected rain, humidity, late-blight risk) over 5 km
squares, a timeline of issue days and days ahead, a per-square 7-day outlook with a plain-language
**"Why this forecast?"** SHAP breakdown, verification charts, and sign-in with invite-only accounts.

```bash
python -m app.build_data                      # dashboard data from the saved test forecasts
python -m app.build_shap                      # per-forecast explanations (~10 min, memory-heavy)
python -m web.users add <name> --role admin   # create an account (password typed at a hidden prompt)
python -m web.server                          # open http://127.0.0.1:8765 and sign in
```

Hosting (HTTPS, Docker, first admin, checks): [`deploy/README.md`](deploy/README.md). The image has not
been built or deployed yet.

## Repository

```
config/      study sites and experiment settings
src/         ingestion, features, models, explanations
app/         dashboard data builders
web/         dashboard server (sign-in, API) and front end
tests/       test suite
reports/     result tables and figures (committed)
docs/        project context and decision log
deploy/      hosting guide
```

## Acknowledgements and attribution

CHIRPS (Funk et al., 2015; Climate Hazards Center, UC Santa Barbara). ERA5 (Hersbach et al., 2020;
contains modified Copernicus Climate Change Service information). SRTM (NASA / USGS). Basemap
© OpenStreetMap contributors © CARTO.
