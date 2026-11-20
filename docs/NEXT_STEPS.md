# Next steps after the supervisor's demo review (20 November 2026)

Written 20 November 2026, after reading `AGENTS.md`, `docs/PROJECT_CONTEXT.md`, `docs/DEMO_GUIDE.md` and `docs/report/latex/chapter5.tex`, and the code they refer to. No code, configuration or other file was changed. Facts about outside data sources were checked online on 20 November 2026. The sources are listed in section 0, and anything that could not be verified is marked **(not verified)**.

Effort scale: **small** is a few hours, **medium** is one to three days, and **large** is a week or more. Risk means the chance that the work fails, takes much longer than planned, or weakens a reported result.

---

## 0. Facts checked for this plan

### 0.1 Two corrections to the project record

1. **The rainfall data is CHIRPS version 2.0, not v3.** `docs/PROJECT_CONTEXT.md` section 4 says "CHIRPS v3 daily via Earth Engine, collection `UCSB-CHG/CHIRPS/DAILY`". The Earth Engine catalogue page for that collection is titled "Version 2.0 Final". The v3 collections have other IDs: `UCSB-CHC/CHIRPS/V3/DAILY_SAT` and `UCSB-CHC/CHIRPS/V3/DAILY_RNL`. The report chapters name the collection, cite Funk et al. (2015) and give no version number, so they are not wrong. They should still say "v2.0".
2. **CHIRPS v2 is being retired.** The CHC says: "Its last month of operation will be December 2026" (`README-CHIRPS2-Daily.txt`). Any live or extended system built on v2 therefore stops working in early 2027. This shapes the retraining question (section 12).

### 0.2 How daily CHIRPS is made (needed for points 6 and 7)

Both CHC readme files say that CHIRPS is "fundamentally a pentad and monthly product". Station blending happens only at the pentad and monthly steps. The daily values are made by splitting each pentad's total across its days:

- **v2** splits the totals using the daily pattern of the **CFSv2** weather model.
- **v3** uses ERA5 for the `rnl` product and IMERG Late for the `sat` product.

The v3 readme adds: "The 0.05° grid spacing of the daily products therefore should not be interpreted as indicating that the ERA5 or IMERG data used for temporal disaggregation contain information at 0.05° resolution."

The v2 readme warns that daily extremes, onset dates, dry spells and wet-day counts are all shaped by how the totals are split across days. It recommends that users "state the disaggregation method in your methods".

### 0.3 Latest data available on 20 November 2026

| Product | Latest day available | Typical delay (official wording) | Where checked |
|---|---|---|---|
| CHIRPS v2.0 final, Earth Engine `UCSB-CHG/CHIRPS/DAILY` (what `chirps.py` uses) | **30 Sep 2026** (catalogue "Dataset Availability") | Not stated on the page; Earth Engine lags the CHC server | Earth Engine catalogue |
| CHIRPS v2.0 final, CHC server (`products/CHIRPS/v2.0/global_daily/tifs/p05/2026/`, also `africa_daily`) | **31 Oct 2026** (files posted 13 Nov 2026) | Not stated for v2. The October files appeared on 13 November | CHC directory listing |
| CHIRPS v2.0 preliminary, CHC server (`products/CHIRPS/v2.0/prelim/global_daily/tifs/p05/2026/`) | **15 Nov 2026** (posted 17 Nov 2026) | Same pentad schedule as v3 preliminary (see below) | CHC directory listing |
| CHIRPS v3.0 final, `sat` and `rnl` (CHC server and Earth Engine) | **31 Oct 2026** (`sat` posted 18 Nov, `rnl` 13 Nov; both Earth Engine collections end 2026-10-31) | "produced once a month, typically on the third week of the following month" | chc.ucsb.edu/data/chirps3, listings, Earth Engine catalogue |
| CHIRPS v3.0 preliminary (only `sat` exists) | **15 Nov 2026** (posted 17 Nov 2026) | "available 2 days after the end of a pentad (on the 2nd, 7th, 12th, 17th, 22nd and 27th day of each month)" | chc.ucsb.edu/data/chirps3, listing |
| ERA5 / ERA5T, CDS (`reanalysis-era5-single-levels` and the `-timeseries` dataset used here) | About **15 Nov 2026** (5-day latency; not checked by download) | "updated daily with a latency of about 5 days". ERA5T for a month is "overwritten with the final ERA5 data about two months after the month in question" | CDS dataset pages, ECMWF "ERA5: data documentation" |
| ERA5 already on disk | **21 Oct 2026, 23:00** (`data/raw/era5_manual/<site>/ts_<site>_2016-01-01_2026-10-21.nc`, downloaded 27 Oct) | | Inspected the file |
| CHIRPS already on disk | **30 Sep 2026** (`data/raw/chirps/<site>/20260901_20261001.parquet`) | | Directory listing |

The next v2 and v3 preliminary pentad (16 to 20 November) should appear on about 22 November. Final November data should appear in the third week of December.

**Are the newest days revised later?** Yes, in two ways:

- **CHIRPS preliminary is replaced by CHIRPS final.** The preliminary product uses "rapidly updated station observations". The final product "blends in best available station inputs".
- **ERA5T can differ from final ERA5.** The ECMWF page says ERA5T "could be different to the final ERA5 data" if serious flaws are found. It lists two such cases, both caused by snow observations (central Asia in Sep to Dec 2021, the Alps in Jul to Nov 2024). The CDS time-series file on disk carries **no `expver` flag** (its variables are `u10, v10, d2m, t2m, sp, tcc`), so ERA5T rows cannot be told apart inside it.

The file was downloaded on 27 October. Given the two-month overwrite, the last two to three months it covers (about August to October 2026) were very likely ERA5T when downloaded. **This includes the end of the reported test period.** The effect is expected to be negligible, because both known revisions were snow events outside East Africa. It should still be stated (section 13), and it can be checked later without changing any result (section 11).

### 0.4 Sources (all read on 20 November 2026)

- Earth Engine catalogue: `developers.google.com/earth-engine/datasets/catalog/UCSB-CHG_CHIRPS_DAILY`, `.../UCSB-CHC_CHIRPS_V3_DAILY_SAT`, `.../UCSB-CHC_CHIRPS_V3_DAILY_RNL`
- CHC: `www.chc.ucsb.edu/data/chirps3`, `www.chc.ucsb.edu/data/chirps`, `www.chc.ucsb.edu/data/chirps-gefs`, `data.chc.ucsb.edu/products/CHIRPS/v2.0/README-CHIRPS2-Daily.txt`, `data.chc.ucsb.edu/products/CHIRPS/v3.0/daily/readme.txt`, and the directory listings named in the table
- ERA5: `cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels`, `.../reanalysis-era5-single-levels-timeseries`, `confluence.ecmwf.int/display/CKB/ERA5%3A+data+documentation`
- Open-Meteo: `open-meteo.com/en/pricing`, `/en/terms`, `/en/licence`, `/en/docs`, `/en/docs/historical-forecast-api`, `/en/docs/previous-runs-api`, `/en/docs/historical-weather-api`, `/en/docs/ensemble-api`
- ECMWF open data: `www.ecmwf.int/en/forecasts/datasets/open-data`, ECMWF Confluence "ECMWF open data: real-time forecasts from IFS and AIFS"
- NOAA: `registry.opendata.aws/noaa-gfs-bdp-pds/`, `www.emc.ncep.noaa.gov/emc/pages/numerical_forecast_systems/gfs.php`, `nomads.ncep.noaa.gov`
- Others: `gpm.nasa.gov/data/imerg`, `tahmo.org/climate-data/`, `openweathermap.org/price`, `www.tomorrow.io/weather-api/`

---

## 1. Point 1: Real-time forecasts on a separate branch, and the choice of live data source

### What "live" means for these models

The models do **not** take a weather forecast as input. A forecast issued on day t uses **observed conditions up to day t**: CHIRPS rain history (up to 30 days), ERA5 weather history (up to 14 days), area-wide and neighbourhood rain on day t, elevation and climatology (`src/features/build.py`; Chapter 5 Table "Model Inputs by Group").

Going live therefore needs a near-real-time **record of the last 30 days, up to today**, not a forecast feed. This matters for the choice of source.

SHAP shows that day t is the most important day:

- Area-wide conditions carry 31% (Nakuru) and 46% (Busia) of day-1 rain-probability SHAP.
- The pixel's own ERA5 weather carries 25% and 19% (`PROJECT_CONTEXT.md` section 6).

So the quality of the live substitute for *today's* rain and weather decides how much skill survives.

### The options compared

| Source | Cost and licence | Variables | Resolution | Updates | Recent past | Historical archive | Reliability | Ease of use |
|---|---|---|---|---|---|---|---|---|
| **Open-Meteo** (forecast API plus historical APIs) | Free for non-commercial use. Data under CC BY 4.0, with a link "next to any location Open-Meteo data are displayed". Free limits: under 10,000 calls a day, 5,000 an hour and 600 a minute. Paid plans exist (prices were not visible). Upstream licences apply (ECMWF CC-BY, NOAA, Copernicus) | Hourly `temperature_2m`, `relative_humidity_2m`, `dew_point_2m`, `precipitation`, `surface_pressure`, `cloud_cover`, `wind_speed_10m`: every ERA5 variable the models use | Varies by model. ECMWF IFS HRES 9 km, IFS 0.25°, AIFS 0.25°, GFS 0.11°/0.25° | ECMWF every 6 h. The series is "stitched to the previous run without gaps" | `past_days` 0 to 92 on the forecast API | Historical Weather API: **"ECMWF IFS, 9 km, hourly, 2017 to present, every 6 hours with no delay"**, plus ERA5 and ERA5-Land (5-day delay). Historical Forecast API: IFS HRES 9 km from 2017-01-01, GFS 0.11° from 2021-03-23. Previous Runs API (lead-time aligned, 1 to 7 days): "most models are archived from January 2024" | Free tier "carries no uptime guarantee". Paid plans target 99.9%. DWD ICON Global is listed as discontinued from 2026-11-30, so a pinned model can disappear | Easiest: JSON over HTTPS, no key, several locations per request |
| **ECMWF open data** (direct) | Free, CC-BY-4.0. Note: "may be subject to Service Charges" for delivery arrangements | 2t, 2d, tp, sp, 10u, 10v, tcc (HRES, ENS, AIFS). RH only on pressure levels, so derive 2 m RH as now | 0.25°, and **0.1° for IFS since 20 October 2026** | 4 runs a day | Only what is still on the server: "the most recent 12 forecast runs ... approximately 2–3 days". Confluence says about 4 days | Cloud mirrors (AWS, Azure, Google) are "kept much longer", but no start date was found **(not verified)**. The full archive is paid | ECMWF operational service | Harder: GRIB2, the `ecmwf-opendata` client, own regridding and storage |
| **NOAA GFS** (NOMADS, AWS) | Free; "open to the public and can be used as desired" | Exact variable names (for example 2 m RH, APCP) **(not verified)** | 13 km model, 0.25° files | 4 runs a day, to 16 days | NOMADS retention **(not verified)** | AWS bucket `noaa-gfs-bdp-pds`; start date **(not verified)**. GFS model versions change over time | Operational | Harder: GRIB2 |
| **CHIRPS-GEFS v3** (CHC) | CC BY 4.0 | **Rain only**: GEFS ensemble-mean forecasts bias-corrected to CHIRPS (v3, since CHIRPS2-GEFS "ended on July 1, 2026") | 0.05°, the same grid as the project | Daily (5-, 10- and 15-day totals, plus daily fields) | Not applicable (it is a forecast) | Archive "begins January 2001" (the page also says "2020 is not included", which conflicts with the listing) | Same producer as CHIRPS | GeoTIFF files |
| **CHIRPS v3 preliminary** (CHC) | Same as CHIRPS (CC BY 4.0) | Rain | 0.05° | Every pentad, 2 days after it ends | 2 to 7 days behind | Same as CHIRPS v3 | Same as CHIRPS | GeoTIFF (new reader needed) |
| **IMERG Early** (NASA) | Free; registration with PPS / Earthdata | Rain | 0.1° (from the CHIRPS v3 readme) | Half-hourly; "lowest latency available (4 hours)" | Yes | Yes (Final run about 2.5 to 3.5 months behind; the NASA page gives both figures) | Operational | Registration and GeoTIFF/HDF5 |
| **ERA5T** (CDS, current route) | Free, CC-BY | All current variables | 0.25° | Daily | About 5 days behind | 1940 onward | CDS queue | Already built |
| **OpenWeatherMap / Tomorrow.io** | Free tiers (OpenWeatherMap: 1,000,000 calls a month; Tomorrow.io: 5-day forecast, 24 h history). Archive is paid. Storage terms not found | Point weather | Not stated | | 24 h (Tomorrow.io free) | Paid | Commercial | Easy, but no affordable archive to retrain on |
| **TAHMO stations** | Portal access with a request form. The terms PDF could not be read **(not verified)** | Station observations | Points | | | | | `PROJECT_CONTEXT.md` section 2: "TAHMO data will not be available" |
| **NASA POWER, Kenya Meteorological Department** | Not verified (the POWER methodology page returned 404; no public KMD API was found or checked) | | | | | | | |

### Recommendation: Open-Meteo for the weather inputs, with ECMWF IFS chosen explicitly, plus CHIRPS preliminary for rain

1. **It is the only option that gives the same product for both the archive and the live feed, with no delay.** The Historical Weather API's ECMWF IFS 9 km series runs from 2017 to the present "with no delay". A model could therefore be trained and run on matching inputs (option b in section 12), which no other free source offers. ECMWF open data keeps only 2 to 4 days on its own server, and GFS needs GRIB handling and unverified archive dates.
2. **It covers every variable the models use**, as plain JSON, without a key. Not needing a key keeps the dashboard server credential-free (`chapter4.tex`, architecture section).
3. **It fits the free limits.** ERA5 is used on its 0.25° grid and interpolated to the pixels, so the live job only needs the ERA5 grid nodes: 81 for Nakuru (9 × 9, read from the file on disk) and 48 for Busia (8 × 6, from `config/sites.yaml`). That is 129 locations. A daily update of a few days per location is about 129 calls. A 2024 backfill for the backtest is about 129 × 27 two-week chunks, roughly 3,500 calls, inside one day's limit.
4. **Ask for the model explicitly instead of `best_match`.** The `best_match` setting mixes ECMWF, GFS and ICON, and ICON Global is being discontinued on 2026-11-30. Requesting ECMWF IFS keeps the inputs stable.
5. **Rain still comes from CHIRPS**, because Open-Meteo's rain is model rain, not the CHIRPS estimate the models were trained on. Use CHIRPS preliminary (2 to 7 days behind). Fill the missing last days either from Open-Meteo IFS precipitation, or from IMERG Early (4 h latency; IMERG is what CHIRPS v3 `sat` uses to split pentad totals into days). Which gap-filler hurts least is part of the backtest (section 12).

**Caveats to state:**

- Free use is for non-commercial purposes. The terms list "public research conducted at public institutions" and "use in educational content" as examples. Confirm that a Strathmore student capstone fits, and email Open-Meteo if unsure.
- The free tier has no uptime guarantee.
- Open-Meteo warns that archived series are "not suitable for long time series due to model version changes".
- How Open-Meteo builds a value for one coordinate (nearest grid cell, interpolation, or elevation correction) **(not verified)**. Check it before comparing with ERA5.

**Option to include in the report:** use CHIRPS-GEFS v3 as an outside **benchmark** of rain forecasts on the same 0.05° grid, and later as a forecast **input**. It is the "different information source" that `PROJECT_CONTEXT.md` section 14 says fusion would need. Its archive from 2001 would allow clean training splits, but the gap noted for 2020 must be checked first.

### Build options

| Option | What | Effort | Risk | Evidence and files |
|---|---|---|---|---|
| 1a. Live branch, frozen models (recommended for submission) | New branch `feature/live`. A scheduled job fetches the last 30+ days from Open-Meteo IFS (weather) and CHIRPS preliminary (rain), fills the gap days, rebuilds features for the latest issue day only, and runs the saved final models (`models/<site>/final_train60_score<N>/`). It writes forecasts to a separate store that the web server reads. The replay stays unchanged on `main` | Medium to large | Input mismatch (section 12); the free API has no uptime guarantee; CHIRPS v2 preliminary ends with v2 retirement (December 2026) | `src/features/build.py`, `src/models/final_test.py` (how saved models are loaded and smearing recomputed), `src/models/transfer.py::predict_all`, `src/ingestion/era5.py::interpolate_to_pixels` |
| 1b. "Latest possible" forecast from the existing pipeline | No new source. Issue a forecast from the newest day that has CHIRPS preliminary and ERA5T (15 Nov 2026 today), for 16 to 22 Nov. Days 21 and 22 are still in the future on 20 Nov. This shows the latency gap honestly | Small to medium (reuses the point 11 work) | Low | Same as point 11 |
| 1c. Spoken answer only | Explain the latency, as in point 9 | Small | Supervisor asked for a build | `DEMO_GUIDE.md` section 6 |

---

## 2. Point 2: A system built for farmers

**What exists.** The dashboard is a Windy-style map built for exploration: layers, views, a timeline and "How good is it?" (`PROJECT_CONTEXT.md` section 16). Farmer features were added on top of it (section 19 and `web/static/hud.js`):

- an "At a glance" panel with a 7-day rain strip, and crop cards (`web/crops.py`);
- onboarding with user type, saved places and "use my location".

The first screen is still the research map.

| Option | What | Effort | Risk | Evidence |
|---|---|---|---|---|
| 2a. **Farmer mode as the default for farmer accounts** (recommended) | Opening the app shows "My farm": seven day cards. Each card gives the day, the chance of rain in words and numbers ("6 in 10 chance"), the expected mm, humidity, potato and tomato blight risk, and a confidence label from the tested skill. The map and research layers move behind "Explore". Remove the sign-up and tour friction from the farmer route (the tour is already a setting) | Medium | Low; reuses `/api/advice` and `/api/pixel` | `web/crops.py` (`outlook`, `confidence`, `RAIN_SKILL_DAYS`), `hud.js`, `user.js` |
| 2b. Text-message style summary | A 3-line SMS or WhatsApp-style text per farm per day, shown in the app as a preview (not actually sent). The supervisor left the delivery channel to the student (`PROJECT_CONTEXT.md` section 2) | Small (after 2a) | Low | `web/crops.py::outlook` already writes a one-line week summary |
| 2c. Kiswahili text | Translate the farmer-mode strings | Medium | Needs a fluent checker | None yet |
| 2d. Report-only change | Rewrite the users section so the farmer is the primary actor, and the extension officer and researcher are secondary | Small | None | `chapter4.tex` lines 45 and 218 (Viewer/Administrator replaced the farmer actor) |

Be honest about one thing: at Nakuru, rain is clearly skilful only on days 1 and 2 (`reports/test/skill.csv`; Chapter 5 Table "Test Summary"). Farmer mode should say plainly that days 3 to 7 are "about the usual for this time of year", not dress them up as precise forecasts.

---

## 3. Point 3: Who the explanations are for, and how to show they are understood

**Current state.** Every forecast has an exact SHAP waterfall in plain-language groups (`app/build_shap.py`; `PROJECT_CONTEXT.md` section 18). Chapter 5 already says the explanations are shown to be **correct** (they add up exactly; TC-12), but not **clear or trusted**, because no user was tested (`chapter5.tex` "Explanation" paragraph; Chapter 6 future work "User Evaluation").

**Recommended framing (report change and spoken answer; small effort):**

| Audience | What they see | Purpose |
|---|---|---|
| Farmers | One sentence: the single largest reason ("Rain is likely mainly because it rained across the area today"), plus a confidence label | Decide whether to act on the forecast |
| Extension officers (main audience for the full explanation) | The group waterfall, and the skill and normal next to each forecast | Judge and pass on the forecast; spot when the model leans only on "normal for the date" |
| Researchers, Kenya Met, examiners | The overall SHAP shares by lead time (`reports/shap_groups_validation.csv`) and the skill panel | Check that the model behaves sensibly (day-to-day skill comes from weather; at long lead times it relies on climatology) |

**How to establish understanding (build a protocol; medium effort):**

1. **Comprehension tasks.** For 6 to 8 screenshots, ask:
   - "Will it rain tomorrow at this farm?"
   - "What is the main reason?"
   - "Is the day-6 forecast more or less reliable than day 1?"
   - "Is this a measurement at your farm?"
   
   Score answers against a key written in advance.
2. **Trust that matches skill.** Check whether participants rate day-1 forecasts as more reliable than day-5 forecasts, and heavy-rain forecasts as least reliable, as the measured skill says they should.
3. **A success criterion fixed in advance.** For example, at least 80% correct on the main tasks. This is a proposal: no stakeholder has set a figure.
4. **Who and when.** The constraints in `PROJECT_CONTEXT.md` section 2 and Chapter 6 R1 still apply: no farmer testing before Kenya Met sign-off, plus the university's ethics process. Options:
   - (i) write and include the protocol (instrument, tasks, scoring) as future work (small);
   - (ii) if ethics allows, run a **pilot with non-farmer volunteers** (for example classmates), clearly labelled as a pilot and not as evidence about farmers (medium).

---

## 4. Point 4: Demo scenario, "I am a farmer in Nakuru today: will it rain in the next 7 days?"

| Option | Demo | Effort | Risk | Evidence |
|---|---|---|---|---|
| 4a. Replay, framed as "today" (minimum) | Pick a replay issue day, sign in as a farmer, and open "My farm" (2a) or "At a glance". Walk through all seven day cards: chance, amount, confidence, what happened. Use the 14 April 2025 day from `DEMO_GUIDE.md` (54% average forecast, 68% of squares wet), and be honest with 15 April as well | Small (medium with 2a) | Low | `DEMO_GUIDE.md` section 4; `app/data/nakuru_replay.parquet` |
| 4b. **Latest forecast from current data** (recommended for the meeting) | After point 11, show the forecast issued on the newest data day (about 15 November) for the following 7 days, some of which are still ahead. Show the "preliminary data" labels and "not yet observed" for future days | Medium (needs point 11 plus allowing issue days without observations in `app/build_data.py`) | Data timing: the preliminary release on the 22nd moves the date | Section 11 |
| 4c. Live branch | If 1a is built: the forecast issued today, 20 November, for 21 to 27 November | Large (depends on 1a) | Live API on the day; keep 4b as backup | Section 1 |

Whichever option is used, the demo must say that rain is skilful for about 2 days at Nakuru, and that days 3 to 7 are close to the normal for the date.

---

## 5. Point 5: Bringing out the tomato and potato late-blight use

**What exists:**

- Hutton and humidity-only risk layers, and a "Tomato blight" layer (the potato model, labelled as such).
- Crop cards for potato and tomato (`web/crops.py::blight_advice`).
- Post-hoc test skill: Hutton day 2 +0.307 [+0.25, +0.36], clear up to day 6 (`reports/late_blight/test_skill.csv`; `PROJECT_CONTEXT.md` section 15).

| Option | What | Effort | Risk |
|---|---|---|---|
| 5a. Put blight in farmer mode (recommended) | Potato and tomato rows on each day card ("blight weather: high / medium / low"), with the action from `crops.py`. Show it in the demo straight after the rain question | Small (part of 2a) | Low |
| 5b. Demo step | Choose a replay day in April or May (the seasonal peak, 33 to 37% of days) at a highland potato square | Small | None |
| 5c. Report change | Move late blight from "post-hoc extension" to a named farmer use case in Chapters 4 and 6, keeping the caveats: British thresholds, tomato reuses the potato model, no outbreak data, and the rule is met on only 3% of test days above 2,600 m | Small | None |

---

## 6. Point 6: Outputs are estimates for a 5 km area, not farm observations

| Option | What | Effort | Evidence |
|---|---|---|---|
| 6a. UI wording (recommended) | Under every farmer-facing number: "Estimate for the 5 km square around this point, from satellite and weather-model data. Not a measurement at your farm." The square's outline is already drawn on hover | Small | `PROJECT_CONTEXT.md` section 19 (hover shows the square's range) |
| 6b. Report and spoken answer | Three facts: (1) the "truth" is gridded CHIRPS and ERA5, not gauges (`PROJECT_CONTEXT.md` section 9); (2) daily CHIRPS splits pentad totals across days using CFSv2 (v2), so daily wet and dry patterns at a pixel are partly model-made (section 0.2); (3) ERA5 humidity correlates 0.93 to 0.98 with the area average, so humidity differences between squares are not measured microclimate | Small | `chapter5.tex` RQ2 section; CHC readmes |

---

## 7. Point 7: Why 5 km, and could it be finer?

**Spoken answer (small effort):**

1. 5 km is the grid of the only long, daily, free gridded rainfall record used for both training and scoring: CHIRPS at 0.05°. A model can only be trained and *checked* at the resolution of its reference data.
2. Even 5 km is optimistic for daily rain. The station blending happens at pentad scale, and the daily split comes from coarser data (CFSv2 for v2, ERA5 0.25° or IMERG 0.1° for v3). The CHC itself says the 0.05° spacing does not mean the daily timing information is that fine (section 0.2).
3. Humidity and temperature come from ERA5 at about 25 km, interpolated bilinearly (`src/ingestion/era5.py`). Their effective resolution is coarser still.
4. The evidence fits this. The pixel-versus-area rain correlation is 0.62 at Nakuru and 0.83 at Busia, and area-wide conditions are the largest SHAP group for day-1 rain.

**Could it be finer?**

| Route | Effort | Risk |
|---|---|---|
| Finer weather inputs: Open-Meteo lists ECMWF IFS at 9 km (2017 onward) and ERA5-Land at 0.1° (1950 onward, 5-day delay) | Large (re-ingest, retrain) | The rain reference is still 5 km |
| Downscale with elevation (SRTM 30 m is already used as summary statistics) | Large | **No 1 km reference to test against**, so the result could not be verified, which would go against the project's "honest evaluation" priority |
| Station data (Busia station deferred; TAHMO via portal request; a KMD–TAHMO data-sharing MOU was announced in September 2026) | Large | Access outside the project's control |

Conclusion for the answer: finer is technically possible, but without a finer reference it would be false precision. The honest next step is station validation, not a smaller grid.

---

## 8. Point 8: Defending CatBoost more strongly

**Evidence already in the repo (spoken answer and report change; small effort):**

1. **Chosen by a stated rule on validation, then confirmed out of sample.**
   - Validation 2024: 33 of 294 paired differences were clear; CatBoost was better in 23 and worse in none.
   - Test 2025 to 2026, untouched during selection: 51 clear differences, of which CatBoost won 40, XGBoost 9 and LightGBM 2 (`PROJECT_CONTEXT.md` sections 6 and 13; Chapter 5 Table "Model Selection").
   
   This is a replicated result, not a choice made on one year.
2. **Robust without tuning.** Optuna gained 0.2 to 1.4% in cross-validation, and only 1 of 24 validation comparisons was clear. The default settings are not a weakness.
3. **Well calibrated without recalibration.** No rain-probability bias interval on the test period excludes zero, and the slopes are mostly 0.9 to 1.06. Platt scaling made things worse (4 of 42 comparisons worse, none better).
4. **Practical fit.** It ran on an 8 GB CPU laptop (6 to 7 minutes per site, 1.1 to 2.2 GB peak). CatBoost's built-in exact TreeSHAP produced 4.0 million explanations for the dashboard (`app/build_shap.py`).
5. **Design reason (from the cited paper, `prokhorenkova2018catboost` in `references.bib`):** ordered boosting was designed to reduce prediction shift (overfitting to its own training targets), which suits data with strongly correlated rows such as pixel-days.
6. **Say plainly that the margins are small** (at most 0.038 skill on validation). Any of the three libraries would give similar forecasts. The defence is that the choice was principled and was replicated on unseen data, not that CatBoost is dramatically better.

**Optional new evidence (validation 2024 only, never the test period):**

| Option | Effort | Risk |
|---|---|---|
| 8a. Repeat the three-algorithm comparison with 3 to 5 random seeds, to show that CatBoost's wins exceed seed-to-seed noise. This addresses the "single random seed" limitation in Chapter 5 | Medium (about 25 minutes per seed for three algorithms on 60 pixels, `appendix_ch5.tex` runtimes table) | Low; could show some wins are noise, which would still be reportable |
| 8b. Add a simple logistic or linear regression on the same features, to show what boosting adds | Medium | Low |
| 8c. Record per-algorithm training time (currently only "about 25 minutes for three") | Small | None |

---

## 9. Point 9: Why the current system is not live

The current answer ("research prototype showing historical prediction performance, live as a future add-on") is correct but weak. It can be made stronger in three ways (spoken answer and report; small effort):

1. **Give exact latencies.**
   - CHIRPS preliminary arrives 2 days after each pentad, so rain is 2 to 7 days behind.
   - CHIRPS final arrives in the third week of the following month.
   - ERA5T is about 5 days behind.
   - On 20 November the newest complete inputs are for about 15 November.
   
   The models need *today's* conditions, which carry most of the day-1 skill (point 1).
2. **Present the replay as a deliberate design for honest evaluation.** Every forecast can be shown next to what happened, with skill computed once under a pre-registered protocol (`dffea25`).
3. **Name the authority question.** Kenya Met is the national forecasting authority, and public forecasts need its sign-off (`PROJECT_CONTEXT.md` section 2; Chapter 6 R1).

Then **show the bridge**: the "latest possible" forecast (4b), and, if built, the live branch (1a) with its backtest numbers. These turn "future add-on" into "here is what live costs in accuracy".

---

## 10. Point 10 (for information)

No work is needed. One note: the tutorial can be skipped for the demo by turning off "Tutorial when I sign in" in the menu (`PROJECT_CONTEXT.md` section 19).

---

## 11. Point 11: Extending the replay to October and November 2026

### 11.1 Achievable period (as of 20 November 2026)

- **Post-test period A: target days 1 to 31 October 2026, final CHIRPS v2.**
  - The CHIRPS final data is available now, but only on the CHC server; the Earth Engine collection still ends on 30 September.
  - ERA5T is already on disk to 21 October. 22 to 31 October needs a small new download.
- **Post-test period B: target days 1 to 15 November 2026, preliminary CHIRPS and ERA5T.**
  - Label this period provisional.
  - Re-score it once final November CHIRPS appears (expected in the third week of December).
- Issue days run from 24 September (so day-7 forecasts land on 1 October) to 7 November (day 7 = 15 November). Forecasts issued 8 to 15 November have targets that are not yet observed (useful for demo 4b).

Use the same CHIRPS **v2** as the models were trained on. Mixing in v3 would silently change the input data.

### 11.2 Downloading with the existing scripts, and what must change

**Approvals needed first (AGENTS.md rule 8):**

- `config/sites.yaml` `period.end` (2026-09-30 → 2026-11-15).
- A new `posttest` split in `config/experiment.yaml`. The `test` split must stay 2025-01-01..2026-09-30.

**CHIRPS (`src/ingestion/chirps.py`):**

- **Route 1: wait for Earth Engine** to add October to `UCSB-CHG/CHIRPS/DAILY`. No code change is needed. The completeness check (`chunk_complete`) already refuses a month Earth Engine has not finished. When the delay will end is not known, and this route never gives preliminary data.
- **Route 2 (needed for November, and for the live branch): add a CHC-server reader.**
  - Download the daily GeoTIFFs: `products/CHIRPS/v2.0/africa_daily/tifs/p05/<year>/chirps-v2.0.YYYY.MM.DD.tif.gz` for final data, and `products/CHIRPS/v2.0/prelim/global_daily/tifs/p05/<year>/` for preliminary data.
  - Gunzip, crop to the bbox, and sample at the same pixel centres (`era5.py::pixel_grid` gives the identical ids).
  - This needs a raster library (for example `rasterio`), which is **not** in `requirements.txt`.
  - The download is small: about 0.8 MB per day for Africa final and 2.8 MB for global preliminary, so about 100 MB in all.
  - **Guard:** before using the reader for new data, read September 2026 through it and check that every value equals the Earth Engine cache in `data/raw/chirps/<site>/20260901_20261001.parquet`.
- **Stale cache for partial months.** The cache is keyed by dates only. Since `b064984` an incomplete month is no longer cached, but a **complete preliminary chunk would be cached under the same name as the final one**, and never replaced when the final data arrives. Fixes:
  1. Keep preliminary data in its own cache folder (`data/raw/chirps_prelim/<site>/`), never in `data/raw/chirps/`.
  2. Add a `rain_source` column (`final` or `prelim`) that is carried through the merge.
  3. Whenever final data covers a date, prefer it and delete that preliminary chunk.
  4. Partial final months (for example `20261101_20261116`) get their own file names, so they never clash with the full month.

**ERA5 (`src/ingestion/era5.py`):**

1. Download **only the new days**: `python test_timeseries.py --site <site> --start 2026-10-22 --end 2026-11-15`, then move the file to `data/raw/era5_manual/<site>/`. Do **not** download the whole period again.
   - The newer August and September values may now be final ERA5 rather than the ERA5T used in the test, and could differ.
   - `ingest_local` drops overlapping days without a defined priority (`np.unique` after concatenation).
2. `python -m src.ingestion.era5 --local --site <site>`, then rerun `python -m src.features.humid_hours --site nakuru` (late blight).
3. Record the download date. Every ERA5 day within about two months of that date is ERA5T.

**Merge and features:**

- Run `python -m src.features.merge` and `python -m src.features.build`.
- **Guard:** every row of `*_daily.parquet`, `features_*.parquet` and `nakuru_humid_hours.parquet` dated up to 2026-09-30 must be **identical** to before. This is the same check as the November extension (`PROJECT_CONTEXT.md` section 11). The future-invariance test `test_features_never_use_the_future` supports this.

### 11.3 Scoring with the frozen final models (no retraining)

Write a new script, `src/models/posttest.py`. **Never rerun `final_test train`** (it retrains), and never write to `test_predictions.parquet` or `reports/test/`. The script should:

1. Load the saved models from `models/<site>/final_train60_score<N>/h<h>/` (all three algorithms) and the late-blight models from `models/nakuru/late_blight_final/<rule>/h<h>/`.
2. Use the **same** per-pixel climatology as the final test (fitted on 2016 to 2024), both as a model input and as the baseline. Recompute the smearing factor exactly as `final_test.py` step 2 does.
3. **Guard:** re-predict a sample of test-period rows and require that they match the saved `test_predictions.parquet` (difference 0). This proves the models and inputs are unchanged.
4. Predict post-test targets with `transfer.predict_all` and `train.baseline_preds`. Write `posttest_predictions.parquet` next to the test file, with a `period` column (`posttest_final`, `posttest_prelim`).
5. Score with the same machinery (`uncertainty.block_bootstrap_weights`, `bootstrap_site_horizon`, calibration functions) into `reports/posttest/`. Report A and B separately, and both baselines.
6. **Expect wide intervals.** 31 target days is only about 4 or 5 seven-day blocks, so the moving-block bootstrap will be unstable. Present these as a short additional check in the October–December short-rains season, not as a new primary result.

Effort: medium. Risk: low for the reported results, because they are not touched.

### 11.4 Dashboard

- **`app/build_data.py`:** read the test and post-test prediction files, add the `period` and `rain_source` columns, and allow issue days whose targets are not yet observed. `meta.json` then extends automatically, because `web/server.py::check_date` reads it.
- **Tests:** `tests/test_web.py` (TC-08 uses "date outside the test period") and the replay check (TC-13) need updating.
- **`app/build_shap.py`:** extend it to the new rows. The full rebuild takes about 10 minutes and peaks near the memory limit.
- **Labels in the app:**
  - A banner on post-test days: "After the reported test period. Not part of the evaluated results."
  - On days where observed rain is preliminary or ERA5 is ERA5T: "Preliminary data: may change when final data is published (CHIRPS final about mid-December; ERA5 about two months after the month)."
  - "Not yet observed" for future targets.
- **"How good is it?":** keep the test results as the headline, and add the post-test table separately with its sample-size caveat.

Effort: medium. Risk: low.

---

## 12. Retraining question

The final models were trained on CHIRPS **v2** and ERA5 (2016 to 2024, early stopping on 2024). Two hard constraints apply:

- The 2025 to 2026 test period and the October to November 2026 period must never be used for training.
- CHIRPS v2 ends after December 2026, so **any option that keeps CHIRPS v2 has a natural end date in early 2027.**

| | a. Keep the models, feed live inputs | b. Retrain on the live source's own archive | c. a or b now, plus periodic retraining |
|---|---|---|---|
| **What** | Frozen final models. Open-Meteo IFS for weather, CHIRPS v2 preliminary plus gap-fill for rain | Weather inputs from Open-Meteo ECMWF IFS 9 km (archive 2017 onward, no delay). Rain from CHIRPS **v3 `sat`** (the only v3 product with a preliminary feed). Fit 2017 to 2023, early stopping 2023, validation 2024, final refit to 2024. Keep the decided settings (CatBoost defaults, 60 pixels) | Add each new full year of *final* data once it exists, then evaluate on the following, still unseen period (rolling origin). Keep a model registry |
| **Effort** | Medium to large (live job plus backtest) | Large: re-ingest about 27,000 Open-Meteo calls (about 3 days at the free limit), switch CHIRPS to v3 (Earth Engine `UCSB-CHC/CHIRPS/V3/DAILY_SAT`; band names **(not verified)**), then rebuild features, retrain, re-validate, retrain the late-blight models, rebuild the dashboard | Small code (`final_test.py` already retrains), but needs a written protocol for each cycle |
| **Main risk** | **Input mismatch.** IFS 9 km values differ from ERA5 0.25°. Surface pressure is the main danger: it depends on model terrain height, and the transfer experiment showed that trees fail on inputs outside their training range (96 to 98% of rows outside range ruined transfer). The late-blight lapse-rate correction uses ERA5 terrain height. Rain for the last 2 to 7 days is model or IMERG rain, not CHIRPS. Ends with CHIRPS v2 | One fewer training year (IFS archive starts 2017). IFS model upgrades within the archive. The v3 "truth" differs from v2, so the new scores are not directly comparable with the reported ones. Free-tier availability | Retraining without a new unseen period would erode the evaluation; drift monitoring is needed |
| **How to evaluate** | **Input-swap backtest on validation 2024 using the *development* models** (`models/<site>/train60_score696/`, fitted to 2022, early stopping 2023), so that 2024 is unseen by them. (1) Replace the ERA5 inputs with Open-Meteo IFS values for late 2023 to 2024. (2) Also replace CHIRPS for the last k = 2 to 7 days before each issue day with the gap-filler. (3) Score paired against the standard development model with `uncertainty.py`. Also report per-variable bias and RMSE of IFS against ERA5 for 2017 to 2023 (training years). If pressure is out of range, test a per-pixel offset fitted on 2017 to 2023 only. Then **prospective scoring**: store each live forecast before its target day, score it after final data arrives, and record the share of days with an on-time forecast (Chapter 6 future work already describes this) | Same validation protocol as the main study (2024, both baselines, paired intervals). Optionally score once on the test period, **clearly labelled post-hoc and secondary**, never replacing the reported results. Then prospective scoring as in a | Each cycle: retrain on everything up to year Y, score prospectively on Y + 1 |
| **Keeps reported results valid?** | Yes (no retraining) | Yes, if the test and post-test periods are never in the fit or early-stopping windows | Yes, if every new evaluation period is after its training data |

**Recommendation: option a now, with the input-swap backtest; option b as the planned next step; then c.**

- **a** answers the supervisor's request in the time left, keeps every reported number, and the backtest turns "the inputs differ" into a measured skill loss.
- **b** is required anyway once CHIRPS v2 stops after December 2026, and it removes the mismatch. Present it as the operational plan (Chapter 6), and start it only after the must-do items below.
- **c** only makes sense on top of b (v3).

If the backtest shows a large loss (for example, day-1 rain skill mostly gone), the live branch should say so on screen, and b moves up in priority.

---

## 13. Recommended plan

### Must-do before final submission (rough order of work)

1. **Fix the record (small).** Change CHIRPS to v2.0 in `PROJECT_CONTEXT.md` section 4 and in the report's data table. Add the daily-split caveat (CFSv2) and the ERA5T note for the end of the test period. Get approval for the config changes in section 11.2.
2. **Prepare the spoken answers for points 6 to 9 (small).** Sections 6 to 9 above, and an updated `DEMO_GUIDE.md` section 6.
3. **Extend ingestion (medium).** CHC GeoTIFF reader with the September guard, a separate preliminary cache and a `rain_source` flag, then the new ERA5 days and humid hours. This is shared by point 11 and the live branch.
4. **Post-test scoring (medium).** `src/models/posttest.py` with the test-reproduction guard, writing to `reports/posttest/`.
5. **Dashboard (medium).**
   - Post-test period, labels, "not yet observed" days.
   - **Farmer mode**: 7-day farm cards, potato and tomato blight, 5 km estimate wording, confidence from tested skill.
   - The latest-forecast demo (4b).
6. **Live branch, option a (medium to large).**
   - Open-Meteo client with ECMWF IFS requested explicitly, for the 129 ERA5 grid nodes.
   - Input-swap backtest on 2024 with the development models.
   - A live forecast job writing to its own store.
   - A banner saying "experimental live mode" with the backtest result.
7. **Explainability (small to medium).** The audience table in the report, plus the comprehension-test protocol. Run a non-farmer pilot only if ethics allows.
8. **Report (medium).** Chapters 4 to 6 changes (section 14). Rehearse the demo with the farmer scenario and no tutorial.

### Nice-to-have (if time remains)

- Option b: retrain on CHIRPS v3 `sat` and Open-Meteo IFS (large). This becomes necessary in early 2027.
- CHIRPS-GEFS v3 as an outside benchmark on validation 2024 (and post-hoc on the test period) (medium).
- CatBoost seeds experiment and a logistic or linear baseline on validation (medium each).
- Text-message preview and Kiswahili strings (small and medium).
- When final ERA5 has replaced ERA5T (about two months after each month), download August to October 2026 again **into a separate folder** and report the size of the ERA5T-to-ERA5 differences. This is a data check only, with no re-scoring.
- Re-score post-test period B when final November CHIRPS is published (small).
- Hosting (container never built; Chapter 6 R4).

---

## 14. Changes to Chapters 4 to 6

**Chapter 4 (`chapter4.tex`):**

- **Users:** make the farmer the primary actor again, with the extension officer and researcher as secondary (lines 45 and 218 currently replace them with Viewer and Administrator). Update the use case diagram.
- **Table "Changes from Chapter 3", "Forecasting mode" row (line 62):** replay remains the evaluated mode; add the experimental live mode (branch) and the post-test replay extension.
- **New functional requirements:** farmer 7-day view; labels for preliminary or unobserved data; 5 km estimate wording; late-blight crop cards.
- **IR-03 and the architecture section:** "external services used only in the offline ingestion stage" and "there is no live forecast to compute" must allow a separate scheduled live job. The web server can stay free of credentials, because Open-Meteo needs no key.
- **Data requirements:** add Open-Meteo (CC BY 4.0, link attribution required next to displayed data) and CHIRPS preliminary.
- **Sign-up:** the invite-only versus sign-up inconsistency already noted in `docs/report/REMAINING.md`.

**Chapter 5 (`chapter5.tex`):**

- **Data sources table:** "CHIRPS v2.0" and its CFSv2 daily split.
- **RQ2 limitations:** daily CHIRPS timing is partly model-made; the last two to three months of ERA5 were ERA5T when downloaded.
- **New subsection** "Post-test period (October–November 2026)": separate, frozen models, small-sample caveat, preliminary labels.
- **Live-input backtest results**, if built, kept separate from the protocol results.
- **Workflow section 5.5.3:** rewrite around the farmer scenario.
- **"Explanation" paragraph:** state the audiences.
- **Selection section:** add the CatBoost defence points (replicated on test, robust to tuning, calibrated, TreeSHAP), and seed results if run.
- **Limitations "Data latency":** give the exact latencies.
- **New tests:** live and post-test test cases if built (the test count changes from 105).

**Chapter 6 (`chapter6_draft/chapter6.tex`):**

- **"Operational Forecasting" future work (line 158):** replace with the concrete plan (option b on CHIRPS v3 and Open-Meteo IFS, forced by the CHIRPS v2 retirement; then c with prospective scoring), plus CHIRPS-GEFS as a benchmark or fusion input.
- **O5 paragraph (line 81) and R1 / R4:** mention the live branch and how it is labelled.
- **"User Evaluation" future work:** add the comprehension protocol and success criterion.
- **Conclusions:** add the post-test result as a short additional check.
- **"5 km" boundary statement**, and the finer-resolution discussion from point 7.

---

## Summary

1. On 20 November 2026 the newest data is CHIRPS v2 final to 31 October (CHC server; Earth Engine still ends 30 September), CHIRPS preliminary to 15 November, and ERA5T to about 15 November; the replay can add October (final) and 1 to 15 November (preliminary, labelled).
2. The project's CHIRPS is v2.0, not v3, and v2 ends after December 2026; this must be corrected in the record and shapes every live or retraining plan.
3. The post-test period should be scored by a new script using the frozen final models and the 2016 to 2024 climatology, with a guard that reproduces the saved test forecasts, written to `reports/posttest/`, so no reported 2025 to 2026 result changes.
4. For live inputs, use Open-Meteo with ECMWF IFS requested explicitly for the weather variables (free for non-commercial use, CC BY 4.0, archive from 2017 with no delay), plus CHIRPS preliminary for rain with a gap-fill for the last days.
5. Retraining: keep the frozen models live now (option a) with an input-swap backtest on 2024 using the development models, and plan option b (CHIRPS v3 and Open-Meteo IFS) as the next step, followed by periodic retraining.
6. Must-do before submission: farmer-first 7-day farm view with potato and tomato blight and 5 km wording, an explanation audience and comprehension protocol, stronger answers for 5 km, CatBoost and "why not live", and matching changes to Chapters 4 to 6.

---

## Progress (20 November 2026, same day, after approval)

Done on branch `feature/posttest-farmer` (not committed). Details are in `docs/PROJECT_CONTEXT.md` section 20.

- **Record fixed:** the data are CHIRPS v2.0 (`PROJECT_CONTEXT.md`, Chapter 5 data table).
- **Point 11:**
  - **Data:** extended to 14 November 2026 (CHIRPS final to 31 Oct, preliminary to 14 Nov; ERA5T to 14 Nov).
  - **Checks:** everything up to 30 September 2026 is identical, and the frozen models reproduce the saved test forecasts with a difference of 0.
  - **Results:** post-test results are in `reports/posttest/`. October rain at Nakuru was no better than climatology (a quiet month); humidity and late blight held up.
  - **Dashboard:** the post-test period is labelled in the dashboard, with "preliminary" and "not observed yet" tags.
- **Points 2, 4, 5, 6:** "My farm" view for farmers, with seven day cards, reliability, the main reason, potato and tomato blight, the 5 km notice and "what happened". The demo script is in `docs/DEMO_GUIDE.md` section 4a.
- **Point 3:** `docs/EXPLAINABILITY_EVALUATION.md`.
- **Points 7-9:** answers in `docs/DEMO_GUIDE.md` section 6 (items 1, 3, 9-11), and the CatBoost defence in Chapter 5.
- **Point 1 / retraining:**
  - Open-Meteo ECMWF IFS 0.25 is used (it answers at ERA5's own grid points).
  - The backtest on 2024 is `src/live/backtest.py`, with results in `reports/live/`.
  - The live forecast job is `src/live/forecast.py`.
  - The dashboard shows live forecasts only with `HL_LIVE=1`.
  - **Backtest run (20 Nov 2026, later the same day):** guards pass, but with the realistic live inputs Nakuru day-1 rain skill falls from 0.215 to -0.131 (worse than climatology), and Busia loses its rain and day-1 humidity skill. Filling the rain gap with IFS is the main cause at Nakuru; at Busia the weather inputs (most likely the +13.6 hPa surface pressure offset) are. Nakuru humidity keeps skill on days 1-2. Details are in `PROJECT_CONTEXT.md` section 20. The live forecast job has not been run since.
  - **Fixes tested and applied (same day):**
    - A cross-fitted per-pixel weather bias correction restores rain skill fully when CHIRPS is available.
    - The best live setup is ERA5 to t-5, then corrected IFS, with the gap rain quantile-mapped to CHIRPS (S5). Its rain is about as good as climatology at every lead time, and Nakuru humidity stays skilful on days 1-2.
    - The live job now uses S5, and forecasts for 15-19 Nov 2026 were made.
    - The dashboard labels every live day a rough guide and shows the backtest table.
  - **Still to do:** split the live code into its own branch when committing. Optional future work: try IMERG Early or CHIRPS v3 preliminary as real-time rain, and backtest late blight with live inputs.
