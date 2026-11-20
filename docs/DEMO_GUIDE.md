# Demo guide: 20-minute read before the supervisor meeting

Sources are in brackets: **ctx** = `docs/PROJECT_CONTEXT.md`, **ch4** / **ch5** = the report chapters.

## 1. The project in five sentences

1. Smallholder farmers need to know whether it will rain and how humid it will be in the next few days, at the scale of their own area rather than a whole county. This project forecasts daily rainfall and relative humidity 1 to 7 days ahead for every ~5 km square ("pixel"): 696 in Nakuru County and 216 in a Busia area.
2. The data is CHIRPS daily rainfall (satellite plus gauge, 0.05°), ERA5 reanalysis weather (temperature, dewpoint, pressure, wind, cloud; humidity computed from temperature and dewpoint) interpolated onto each pixel, and SRTM elevation, for 2016-01-01 to 2026-09-30.
3. Gradient boosting models (CatBoost, LightGBM, XGBoost) were trained per site, lead time and target. CatBoost with default settings is the final model, and SHAP explains each forecast.
4. The dashboard is a **replay**: it shows the final models' saved forecasts for the unseen test period (issue days 7 Jan 2025 to 23 Sep 2026) next to what actually happened. Since 20 Nov 2026 it continues **after the test** to the newest data (issue days to 14 Nov 2026), scored separately and labelled. Forecasts made on the newest data reach days that have not happened yet.
5. The models start from the observed weather up to the day the forecast is made. Preliminary CHIRPS rain arrives 2-7 days late and ERA5 about 5 days late, so the evaluated system replays rather than forecasts live. A live mode on a separate branch feeds the same frozen models with Open-Meteo ECMWF IFS weather and fills the newest rain days. A backtest on 2024 measures what that costs (`docs/NEXT_STEPS.md` section 12; `reports/live/`).

## 2. How it works end to end

1. **Rainfall:** CHIRPS per pixel from Earth Engine (`src.ingestion.chirps`).
2. **Weather:** hourly ERA5 from the Copernicus time-series dataset, converted to daily values and interpolated onto each pixel (`src.ingestion.era5`).
3. **Elevation:** SRTM mean, std, min and max per pixel (`src.ingestion.elevation`).
4. **Merge:** one row per pixel per day, 21 columns, no missing values. The step stops if anything does not line up (`src.features.merge`).
5. **Features:** 72 features (lags, rolling windows, area-wide and neighbourhood rain, season) built using only data up to the issue day (`src.features.build`).
6. **Baselines:** persistence ("same as today") and day-of-year climatology ("normal for the date") (`src.models.baselines`).
7. **Training and selection:** one model per site × lead time × task. Each model is fitted on 60 pixels per site and chosen on validation 2024 only (`src.models.train`).
8. **Final test:** run once on Jan 2025 to Sep 2026, with the protocol committed beforehand (`src.models.final_test`).
9. **Dashboard data:** compact files built from the saved test forecasts (`app.build_data`), plus exact SHAP explanations for every forecast (`app.build_shap`).
10. **Dashboard:** a Starlette server with sign-in that serves map images and JSON. It never runs a model (`web.server`).

## 3. Starting the demo

In PowerShell, from the repo folder (`C:\Users\ericm\hyperlocal-climate-predictions`):

```powershell
.venv\Scripts\activate
python -m web.server
```

It prints `Dashboard running at http://127.0.0.1:8765  (Ctrl+C to stop)`. Open **http://127.0.0.1:8765** in a browser and sign in as **MutungaEric**, the only account (role: admin). Use your own password. Stop the server with **Ctrl+C**.

To land directly on the demo view, open:
`http://127.0.0.1:8765/#site=nakuru&date=2025-04-14&h=1&layer=rain1&view=forecast&mode=smooth`

**Farmer demo account (recommended for the meeting).** The farmer view opens by itself for accounts whose user type is *Farmer*. To set one up once:
1. Run `python -m web.users add demo.farmer` and type the password at the hidden prompt.
2. Sign in and choose *Farmer* in onboarding.
3. Save a place near Njoro: search "njoro", or click the map and choose *Save place*.
4. Tick Irish potato and tomato.
5. Turn off "Tutorial when I sign in" in the menu (the supervisor asked for no tutorial).

**Rehearse once before the meeting:** sign in, follow section 4, and check that the map draws.

## 4a. Farmer scenario first (about 3 minutes): "I am a farmer in Nakuru today; will it rain in the next 7 days?"

Sign in as the farmer account. "My farm" opens, headed **Will it rain at (place)?**

| Step | Do | Say |
|---|---|---|
| 1 | Point at the blue box | "Every number is an estimate for the 5 km square around the farm, from satellite and weather-model data. It is not a measurement at the farm." |
| 2 | Press **Newest** | "This is the newest forecast the data allow, made on 14 November from data up to that day. Rain data arrive a few days late, so the first days are already past and the rest are still ahead. Those say *Not observed yet*." |
| 3 | Read the seven cards left to right | "For each day: the chance of rain in numbers and words, the usual chance for the date, the average amount, humidity, and blight weather for potato and tomato. Days 1-2 are marked *Reliable*, because the test showed they beat the normal for the date. From day 3, rain is a rough guide, close to the normal." |
| 4 | Point at *Main reason* on a card | "This is a one-sentence explanation for the farmer. The full breakdown is for extension officers, under *Why this forecast?*" |
| 5 | Scroll to *Your crops* | "For potato and tomato, the late-blight risk comes from the tested model, with a spray window on a dry day. Tomato uses the potato model, and the card says so." |
| 6 | Press ← a few times, or open 14 Apr 2025 | "Because it is a replay, I can show what happened: *What happened: rain, 16 mm* appears under each card." |

Then press **Map** and continue with the map script below, which is for the researcher view.

## 4b. Map demo script (about 5 minutes)

**Start:** Nakuru, issued 14 Apr 2025, 1 day ahead, layer *Chance ≥ 1 mm*, view *Forecast*.
- Why this view: Nakuru is the pilot site, rain at day 1 is the headline result, and this is the long rains. I checked the replay files: on this day the average forecast was 54% and 68% of squares got ≥ 1 mm. The pattern of the forecast matched where it rained better than on most days (correlation 0.72).
- Be honest that this is a good day. Press **→** to go to 15 April: the average forecast there was 39%, but only 17% of squares got rain.

| Step | Do | Say |
|---|---|---|
| 1 | Show the map and the "Research replay · not an official forecast" badge | "Each 5 km square shows the chance of at least 1 mm of rain tomorrow, as forecast on 14 April 2025. This is a replay of unseen test data, not an official forecast." |
| 2 | Click *Observed* in the right panel, then back to *Forecast* | "Because it's a replay, I can show what actually fell on the same squares." |
| 3 | Click *Normal* | "This is climatology, the usual chance for this date. It's the baseline the model has to beat." |
| 4 | Click day pills **+1 → +7**, or press **Space** to play | "As lead time grows, the forecast fades towards the normal. That's the skill decay you'll see in the numbers." |
| 5 | Switch layer to *Humidity*, then *Late blight* | "Humidity is much more predictable than rain. Late blight is a downstream risk layer for Nakuru potatoes, based on the Hutton criteria." |
| 6 | Back to *Chance ≥ 1 mm*. Click a square to open the **location drawer** | "For one square: the 7-day rain chance against the normal and what happened, plus humidity, blight risk and the whole test period." |
| 7 | In the drawer, read **Why this forecast?** | "SHAP breaks this forecast down from a typical forecast into plain-language input groups, and the parts add up exactly. It shows what the model used, not what causes the weather." |
| 8 | Open **How good is it?** and switch *Compared with* between climatology and persistence | "These are the test-period scores, computed once, with 95% intervals. Rain beats the normal on days 1–2, humidity all week, and both beat 'same as today' at every lead time." |

## 5. Headline results (final test, 8 Jan 2025 to 30 Sep 2026, CatBoost, all pixels; ch5 Table "Primary Outcomes" and test summary)

Skill is 0 when the model is no better than the reference and 1 when it is perfect. Brackets are 95% intervals.
- **Day-1 rain (≥ 1 mm):** Nakuru +0.205 [+0.155, +0.258] vs climatology and +0.407 vs persistence. Busia +0.170 [+0.119, +0.222] vs climatology and +0.384 vs persistence.
- **Rain skill fades fast:** Nakuru +0.078 on day 2 and +0.021 [−0.003, +0.046] on day 3, which is not clear (the only one of the 12 primary outcomes that misses). Busia +0.077 on day 2 and +0.048 on day 3, not clear from day 4. Calibration stays fine (no bias interval excludes 0); the forecasts lose resolution, not reliability.
- **Humidity:** day 1 Nakuru +0.521 and Busia +0.389 vs climatology (+0.110 and +0.121 vs persistence). Clear skill on all 7 days at both sites (day 7: +0.116 and +0.049).
- **Heavy rain is the weak spot:** ≥ 25 mm day-1 skill is only +0.052 (Nakuru) and +0.066 (Busia). With a 50% warning level the model caught just 0.4% and 1.8% of heavy-rain days (post-hoc check). Busia day 7 is slightly worse than climatology (−0.012).
- **Pre-registered:** the protocol was committed before the test (`dffea25`), and 11 of 12 primary outcomes exclude zero. Skill held up as well as or better than in validation.

## 6. Likely supervisor questions

1. **Why not live forecasts?**
   - The models start from the weather observed up to the forecast day. Preliminary CHIRPS is published 2 days after each pentad (2-7 days late), final CHIRPS in the third week of the next month, and ERA5T about 5 days late. On 20 November the newest complete inputs were for 14 November.
   - The replay is deliberate. Every forecast can be checked against what happened, and the scores were computed once under a protocol committed in advance (`dffea25`).
   - Kenya Met is the forecasting authority, so a public live forecast needs its sign-off.
   - To show the bridge, use the newest-data forecast (farmer view, *Newest*) and the live branch. The live branch feeds the frozen models Open-Meteo ECMWF IFS weather, and its cost in skill was measured on 2024 (`reports/live/backtest_skill.csv`).
2. **Why no TAHMO or station data?** TAHMO data was not available. Busia station data was offered but is deferred. So the "truth" is gridded CHIRPS and ERA5, not gauges, and skill at a farm may be lower.
3. **Why CatBoost?**
   - It was chosen by a rule fixed before the test: better whenever the algorithms differed clearly on validation (23 of 33 clear differences, none lost).
   - The choice was then **replicated** on the unseen test: 40 of 51 clear differences, against 9 for XGBoost and 2 for LightGBM.
   - It is robust without tuning (1 of 24 comparisons) and well calibrated without recalibration.
   - It trains in 6-7 minutes per site on the CPU laptop, and its built-in exact TreeSHAP explains 4.3 million forecasts.
   - Ordered boosting was designed against prediction shift (Prokhorenkova et al., 2018).
   - Be honest: the margins are small (at most 0.038), and in the short post-test period the three did not differ consistently (LightGBM 22, CatBoost 20, XGBoost 18 clear wins).
4. **How did you prevent leakage?** Splits are by time only: train 2016–2023, validate 2024, test 2025–Sep 2026, with 7-day gaps and no random splits. Features use only data up to the issue day, and a test changes all future data and checks that no feature moves. The test was run once, with the protocol committed first.
5. **What does SHAP show?** Day-1 rain is driven mostly by area-wide conditions (31% Nakuru, 46% Busia) and the pixel's ERA5 weather. Differences between squares come from ERA5, climatology and own rain history. By day 7, climatology and season carry 44–57%. This shows what the model used, not causes.
6. **Does it work in places it wasn't trained on?** Within a site, yes: an unseen-block hold-out kept +0.176 of Nakuru's +0.201 day-1 rain skill. Between sites, poorly: Busia → Nakuru keeps about half, and Nakuru → Busia is not significant, so a new region needs local training.
7. **Limitations?** Data latency (replay only), gridded truth, smooth ERA5 humidity (not real microclimate), intervals that cover day-to-day but not year-to-year variation, no user testing (needs Kenya Met sign-off), not deployed, and no map without internet.
8. **What comes next?**
   - Station data to check the forecasts against.
   - Retraining on CHIRPS v3 (v2 ends after December 2026) with Open-Meteo IFS inputs, so training and live inputs match.
   - Hosting (the Dockerfile exists but has not been built) and serving the map libraries locally.
   - The comprehension study in `docs/EXPLAINABILITY_EVALUATION.md`, after Kenya Met sign-off.
9. **Why 5 km, and could it be finer?**
   - 5 km is the grid of the only long, free, daily gridded rain record the models are trained and scored against (CHIRPS 0.05°).
   - Even that is optimistic for daily rain. Gauges are blended into five-day totals, and the split into days comes from a coarser weather model (CFSv2).
   - Humidity comes from ERA5 at about 25 km.
   - Finer inputs exist (ECMWF IFS 9 km, ERA5-Land about 10 km). But without a finer reference, a finer map could not be checked: it would be false precision.
   - The honest next step is station validation.
10. **What happened after the test?** October and 1-14 November 2026, frozen models, reported separately (`reports/posttest/`):
   - Humidity day 1 stayed clearly skilful at both sites.
   - Nakuru day-1 rain was no better than the normal in October, a quiet month with one widespread rain day. It was +0.198 in early November, on preliminary data.
   - Busia rain was slightly worse than the normal from day 2 in early November.
   - Late blight was skilful all week.
   - These are only 31 + 14 days, so the intervals are wide.
11. **Who are the explanations for?**
   - Farmers: one sentence per day (the main reason) and a reliability label.
   - Extension officers: the full waterfall.
   - Researchers and Kenya Met: the overall SHAP shares.
   - All three are exact and add up, but whether people understand them is untested. `docs/EXPLAINABILITY_EVALUATION.md` gives the test protocol, with success criteria.

## 7. If something breaks

- **"No accounts yet" message:** the user database is missing. Run `python -m web.users add <name> --role admin` (the password is typed at a hidden prompt).
- **Forgot the password, or locked out:** 5 wrong tries lock the account for 15 minutes. Reset it with `python -m web.users reset-password MutungaEric`.
- **Port already in use:** another server is running, so close its window. Or start on another port with `$env:HL_PORT = "8766"; python -m web.server` and open http://127.0.0.1:8766.
- **Errors about missing `app/data` files:** rebuild with `python -m app.build_data`, then `python -m app.build_shap` (about 10 minutes, memory-heavy, so close other programs).
- **Map is blank but the page frame loads:** the browser needs internet for MapLibre and Observable Plot (from unpkg/jsDelivr) and the CARTO basemap. There's no error message for this (ch5 TC-15). Connect to Wi-Fi or a phone hotspot and reload.
- **Backup:** before the meeting, while online, run `python -m web.screenshot` (needs Playwright, already in `.playwright/`). It saves screenshots of every view to `web/screenshots/`, and `docs/report/latex/figures/ch4/ui_*.png` already has the map, location drawer and "How good is it?" screens (issued 15 Apr 2025).
