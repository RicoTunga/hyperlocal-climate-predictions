# Demo guide: 20-minute read before the supervisor meeting

Sources are in brackets: **ctx** = `docs/PROJECT_CONTEXT.md`, **ch4** / **ch5** = the report chapters.

## 1. The project in five sentences

1. Smallholder farmers need to know whether it will rain and how humid it will be in the next few days, at the scale of their own area rather than a whole county. This project forecasts daily rainfall and relative humidity 1 to 7 days ahead for every ~5 km square ("pixel"): 696 in Nakuru County and 216 in a Busia area.
2. The data is CHIRPS daily rainfall (satellite plus gauge, 0.05°), ERA5 reanalysis weather (temperature, dewpoint, pressure, wind, cloud; humidity computed from temperature and dewpoint) interpolated onto each pixel, and SRTM elevation, for 2016-01-01 to 2026-09-30.
3. Gradient boosting models (CatBoost, LightGBM, XGBoost) were trained per site, lead time and target. CatBoost with default settings is the final model, and SHAP explains each forecast.
4. The dashboard is a **replay**: it shows the final models' saved forecasts for the unseen test period (issue days 7 Jan 2025 to 23 Sep 2026) next to what actually happened.
5. It is a replay rather than live because CHIRPS and ERA5 are published days to weeks after the dates they describe, so they cannot feed a forecast for the coming week. Live use would need a forecast input such as an NWP feed (ch4, ch5 limitations; ctx §9).

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

**Rehearse once before the meeting:** sign in, follow section 4, and check that the map draws.

## 4. Click-by-click demo script (about 5 minutes)

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

1. **Why not live forecasts?** CHIRPS and ERA5 arrive days to weeks late, so there is nothing to forecast from for next week. The replay lets every forecast be checked against what happened. Going live needs an NWP forecast feed, which is future work.
2. **Why no TAHMO or station data?** TAHMO data was not available. Busia station data was offered but is deferred. So the "truth" is gridded CHIRPS and ERA5, not gauges, and skill at a farm may be lower.
3. **Why CatBoost?** The three algorithms are almost equal. In paired comparisons CatBoost won 23 of 33 clear differences on validation and lost none, and 40 of 51 on the test. Tuning gave no measurable gain (1 of 24 comparisons), so I used default settings.
4. **How did you prevent leakage?** Splits are by time only: train 2016–2023, validate 2024, test 2025–Sep 2026, with 7-day gaps and no random splits. Features use only data up to the issue day, and a test changes all future data and checks that no feature moves. The test was run once, with the protocol committed first.
5. **What does SHAP show?** Day-1 rain is driven mostly by area-wide conditions (31% Nakuru, 46% Busia) and the pixel's ERA5 weather. Differences between squares come from ERA5, climatology and own rain history. By day 7, climatology and season carry 44–57%. This shows what the model used, not causes.
6. **Does it work in places it wasn't trained on?** Within a site, yes: an unseen-block hold-out kept +0.176 of Nakuru's +0.201 day-1 rain skill. Between sites, poorly: Busia → Nakuru keeps about half, and Nakuru → Busia is not significant, so a new region needs local training.
7. **Limitations?** Data latency (replay only), gridded truth, smooth ERA5 humidity (not real microclimate), intervals that cover day-to-day but not year-to-year variation, no user testing (needs Kenya Met sign-off), not deployed, and no map without internet.
8. **What comes next?** Station data to check against, a live NWP input, hosting (the Dockerfile exists but has not been built), serving the map libraries locally, and user testing after Kenya Met sign-off.

## 7. If something breaks

- **"No accounts yet" message:** the user database is missing. Run `python -m web.users add <name> --role admin` (the password is typed at a hidden prompt).
- **Forgot the password, or locked out:** 5 wrong tries lock the account for 15 minutes. Reset it with `python -m web.users reset-password MutungaEric`.
- **Port already in use:** another server is running, so close its window. Or start on another port with `$env:HL_PORT = "8766"; python -m web.server` and open http://127.0.0.1:8766.
- **Errors about missing `app/data` files:** rebuild with `python -m app.build_data`, then `python -m app.build_shap` (about 10 minutes, memory-heavy, so close other programs).
- **Map is blank but the page frame loads:** the browser needs internet for MapLibre and Observable Plot (from unpkg/jsDelivr) and the CARTO basemap. There's no error message for this (ch5 TC-15). Connect to Wi-Fi or a phone hotspot and reload.
- **Backup:** before the meeting, while online, run `python -m web.screenshot` (needs Playwright, already in `.playwright/`). It saves screenshots of every view to `web/screenshots/`, and `docs/report/latex/figures/ch4/ui_*.png` already has the map, location drawer and "How good is it?" screens (issued 15 Apr 2025).
