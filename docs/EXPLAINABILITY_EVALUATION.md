# Who the explanations are for, and how to test that they are understood

Written 20 November 2026 in response to the supervisor's point 3 (demo review, 20 November 2026). The explanations are computed in `app/build_shap.py` and shown by `web/static/farm.js` (farmers) and `web/static/panels.js` (map drawer and "Why these forecasts?").

## 1. What is already established, and what is not

- **Faithful:** every explanation adds up exactly to its forecast. TreeSHAP values plus the expected value equal the model output for every row, and the rebuilt forecasts equal the saved ones (maximum difference about 1e-13 points). This is test TC-12 in Chapter 5.
- **Not established:** that people *understand* the explanations or trust them appropriately. No user has seen them. Chapter 5 and Chapter 6 R1 already say so, and testing with farmers needs Kenya Meteorological Department sign-off and ethics approval (`docs/PROJECT_CONTEXT.md` section 2).

## 2. Three audiences, three levels of explanation

| Audience | What they see | Where | Purpose | Built from |
|---|---|---|---|---|
| **Farmers** | One sentence per day: "Main reason: *Rain and humidity across the area* (raises the chance)", plus a reliability label ("Reliable: tested better than the usual" or "Rough guide: close to the usual") | "My farm" day cards | Decide whether to act on a day's forecast, and know when not to rely on it | The largest absolute SHAP group for that day's rain chance; reliability from the test skill (`web/crops.py` `RAIN_SKILL_DAYS`) |
| **Extension officers** (main audience for the full explanation) | The waterfall from "a typical forecast" to this forecast, in 8 plain-language groups and percentage points, next to the normal for the date and what happened | Map → click a square → "Why this forecast?" | Judge a forecast before passing it on, and notice when it only repeats the normal for the date | Exact per-forecast SHAP (`app/build_shap.py`) |
| **Researchers, Kenya Met, examiners** | Share of each input group at 1, 3 and 7 days ahead (overall and within-day), plus skill and calibration | "Why these forecasts?" and "How good is it?"; Chapter 5 | Check that the model behaves sensibly (weather drives day 1, climatology takes over by day 7) | Validation SHAP analysis (`reports/shap_groups_validation.csv`), test results |

The farmer sentence is a summary of the extension-officer waterfall, not a different method. Both come from the same SHAP values.

## 3. Protocol: do people understand the explanations?

Fixed before any session is run, so the result cannot be tuned afterwards.

**Research question.** Can the intended users read (a) the forecast and (b) its explanation correctly? And does their trust follow the measured skill: higher for day 1 than day 5, lowest for heavy rain?

**Participants.**
- *Pilot (possible before submission, if the university's ethics process allows):* 5 to 8 non-farmer volunteers, for example students outside the project. Report it as a pilot of the method only, never as evidence about farmers.
- *Main study (future work, after Kenya Met sign-off and ethics approval):* about 8 extension officers per site, then farmers through them.

**Materials.** Six screenshots made with `python -m web.screenshot`:
- the farm view for two replay days, one wet (14 April 2025) and one quiet (October 2026);
- two map drawers with "Why this forecast?";
- the "How good is it?" panel;
- one day-6 card.

Sessions use the replay, so every question has a known answer and no real decision depends on it.

**Tasks and answer key.** The key is written from the saved data before the sessions.

| # | Task (spoken or written) | Correct if |
|---|---|---|
| T1 | "Will it rain at this farm tomorrow? How sure is the app?" | Gives the chance (within 10 points) and calls it "likely", "possible" or "unlikely" consistently with the card |
| T2 | "What is the main reason for tomorrow's forecast?" | Names the group shown as the main reason |
| T3 | "Is the forecast for day 6 as reliable as the one for tomorrow?" | Says day 6 is less reliable, or close to the usual for the time of year |
| T4 | "Is this number measured at your farm?" | Says no, it is an estimate for the 5 km square |
| T5 | (Drawer) "Which inputs pushed the chance up, and which pushed it down?" | Names at least one group of each sign correctly |
| T6 | "Was this forecast right?" (a past day) | Compares the forecast with "What happened" correctly |
| T7 | Rate your trust, 1 to 5, in tomorrow's forecast, the day-5 forecast and a heavy-rain warning | Scored as calibrated trust: day 1 > day 5, and heavy rain lowest |

**Measures.**
- Share correct per task.
- Time per task.
- Calibrated-trust rate (T7).
- Free comments, coded into confusions (for example, mistaking percentage points for percent).
- Optional: the System Usability Scale for the farm view.

**Success criteria (proposed; no stakeholder has set them).**
- At least 80% correct on T1, T3 and T4 (the safety-relevant tasks).
- At least 70% on T2, T5 and T6.
- At least two thirds of participants with calibrated trust on T7.

If a criterion is missed, change the wording in `web/static/farm.js` or `web/static/panels.js` and re-test with new participants.

**Procedure.**
1. Consent and a short introduction, with no tutorial (the supervisor asked for none in the demo).
2. The tasks in a fixed order, recording answers on a sheet.
3. A two-minute debrief.

About 15 minutes per person. No personal data is stored beyond the answer sheets, and no account is needed: a demo account is used.

**Analysis.** Report counts and proportions with exact binomial 95% intervals. With 8 participants, an interval around 7 of 8 correct is roughly 0.47 to 1.00, so a pilot can show large problems but cannot prove understanding. Say this plainly.

## 4. What changes in the report

- **Chapter 4 (requirements):** keep NFR-09 (faithful explanations). Add a requirement that explanations are pitched per audience, verified by the protocol above.
- **Chapter 5:** keep "correct, not yet shown to be clear". Add the audience table, and the pilot results if a pilot is run.
- **Chapter 6:** the main study becomes the concrete form of the future work "User Evaluation with Extension Officers and Farmers".
