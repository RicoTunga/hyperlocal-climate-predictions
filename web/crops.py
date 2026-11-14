"""Crop insights: plain-language advice for one square and issue day, built from the saved forecasts.

Two kinds of evidence, always labelled in the output:
- "tested": the late-blight risk model (src/models/late_blight.py), scored on the final test period
  (Nakuru only). Used for Irish potato and for tomato (same pathogen, Phytophthora infestans; the Hutton
  criteria were developed for potato, so this is flagged for tomato).
- "rule": a simple guidance rule on the rain and humidity forecasts. The forecasts themselves were tested
  (reports/test/skill.csv); the rules on top of them (thresholds below) were NOT separately evaluated
  against crop outcomes, because no field data were available. They summarise standard agronomic advice;
  they are not a substitute for an extension officer.

The thresholds are module constants so they are visible, testable and easy to change. Everything is a
pure function of the forecast values, so the same input always gives the same advice.
"""
from dataclasses import dataclass

import pandas as pd

# ---------------------------------------------------------------- thresholds (guidance rules)
DRY_P = 0.30         # chance of rain >= 1 mm below this: a "dry" day (spraying, harvesting, drying)
WET_P = 0.60         # at least this: a "wet" day
HEAVY_P10 = 0.35     # chance of >= 10 mm at least this, or ...
HEAVY_P25 = 0.15     # ... chance of >= 25 mm at least this: heavy rain likely
HUMID_RH = 85.0      # daily mean relative humidity at least this: long leaf wetness likely
BLIGHT_HIGH = 0.50   # late-blight model: chance the Hutton criteria are met
BLIGHT_MOD = 0.25
ONSET_MM = 20.0      # expected rain over days 1-3 at least this: planting rains look established
DRY_RUN = 3          # consecutive dry days that make a harvesting / drying window
# lead times with clear skill over the normal for the date on the final test (reports/test/skill.csv):
# rain >= 1 mm days 1-2 in Nakuru, 1-3 in Busia; humidity the whole week; late blight about days 1-5
RAIN_SKILL_DAYS = {"nakuru": 2, "busia": 3}
BLIGHT_SKILL_DAYS = 5


@dataclass(frozen=True)
class Crop:
    id: str
    name: str
    local: str      # Kiswahili name
    where: str      # where it is grown in the study areas
    basis: str      # "tested" or "rule"
    provides: str   # what the app gives for this crop
    blight: bool = False


CATALOGUE = (
    Crop("potato", "Irish potato", "Viazi", "Molo, Kuresoi, Njoro, Bahati and the Mau highlands", "tested",
         "Late-blight risk for the next 7 days (tested model) and spray-timing windows.", blight=True),
    Crop("tomato", "Tomato", "Nyanya", "Naivasha, Rongai, Subukia and Gilgil", "tested",
         "Late-blight risk (same disease as potato; model built for potato) and spray-timing windows.", blight=True),
    Crop("maize", "Maize", "Mahindi", "Grown across the county; Busia too", "rule",
         "Planting-rain signal, fertiliser timing around heavy rain, and dry windows for harvesting and drying."),
    Crop("beans", "Beans", "Maharagwe", "Often intercropped with maize; Nakuru and Busia", "rule",
         "Wet-and-humid disease warnings and dry windows for harvesting pods."),
    Crop("wheat", "Wheat", "Ngano", "Njoro, Mau Narok, Rongai and Molo", "rule",
         "Wet-and-humid disease warnings (rusts, Septoria), heavy-rain lodging risk and harvest windows."),
)
CROPS = {c.id: c for c in CATALOGUE}
CROP_IDS = tuple(CROPS)


def catalogue() -> list[dict]:
    return [c.__dict__ for c in CATALOGUE]


# ---------------------------------------------------------------- inputs
def pixel_days(day: pd.DataFrame, pixel_id: str) -> list[dict]:
    """The 7 forecast days for one square from one issue day's table (app.components.issue_day)."""
    g = day[day["pixel_id"] == pixel_id].sort_values("horizon")
    if len(g) != 7:
        raise KeyError(pixel_id)
    num = lambda r, c: (None if c not in r or pd.isna(r[c]) else float(r[c]))  # noqa: E731
    out = []
    for _, r in g.iterrows():
        t = pd.Timestamp(r["target_date"])
        out.append({"h": int(r["horizon"]), "date": str(t.date()), "label": f"{t:%a} {t.day} {t:%b}",
                    "p1": num(r, "p_rain1"), "p10": num(r, "p_rain10"), "p25": num(r, "p_rain25"),
                    "mm": num(r, "rain_mm"), "rh": num(r, "rh"), "clim_p1": num(r, "clim_rain1"),
                    "clim_rh": num(r, "clim_rh"), "blight": num(r, "blight_hutton")})
    return out


# ---------------------------------------------------------------- day classes
def dry(d) -> bool:
    return d["p1"] is not None and d["p1"] < DRY_P


def wet(d) -> bool:
    return d["p1"] is not None and d["p1"] >= WET_P


def heavy(d) -> bool:
    return (d["p10"] or 0) >= HEAVY_P10 or (d["p25"] or 0) >= HEAVY_P25


def humid(d) -> bool:
    return d["rh"] is not None and d["rh"] >= HUMID_RH


def dry_runs(days, n=DRY_RUN) -> list[tuple[int, int]]:
    """(first, last) horizons of every run of at least n consecutive dry days."""
    runs, start, last = [], None, None
    for d in days:
        if dry(d):
            start = d["h"] if start is None else start
            last = d["h"]
        else:
            if start is not None and last - start + 1 >= n:
                runs.append((start, last))
            start = None
    if start is not None and last - start + 1 >= n:  # a run that lasts to the end of the week
        runs.append((start, last))
    return runs


def span(days, a, b) -> str:
    la = days[a - 1]["label"]
    return la if a == b else f"{la} to {days[b - 1]['label']}"


def names(days, hs) -> str:
    """Readable list of days; runs of 3 or more consecutive days become a range ("Mon 21 Apr to Thu 24 Apr")."""
    groups, hs = [], sorted(hs)
    for h in hs:
        if groups and h == groups[-1][-1] + 1:
            groups[-1].append(h)
        else:
            groups.append([h])
    parts = []
    for g in groups:
        if len(g) >= 3:
            parts.append(span(days, g[0], g[-1]))
        else:
            parts += [days[h - 1]["label"] for h in g]
    return parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + " and " + parts[-1]


def confidence(site, h, kind="rain") -> str:
    limit = BLIGHT_SKILL_DAYS if kind == "blight" else RAIN_SKILL_DAYS.get(site, 2)
    return "good" if h <= limit else "low"


def tag(days, site):
    """One short tag per day for the crop cards' 7-day strip."""
    out = []
    for d in days:
        t = "heavy" if heavy(d) else "wet" if wet(d) else "dry" if dry(d) else "mixed"
        out.append({"h": d["h"], "label": d["label"], "tag": t, "p1": d["p1"], "humid": humid(d),
                    "blight": d["blight"], "confidence": confidence(site, d["h"])})
    return out


# ---------------------------------------------------------------- advice per crop
def _result(crop, status, headline, actions, days, site, extra_caveat=None):
    c = CROPS[crop]
    caveats = []
    if c.basis == "rule":
        caveats.append("Guidance rule on the tested rain and humidity forecasts; the rule itself was not checked "
                       "against crop outcomes.")
    if extra_caveat:
        caveats.append(extra_caveat)
    caveats.append(f"Rain forecasts beat the normal for the date only {RAIN_SKILL_DAYS.get(site, 2)} days ahead; "
                   "treat later days as a rough guide.")
    return {"crop": crop, "name": c.name, "local": c.local, "basis": c.basis, "status": status,
            "headline": headline, "actions": actions, "days": tag(days, site), "caveats": caveats}


def blight_advice(crop, days, site, has_blight):
    if not has_blight:
        n = [d["h"] for d in days if humid(d) and wet(d)]
        status = "watch" if len(n) >= 2 else "ok"
        head = (f"Wet, humid days ({names(days, n)}) favour late blight" if n and status == "watch"
                else "No long wet and humid spell in the forecast")
        return _result(crop, status, head, [
            "The late-blight model covers Nakuru only; this is a humidity-and-rain warning instead.",
            "Check the crop for dark, water-soaked leaf spots after wet spells."], days, site,
            "No tested late-blight model for this site.")
    risk = [(d["h"], d["blight"] or 0.0) for d in days]
    high = [h for h, p in risk if p >= BLIGHT_HIGH]
    mod = [h for h, p in risk if BLIGHT_MOD <= p < BLIGHT_HIGH]
    actions = []
    if high:
        first = high[0]
        before = [d["h"] for d in days if d["h"] <= first and dry(d)]
        status, head = "act", f"High late-blight risk on {names(days, high)}"
        if before:
            actions.append(f"Spray a protectant fungicide on {days[before[0] - 1]['label']} (a dry day), before the risk "
                           "builds. Follow the product label and your extension officer's advice.")
        else:
            actions.append("No dry day before the risk: spray at the first break in the rain. Rain within a few hours of "
                           "spraying washes protectant off.")
        actions.append("Walk the field and remove plants with blight lesions; do not leave culls near the crop.")
    elif mod:
        status, head = "watch", f"Moderate late-blight risk on {names(days, mod)}"
        actions.append("Scout the crop for dark, water-soaked leaf spots and white mould under leaves.")
        dry_h = [d["h"] for d in days if dry(d)]
        if dry_h:
            actions.append(f"If you need to spray, {days[dry_h[0] - 1]['label']} looks dry.")
    else:
        status, head = "ok", "Low late-blight risk this week"
        actions.append("Keep up routine scouting; risk can rise quickly after warm, humid nights.")
    if crop == "tomato":
        actions.append("Water at the base in the morning, not over the leaves; staking and pruning help leaves dry.")
    late = [h for h in high + mod if h > BLIGHT_SKILL_DAYS]
    extra = "Blight risk beyond day 5 is less reliable." if late else None
    if crop == "tomato":
        extra = ("The late-blight model was built and tested for potato; tomato gets the same disease, but the "
                 "thresholds were not checked on tomato.") + (" " + extra if extra else "")
    return _result(crop, status, head, actions, days, site, extra)


def maize_advice(days, site):
    actions, status = [], "ok"
    onset = sum(d["mm"] or 0 for d in days[:3])
    heavy_h = [d["h"] for d in days if heavy(d)]
    runs = dry_runs(days)
    if onset >= ONSET_MM:
        head = f"Good rains expected in the next 3 days (about {onset:.0f} mm)"
        actions.append("Planting: if the soil is moist to about a hand's depth (15 cm), conditions favour planting.")
    elif runs:
        head = f"Dry spell {span(days, *runs[0])}"
    else:
        head = "Mixed week: some rain, no long dry spell"
    light = [d["h"] for d in days if 0.4 <= (d["p1"] or 0) < 0.8 and not heavy(d)]
    if heavy_h:
        status = "watch"
        actions.append(f"Top-dressing: avoid applying fertiliser just before {names(days, heavy_h)} (heavy rain likely; "
                       "it can wash fertiliser away).")
    if light:
        actions.append(f"Top-dressing: {days[light[0] - 1]['label']} has a fair chance of light rain, which helps "
                       "fertiliser into the soil.")
    if runs:
        actions.append(f"Harvesting: {span(days, *runs[0])} looks dry, good for harvesting and drying cobs. Dry grain "
                       "well and off the ground to limit aflatoxin.")
    elif sum(wet(d) for d in days) >= 3:
        actions.append("Harvesting: rain on several days; if you must harvest, dry grain under cover, off the ground.")
    return _result("maize", status, head, actions, days, site)


def humid_disease(days):
    return [d["h"] for d in days if humid(d) and wet(d)]


def beans_advice(days, site):
    hd, runs, heavy_h = humid_disease(days), dry_runs(days), [d["h"] for d in days if heavy(d)]
    actions = []
    if len(hd) >= 2:
        status, head = "watch", f"Wet, humid days ({names(days, hd)}) favour leaf diseases"
        actions.append("Angular leaf spot, anthracnose and bacterial blight spread in wet weather: stay out of the field "
                       "while leaves are wet, since hands and tools carry the disease.")
    else:
        status, head = "ok", "No long wet and humid spell"
    if runs:
        actions.append(f"Harvesting: {span(days, *runs[0])} looks dry, good for pulling and drying mature pods.")
    elif sum(wet(d) for d in days) >= 3:
        actions.append("Harvesting: rain on several days; if pods are mature, dry them under cover to avoid rotting.")
    if heavy_h:
        actions.append(f"Heavy rain likely on {names(days, heavy_h)}: open drainage furrows to avoid waterlogging.")
    return _result("beans", status, head, actions, days, site)


def wheat_advice(days, site):
    hd, runs, heavy_h = humid_disease(days), dry_runs(days), [d["h"] for d in days if heavy(d)]
    actions = []
    if len(hd) >= 2:
        status, head = "watch", f"Wet, humid days ({names(days, hd)}) favour rusts and Septoria"
        actions.append("Scout for yellow or orange rust pustules and Septoria leaf blotch; spray only if you find them, "
                       "as your extension officer advises.")
    else:
        status, head = "ok", "No long wet and humid spell"
    if runs:
        actions.append(f"Harvesting: {span(days, *runs[0])} looks dry, a window for combining ripe wheat.")
    if heavy_h:
        if status == "ok":
            status, head = "watch", f"Heavy rain likely on {names(days, heavy_h)}"
        actions.append(f"Heavy rain likely on {names(days, heavy_h)}: ripe crops may lodge (fall over); harvest ripe "
                       "fields first.")
    return _result("wheat", status, head, actions, days, site)


def advise(crop: str, days: list[dict], site: str, has_blight: bool) -> dict:
    if crop not in CROPS:
        raise KeyError(crop)
    if CROPS[crop].blight:
        return blight_advice(crop, days, site, has_blight)
    return {"maize": maize_advice, "beans": beans_advice, "wheat": wheat_advice}[crop](days, site)


# ---------------------------------------------------------------- general outlook (everyone)
def outlook(days: list[dict], site: str) -> dict:
    """A one-line summary of the week plus the facts the at-a-glance panel shows."""
    wettest = max(days, key=lambda d: d["p1"] or 0)
    heavy_h = [d["h"] for d in days if heavy(d)]
    runs = dry_runs(days, 2)
    parts = []
    if heavy_h:
        parts.append(f"heavy rain possible {names(days, heavy_h)}")
    if (wettest["p1"] or 0) >= WET_P:
        parts.append(f"wettest {wettest['label']} ({round(100 * wettest['p1'])}% chance)")
    if runs:
        parts.append(f"dry {span(days, *runs[0])}")
    if not parts:
        parts.append("no strong signal; close to normal for the time of year")
    line = "; ".join(parts)
    return {"summary": line[0].upper() + line[1:] + ".", "days": tag(days, site), "rain_skill_days": RAIN_SKILL_DAYS.get(site, 2)}
