"""Tests for the crop advice rules (web/crops.py): each rule fires on the forecast pattern it describes."""
import pytest

from web import crops as C


def week(over=None, p1=0.5, p10=0.0, p25=0.0, mm=2.0, rh=70.0, blight=0.0):
    """Seven forecast days; `over` overrides values on given horizons, e.g. over={"p1": {3: 0.9}}."""
    per_day, days = over or {}, []
    for h in range(1, 8):
        d = {"h": h, "date": f"2025-04-{10 + h:02d}", "label": f"Day{h}", "p1": p1, "p10": p10, "p25": p25, "mm": mm,
             "rh": rh, "clim_p1": 0.5, "clim_rh": 75.0, "blight": blight}
        for k, v in per_day.items():
            if h in v:
                d[k] = v[h]
        days.append(d)
    return days


def test_catalogue_has_potato_tomato_and_three_rule_crops():
    assert C.CROP_IDS == ("potato", "tomato", "maize", "beans", "wheat")
    assert {c.id for c in C.CATALOGUE if c.basis == "tested"} == {"potato", "tomato"}


def test_dry_runs():
    days = week(p1=0.8, over={"p1": {2: 0.1, 3: 0.2, 4: 0.1, 6: 0.1, 7: 0.1}})
    assert C.dry_runs(days, 3) == [(2, 4)]
    assert C.dry_runs(days, 2) == [(2, 4), (6, 7)]


def test_high_blight_risk_recommends_spraying_on_a_dry_day_before_it():
    days = week(p1=0.7, over={"blight": {4: 0.8}, "p1": {2: 0.1}})
    a = C.advise("potato", days, "nakuru", has_blight=True)
    assert a["status"] == "act" and "Day4" in a["headline"]
    assert "Day2" in a["actions"][0] and a["basis"] == "tested"


def test_high_blight_with_no_dry_day_says_spray_at_first_break():
    a = C.advise("potato", week(p1=0.9, blight=0.6), "nakuru", has_blight=True)
    assert a["status"] == "act" and "first break" in a["actions"][0]


def test_low_and_moderate_blight_and_tomato_caveat():
    assert C.advise("potato", week(blight=0.05), "nakuru", True)["status"] == "ok"
    assert C.advise("potato", week(over={"blight": {3: 0.3}}), "nakuru", True)["status"] == "watch"
    t = C.advise("tomato", week(blight=0.05), "nakuru", True)
    assert any("tomato" in c for c in t["caveats"]) and any("base" in x for x in t["actions"])


def test_busia_has_no_blight_model_so_falls_back_to_humidity_warning():
    a = C.advise("potato", week(p1=0.8, rh=90), "busia", has_blight=False)
    assert a["status"] == "watch" and any("No tested" in c for c in a["caveats"])


def test_maize_planting_heavy_rain_and_harvest_window():
    a = C.advise("maize", week(mm=9.0, p1=0.7), "nakuru", False)
    assert "Good rains" in a["headline"] and any("Planting" in x for x in a["actions"])
    a = C.advise("maize", week(p1=0.7, over={"p10": {2: 0.5}}), "nakuru", False)
    assert a["status"] == "watch" and any("avoid applying fertiliser just before Day2" in x for x in a["actions"])
    a = C.advise("maize", week(p1=0.1, mm=0.1), "nakuru", False)
    assert any("Harvesting" in x and "aflatoxin" in x for x in a["actions"])
    assert a["basis"] == "rule" and any("not checked" in c for c in a["caveats"])


def test_beans_and_wheat_disease_warnings_need_two_wet_humid_days():
    one = week(p1=0.2, rh=60, over={"p1": {1: 0.9}, "rh": {1: 90}})
    two = week(p1=0.2, rh=60, over={"p1": {1: 0.9, 2: 0.9}, "rh": {1: 90, 2: 90}})
    for crop in ("beans", "wheat"):
        assert C.advise(crop, one, "nakuru", False)["status"] == "ok"
        assert C.advise(crop, two, "nakuru", False)["status"] == "watch"


def test_wheat_heavy_rain_lodging():
    a = C.advise("wheat", week(p1=0.2, over={"p25": {5: 0.3}}), "nakuru", False)
    assert any("lodge" in x for x in a["actions"])


def test_outlook_and_confidence():
    o = C.outlook(week(p1=0.2, over={"p1": {3: 0.9}, "p10": {3: 0.5}}), "nakuru")
    assert "Heavy rain possible Day3" in o["summary"] and "wettest Day3" in o["summary"]
    assert [d["confidence"] for d in o["days"]][:3] == ["good", "good", "low"]
    assert C.outlook(week(p1=0.45), "busia")["summary"].startswith("No strong signal")


def test_unknown_crop():
    with pytest.raises(KeyError):
        C.advise("coffee", week(), "nakuru", False)
