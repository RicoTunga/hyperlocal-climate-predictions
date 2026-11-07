"""Shared pieces of the dashboard: data access, colour ramps, map layer, legend and layer definitions."""
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "app" / "data"
REPORTS = ROOT / "reports"
PIXEL_HALF = 0.025  # CHIRPS cells are 0.05 degrees; each square spans +/- half a cell around its centre


# ---------------------------------------------------------------- data access
def meta() -> dict:
    return json.loads((DATA / "meta.json").read_text(encoding="utf-8"))


def pixels(site: str) -> pd.DataFrame:
    return pd.read_parquet(DATA / f"{site}_pixels.parquet")


def issue_day(site: str, issue_date) -> pd.DataFrame:
    """All pixels and horizons forecast on one issue day (reads only that day's row groups)."""
    return pd.read_parquet(DATA / f"{site}_replay.parquet", filters=[("issue_date", "==", pd.Timestamp(issue_date))])


def pixel_series(site: str, pixel_id: str, horizon: int, columns: list) -> pd.DataFrame:
    """One pixel at one horizon over the whole test period."""
    return pd.read_parquet(DATA / f"{site}_replay.parquet", columns=["issue_date", "target_date"] + columns,
                           filters=[("pixel_id", "==", pixel_id), ("horizon", "==", horizon)])


# ---------------------------------------------------------------- colour (validated palette, OKLab ramps)
def _srgb_to_lin(c):
    c = np.asarray(c, float)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _lin_to_srgb(c):
    c = np.clip(np.asarray(c, float), 0, 1)
    return np.where(c <= 0.0031308, 12.92 * c, 1.055 * c ** (1 / 2.4) - 0.055)


def hex_to_oklab(h: str) -> np.ndarray:
    r, g, b = _srgb_to_lin([int(h[i:i + 2], 16) / 255 for i in (1, 3, 5)])
    l_ = np.cbrt(0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b)
    m_ = np.cbrt(0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b)
    s_ = np.cbrt(0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b)
    return np.array([0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_,
                     1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_,
                     0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_])


def oklab_to_hex(lab) -> str:
    L, a, b = lab
    l_, m_, s_ = (L + 0.3963377774 * a + 0.2158037573 * b) ** 3, (L - 0.1055613458 * a - 0.0638541728 * b) ** 3, \
                 (L - 0.0894841775 * a - 1.2914855480 * b) ** 3
    rgb = _lin_to_srgb([4.0767416621 * l_ - 3.3077115913 * m_ + 0.2309699292 * s_,
                        -1.2684380046 * l_ + 2.6097574011 * m_ - 0.3413193965 * s_,
                        -0.0041960863 * l_ - 0.7034186147 * m_ + 1.7076147010 * s_])
    return "#" + "".join(f"{int(round(v * 255)):02x}" for v in rgb)


def oklab_ramp(stops: list, n: int) -> list:
    """n colours interpolated in OKLab through the given hex stops (evenly spaced)."""
    labs = [hex_to_oklab(s) for s in stops]
    pos = np.linspace(0, len(labs) - 1, n)
    out = []
    for p in pos:
        i = min(int(np.floor(p)), len(labs) - 2)
        f = p - i
        out.append(oklab_to_hex((1 - f) * labs[i] + f * labs[i + 1]))
    return out


# blue: the palette's sequential steps 100, 200, 300, 400, 500, 600, 700
BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
# orange: second sequential context = categorical slot 2 (#eb6834), light -> dark in OKLab
ORANGE = oklab_ramp(["#fde6da", "#eb6834", "#7a2a0b"], 7)
# diverging blue <-> red with the palette's neutral grey midpoint (#f0efec)
DIVERGING = ["#184f95", "#3987e5", "#9ec5f4", "#f0efec"] + oklab_ramp(["#f6c9c8", "#e34948", "#8f1d1c"], 3)
INK = {"primary": "#0b0b0b", "secondary": "#52514e", "muted": "#898781"}
SITE_COLOURS = {"nakuru": "#2a78d6", "busia": "#eb6834"}  # categorical slots 1 and 2


def lightness(hexes: list) -> list:
    return [float(hex_to_oklab(h)[0]) for h in hexes]


# ---------------------------------------------------------------- layers
@dataclass(frozen=True)
class Layer:
    label: str
    forecast: str          # column with the forecast
    observed: str          # column with what happened
    clim: str              # column with climatology
    breaks: tuple          # 8 edges -> 7 classes
    unit: str
    ramp: tuple
    err_breaks: tuple      # 8 edges for forecast minus observed
    observed_label: str
    blight: bool = False


PROB = (0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0001)
PROB_ERR = (-1.0001, -0.5, -0.2, -0.05, 0.05, 0.2, 0.5, 1.0001)
LAYERS = {
    "Chance of rain (>= 1 mm)": Layer("Chance of rain (>= 1 mm)", "p_rain1", "obs_rain_mm", "clim_rain1", PROB, "%",
                                      tuple(BLUE), PROB_ERR, "observed rain (mm)"),
    "Chance of heavy rain (>= 10 mm)": Layer("Chance of heavy rain (>= 10 mm)", "p_rain10", "obs_rain_mm", "clim_rain10",
                                             PROB, "%", tuple(BLUE), PROB_ERR, "observed rain (mm)"),
    "Chance of very heavy rain (>= 25 mm)": Layer("Chance of very heavy rain (>= 25 mm)", "p_rain25", "obs_rain_mm",
                                                  "clim_rain25", PROB, "%", tuple(BLUE), PROB_ERR, "observed rain (mm)"),
    "Expected rain (mm)": Layer("Expected rain (mm)", "rain_mm", "obs_rain_mm", "clim_rain_mm",
                                (0, 1, 2, 5, 10, 20, 40, 1e9), "mm", tuple(BLUE), (-1e9, -10, -5, -1, 1, 5, 10, 1e9),
                                "observed rain (mm)"),
    "Humidity (%)": Layer("Humidity (%)", "rh", "obs_rh", "clim_rh", (0, 50, 60, 70, 80, 85, 90, 100.001), "%",
                          tuple(BLUE), (-100, -10, -5, -2, 2, 5, 10, 100), "observed humidity (%)"),
    "Late-blight risk (Hutton)": Layer("Late-blight risk (Hutton)", "blight_hutton", "obs_blight_hutton",
                                       "clim_blight_hutton", PROB, "%", tuple(ORANGE), PROB_ERR,
                                       "rule met (1) or not (0)", blight=True),
    "Late-blight risk (humidity only)": Layer("Late-blight risk (humidity only)", "blight_humid", "obs_blight_humid",
                                              "clim_blight_humid", PROB, "%", tuple(ORANGE), PROB_ERR,
                                              "rule met (1) or not (0)", blight=True),
}
RAIN_EVENT = {"p_rain1": 1.0, "p_rain10": 10.0, "p_rain25": 25.0}


def view_values(df: pd.DataFrame, layer: Layer, view: str) -> tuple[np.ndarray, tuple, tuple, str]:
    """(values, breaks, ramp, unit) for the chosen view of a layer.

    For rain-chance layers the 'observed' view shows the rain that fell (mm) and the error view
    compares the forecast probability with whether the event happened (1) or not (0)."""
    if view == "Forecast":
        return df[layer.forecast].to_numpy(), layer.breaks, layer.ramp, layer.unit
    if view == "Climatology (normal for the date)":
        return df[layer.clim].to_numpy(), layer.breaks, layer.ramp, layer.unit
    if view == "Observed":
        if layer.forecast in RAIN_EVENT or layer.forecast == "rain_mm":
            return df["obs_rain_mm"].to_numpy(), LAYERS["Expected rain (mm)"].breaks, tuple(BLUE), "mm"
        return df[layer.observed].to_numpy(), layer.breaks, layer.ramp, layer.unit
    # error: forecast minus what happened
    if layer.forecast in RAIN_EVENT:
        happened = (df["obs_rain_mm"] >= RAIN_EVENT[layer.forecast]).astype(float)
        return (df[layer.forecast] - happened).to_numpy(), layer.err_breaks, tuple(DIVERGING), "points"
    return (df[layer.forecast] - df[layer.observed]).to_numpy(), layer.err_breaks, tuple(DIVERGING), layer.unit


def classify(values: np.ndarray, breaks: tuple) -> np.ndarray:
    """Class index 0..6 for each value (edges are left-closed; the last edge closes the top class)."""
    return np.clip(np.digitize(values, breaks[1:-1], right=False), 0, len(breaks) - 2)


def fmt_value(v: float, unit: str) -> str:
    if unit == "%" and abs(v) <= 1.0001:
        return f"{100 * v:.0f}%"
    if unit == "points":
        return f"{100 * v:+.0f} pts"
    return f"{v:.1f} {unit}" if unit != "%" else f"{v:.0f}%"


def legend_html(breaks: tuple, ramp: tuple, unit: str, title: str) -> str:
    """A horizontal legend: one swatch per class with its range."""
    def edge(x):
        if abs(x) >= 1e8:
            return ""
        if unit in ("%", "points") and abs(x) <= 1.0001 and breaks[-1] <= 1.0001:
            return f"{100 * x:.0f}" + ("%" if unit == "%" else "")
        return f"{x:g}"
    cells = []
    for i, colour in enumerate(ramp):
        lo, hi = edge(breaks[i]), edge(breaks[i + 1])
        label = f"&lt; {hi}" if lo == "" else (f"&ge; {lo}" if hi == "" else f"{lo}-{hi}")
        cells.append(f"<div style='flex:1;text-align:center'><div style='height:14px;background:{colour};"
                     f"border:1px solid rgba(11,11,11,0.1)'></div><div style='font-size:11px;color:{INK['secondary']}'>"
                     f"{label}</div></div>")
    return (f"<div style='font-size:12px;color:{INK['secondary']};margin:4px 0'>{title}</div>"
            f"<div style='display:flex;gap:2px'>{''.join(cells)}</div>")


def squares(df: pd.DataFrame, px: pd.DataFrame) -> pd.DataFrame:
    """Add a square polygon (lon/lat corners) for every pixel."""
    d = df.merge(px, on="pixel_id", how="left", validate="one_to_one")
    d["polygon"] = [[[lo - PIXEL_HALF, la - PIXEL_HALF], [lo + PIXEL_HALF, la - PIXEL_HALF],
                     [lo + PIXEL_HALF, la + PIXEL_HALF], [lo - PIXEL_HALF, la + PIXEL_HALF]]
                    for la, lo in zip(d["lat"], d["lon"])]
    return d


def hex_rgba(h: str, alpha: int = 175) -> list:
    return [int(h[1:3], 16), int(h[3:5], 16), int(h[5:7], 16), alpha]
