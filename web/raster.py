"""Colour-field PNGs for the map: one site, one issue day, one horizon, one layer and view.

Values come from app.components.view_values (the same layers, class breaks and views as the data
checks). Colours use dark-mode ramps: on the dark basemap the lowest class is the darkest and the
highest the brightest, so attention goes to high values (palette rule: dark mode gets its own
steps from the same ramps, checked against the dark surface).

Two modes:
- smooth: the value grid is upsampled with bilinear interpolation before colouring (the Windy
  look); a display choice only, the data are 5 km pixels;
- cells:  each 5 km pixel is a crisp block, showing the true resolution.
"""
from io import BytesIO

import numpy as np
from PIL import Image
from scipy.ndimage import zoom

from app import components as C
from src.features.build import grid_index

DARK_SURFACE = "#1a1a19"
# dark-mode ramps: low = dark, high = bright (built in OKLab from the palette's own hues)
RAMPS_DARK = {  # tops stay saturated (not near-white), so high values read as intense on the dark map
    "blue": C.oklab_ramp(["#0f2a4d", "#2a78d6", "#7cc0ff"], 7),
    "aqua": C.oklab_ramp(["#0b3326", "#1baf7a", "#7ff0c3"], 7),
    "orange": C.oklab_ramp(["#3d1607", "#eb6834", "#ffb07a"], 7),
}
# diverging for errors: dark neutral midpoint, arms brighten outward (blue = too low, red = too high)
DIVERGING_DARK = ["#cde2fb", "#3987e5", "#1c5cab", "#383835", "#a9302f", "#e66767", "#f8d3d2"]
SCALE = 12  # output pixels per 5 km cell
FEATHER_CELLS = 0.6  # smooth mode: fade the field's border into the map over this many cells


def ramp_for(layer: C.Layer, view: str) -> list:
    if view == "error":
        return DIVERGING_DARK
    if layer.blight and view != "observed":
        return RAMPS_DARK["orange"]
    if layer.forecast == "rh":
        return RAMPS_DARK["aqua"]
    return RAMPS_DARK["blue"]


def finite_breaks(breaks: tuple) -> np.ndarray:
    """Replace open-ended edges (+/- 1e9) with one class-width beyond their neighbour."""
    b = np.array(breaks, float)
    if abs(b[0]) >= 1e8:
        b[0] = b[1] - (b[2] - b[1])
    if abs(b[-1]) >= 1e8:
        b[-1] = b[-2] + (b[-2] - b[-3])
    return b


def class_position(values: np.ndarray, breaks: tuple) -> np.ndarray:
    """Continuous position 0..6 along the 7 classes: class k spans [k, k+1) around its centre k+0.5."""
    b = finite_breaks(breaks)
    pos = np.interp(values, b, np.arange(len(b)))  # 0 at the first edge, 7 at the last
    return np.clip(pos - 0.5, 0, len(b) - 2)


def colourise(pos: np.ndarray, ramp: list, alpha: np.ndarray) -> np.ndarray:
    rgb = np.array([[int(h[i:i + 2], 16) for i in (1, 3, 5)] for h in ramp], float)
    lo = np.floor(pos).astype(int)
    hi = np.minimum(lo + 1, len(ramp) - 1)
    f = (pos - lo)[..., None]
    out = (1 - f) * rgb[lo] + f * rgb[hi]
    return np.dstack([out, alpha]).astype(np.uint8)


def value_grid(day, pixels, layer: C.Layer, view: str):
    """(2-D value grid with north at the top, breaks) for one horizon's rows of an issue day."""
    names = {"forecast": "Forecast", "observed": "Observed", "error": "Forecast minus observed",
             "normal": "Climatology (normal for the date)"}
    d = day.merge(pixels, on="pixel_id", how="left", validate="one_to_one")
    values, breaks, _, _ = C.view_values(d, layer, names[view])
    gi, gj = grid_index(d["lat"].to_numpy()), grid_index(d["lon"].to_numpy())
    grid = np.full((gi.max() + 1, gj.max() + 1), np.nan)
    grid[gi, gj] = values
    return grid[::-1], breaks  # row 0 = northernmost


def render_png(day, pixels, layer: C.Layer, view: str, mode: str = "smooth") -> bytes:
    grid, breaks = value_grid(day, pixels, layer, view)
    missing = np.isnan(grid)
    filled = np.where(missing, np.nanmean(grid), grid)
    if mode == "smooth":
        big = zoom(filled, SCALE, order=1, grid_mode=True, mode="nearest")
        big_missing = zoom(missing.astype(float), SCALE, order=0, grid_mode=True, mode="nearest") > 0.5
    else:
        big = np.kron(filled, np.ones((SCALE, SCALE)))
        big_missing = np.kron(missing, np.ones((SCALE, SCALE), bool))
    pos = class_position(big, breaks)
    ramp = ramp_for(layer, view)
    if view == "error" or layer.forecast == "rh":
        alpha = np.full(pos.shape, 205.0)
    else:  # low amounts and chances fade into the map, high ones stand out
        alpha = 70 + 150 * pos / (len(ramp) - 1)
    alpha[big_missing] = 0
    if mode == "smooth":  # soft edge, so the field blends into the basemap instead of ending in a hard box
        rows, cols = np.indices(alpha.shape)
        edge = np.minimum.reduce([rows, cols, alpha.shape[0] - 1 - rows, alpha.shape[1] - 1 - cols]).astype(float)
        alpha *= np.clip(edge / (FEATHER_CELLS * SCALE), 0, 1)
    buf = BytesIO()
    Image.fromarray(colourise(pos, ramp, alpha), "RGBA").save(buf, "PNG", optimize=False)
    return buf.getvalue()


def bounds(pixels) -> list:
    """Image corner coordinates for MapLibre: top-left, top-right, bottom-right, bottom-left."""
    w, e = pixels["lon"].min() - C.PIXEL_HALF, pixels["lon"].max() + C.PIXEL_HALF
    s, n = pixels["lat"].min() - C.PIXEL_HALF, pixels["lat"].max() + C.PIXEL_HALF
    return [[w, n], [e, n], [e, s], [w, s]]
