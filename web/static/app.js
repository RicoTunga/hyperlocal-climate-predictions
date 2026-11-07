// Map-first forecast replay (Windy/Ventusky style). Data come from the local server (web/server.py).
import { openDrawer, refreshDrawer, closeDrawer, openSkill, openAbout } from "./panels.js";

const $ = (s) => document.querySelector(s);
const $$ = (s) => [...document.querySelectorAll(s)];
const BASEMAP = "https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json";
const EVENT_MM = { p_rain1: 1, p_rain10: 10, p_rain25: 25 };
const VIEW_NAMES = { forecast: "forecast", observed: "observed", error: "forecast minus observed", normal: "normal for the date" };
export const LABELS = { rain1: "Chance of rain ≥ 1 mm", rain10: "Chance of heavy rain ≥ 10 mm",
  rain25: "Chance of very heavy rain ≥ 25 mm", rainmm: "Expected rain", rh: "Humidity",
  blight_hutton: "Late-blight risk (Hutton)", blight_humid: "Late-blight risk (humidity only)" };
const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
let lastUrl = null;

export const state = {
  meta: null, pixels: {}, index: {}, dates: {},
  site: "nakuru", issue: "2025-04-15", h: 1, layer: "rain1", view: "forecast", mode: "smooth",
  day: null, selected: null, playing: null, map: null,
};
const dayCache = new Map();

export const api = (path) => fetch(path).then((r) => {
  if (!r.ok) throw new Error(`${path}: ${r.status}`);
  return r.json();
});

// ---------------------------------------------------------------- dates and formatting
const toDate = (s) => new Date(`${s}T00:00:00`);
const iso = (d) => d.toISOString().slice(0, 10);
const addDays = (s, n) => { const d = toDate(s); d.setDate(d.getDate() + n); return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, "0")}-${String(d.getDate()).padStart(2, "0")}`; };
export const fmtDate = (s, opts = { weekday: "short", day: "numeric", month: "short", year: "numeric" }) =>
  toDate(s).toLocaleDateString("en-GB", opts);

export function layerInfo(lid = state.layer) { return state.meta.layers[lid]; }
const isProb = (L) => L.forecast.startsWith("p_") || L.blight;

/** Value of one pixel (position i) for a layer and view, mirroring app/components.view_values. */
export function valueOf(lid, view, H, i) {
  const L = layerInfo(lid);
  const get = (c) => (H[c] ? H[c][i] : null);
  if (view === "forecast") return get(L.forecast);
  if (view === "normal") return get(L.clim);
  const ev = EVENT_MM[L.forecast];
  if (view === "observed") return (ev || L.forecast === "rain_mm") ? get("obs_rain_mm") : get(L.observed);
  if (ev) { const o = get("obs_rain_mm"), f = get(L.forecast); return o == null || f == null ? null : f - (o >= ev ? 1 : 0); }
  const f = get(L.forecast), o = get(L.observed);
  return f == null || o == null ? null : f - o;
}

export function fmtValue(lid, view, v) {
  if (v == null || Number.isNaN(v)) return "–";
  const L = layerInfo(lid);
  const unit = L.views[view].unit;
  const sign = (x) => (x >= 0 ? "+" : "−");
  if (unit === "points") return `${sign(v)}${Math.abs(Math.round(100 * v))} pts`;
  if (unit === "mm") return `${v.toFixed(1)} mm`;
  if (L.blight && view === "observed") return v >= 0.5 ? "rule met" : "not met";
  if (isProb(L)) return view === "error" ? `${sign(v)}${Math.abs(Math.round(100 * v))} pts` : `${Math.round(100 * v)}%`;
  if (view === "error") return `${sign(v)}${Math.abs(v).toFixed(1)}%`;
  return `${v.toFixed(0)}%`;
}

// ---------------------------------------------------------------- data
async function loadDay(site, issue) {
  const key = `${site}/${issue}`;
  if (!dayCache.has(key)) {
    dayCache.set(key, api(`/api/day/${site}/${issue}`));
    if (dayCache.size > 12) dayCache.delete(dayCache.keys().next().value);
  }
  return dayCache.get(key);
}

function buildIndex(site) {
  const s = state.meta.sites[site], P = state.pixels[site], idx = new Map();
  P.ids.forEach((_, k) => {
    const gi = Math.round((P.lat[k] - s.lat0) / s.step), gj = Math.round((P.lon[k] - s.lon0) / s.step);
    idx.set(`${gi}_${gj}`, k);
  });
  state.index[site] = idx;
  const dates = [];
  for (let d = s.issue_min; d <= s.issue_max; d = addDays(d, 1)) dates.push(d);
  state.dates[site] = dates;
}

export function pixelAt(lngLat) {
  const s = state.meta.sites[state.site];
  const gi = Math.round((lngLat.lat - s.lat0) / s.step), gj = Math.round((lngLat.lng - s.lon0) / s.step);
  return state.index[state.site].get(`${gi}_${gj}`) ?? null;
}

// ---------------------------------------------------------------- map
function boxGeoJSON(b) { return { type: "Feature", geometry: { type: "Polygon", coordinates: [[...b, b[0]]] } }; }
function cellGeoJSON(i) {
  if (i == null) return { type: "FeatureCollection", features: [] };
  const P = state.pixels[state.site], h = state.meta.sites[state.site].step / 2;
  const lo = P.lon[i], la = P.lat[i];
  return boxGeoJSON([[lo - h, la + h], [lo + h, la + h], [lo + h, la - h], [lo - h, la - h]]);
}
const rasterUrl = () =>
  `/api/raster/${state.site}/${state.issue}/${state.h}/${state.layer}/${state.view}.png?mode=${state.mode}`;

function fitSite(animate = true) {
  const b = state.meta.sites[state.site].bounds;
  state.map.fitBounds([b[3], b[1]], { padding: { top: 40, bottom: 130, left: 370, right: 250 }, duration: animate ? 900 : 0 });
}

function initMap() {
  const site = state.meta.sites[state.site];
  const map = new maplibregl.Map({
    container: "map", style: BASEMAP, center: site.center, zoom: 8.4, minZoom: 5, maxZoom: 12,
    attributionControl: { compact: true }, dragRotate: false, pitchWithRotate: false,
  });
  state.map = map;
  map.addControl(new maplibregl.NavigationControl({ showCompass: false }), "bottom-right");
  return new Promise((resolve) => map.on("load", () => {
    const firstLabel = map.getStyle().layers.find((l) => l.type === "symbol")?.id;
    lastUrl = rasterUrl();
    map.addSource("field", { type: "image", url: lastUrl, coordinates: site.bounds });
    map.addLayer({ id: "field", type: "raster", source: "field",
      paint: { "raster-opacity": 0.92, "raster-resampling": "linear", "raster-fade-duration": 180 } }, firstLabel);
    map.addSource("box", { type: "geojson", data: boxGeoJSON(site.bounds) });
    map.addLayer({ id: "box", type: "line", source: "box",
      paint: { "line-color": "#f0efec", "line-opacity": 0.16, "line-width": 1, "line-dasharray": [3, 3] } }, firstLabel);
    map.addSource("selected", { type: "geojson", data: cellGeoJSON(null) });
    map.addLayer({ id: "selected", type: "line", source: "selected", paint: { "line-color": "#ffffff", "line-width": 2 } });
    fitSite(false);
    map.on("mousemove", onHover);
    map.on("mouseout", () => { $("#readout").hidden = true; map.getCanvas().style.cursor = ""; });
    map.on("click", (e) => { const i = pixelAt(e.lngLat); if (i != null) selectPixel(i); });
    resolve();
  }));
}

function updateMapLayer() {
  const map = state.map, s = state.meta.sites[state.site];
  const url = rasterUrl();
  if (url !== lastUrl) {  // only reload the image when something changed (avoids aborting a load in progress)
    document.body.dataset.ready = "0";
    lastUrl = url;
    map.getSource("field").updateImage({ url, coordinates: s.bounds });
  }
  map.getSource("box").setData(boxGeoJSON(s.bounds));
  map.setPaintProperty("field", "raster-resampling", state.mode === "cells" ? "nearest" : "linear");
  map.once("idle", () => { document.body.dataset.ready = "1"; });
  map.triggerRepaint();
}

// ---------------------------------------------------------------- hover readout
function onHover(e) {
  const i = pixelAt(e.lngLat), box = $("#readout");
  if (i == null || !state.day) { box.hidden = true; state.map.getCanvas().style.cursor = ""; return; }
  state.map.getCanvas().style.cursor = "crosshair";
  box.innerHTML = readoutHTML(i);
  box.hidden = false;
  const pad = 16, w = box.offsetWidth, h = box.offsetHeight;
  let x = e.originalEvent.clientX + pad, y = e.originalEvent.clientY + pad;
  if (x + w > innerWidth - 8) x = e.originalEvent.clientX - w - pad;
  if (y + h > innerHeight - 8) y = e.originalEvent.clientY - h - pad;
  box.style.left = `${x}px`; box.style.top = `${y}px`;
}

function readoutHTML(i) {
  const H = state.day.horizons[state.h], L = layerInfo(), P = state.pixels[state.site];
  const v = valueOf(state.layer, state.view, H, i);
  const f = valueOf(state.layer, "forecast", H, i), n = valueOf(state.layer, "normal", H, i);
  const o = valueOf(state.layer, "observed", H, i);
  return `<div>${LABELS[state.layer]} · ${VIEW_NAMES[state.view]}</div>
    <div class="big">${fmtValue(state.layer, state.view, v)}</div>
    <div class="row"><span>Forecast</span><b>${fmtValue(state.layer, "forecast", f)}</b></div>
    <div class="row"><span>Normal for the date</span><b>${fmtValue(state.layer, "normal", n)}</b></div>
    <div class="row"><span>Observed</span><b>${fmtValue(state.layer, "observed", o)}</b></div>
    <div class="where">${P.lat[i].toFixed(3)}°, ${P.lon[i].toFixed(3)}° · ${P.elev[i].toLocaleString()} m · click for 7-day detail</div>`;
}

// ---------------------------------------------------------------- legend, headline, controls
function renderLegend() {
  const L = layerInfo(), V = L.views[state.view], ramp = V.ramp, br = V.breaks;
  const stops = ramp.map((c, k) => `${c} ${((k + 0.5) / ramp.length) * 100}%`).join(", ");
  const label = (x, k) => {
    if (x == null) return "";
    let t = V.unit === "points" ? `${x > 0 ? "+" : ""}${Math.round(100 * x)}`
      : isProb(L) && V.unit === "%" ? `${Math.round(100 * x)}%`
        : `${Math.round(x * 10) / 10}`;
    if (k === br.length - 1 && V.unit === "mm") t += "+";  // open-ended top class
    return t;
  };
  const title = state.view === "observed" && (EVENT_MM[L.forecast] || L.forecast === "rain_mm") ? "Observed rain (mm)"
    : state.view === "error" ? `${LABELS[state.layer]}: forecast minus observed${V.unit === "points" ? " (percentage points)" : ""}`
      : `${LABELS[state.layer]} · ${VIEW_NAMES[state.view]}${V.unit === "mm" ? " (mm)" : V.unit === "%" && !isProb(L) ? " (%)" : ""}`;
  $("#legend").innerHTML = `<div class="legend-title">${title}</div>
    <div class="legend-bar" style="background: linear-gradient(90deg, ${stops})"></div>
    <div class="legend-ticks">${br.map((x, k) => `<span>${label(x, k)}</span>`).join("")}</div>`;
}

function renderHeadline() {
  const H = state.day?.horizons[state.h];
  const target = H ? H.target : addDays(state.issue, state.h);
  $("#headline").textContent = fmtDate(target, { weekday: "long", day: "numeric", month: "long", year: "numeric" });
  $("#subline").textContent = `Issued ${fmtDate(state.issue, { day: "numeric", month: "short", year: "numeric" })} · ${state.h} day${state.h > 1 ? "s" : ""} ahead`;
}

function renderControls() {
  $$("#site-seg button").forEach((b) => b.setAttribute("aria-checked", String(b.dataset.site === state.site)));
  $$("#view-seg button").forEach((b) => b.setAttribute("aria-checked", String(b.dataset.view === state.view)));
  $$(".layer").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.layer === state.layer)));
  $("#crop-group").hidden = !state.meta.sites[state.site].has_blight;
  $("#cells-toggle").checked = state.mode === "cells";
  const dates = state.dates[state.site];
  const range = $("#issue-range");
  range.max = String(dates.length - 1);
  range.value = String(dates.indexOf(state.issue));
  const input = $("#issue-input");
  input.min = dates[0]; input.max = dates[dates.length - 1]; input.value = state.issue;
  $("#day-pills").innerHTML = Array.from({ length: 7 }, (_, k) => {
    const h = k + 1, t = addDays(state.issue, h);
    return `<button role="radio" data-h="${h}" aria-checked="${h === state.h}">
      <b>${fmtDate(t, { weekday: "short", day: "numeric" })}</b><small>+${h} day${h > 1 ? "s" : ""}</small></button>`;
  }).join("");
  $$("#day-pills button").forEach((b) => b.addEventListener("click", () => setState({ h: Number(b.dataset.h) })));
}

function renderMonths() {
  const dates = state.dates[state.site], n = dates.length - 1;
  $("#tl-months").innerHTML = dates.map((d, k) => {
    if (!d.endsWith("-01") && k !== 0) return "";
    const isJan = d.slice(5, 7) === "01";
    const text = isJan ? d.slice(0, 4) : MONTHS[Number(d.slice(5, 7)) - 1];
    return `<span class="${isJan ? "year" : ""}" style="left:${(k / n) * 100}%">${k === 0 ? "" : text}</span>`;
  }).join("");
}

function updateHash() {
  const p = new URLSearchParams({ site: state.site, date: state.issue, h: state.h, layer: state.layer, view: state.view, mode: state.mode });
  history.replaceState(null, "", `#${p}`);
}

// ---------------------------------------------------------------- state changes
export async function setState(patch) {
  const siteChanged = patch.site && patch.site !== state.site;
  Object.assign(state, patch);
  if (siteChanged) {
    const dates = state.dates[state.site];
    if (!dates.includes(state.issue)) state.issue = state.issue < dates[0] ? dates[0] : dates[dates.length - 1];
    if (!state.meta.sites[state.site].has_blight && state.layer.startsWith("blight")) state.layer = "rain1";
    state.selected = null;
    state.map.getSource("selected").setData(cellGeoJSON(null));
    closeDrawer();
    fitSite();
    renderMonths();
  }
  renderControls();
  renderLegend();
  updateHash();
  updateMapLayer();
  state.day = await loadDay(state.site, state.issue);
  renderHeadline();
  if (state.selected != null) refreshDrawer();
  // warm the cache for the neighbouring issue days
  const dates = state.dates[state.site], k = dates.indexOf(state.issue);
  [k - 1, k + 1].filter((j) => j >= 0 && j < dates.length).forEach((j) => loadDay(state.site, dates[j]));
}

function selectPixel(i) {
  state.selected = i;
  state.map.getSource("selected").setData(cellGeoJSON(i));
  openDrawer();
}

function stepIssue(delta) {
  const dates = state.dates[state.site], k = dates.indexOf(state.issue);
  const j = Math.min(Math.max(k + delta, 0), dates.length - 1);
  if (j !== k) setState({ issue: dates[j] });
}

function togglePlay() {
  const btn = $("#play");
  if (state.playing) {
    clearInterval(state.playing); state.playing = null;
    btn.innerHTML = '<svg viewBox="0 0 24 24"><path d="M8 5l11 7-11 7z"/></svg>';
    btn.setAttribute("aria-label", "Play through days ahead");
    return;
  }
  btn.innerHTML = '<svg viewBox="0 0 24 24"><path d="M7 5h3v14H7zM14 5h3v14h-3z"/></svg>';
  btn.setAttribute("aria-label", "Pause");
  state.playing = setInterval(() => setState({ h: state.h % 7 + 1 }), 1200);
}

function wireControls() {
  $$("#site-seg button").forEach((b) => b.addEventListener("click", () => setState({ site: b.dataset.site })));
  $$("#view-seg button").forEach((b) => b.addEventListener("click", () => setState({ view: b.dataset.view })));
  $$(".layer").forEach((b) => b.addEventListener("click", () => setState({ layer: b.dataset.layer })));
  $("#cells-toggle").addEventListener("change", (e) => setState({ mode: e.target.checked ? "cells" : "smooth" }));
  $("#issue-range").addEventListener("input", (e) => {
    const d = state.dates[state.site][Number(e.target.value)];
    $("#issue-input").value = d;
  });
  $("#issue-range").addEventListener("change", (e) => setState({ issue: state.dates[state.site][Number(e.target.value)] }));
  $("#issue-input").addEventListener("change", (e) => { if (state.dates[state.site].includes(e.target.value)) setState({ issue: e.target.value }); });
  $("#prev-day").addEventListener("click", () => stepIssue(-1));
  $("#next-day").addEventListener("click", () => stepIssue(1));
  $("#play").addEventListener("click", togglePlay);
  $("#open-skill").addEventListener("click", openSkill);
  $("#open-about").addEventListener("click", openAbout);
  $("#close-drawer").addEventListener("click", () => {
    state.selected = null; state.map.getSource("selected").setData(cellGeoJSON(null)); closeDrawer();
  });
  $$(".modal").forEach((m) => {
    m.addEventListener("click", (e) => { if (e.target === m || e.target.closest(".close-modal")) m.hidden = true; });
  });
  document.addEventListener("keydown", (e) => {
    if (e.target.matches("input, select")) return;
    if (e.key === "Escape") { $$(".modal").forEach((m) => (m.hidden = true)); return; }
    if (e.key === "ArrowLeft") stepIssue(-1);
    else if (e.key === "ArrowRight") stepIssue(1);
    else if (e.key === "ArrowUp") setState({ h: Math.min(7, state.h + 1) });
    else if (e.key === "ArrowDown") setState({ h: Math.max(1, state.h - 1) });
    else if (e.key === " ") { e.preventDefault(); togglePlay(); }
    else return;
    e.preventDefault();
  });
}

function readHash() {
  const p = new URLSearchParams(location.hash.slice(1));
  const patch = {};
  if (["nakuru", "busia"].includes(p.get("site"))) patch.site = p.get("site");
  if (p.get("date")) patch.issue = p.get("date");
  if (p.get("h")) patch.h = Math.min(7, Math.max(1, Number(p.get("h")) || 1));
  if (p.get("layer") && state.meta.layers[p.get("layer")]) patch.layer = p.get("layer");
  if (["forecast", "observed", "error", "normal"].includes(p.get("view"))) patch.view = p.get("view");
  if (["smooth", "cells"].includes(p.get("mode"))) patch.mode = p.get("mode");
  Object.assign(state, patch);
  const dates = state.dates[state.site];
  if (!dates.includes(state.issue)) state.issue = dates[Math.floor(dates.length * 0.7)];
  if (!state.meta.sites[state.site].has_blight && state.layer.startsWith("blight")) state.layer = "rain1";
}

async function main() {
  state.meta = await api("/api/meta");
  for (const s of Object.keys(state.meta.sites)) {
    state.pixels[s] = await api(`/api/pixels/${s}`);
    buildIndex(s);
  }
  readHash();
  await initMap();
  wireControls();
  renderMonths();
  await setState({});
  window.__app = { state, setState, selectPixel, openSkill, openAbout };  // used by the screenshot script
}

main().catch((err) => {
  console.error(err);
  document.body.insertAdjacentHTML("beforeend", `<div class="panel" style="top:40%;left:50%;transform:translate(-50%,-50%);padding:16px">Could not load the dashboard data: ${err.message}</div>`);
});
