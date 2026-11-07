// Drawer (meteogram for one location), "How good is it?" and "About" overlays.
import * as Plot from "https://cdn.jsdelivr.net/npm/@observablehq/plot@0.6.16/+esm";
import { state, api, fmtDate } from "./app.js";

const $ = (s) => document.querySelector(s);
const C = {  // dark-mode tokens and series colours (validated palette)
  ink: "#f0efec", ink2: "#c3c2b7", ink3: "#898781", grid: "#2c2c2a", axis: "#383835",
  rain: "#3987e5", rainLight: "#9ec5f4", humid: "#2ec28e", blight: "#d95926",
  site: { Nakuru: "#3987e5", Busia: "#d95926" },
};
const style = { background: "transparent", color: C.ink2, fontSize: "11px", fontFamily: "inherit", overflow: "visible" };
const axisDefaults = { tickSize: 0, labelArrow: "none" };

// ---------------------------------------------------------------- drawer
let seriesH = 1;

export function openDrawer() {
  $("#drawer").classList.add("open");
  $("#drawer").setAttribute("aria-hidden", "false");
  refreshDrawer();
}

export function closeDrawer() {
  $("#drawer").classList.remove("open");
  $("#drawer").setAttribute("aria-hidden", "true");
}

function outlookRows(i) {
  return Array.from({ length: 7 }, (_, k) => {
    const H = state.day.horizons[k + 1];
    const g = (c) => (H[c] ? H[c][i] : null);
    return {
      h: k + 1, label: fmtDate(H.target, { weekday: "short", day: "numeric" }),
      p: g("p_rain1"), clim: g("clim_rain1"), obs: g("obs_rain_mm"),
      rh: g("rh"), rhObs: g("obs_rh"), rhClim: g("clim_rh"),
      bl: g("blight_hutton"), blClim: g("clim_blight_hutton"), blObs: g("obs_blight_hutton"),
    };
  });
}

export function refreshDrawer() {
  const i = state.selected;
  if (i == null || !state.day) return;
  const P = state.pixels[state.site];
  $("#drawer-title").textContent = `${P.lat[i].toFixed(3)}°, ${P.lon[i].toFixed(3)}° · ${P.elev[i].toLocaleString()} m`;
  const rows = outlookRows(i), labels = rows.map((r) => r.label);
  const body = $("#drawer-body");
  body.innerHTML = `
    <div class="section-title">Chance of rain (≥ 1 mm), issued ${fmtDate(state.issue, { day: "numeric", month: "short" })}</div>
    <div id="m-rain"></div>
    <div class="section-note">Bars: forecast · dashed line: normal for the date · dot: what happened (filled = rained)</div>
    <div class="section-title">Humidity (%)</div>
    <div id="m-rh"></div>
    ${state.meta.sites[state.site].has_blight ? `<div class="section-title">Late-blight risk (Hutton criteria)</div>
      <div id="m-bl"></div><div class="section-note">Orange: forecast chance the criteria are met · dashed: normal · dot: met</div>` : ""}
    <div class="section-title">Whole test period</div>
    <div class="chip-row" role="radiogroup" aria-label="Days ahead">${[1, 2, 3, 4, 5, 6, 7].map((h) =>
      `<button class="chip" role="radio" data-h="${h}" aria-checked="${h === seriesH}">+${h}d</button>`).join("")}</div>
    <div id="m-series"><div class="section-note">Loading…</div></div>`;
  const W = 400;
  $("#m-rain").append(Plot.plot({
    width: W, height: 170, style, marginLeft: 34, marginTop: 22,
    x: { domain: labels, label: null, ...axisDefaults, padding: 0.25 },
    y: { domain: [0, 1], ticks: [0, 0.5, 1], tickFormat: ".0%", label: null, grid: true, ...axisDefaults },
    marks: [
      Plot.gridY({ stroke: C.grid, strokeOpacity: 1, ticks: [0, 0.5, 1] }),
      Plot.barY(rows, { x: "label", y: "p", fill: C.rain, insetLeft: 2, insetRight: 2 }),
      Plot.tickY(rows, { x: "label", y: "clim", stroke: C.ink, strokeWidth: 1.5, strokeDasharray: "3,2" }),
      Plot.dot(rows, { x: "label", y: () => 1, dy: -12, r: 4.5, fill: (d) => (d.obs >= 1 ? C.rainLight : C.ink3 + "00"),
        stroke: (d) => (d.obs >= 1 ? C.rainLight : C.ink3), strokeWidth: 1.3 }),
      Plot.tip(rows, Plot.pointerX({ x: "label", y: "p", fill: "#111", stroke: C.axis,
        title: (d) => `${d.label}\nforecast ${Math.round(100 * d.p)}%\nnormal ${Math.round(100 * d.clim)}%\nobserved ${d.obs?.toFixed(1)} mm` })),
    ],
  }));
  const rh = rows.flatMap((r) => [
    { label: r.label, series: "forecast", v: r.rh }, { label: r.label, series: "observed", v: r.rhObs },
    { label: r.label, series: "normal", v: r.rhClim }]);
  $("#m-rh").append(Plot.plot({
    width: W, height: 150, style, marginLeft: 34,
    x: { domain: labels, label: null, ...axisDefaults, type: "point", padding: 0.5 },
    y: { label: null, grid: true, nice: true, ...axisDefaults },
    color: { domain: ["forecast", "observed", "normal"], range: [C.humid, C.ink, C.ink3], legend: true, style: { ...style, marginLeft: "34px" } },
    marks: [
      Plot.lineY(rh, { x: "label", y: "v", stroke: "series", strokeWidth: 2, strokeDasharray: (d) => (d.series === "normal" ? "4,3" : null) }),
      Plot.dot(rh.filter((d) => d.series !== "normal"), { x: "label", y: "v", fill: "series", r: 3 }),
      Plot.tip(rh, Plot.pointer({ x: "label", y: "v", fill: "#111", stroke: C.axis, title: (d) => `${d.label} ${d.series}: ${d.v?.toFixed(1)}%` })),
    ],
  }));
  if (state.meta.sites[state.site].has_blight) {
    $("#m-bl").append(Plot.plot({
      width: W, height: 130, style, marginLeft: 34, marginTop: 22,
      x: { domain: labels, label: null, ...axisDefaults, padding: 0.25 },
      y: { domain: [0, 1], ticks: [0, 0.5, 1], tickFormat: ".0%", label: null, ...axisDefaults },
      marks: [
        Plot.gridY({ stroke: C.grid, strokeOpacity: 1, ticks: [0, 0.5, 1] }),
        Plot.barY(rows, { x: "label", y: "bl", fill: C.blight, insetLeft: 2, insetRight: 2 }),
        Plot.tickY(rows, { x: "label", y: "blClim", stroke: C.ink, strokeWidth: 1.5, strokeDasharray: "3,2" }),
        Plot.dot(rows, { x: "label", y: () => 1, dy: -12, r: 4.5, fill: (d) => (d.blObs > 0.5 ? C.blight : C.ink3 + "00"),
          stroke: (d) => (d.blObs > 0.5 ? C.blight : C.ink3), strokeWidth: 1.3 }),
      ],
    }));
  }
  body.querySelectorAll(".chip").forEach((b) => b.addEventListener("click", () => { seriesH = Number(b.dataset.h); refreshDrawer(); }));
  renderSeries(i, W);
}

async function renderSeries(i, W) {
  const site = state.site, id = state.pixels[site].ids[i];
  const s = await api(`/api/pixel/${site}/${encodeURIComponent(id)}?h=${seriesH}`);
  if (state.selected !== i || state.site !== site) return;  // the user moved on
  const rows = s.target.map((t, k) => ({ date: new Date(`${t}T00:00:00`), rh: s.rh[k], rhObs: s.obs_rh[k], p: s.p_rain1[k], wet: s.obs_rain_mm[k] >= 1 ? 1 : 0 }));
  const months = Plot.groupX({ y: "mean" }, { x: (d) => new Date(d.date.getFullYear(), d.date.getMonth(), 1) });
  const box = $("#m-series");
  box.innerHTML = `<div class="section-note">Humidity: forecast (green) vs observed (white), ${seriesH} day${seriesH > 1 ? "s" : ""} ahead</div>`;
  box.append(Plot.plot({
    width: W, height: 140, style, marginLeft: 34,
    x: { label: null, ...axisDefaults }, y: { label: null, grid: true, nice: true, ...axisDefaults },
    marks: [
      Plot.lineY(rows, { x: "date", y: "rhObs", stroke: C.ink, strokeWidth: 1, strokeOpacity: 0.8 }),
      Plot.lineY(rows, { x: "date", y: "rh", stroke: C.humid, strokeWidth: 1.3 }),
    ],
  }));
  box.insertAdjacentHTML("beforeend", `<div class="section-note">Rain by month: average forecast chance (blue) vs share of days it rained (white). Reliable forecasts track each other.</div>`);
  box.append(Plot.plot({
    width: W, height: 140, style, marginLeft: 34,
    x: { label: null, ...axisDefaults }, y: { label: null, grid: true, tickFormat: ".0%", domain: [0, 1], ...axisDefaults },
    marks: [
      Plot.lineY(rows, { ...months, y: "wet", stroke: C.ink, strokeWidth: 1.5, marker: "circle-stroke" }),
      Plot.lineY(rows, { ...months, y: "p", stroke: C.rain, strokeWidth: 2, marker: "circle" }),
    ],
  }));
}

// ---------------------------------------------------------------- skill overlay
const TARGETS = { "rain>=1mm": "Rain (≥ 1 mm)", "rain>=10mm": "Heavy rain (≥ 10 mm)", "rain>=25mm": "Very heavy rain (≥ 25 mm)",
  rain_mm_wet: "Rain amount on wet days", rh_pct: "Humidity" };
let skillData = null;
const pick = { target: "rain>=1mm", reference: "climatology", period: "full test period" };

export async function openSkill() {
  $("#skill-modal").hidden = false;
  if (!skillData) {
    $("#skill-body").innerHTML = `<div class="section-note">Loading…</div>`;
    const [skill, cal, blight] = await Promise.all([api("/api/skill"), api("/api/calibration"), api("/api/blight_skill")]);
    skillData = { skill, cal, blight };
  }
  renderSkill();
}

function skillChart(rows, colourKey, colours, height = 280, width = 620) {
  // nudge each series sideways on a numeric axis so the two sets of intervals do not overlap
  const first = Object.keys(colours)[0];
  const data = rows.map((d) => ({ ...d, xo: d.horizon + (d[colourKey] === first ? -0.09 : 0.09) }));
  return Plot.plot({
    width, height, style, marginLeft: 44, marginRight: 24, marginTop: 24, marginBottom: 36,
    x: { label: "days ahead", domain: [0.6, 7.4], ticks: [1, 2, 3, 4, 5, 6, 7], tickFormat: "d", ...axisDefaults },
    y: { label: "skill score", grid: true, nice: true, ...axisDefaults },
    color: { domain: Object.keys(colours), range: Object.values(colours), legend: true,
      style: { background: "transparent", color: C.ink2, fontSize: "12px", marginLeft: "44px" } },
    marks: [
      Plot.gridY({ stroke: C.grid, strokeOpacity: 1 }),
      Plot.ruleY([0], { stroke: C.ink3, strokeDasharray: "4,3" }),
      Plot.ruleX(data, { x: "xo", y1: "ci_low", y2: "ci_high", stroke: colourKey, strokeWidth: 2.5, strokeOpacity: 0.55 }),
      Plot.lineY(data, { x: "xo", y: "estimate", stroke: colourKey, strokeWidth: 2 }),
      Plot.dot(data, { x: "xo", y: "estimate", fill: colourKey, r: 4.5, stroke: "#1a1a19", strokeWidth: 1.5 }),
      Plot.tip(data, Plot.pointer({ x: "xo", y: "estimate", fill: "#111", stroke: C.axis,
        title: (d) => `${d[colourKey]}, day ${d.horizon}\n${d.estimate >= 0 ? "+" : ""}${d.estimate.toFixed(3)}  [${d.ci_low.toFixed(3)}, ${d.ci_high.toFixed(3)}]` })),
    ],
  });
}

function renderSkill() {
  const { skill, cal, blight } = skillData;
  const periods = [...new Set(skill.map((d) => d.period))];
  const rows = skill.filter((d) => d.target === pick.target && d.reference === pick.reference && d.period === pick.period)
    .map((d) => ({ ...d, site: d.site[0].toUpperCase() + d.site.slice(1) }));
  const opt = (obj, cur) => Object.entries(obj).map(([k, v]) => `<option value="${k}" ${k === cur ? "selected" : ""}>${v}</option>`).join("");
  $("#skill-body").innerHTML = `
    <p class="summary">Scored <b>once</b> on 21 months the models never saw (8 Jan 2025 – 30 Sep 2026), every pixel, with 95% intervals
    from resampling blocks of days. <b>0</b> = no better than the reference, <b>1</b> = perfect. Rain forecasts are clearly better than the
    normal for the date on days 1–2 (Busia to day 3); humidity for the whole week; both beat “same as today” at every horizon.</p>
    <div class="controls">
      <label>What<select id="s-target">${opt(TARGETS, pick.target)}</select></label>
      <label>Compared with<select id="s-ref">${opt({ climatology: "the normal for the date", persistence: "“same as today”" }, pick.reference)}</select></label>
      <label>Period<select id="s-period">${opt(Object.fromEntries(periods.map((p) => [p, p])), pick.period)}</select></label>
    </div>
    <div id="s-chart"></div>
    <div class="grid2" style="margin-top:14px">
      <div class="card"><h3>Late-blight risk (Nakuru)</h3><p>Post-hoc check on the same period. Day 1 is partly known on the issue day; day 2 onwards is the fair measure.</p><div id="s-blight"></div></div>
      <div class="card"><h3>Are the chances honest? (rain ≥ 1 mm)</h3><p>Bias near 0 and slope near 1 mean the stated chances match how often it rained.</p><div id="s-cal"></div></div>
    </div>`;
  $("#s-chart").append(skillChart(rows, "site", C.site, 300, 1000));
  const bl = blight.map((d) => ({ horizon: d.horizon, estimate: d.bss_vs_climatology, ci_low: d.bss_vs_climatology_low,
    ci_high: d.bss_vs_climatology_high, rule: d.rule === "hutton" ? "Hutton" : "Humidity only" }));
  $("#s-blight").append(skillChart(bl, "rule", { Hutton: C.blight, "Humidity only": C.humid }, 220, 470));
  const c = cal.filter((d) => d.threshold_mm === 1);
  $("#s-cal").innerHTML = `<table class="data"><thead><tr><th>Site</th><th>Day</th><th>Rained</th><th>Forecast</th><th>Bias</th><th>Slope</th></tr></thead><tbody>${
    c.map((d) => `<tr><td>${d.site[0].toUpperCase() + d.site.slice(1)}</td><td>${d.horizon}</td><td>${Math.round(100 * d.observed_rate)}%</td>
      <td>${Math.round(100 * d.mean_forecast)}%</td><td>${d.bias >= 0 ? "+" : "−"}${Math.abs(100 * d.bias).toFixed(1)} pts</td><td>${d.slope.toFixed(2)}</td></tr>`).join("")
  }</tbody></table>`;
  $("#s-target").addEventListener("change", (e) => { pick.target = e.target.value; renderSkill(); });
  $("#s-ref").addEventListener("change", (e) => { pick.reference = e.target.value; renderSkill(); });
  $("#s-period").addEventListener("change", (e) => { pick.period = e.target.value; renderSkill(); });
}

// ---------------------------------------------------------------- about overlay
export function openAbout() {
  const m = state.meta.sites;
  $("#about-body").innerHTML = `
    <div class="notice">Research prototype (Strathmore University). A replay of past forecasts for 2025–2026, not a live or official
    forecast: the Kenya Meteorological Department is the authority.</div>
    <h3>What it forecasts</h3>
    <p>Daily rain (chance of at least 1, 10 and 25 mm, and expected amount) and relative humidity, 1 to 7 days ahead, for each
    ~5 km square in Nakuru County (${m.nakuru.n_pixels} squares) and a Busia area (${m.busia.n_pixels} squares); plus a potato
    late-blight risk indicator for Nakuru.</p>
    <h3>How</h3>
    <p>CatBoost gradient-boosting models trained on 2016–2024 with inputs from CHIRPS rainfall, ERA5 weather and SRTM elevation.
    Every result is compared with the normal for the date (climatology) and “same as today” (persistence).</p>
    <h3>Why a replay</h3>
    <p>CHIRPS and ERA5 are published days to weeks after the fact, so the models cannot run in real time from these sources. The
    dashboard replays forecasts for 2025–2026, which the models never saw, so each one can be shown next to what happened.</p>
    <h3>Caveats</h3>
    <ul>
      <li>“Observed” is itself an estimate: CHIRPS (satellite plus gauges) for rain, ERA5 reanalysis (~28 km) for humidity.</li>
      <li>Squares are about 5 km; the smooth map is a display choice (switch on “5 km squares” to see the true cells).</li>
      <li>Late-blight risk uses the Hutton criteria from Great Britain; their 10 °C minimum rarely occurs above 2,600 m, so a
      humidity-only version is shown too. No observed outbreak data were available to check it.</li>
      <li>Models trained on one site did not transfer well to the other: a new area needs local training data.</li>
      <li>2026 had unusual months (wet February–March, dry June–September), which makes the normal a weaker reference that year.</li>
    </ul>
    <h3>Keyboard</h3>
    <p>← → previous / next issue day · ↑ ↓ days ahead · space play · Esc close.</p>
    <p style="color:#898781">Basemap © OpenStreetMap contributors © CARTO. Methods and numbers: docs/PROJECT_CONTEXT.md in the project repository.</p>`;
  $("#about-modal").hidden = false;
}
