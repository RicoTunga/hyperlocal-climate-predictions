// "My farm": the farmer's view. It answers one question in plain words: will it rain at my farm over the
// next 7 days? Seven day cards (chance of rain in words and numbers, expected amount, humidity, potato and
// tomato blight weather), how far each day can be trusted (from the measured test skill), what actually
// happened (the app is a replay), and clear labels: a 5 km estimate, not a measurement at the farm;
// days after the reported test; preliminary data; days not observed yet.
// Everything from the server or the user is escaped (esc) before it is put into HTML.
import { state, bus, esc, fmtDate, setState, goToSquare, SITE_NAMES, fmtLatLon, api, isLiveIssue } from "./app.js";
import { homePlace, placeById, placeCoords, getAdvice, ICON, cropsDialog } from "./user.js";
import { locateMe, focusSearch } from "./search.js";
import { openSkill } from "./panels.js";

const $ = (s) => document.querySelector(s);
let shownPlace = null;  // id of the place shown; null = home place
let ticket = 0;
const STATUS = { act: "Act now", watch: "Watch", ok: "OK" };
const BLIGHT = (p) => (p == null ? null : p >= 0.5 ? ["high", "High"] : p >= 0.25 ? ["medium", "Medium"] : ["low", "Low"]);
const pct = (p) => (p == null ? "–" : `${Math.round(100 * p)}%`);
const mmText = (v) => (v == null ? "–" : v < 0.5 ? "average under 0.5 mm" : `average ${v < 10 ? v.toFixed(1) : Math.round(v)} mm`);

export function isFarmOpen() { return !$("#farm").hidden; }

export function openFarm() {
  $("#farm").hidden = false;
  document.body.classList.add("farm-open");
  $("#farm-btn")?.setAttribute("aria-pressed", "true");
  renderFarm();
}

/** Open "My farm" on one saved place at its newest forecast (used by the alerts list). */
export async function showFarmPlace(placeId) {
  const p = placeById(placeId);
  if (!p) return;
  shownPlace = placeId;
  const dates = state.dates[p.site];
  if (dates?.length) await setState({ site: p.site, issue: dates[dates.length - 1] });
  openFarm();
}

export function closeFarm() {
  $("#farm").hidden = true;
  document.body.classList.remove("farm-open");
  $("#farm-btn")?.setAttribute("aria-pressed", "false");
}

export function initFarm() {
  const farm = $("#farm");
  $("#farm-btn")?.addEventListener("click", () => (isFarmOpen() ? closeFarm() : openFarm()));
  farm.addEventListener("click", onClick);
  farm.addEventListener("change", (e) => {
    if (e.target.id === "farm-place") { shownPlace = Number(e.target.value); renderFarm(); }
  });
  let t = null;
  const later = () => { if (isFarmOpen()) { clearTimeout(t); t = setTimeout(renderFarm, 60); } };
  ["change", "places", "crops"].forEach((ev) => bus.addEventListener(ev, later));
  // farmers start in this view (the map is one click away); everyone else starts on the map
  if (state.profile?.user_type === "farmer") openFarm();
}

const target = () => placeById(shownPlace) ?? homePlace();

async function onClick(e) {
  const b = e.target.closest("[data-act]");
  if (!b) return;
  const p = target();
  const dates = p ? state.dates[p.site] : null;
  switch (b.dataset.act) {
    case "close": closeFarm(); break;
    case "newest": if (dates) await setState({ site: p.site, issue: dates[dates.length - 1] }); break;
    case "prev": case "next": {
      if (!dates) break;
      const k = dates.indexOf(state.issue) + (b.dataset.act === "next" ? 1 : -1);
      if (k >= 0 && k < dates.length) await setState({ site: p.site, issue: dates[k] });
      break;
    }
    case "why": case "map": {
      const c = p && placeCoords(p);
      closeFarm();
      if (c) goToSquare(p.site, c.i, { open: b.dataset.act === "why" });
      if (b.dataset.h) setState({ h: Number(b.dataset.h) });
      break;
    }
    case "skill": openSkill(); break;
    case "locate": locateMe({ offerSave: true }); closeFarm(); break;
    case "search": closeFarm(); focusSearch(); break;
    case "crops": cropsDialog(); break;
    default: break;
  }
}

function emptyHTML() {
  return `${headHTML("")}
    <div class="farm-empty">
      <h2>Will it rain at my farm this week?</h2>
      <p>Save your farm once and this page shows its 7-day forecast every time you open the app.</p>
      <div class="farm-btns">
        <button class="btn primary" data-act="locate">${ICON.locate}<span>Use my location</span></button>
        <button class="btn" data-act="search">${ICON.search}<span>Search for my farm</span></button>
      </div>
      <p class="farm-foot">Or open the map, click your area and choose <b>Save place</b>.</p>
    </div>`;
}

function headHTML(selector) {
  return `<div class="farm-head">
    <div class="eyebrow">My farm · next 7 days</div>
    ${selector}
    <button class="btn small" data-act="close" title="Show the map">${ICON.pin}<span>Map</span></button>
  </div>`;
}

/** The farmer-level explanation: the one input group that moved this day's rain chance most away from a
 *  typical forecast, from the exact SHAP explanation the map drawer shows in full (app/build_shap.py). */
function reasonHTML(why, h) {
  const t = why?.targets?.rain1?.[String(h)];
  if (!t) return "";
  const labels = Object.fromEntries(why.groups.map((g) => [g.key, g.label]));
  const [key, v] = Object.entries(t.contributions).sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]))[0] ?? [];
  if (key == null || Math.abs(v) < 2) return `<div class="dc-reason">Main reason: nothing stands out; close to a typical forecast</div>`;
  return `<div class="dc-reason" title="The input group that moved this forecast most (SHAP): ${v > 0 ? "+" : "−"}${Math.abs(v).toFixed(0)} percentage points">
    Main reason: <b>${esc(labels[key] ?? key)}</b> (${v > 0 ? "raises" : "lowers"} the chance)</div>`;
}

/** What happened on a past day (the app is a replay), or that it is still ahead. */
function outcomeHTML(d) {
  const s = d.status || {};
  if (s.period === "forecast" || d.obs_mm == null) return `<div class="obs future">Not observed yet</div>`;
  const wet = d.obs_mm >= 1;
  return `<div class="obs ${wet ? "wet" : "dry"}" title="Estimated rain for this 5 km square that day (CHIRPS${s.rain === "preliminary" ? ", preliminary" : ""})">
    What happened: <b>${wet ? `rain, ${d.obs_mm < 10 ? d.obs_mm.toFixed(1) : Math.round(d.obs_mm)} mm` : "dry"}</b>${s.rain === "preliminary" ? ' <span class="prelim">preliminary</span>' : ""}</div>`;
}

function cardHTML(d, k, hasBlight, why) {
  const good = d.confidence === "good";
  const bl = hasBlight ? BLIGHT(d.blight) : null;
  const when = k === 0 ? "Tomorrow" : d.label.split(" ")[0];
  const s = d.status || {};
  return `<li class="day-card ${good ? "good" : "rough"}${s.period === "forecast" ? " ahead" : ""}">
    <div class="dc-when"><b>${esc(when)}</b><span>${esc(d.label.split(" ").slice(1).join(" "))}</span></div>
    <div class="dc-rain">
      <div class="dc-bar" aria-hidden="true"><i style="width:${Math.max(3, Math.round(100 * (d.p1 ?? 0)))}%"></i></div>
      <div class="dc-p"><b>${pct(d.p1)}</b> chance of rain</div>
      <div class="dc-words">${esc(d.words || "")}${d.tag === "heavy" ? ' · <span class="heavy">heavy rain possible</span>' : ""}</div>
      <div class="dc-sub">Usually ${pct(d.clim_p1)} at this time of year</div>
      <div class="dc-sub" title="Average over all outcomes, dry days included">Rain amount: ${esc(mmText(d.mm))}</div>
    </div>
    <div class="dc-more">
      <span title="Average humidity of the day">Humidity <b>${d.rh == null ? "–" : Math.round(d.rh)}%</b></span>
      ${bl ? `<span class="blight ${bl[0]}" title="Chance that the weather favours late blight (Hutton criteria; potato model, also shown for tomato)">Blight weather <b>${bl[1]}</b></span>` : ""}
    </div>
    <div class="dc-trust ${good ? "good" : "rough"}">${good
      ? "Reliable: tested better than the usual"
      : "Rough guide: close to the usual"}</div>
    ${reasonHTML(why, d.h)}
    ${outcomeHTML(d)}
    <button class="link-btn dc-why" data-act="why" data-h="${d.h}">Why this forecast?</button>
  </li>`;
}

export async function renderFarm() {
  const farm = $("#farm");
  if (!state.meta || !state.profile || farm.hidden) return;
  const p = target();
  if (!p) { farm.innerHTML = emptyHTML(); return; }
  const c = placeCoords(p);
  if (!c) return;
  const my = ++ticket;
  const dates = state.dates[p.site];
  const issue = dates.includes(state.issue) ? state.issue : dates[dates.length - 1];
  let adv, why = null;
  try {
    [adv, why] = await Promise.all([getAdvice(p.site, p.pixel_id, issue),
      api(`/api/why/${p.site}/${encodeURIComponent(p.pixel_id)}/${issue}`).catch(() => null)]);
  } catch (err) {
    if (my === ticket) farm.innerHTML = `${headHTML("")}<p class="farm-empty">Could not load the forecast: ${esc(err.message)}</p>`;
    return;
  }
  if (my !== ticket) return;
  const days = adv.outlook.days, skill = adv.outlook.rain_skill_days;
  const hasBlight = state.meta.sites[p.site].has_blight;
  const st = state.meta.sites[p.site].status || {};
  const newest = issue === dates[dates.length - 1];
  const kinds = new Set(days.map((d) => d.status?.period));
  const prelim = days.some((d) => d.status?.rain === "preliminary");
  const selector = state.places.length > 1
    ? `<select id="farm-place" aria-label="Farm shown">${state.places.map((q) => `<option value="${q.id}" ${q.id === p.id ? "selected" : ""}>${esc(q.name)}</option>`).join("")}</select>`
    : "";
  const notes = [];
  const live = isLiveIssue(p.site, issue);
  if (live) {
    const L = state.meta.sites[p.site].live;
    const src = L.rain_source?.[issue];
    notes.push(`Live forecast, made from data up to ${fmtDate(issue, { day: "numeric", month: "short" })}: the newest days' weather comes from the ECMWF IFS weather model (via Open-Meteo, adjusted to match ERA5), and ${src === "ifs_rain" ? "the newest days' rain is the weather model's estimate (adjusted to match CHIRPS), because satellite rain data (CHIRPS) are not published yet" : "rain from CHIRPS" + (src === "chirps_prelim" ? " preliminary data" : "")}. These inputs differ from what the models were trained on; a test on 2024 measures the cost (see How reliable is it?).`);
    notes.push(L.attribution);
  }
  if (kinds.has("posttest") || kinds.has("forecast")) {
    notes.push(`Forecast made after the reported test period (which ended ${fmtDate(st.test_end || "2026-09-30", { day: "numeric", month: "short", year: "numeric" })}); these days are not part of the evaluated results.`);
  }
  if (kinds.has("forecast")) notes.push(`Days after ${fmtDate(st.observed_through, { day: "numeric", month: "short" })} have not been observed yet: the newest data always arrive a few days late.`);
  if (prelim) notes.push("Some observed rain is preliminary and may change when the final data are published (CHIRPS, usually in the third week of the next month).");
  farm.innerHTML = `${headHTML(selector)}
    <div class="farm-place">
      <h2>Will it rain at <span class="name"></span>?</h2>
      <div class="farm-meta">${SITE_NAMES[p.site]} · ${fmtLatLon(c.lat, c.lon, 2)} · ${c.elev.toLocaleString()} m</div>
      <p class="farm-estimate">${ICON.info}<span>These are <b>estimates for the 5 km square</b> around this point, from satellite and weather-model data. They are not measurements at your farm, and rain can differ inside the square.</span></p>
    </div>
    <div class="farm-issue">
      <button class="icon-sm" data-act="prev" aria-label="Forecast made one day earlier"><svg viewBox="0 0 24 24"><path d="M15 5l-7 7 7 7"/></svg></button>
      <div><b>Forecast made ${fmtDate(issue, { weekday: "short", day: "numeric", month: "short", year: "numeric" })}</b>
        <small>${live ? "live forecast from near-real-time inputs" : newest ? "the newest forecast the data allow" : "a replay: what the app would have said that day"}</small></div>
      <button class="icon-sm" data-act="next" aria-label="Forecast made one day later" ${newest ? "disabled" : ""}><svg viewBox="0 0 24 24"><path d="M9 5l7 7-7 7"/></svg></button>
      ${newest ? "" : '<button class="btn small" data-act="newest">Newest</button>'}
    </div>
    <p class="farm-summary">${esc(adv.outlook.summary)}</p>
    <ol class="farm-days">${days.map((d, k) => cardHTML(d, k, hasBlight, why)).join("")}</ol>
    <p class="farm-trust">${skill > 0
      ? `Rain forecasts here beat the usual for the time of year only <b>${skill} day${skill > 1 ? "s" : ""} ahead</b>; later days are a rough guide, close to what is normal for the date. Humidity and blight weather hold up longer.`
      : `In a test on 2024, live rain forecasts were <b>no better than the usual for the time of year</b>, because the newest satellite rain data arrive days late. Treat every day as a rough guide.`}
      <button class="link-btn" data-act="skill">How reliable is it?</button></p>
    ${notes.length ? `<div class="farm-notes">${notes.map((n) => `<p>${esc(n)}</p>`).join("")}</div>` : ""}
    ${adv.crops.length ? `<h3 class="farm-h3">Your crops</h3><div class="farm-crops">${adv.crops.map((a) => `
      <div class="farm-crop ${a.status}"><div class="fc-head"><b>${esc(a.name)}</b><span class="st">${STATUS[a.status]}</span>
        ${a.basis === "tested" ? '<span class="basis tested" title="Built on the late-blight model scored on unseen data">tested model</span>' : '<span class="basis rule" title="Simple rule on the tested rain and humidity forecasts; not checked against crop outcomes">guidance rule</span>'}</div>
        <p class="fc-hl">${esc(a.headline)}</p>
        <ul>${a.actions.map((x) => `<li>${esc(x)}</li>`).join("")}</ul>
        <p class="fc-cav">${a.caveats.map(esc).join(" ")}</p></div>`).join("")}</div>`
      : `<button class="link-btn farm-addcrops" data-act="crops">${ICON.leaf} Add potato, tomato or other crops for crop advice</button>`}
    <p class="farm-foot">Research prototype, not an official forecast: the Kenya Meteorological Department is the forecasting authority.</p>`;
  farm.querySelector(".farm-place .name").textContent = p.name;
}

