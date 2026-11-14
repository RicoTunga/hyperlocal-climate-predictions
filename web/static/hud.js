// "At a glance": the forecast for the user's home place (or another saved place) on the replay date,
// in one compact panel: next day's numbers, a 7-day rain strip, the week in one line and crop alerts.
import { state, bus, esc, fmtDate, loadDay, setState, goToSquare, send, SITE_NAMES, fmtLatLon } from "./app.js";
import { homePlace, placeById, placeCoords, getAdvice, cropsDialog, ICON } from "./user.js";
import { locateMe, focusSearch } from "./search.js";

const $ = (s) => document.querySelector(s);
let shownPlace = null;  // id of the place shown; null = home place
let ticket = 0;         // ignores slow responses that arrive after a newer render started
const STATUS = { act: "Act now", watch: "Watch", ok: "OK" };

const pct = (p) => (p == null ? "–" : `${Math.round(100 * p)}%`);

export function initHud() {
  const hud = $("#hud");
  hud.hidden = false;
  // phones start collapsed to one line, so the map stays visible; larger screens follow the saved choice
  hud.classList.toggle("collapsed", matchMedia("(max-width: 760px)").matches || !state.profile.hud_open);
  let t = null;
  const later = () => { clearTimeout(t); t = setTimeout(renderHud, 60); };
  bus.addEventListener("change", later);
  bus.addEventListener("places", later);
  bus.addEventListener("crops", later);
  hud.addEventListener("click", onClick);
  hud.addEventListener("change", (e) => {
    if (e.target.id === "hud-place") { shownPlace = Number(e.target.value); renderHud(); }
  });
  renderHud();
}

function target() {
  return placeById(shownPlace) ?? homePlace();
}

async function onClick(e) {
  const b = e.target.closest("[data-act]");
  if (!b) return;
  const p = target();
  switch (b.dataset.act) {
    case "toggle": {
      const open = $("#hud").classList.toggle("collapsed") === false;
      $("#hud").querySelector(".hud-toggle")?.setAttribute("aria-expanded", String(open));
      if (!matchMedia("(max-width: 760px)").matches) {  // remember the choice on larger screens only
        send("/api/profile", { hud_open: open }).then((prof) => { state.profile = prof; }).catch(() => {});
      }
      break;
    }
    case "day": {
      const h = Number(b.dataset.h);
      if (p && p.site !== state.site) await setState({ site: p.site, h });
      else setState({ h });
      break;
    }
    case "open": { const c = p && placeCoords(p); if (c) goToSquare(p.site, c.i); break; }
    case "locate": locateMe({ offerSave: true }); break;
    case "search": focusSearch(); break;
    case "crops": cropsDialog(); break;
    default: break;
  }
}

function emptyHTML() {
  return `${headHTML("At a glance", "")}
    <div class="hud-body">
      <p class="hud-empty">Save your farm or a place you care about to see its forecast here every time you open the app.</p>
      <div class="hud-btns">
        <button class="btn small primary" data-act="locate">${ICON.locate}<span>Use my location</span></button>
        <button class="btn small" data-act="search">${ICON.search}<span>Search</span></button>
      </div>
      <p class="hud-foot">Or click any square on the map and choose <b>Save place</b>.</p>
    </div>`;
}

function headHTML(title, extra, mini = "") {
  const open = !$("#hud").classList.contains("collapsed");
  return `<div class="hud-head">
    <div class="eyebrow">At a glance</div>
    ${mini ? `<button class="hud-mini" data-act="toggle" title="Expand">${mini}</button>` : ""}
    ${extra}
    <button class="icon-sm hud-toggle" data-act="toggle" aria-expanded="${open}" aria-label="${open ? "Collapse" : "Expand"} at a glance">
      <svg viewBox="0 0 24 24"><path d="M6 15l6-6 6 6"/></svg></button>
  </div>`;
}

export async function renderHud() {
  const hud = $("#hud");
  if (!state.meta || !state.profile) return;
  const p = target();
  if (!p) { hud.innerHTML = emptyHTML(); hud.dataset.state = "empty"; return; }
  const c = placeCoords(p);
  if (!c) return;
  const my = ++ticket;
  const issue = state.dates[p.site].includes(state.issue) ? state.issue : state.dates[p.site][0];
  let day, adv;
  try {
    [day, adv] = await Promise.all([loadDay(p.site, issue), getAdvice(p.site, p.pixel_id, issue)]);
  } catch (err) {
    if (my === ticket) hud.innerHTML = `${headHTML("At a glance", "")}<div class="hud-body"><p class="hud-empty">Could not load: ${esc(err.message)}</p></div>`;
    return;
  }
  if (my !== ticket) return;
  const i = c.i;
  const H1 = day.horizons["1"];
  const g = (H, k) => (H[k] ? H[k][i] : null);
  const days = adv.outlook.days;
  const skill = adv.outlook.rain_skill_days;
  const selector = state.places.length > 1
    ? `<select id="hud-place" aria-label="Place shown">${state.places.map((q) => `<option value="${q.id}" ${q.id === p.id ? "selected" : ""}>${esc(q.name)}</option>`).join("")}</select>`
    : "";
  const worst = adv.crops.reduce((w, a) => (a.status === "act" ? "act" : w === "act" ? w : a.status === "watch" ? "watch" : w), "ok");
  const h1date = fmtDate(H1.target, { weekday: "short", day: "numeric" });
  const mm = (v) => (v == null ? "–" : v < 0.05 ? "0" : v.toFixed(1));
  hud.dataset.state = worst;
  const worstCrop = adv.crops.find((a) => a.status === worst && worst !== "ok");
  const mini = `<b>${esc(p.name)}</b><span class="rain">${pct(g(H1, "p_rain1"))} rain</span>${worstCrop
    ? `<span class="st ${worst}">${esc(worstCrop.name)}: ${STATUS[worst]}</span>` : ""}`;
  hud.innerHTML = `${headHTML("At a glance", selector, mini)}
    <button class="hud-place" data-act="open" title="Open this square">
      <span class="hud-place-name">${p.id === state.profile.home_place ? ICON.star : ICON.pin}<b></b></span>
      <span class="hud-place-meta">${SITE_NAMES[p.site]} · ${fmtLatLon(c.lat, c.lon, 2)} · ${c.elev.toLocaleString()} m</span>
    </button>
    <div class="hud-body">
      <div class="hud-now" title="Forecast for ${h1date}, issued ${fmtDate(issue, { day: "numeric", month: "short" })}">
        <div><span class="k">${h1date}</span><span class="v rain">${pct(g(H1, "p_rain1"))}</span><span class="u">rain · normal ${pct(g(H1, "clim_rain1"))}</span></div>
        <div><span class="k">Expected</span><span class="v">${mm(g(H1, "rain_mm"))}<small> mm</small></span><span class="u">normal ${mm(g(H1, "clim_rain_mm"))} mm</span></div>
        <div><span class="k">Humidity</span><span class="v humid">${g(H1, "rh") == null ? "–" : Math.round(g(H1, "rh"))}<small>%</small></span><span class="u">normal ${g(H1, "clim_rh") == null ? "–" : Math.round(g(H1, "clim_rh"))}%</span></div>
      </div>
      <div class="hud-strip" role="list" aria-label="Chance of rain, next 7 days">
        ${days.map((d) => `<button class="hud-day ${d.confidence}${d.h === state.h && p.site === state.site ? " on" : ""}" role="listitem" data-act="day" data-h="${d.h}"
            title="${esc(d.label)}: ${pct(d.p1)} chance of rain${d.tag === "heavy" ? ", heavy rain possible" : ""}${d.confidence === "low" ? " (less reliable this far ahead)" : ""}">
          <span class="bar"><i style="height:${Math.max(4, Math.round(100 * (d.p1 ?? 0)))}%"></i>${d.tag === "heavy" ? '<em class="heavy" aria-hidden="true"></em>' : ""}</span>
          <span class="p">${pct(d.p1)}</span><span class="d">${esc(d.label.split(" ")[0])}</span></button>`).join("")}
      </div>
      <p class="hud-summary">${esc(adv.outlook.summary)}</p>
      ${adv.crops.length ? `<div class="hud-crops">${adv.crops.map((a) => `
        <button class="hud-crop ${a.status}" data-act="open" title="${esc(a.headline)}">
          <span class="dot" aria-hidden="true"></span><b>${esc(a.name)}</b><span class="st">${STATUS[a.status]}</span>
          <span class="hl">${esc(a.headline)}</span></button>`).join("")}</div>`
        : `<button class="link-btn hud-addcrops" data-act="crops">${ICON.leaf} Add your crops for crop advice</button>`}
      <p class="hud-foot">Replay · issued ${fmtDate(issue, { day: "numeric", month: "short", year: "numeric" })} ·
        rain reliable to day ${skill}; faded bars are less certain</p>
    </div>`;
  hud.querySelector(".hud-place-name b").textContent = p.name;
}
