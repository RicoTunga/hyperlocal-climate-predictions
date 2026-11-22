// Alerts: a bell in the header with the number of unread alerts, and a list of the alerts the newest forecast
// raises for the user's saved places (late-blight risk, dry windows, heavy rain, humid spells; web/notify.py).
// The server checks the places every time the list is fetched: when the app opens, every few minutes, and
// when places or crops change. New alerts since the last check also show as a toast.
// Everything from the server or the user is escaped (esc) before it is put into HTML.
import { state, bus, api, send, esc, fmtDate, SITE_NAMES } from "./app.js";
import { dialog, closeDialog, toast, ICON } from "./user.js";
import { showFarmPlace } from "./farm.js";

const $ = (s) => document.querySelector(s);
const POLL_MS = 10 * 60 * 1000;
// (ICON comes from user.js, which imports this module: only read it inside functions, never at load time)
const KIND_ICON = {
  spray: '<svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="4"/><path d="M12 2.5v2.5M12 19v2.5M2.5 12H5M19 12h2.5M5.3 5.3l1.8 1.8M16.9 16.9l1.8 1.8M5.3 18.7l1.8-1.8M16.9 7.1l1.8-1.8"/></svg>',
  heavy: '<svg viewBox="0 0 24 24"><path d="M5 13a5 5 0 0 1 5-7 6 6 0 0 1 11 3 4 4 0 0 1-1 8H7a4 4 0 0 1-2-4zM8 19l-1 3M12 19l-1 3M16 19l-1 3"/></svg>',
  humid: '<svg viewBox="0 0 24 24"><path d="M3 9c3-2 6 2 9 0s6-2 9 0M3 14c3-2 6 2 9 0s6-2 9 0M3 19c3-2 6 2 9 0s6-2 9 0"/></svg>',
};
const KIND_HELP = {
  blight: "When the tested late-blight model shows high risk (Nakuru; needs potato or tomato in My crops).",
  spray: "Two or more dry days in a row starting within 3 days: a window for spraying, harvesting or drying.",
  heavy: "Heavy rain likely in the next 3 days.",
  humid: "Three or more days in a row with humidity at 85% or more (fungal disease weather).",
};

let data = { items: [], unread: 0, prefs: {}, kinds: {} };
let seen = null;   // ids already known in this visit (null until the first fetch)
let listOpen = false;
let busy = null;

const dates = (a) => (a.first_date === a.last_date ? fmtDate(a.first_date, { weekday: "short", day: "numeric", month: "short" })
  : `${fmtDate(a.first_date, { day: "numeric", month: "short" })} – ${fmtDate(a.last_date, { day: "numeric", month: "short" })}`);

function reliabilityText(a) {
  if (a.reliability === "good") return ["good", "Reliable this far ahead (tested)"];
  if (a.reliability === "partly") return ["partly", "Reliable for the first days; later days a rough guide"];
  return ["low", a.live ? "Rough guide: live forecast" : "Rough guide: further ahead than tested skill"];
}

function stateText(a) {
  if (a.state === "cleared") return "No longer in the newest forecast";
  if (a.ended) return "Over";
  return "";
}

function renderBell() {
  const b = $("#bell-btn");
  if (!b) return;
  const n = data.unread;
  b.querySelector(".bell-count").textContent = n > 9 ? "9+" : String(n);
  b.classList.toggle("has", n > 0);
  b.setAttribute("aria-label", n ? `Alerts: ${n} unread` : "Alerts");
  b.title = n ? `${n} unread alert${n === 1 ? "" : "s"}` : "Alerts";
}

function itemHTML(a) {
  const [rk, rtext] = reliabilityText(a), st = stateText(a);
  const quiet = a.state === "cleared" || a.ended;
  return `<li class="alert ${esc(a.level)}${a.read ? "" : " unread"}${quiet ? " quiet" : ""}" data-id="${a.id}">
    <button class="alert-go" data-act="open" data-place="${a.place_id}" title="Open this place's 7 days">
      <span class="alert-icon ${esc(a.kind)}" aria-hidden="true">${a.kind === "blight" ? ICON.leaf : KIND_ICON[a.kind] ?? ICON.info}</span>
      <span class="alert-text">
        <span class="alert-top"><b>${esc(a.title)}</b><span class="alert-when">${esc(dates(a))}</span></span>
        <span class="alert-place">${ICON.pin}${esc(a.place_name)} · ${esc(SITE_NAMES[a.site] ?? a.site)}</span>
        <span class="alert-body">${esc(a.body)}</span>
        <span class="alert-tags"><span class="rel ${rk}">${esc(rtext)}</span>${st ? `<span class="rel state">${esc(st)}</span>` : ""}</span>
      </span>
      ${a.read ? "" : '<span class="alert-dot" aria-label="Unread"></span>'}
    </button></li>`;
}

function listHTML() {
  const on = Object.values(data.prefs).some(Boolean);
  let body;
  if (!state.places.length) {
    body = `<p class="menu-empty">Save your farm first (click its square on the map and choose <b>Save place</b>). Alerts then
      come from the newest forecast for every place you save.</p>`;
  } else if (!on) {
    body = '<p class="menu-empty">All alert types are switched off. Turn some on under <b>Alert settings</b>.</p>';
  } else if (!data.items.length) {
    body = `<p class="menu-empty">No alerts for your places in the newest forecast. You will see one here when late-blight
      risk is high, a dry window opens, heavy rain is likely or a humid spell is coming.</p>`;
  } else {
    body = `<ul class="alert-list">${data.items.map(itemHTML).join("")}</ul>`;
  }
  return `<div class="alerts-bar">
      <button class="btn small" data-act="all-read" ${data.unread ? "" : "disabled"}>Mark all read</button>
      <button class="btn small ghost" data-act="settings">Alert settings</button></div>
    ${body}
    <p class="alerts-foot">Alerts are guidance rules on the forecasts, not checked against farm outcomes. Rain-based
      alerts are a heads-up, not a warning; the Kenya Meteorological Department is the authority.</p>`;
}

async function markRead(ids) {
  try {
    await send("/api/notifications/read", ids ? { ids } : { all: true });
    data.items.forEach((a) => { if (!ids || ids.includes(a.id)) a.read = true; });
    data.unread = data.items.filter((a) => !a.read && a.state === "active" && !a.ended).length;
    renderBell();
  } catch (err) { toast(err.message, { kind: "error" }); }
}

export function openAlerts() {
  const body = dialog({ title: "Alerts", wide: true, html: listHTML(), onClose: () => { listOpen = false; } });
  listOpen = true;  // after dialog(): opening it runs the previous dialog's onClose
  body.addEventListener("click", async (e) => {
    const b = e.target.closest("[data-act]");
    if (!b) return;
    if (b.dataset.act === "all-read") { await markRead(null); openAlerts(); }
    else if (b.dataset.act === "settings") settingsDialog();
    else if (b.dataset.act === "open") {
      const id = Number(b.closest(".alert").dataset.id);
      markRead([id]);
      closeDialog();
      showFarmPlace(Number(b.dataset.place));
    }
  });
}

function settingsDialog() {
  const body = dialog({ title: "Alert settings", html: `
    <p class="summary">Choose which alerts you get for your saved places.</p>
    ${Object.entries(data.kinds).map(([k, name]) => `
      <label class="switch-row"><span>${esc(name)}<small>${esc(KIND_HELP[k] ?? "")}</small></span>
        <input type="checkbox" role="switch" data-kind="${esc(k)}" ${data.prefs[k] ? "checked" : ""}></label>`).join("")}
    <div class="dialog-actions"><button class="btn primary" data-act="done">Done</button></div>`,
  onClose: () => { listOpen = false; } });
  body.addEventListener("change", async (e) => {
    const k = e.target.dataset.kind;
    if (!k) return;
    try {
      data.prefs = (await send("/api/notifications/prefs", { [k]: e.target.checked })).prefs;
      await refresh({ quiet: true });
    } catch (err) { toast(err.message, { kind: "error" }); e.target.checked = !e.target.checked; }
  });
  body.querySelector("[data-act=done]").addEventListener("click", () => openAlerts());
}

/** Fetch the alerts (the server refreshes them from the newest forecast first). */
export function refresh({ quiet = false } = {}) {
  if (busy) return busy;
  busy = (async () => {
    try {
      data = await api("/api/notifications");
    } catch { return; } finally { busy = null; }
    renderBell();
    const fresh = data.items.filter((a) => !a.read && a.state === "active" && !a.ended && !(seen && seen.has(a.id)));
    const first = seen === null;
    seen = new Set(data.items.map((a) => a.id));
    if (!quiet && fresh.length && !listOpen) {
      const one = fresh[0];
      const text = fresh.length === 1 ? `${one.title} at ${one.place_name} (${dates(one)})`
        : first ? `${fresh.length} unread alerts for your places` : `${fresh.length} new alerts for your places`;
      toast(text, { timeout: 9000, action: { label: "Show", run: openAlerts } });
    }
    if (listOpen && $("#dialog-title").textContent === "Alerts") openAlerts();
  })();
  return busy;
}

export function initNotify() {
  const b = $("#bell-btn");
  if (!b) return;
  b.innerHTML = `${ICON.bell}<span class="bell-count" aria-hidden="true"></span>`;
  b.addEventListener("click", openAlerts);
  let t = null;
  const soon = () => { clearTimeout(t); t = setTimeout(() => refresh({ quiet: false }), 400); };
  ["places", "crops"].forEach((ev) => bus.addEventListener(ev, soon));
  setInterval(() => { if (!document.hidden) refresh(); }, POLL_MS);
  document.addEventListener("visibilitychange", () => { if (!document.hidden) refresh(); });
  refresh();
}
