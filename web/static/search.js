// Find a place: coordinates in any common format, a town from a small built-in list, a saved place, or the
// device's location ("use my location", which the browser asks permission for; it works on localhost or HTTPS).
// Nothing typed here is sent to any outside service: towns are a fixed list below, coordinates are parsed here.
import { state, nearestSquare, goToSquare, fmtLatLon, esc, SITE_NAMES } from "./app.js";
import { toast, savePlaceDialog, placeAt, placeCoords, ICON } from "./user.js";
import { parseCoords } from "./coords.js";

export { parseCoords };

const $ = (s) => document.querySelector(s);

// Approximate town and landmark centres (rounded to ~1 km) inside the two study areas, for quick navigation only.
export const TOWNS = [
  ["Nakuru town", -0.303, 36.080], ["Naivasha", -0.717, 36.431], ["Gilgil", -0.493, 36.318], ["Njoro", -0.330, 35.944],
  ["Egerton University", -0.370, 35.935], ["Molo", -0.248, 35.733], ["Elburgon", -0.303, 35.811], ["Mau Narok", -0.611, 35.952],
  ["Subukia", 0.002, 36.230], ["Bahati", -0.150, 36.150], ["Rongai", -0.173, 35.864], ["Lanet", -0.300, 36.150],
  ["Mai Mahiu", -1.010, 36.585], ["Olenguruone", -0.583, 35.683], ["Kuresoi", -0.290, 35.570], ["Dundori", -0.250, 36.220],
  ["Busia town", 0.461, 34.111], ["Nambale", 0.456, 34.256], ["Butula", 0.338, 34.336], ["Funyula", 0.290, 34.090],
  ["Port Victoria", 0.100, 33.975], ["Malaba", 0.633, 34.283], ["Bumala", 0.320, 34.200], ["Sio Port", 0.230, 34.020],
];

/** Fly to the square containing a point; explain clearly when the point is outside the forecast areas. */
export async function goToPoint(lat, lng, label, { offerSave = false } = {}) {
  const r = nearestSquare(lat, lng);
  if (!r) return;
  if (r.km > 8) {
    toast(`${label} is about ${Math.round(r.km)} km from the nearest forecast square. The app covers Nakuru County and a Busia study area.`,
      { kind: "error", timeout: 9000 });
    return false;
  }
  showYouAreHere(lat, lng, label);
  await goToSquare(r.site, r.i);
  if (r.km > 0) toast(`${label} is just outside the forecast area; showing the nearest square (${r.km.toFixed(1)} km away).`, { timeout: 7000 });
  const saved = placeAt(r.site, state.pixels[r.site].ids[r.i]);
  if (offerSave && !saved) {
    toast(`${label}: ${SITE_NAMES[r.site]}, square at ${fmtLatLon(state.pixels[r.site].lat[r.i], state.pixels[r.site].lon[r.i], 2)}.`,
      { action: { label: "Save this place", run: () => savePlaceDialog(r.site, r.i) }, timeout: 12000 });
  }
  return true;
}

let hereMarker = null;
function showYouAreHere(lat, lng, label) {
  hereMarker?.remove();
  const el = document.createElement("div");
  el.className = "here";
  el.title = label;
  hereMarker = new maplibregl.Marker({ element: el }).setLngLat([lng, lat]).addTo(state.map);
}

export function locateMe({ offerSave = false } = {}) {
  const btn = $("#locate-btn");
  if (!("geolocation" in navigator)) { toast("This browser cannot share its location. Search by coordinates instead.", { kind: "error" }); return; }
  if (!window.isSecureContext) {
    toast("Your browser only shares location over a secure (HTTPS) connection. Search by coordinates instead.", { kind: "error", timeout: 9000 });
    return;
  }
  btn.classList.add("busy");
  navigator.geolocation.getCurrentPosition(
    (pos) => {
      btn.classList.remove("busy");
      const { latitude, longitude, accuracy } = pos.coords;
      goToPoint(latitude, longitude, "Your location", { offerSave }).then((ok) => {
        if (ok && accuracy > 2000) toast(`Your device's location is approximate (within about ${Math.round(accuracy / 1000)} km).`, { timeout: 7000 });
      });
    },
    (err) => {
      btn.classList.remove("busy");
      const msg = err.code === 1 ? "Location permission was refused. You can allow it in the browser's site settings, or search instead."
        : err.code === 3 ? "Finding your location took too long. Try again outdoors, or search instead."
          : "Your location is not available right now. Search by coordinates instead.";
      toast(msg, { kind: "error", timeout: 9000 });
    },
    { enableHighAccuracy: true, timeout: 15000, maximumAge: 120000 },
  );
}

export function focusSearch() {
  const input = $("#search-input");
  input.focus();
  input.select();
  update();
}

// ---------------------------------------------------------------- results list
let results = [], active = 0;

function candidates(q) {
  const out = [], t = q.trim().toLowerCase();
  const c = parseCoords(q);
  if (c) out.push({ icon: ICON.pin, label: fmtLatLon(c.lat, c.lng, 4), sub: "Go to these coordinates", run: () => goToPoint(c.lat, c.lng, "That point", { offerSave: true }) });
  for (const p of state.places) {
    if (t && !p.name.toLowerCase().includes(t)) continue;
    const pc = placeCoords(p);
    if (pc) out.push({ icon: ICON.star, label: p.name, sub: `Saved place · ${SITE_NAMES[p.site]}`, safe: true, run: () => goToSquare(p.site, pc.i) });
  }
  if (t && !c) {
    for (const [name, lat, lng] of TOWNS) {
      if (!name.toLowerCase().includes(t)) continue;
      const r = nearestSquare(lat, lng);
      if (r && r.km <= 8) out.push({ icon: ICON.pin, label: name, sub: `${SITE_NAMES[r.site]} · approximate town centre`, run: () => goToPoint(lat, lng, name, { offerSave: true }) });
    }
  }
  if (!t || "my location".includes(t) || "use my location".includes(t)) {
    out.push({ icon: ICON.locate, label: "Use my location", sub: "Your browser will ask for permission", run: () => locateMe({ offerSave: true }) });
  }
  if (t && !out.length) out.push({ icon: ICON.info, label: "Nothing found", sub: "Try coordinates like -0.33, 35.94 or a town name", run: null });
  return out.slice(0, 8);
}

function update() {
  const q = $("#search-input").value;
  results = candidates(q);
  active = 0;
  render();
}

function render() {
  const ul = $("#search-results"), input = $("#search-input");
  if (!results.length || document.activeElement !== input) { ul.hidden = true; input.setAttribute("aria-expanded", "false"); return; }
  ul.innerHTML = results.map((r, k) => `<li role="option" id="sr-${k}" aria-selected="${k === active}" data-k="${k}" class="${r.run ? "" : "muted"}">
    <span class="sr-icon">${r.icon}</span><span class="sr-text"><b></b><small>${esc(r.sub)}</small></span></li>`).join("");
  ul.querySelectorAll("li").forEach((li, k) => { li.querySelector("b").textContent = results[k].label; });
  ul.hidden = false;
  input.setAttribute("aria-expanded", "true");
  input.setAttribute("aria-activedescendant", `sr-${active}`);
}

function choose(k) {
  const r = results[k];
  if (!r?.run) return;
  $("#search-input").blur();
  $("#search-results").hidden = true;
  r.run();
}

export function initSearch() {
  const input = $("#search-input"), ul = $("#search-results");
  input.addEventListener("input", update);
  input.addEventListener("focus", update);
  input.addEventListener("blur", () => setTimeout(() => { ul.hidden = true; input.setAttribute("aria-expanded", "false"); }, 150));
  input.addEventListener("keydown", (e) => {
    if (e.key === "ArrowDown") { active = Math.min(results.length - 1, active + 1); render(); e.preventDefault(); }
    else if (e.key === "ArrowUp") { active = Math.max(0, active - 1); render(); e.preventDefault(); }
    else if (e.key === "Enter") { choose(active); e.preventDefault(); }
    else if (e.key === "Escape") { input.blur(); }
  });
  ul.addEventListener("mousedown", (e) => {  // mousedown fires before the input's blur
    const li = e.target.closest("li");
    if (li) { e.preventDefault(); choose(Number(li.dataset.k)); }
  });
  $("#locate-btn").addEventListener("click", () => locateMe({ offerSave: true }));
}
