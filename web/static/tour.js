// Guided tour: a spotlight moves over the parts of the app with a short explanation for each.
// Shown automatically when "Tutorial when I sign in" is on (the default for new accounts), once per visit;
// always available from the menu. Steps whose element is hidden (e.g. on a phone) are skipped.
import { state, send } from "./app.js";
import { closeMenu } from "./user.js";

const $ = (s) => document.querySelector(s);

function steps() {
  const t = state.profile.user_type, farmer = t === "farmer" || t === "extension";
  const first = (state.user.display_name || "").split(" ")[0];
  return [
    { title: `Welcome${first ? `, ${first}` : ""}!`, text: "A one-minute tour of the forecast map. Use the arrow keys or the buttons; you can leave at any time." },
    { el: "#map", area: "map", title: "The map", text: "Colours show the forecast for every ~5 km square. Hover over a square to see its coordinates, the square's edges and the values; click it for a 7-day detail." },
    { el: "#rail", mobileEl: "#layers-btn", title: "Choose what to see", text: "Chance of rain, heavy rain, expected amount, humidity and (in Nakuru) late-blight risk for potato and tomato. Switch between the forecast, what was observed, the error and the normal for the date." },
    { el: "#timeline", title: "Move through time", text: "The newest issue date is the latest forecast. Go back to see what the app forecast on any day since January 2025, next to what happened. Pick the issue date, then 1 to 7 days ahead. Space plays, ← → change the day." },
    { el: "#search", title: farmer ? "Find your farm" : "Find a place", text: "Type coordinates (like -0.33, 35.94, or 0°19′S 35°56′E), a town such as Njoro or Busia, or tap the target button to use your location." },
    { el: "#hud", title: "At a glance", text: farmer ? "Your saved farm's next days: rain chance, humidity, a one-line summary of the week and alerts for your crops. Faded bars are less certain."
      : "A saved place's next days in one panel: rain chance, humidity and a one-line summary of the week. Faded bars are less certain." },
    { el: "#menu-btn", title: "Menu", text: farmer ? "Your saved places and crops, how accurate the forecasts are, and settings, including turning this tutorial on or off."
      : "Saved places, “How good is it?” (accuracy with confidence intervals), “Why these forecasts?” (SHAP), settings and your account." },
    { title: "You are set", text: "Remember: this is a research prototype. For official forecasts and warnings, follow the Kenya Meteorological Department.", last: true },
  ];
}

let k = 0, list = [], after = null, keyHandler = null;

// (offsetParent is null for position: fixed elements, so check the box and the computed style instead)
const visible = (el) => {
  if (!el || el.hidden) return false;
  const r = el.getBoundingClientRect(), cs = getComputedStyle(el);
  return r.width > 0 && r.height > 0 && cs.display !== "none" && cs.visibility !== "hidden";
};
function target(s) {
  const mobile = matchMedia("(max-width: 760px)").matches;
  const el = $(mobile && s.mobileEl ? s.mobileEl : s.el);
  return visible(el) ? el : null;
}

/** The open part of the map between the panels (for the map step). */
function mapArea() {
  const rect = (sel) => { const e = $(sel); return visible(e) ? e.getBoundingClientRect() : null; };
  const brand = rect("#brand"), hud = rect("#hud"), rail = rect("#rail"), tl = rect("#timeline");
  if (matchMedia("(max-width: 760px)").matches || !brand) {
    const top = (brand?.bottom ?? 120) + 16, bottom = (tl?.top ?? innerHeight - 140) - 16;
    return { left: 16, top, width: innerWidth - 32, height: Math.max(100, bottom - top) };
  }
  const left = Math.max(brand.right, hud?.right ?? 0) + 24, right = (rail?.left ?? innerWidth) - 24;
  const top = 40, bottom = (tl?.top ?? innerHeight - 100) - 24;
  return { left, top, width: Math.max(160, right - left), height: Math.max(120, bottom - top) };
}

function place() {
  const s = list[k], spot = $("#tour-spot"), card = $("#tour-card");
  const el = s.el ? target(s) : null;
  card.innerHTML = `
    <div class="tour-progress" aria-hidden="true">${list.map((_, j) => `<span class="${j === k ? "on" : j < k ? "done" : ""}"></span>`).join("")}</div>
    <h2 id="tour-title"></h2><p id="tour-text"></p>
    ${s.last ? `<label class="tour-check"><input type="checkbox" id="tour-again" ${state.profile.tour_on_login ? "" : ""}> Show this tour each time I sign in</label>` : ""}
    <div class="tour-actions">
      ${s.last ? "" : '<button class="link-btn" data-t="skip">Skip tour</button>'}
      <span class="grow"></span>
      ${k > 0 ? '<button class="btn small" data-t="back">Back</button>' : ""}
      <button class="btn small primary" data-t="next">${s.last ? "Start exploring" : k === 0 ? "Show me" : "Next"}</button>
    </div>`;
  card.querySelector("#tour-title").textContent = s.title;
  card.querySelector("#tour-text").textContent = s.text;
  if (el) {
    const box = s.area === "map" ? mapArea() : (() => {
      const r = el.getBoundingClientRect(), pad = 6;
      return { left: r.left - pad, top: r.top - pad, width: r.width + 2 * pad, height: r.height + 2 * pad };
    })();
    Object.assign(spot.style, { left: `${box.left}px`, top: `${box.top}px`, width: `${box.width}px`, height: `${box.height}px`, opacity: 1 });
    spot.classList.remove("center");
    // the card goes on the side with the most room
    const cw = Math.min(340, innerWidth - 32), ch = card.offsetHeight || 190;
    const room = { right: innerWidth - (box.left + box.width), left: box.left, below: innerHeight - (box.top + box.height), above: box.top };
    let x, y;
    if (room.below >= ch + 20) { x = box.left + box.width / 2 - cw / 2; y = box.top + box.height + 14; }
    else if (room.above >= ch + 20) { x = box.left + box.width / 2 - cw / 2; y = box.top - ch - 14; }
    else if (room.right >= cw + 20) { x = box.left + box.width + 14; y = box.top + box.height / 2 - ch / 2; }
    else if (room.left >= cw + 20) { x = box.left - cw - 14; y = box.top + box.height / 2 - ch / 2; }
    else { x = innerWidth / 2 - cw / 2; y = innerHeight / 2 - ch / 2; }
    Object.assign(card.style, { left: `${Math.max(16, Math.min(innerWidth - cw - 16, x))}px`, top: `${Math.max(16, Math.min(innerHeight - ch - 16, y))}px`, width: `${cw}px` });
    card.classList.remove("center");
  } else {
    Object.assign(spot.style, { left: `${innerWidth / 2}px`, top: `${innerHeight / 2}px`, width: "0px", height: "0px" });
    spot.classList.add("center");
    card.classList.add("center");
    Object.assign(card.style, { left: "", top: "", width: "" });
  }
  card.querySelector("[data-t=next]").focus();
}

function go(delta) {
  let j = k + delta;
  while (j > 0 && j < list.length - 1 && list[j].el && !target(list[j])) j += delta;  // skip hidden parts
  if (j < 0) return;
  if (j >= list.length) { finish(); return; }
  k = j;
  place();
}

async function finish(skipped = false) {
  const again = !!$("#tour-again")?.checked;
  $("#tour").hidden = true;
  document.removeEventListener("keydown", keyHandler, true);
  removeEventListener("resize", place);
  try { sessionStorage.setItem("hl-tour-shown", "1"); } catch { /* fine */ }
  try {
    state.profile = await send("/api/profile", { tour_done: true, tour_on_login: skipped ? false : again });
    const box = document.querySelector('[data-setting="tour_on_login"]');
    if (box) box.checked = state.profile.tour_on_login;
  } catch { /* the tour still closes */ }
  const f = after; after = null;
  f?.();
}

export function startTour({ after: then = null } = {}) {
  closeMenu();
  list = steps(); k = 0; after = then;
  try { sessionStorage.setItem("hl-tour-shown", "1"); } catch { /* fine */ }
  $("#tour").hidden = false;
  const card = $("#tour-card");
  card.onclick = (e) => {
    const b = e.target.closest("[data-t]");
    if (!b) return;
    if (b.dataset.t === "next") go(1);
    else if (b.dataset.t === "back") go(-1);
    else finish(true);
  };
  keyHandler = (e) => {
    if ($("#tour").hidden) return;
    if (e.key === "Escape") { e.preventDefault(); e.stopPropagation(); finish(true); }
    else if (e.key === "ArrowRight" && !e.target.matches("input")) { e.preventDefault(); e.stopPropagation(); go(1); }
    else if (e.key === "ArrowLeft") { e.preventDefault(); e.stopPropagation(); go(-1); }
  };
  document.addEventListener("keydown", keyHandler, true);
  addEventListener("resize", place);
  place();
}
