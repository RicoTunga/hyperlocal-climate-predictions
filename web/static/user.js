// The signed-in user's side of the app: menu, saved places (and their map pins), crops, account, settings,
// first-run onboarding, plus small shared UI helpers (toasts, dialogs) and the crop-advice cache.
// Everything the user typed is shown with textContent or esc(), never as raw HTML.
import { state, bus, emit, send, api, esc, signOut, goToSquare, fmtLatLon, plainLatLon, SITE_NAMES, setState } from "./app.js";
import { openSkill, openAbout, openUsers, openWhy, refreshDrawer } from "./panels.js";
import { initHud } from "./hud.js";
import { initSearch, locateMe, focusSearch } from "./search.js";
import { initFx, applyFxSetting } from "./fx.js";
import { startTour } from "./tour.js";

const $ = (s) => document.querySelector(s);
const ICON = {
  pin: '<svg viewBox="0 0 24 24"><path d="M12 21s-6.5-5.6-6.5-11a6.5 6.5 0 0 1 13 0c0 5.4-6.5 11-6.5 11z"/><circle cx="12" cy="10" r="2.3"/></svg>',
  star: '<svg viewBox="0 0 24 24"><path d="M12 3.5l2.6 5.4 5.9.8-4.3 4.1 1 5.8L12 16.8l-5.2 2.8 1-5.8L3.5 9.7l5.9-.8z"/></svg>',
  edit: '<svg viewBox="0 0 24 24"><path d="M4 20h4L19 9l-4-4L4 16v4zM13.5 6.5l4 4"/></svg>',
  trash: '<svg viewBox="0 0 24 24"><path d="M5 7h14M10 7V4.5h4V7M7 7l1 13h8l1-13"/></svg>',
  leaf: '<svg viewBox="0 0 24 24"><path d="M5 19c0-8 6-14 15-14 0 9-6 15-14 15M5 19l8-8"/></svg>',
  chart: '<svg viewBox="0 0 24 24"><path d="M4 20V4M4 20h16M8 16l4-5 3 3 5-7"/></svg>',
  bulb: '<svg viewBox="0 0 24 24"><path d="M9 18h6M10 21h4M12 3a6 6 0 0 0-3.5 10.9c.6.5 1 1.2 1 2.1h5c0-.9.4-1.6 1-2.1A6 6 0 0 0 12 3z"/></svg>',
  info: '<svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="9"/><path d="M12 11v6M12 7.5v.5"/></svg>',
  play: '<svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="9"/><path d="M10 8.5l5.5 3.5-5.5 3.5z"/></svg>',
  user: '<svg viewBox="0 0 24 24"><circle cx="12" cy="8" r="4"/><path d="M4 21c1.5-4 4.5-6 8-6s6.5 2 8 6"/></svg>',
  users: '<svg viewBox="0 0 24 24"><circle cx="9" cy="8" r="3.5"/><path d="M2.5 20c1-3.5 3.5-5.5 6.5-5.5s5.5 2 6.5 5.5M16 4.8a3.5 3.5 0 0 1 0 6.4M18 14.8c1.8.8 3 2.5 3.5 5.2"/></svg>',
  out: '<svg viewBox="0 0 24 24"><path d="M14 4h5v16h-5M10 8l-4 4 4 4M6 12h10"/></svg>',
  plus: '<svg viewBox="0 0 24 24"><path d="M12 5v14M5 12h14"/></svg>',
  locate: '<svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="3.2"/><path d="M12 2.5v3M12 18.5v3M2.5 12h3M18.5 12h3"/><circle cx="12" cy="12" r="7"/></svg>',
  search: '<svg viewBox="0 0 24 24"><circle cx="11" cy="11" r="6.5"/><path d="M16 16l4.5 4.5"/></svg>',
  close: '<svg viewBox="0 0 24 24"><path d="M6 6l12 12M18 6L6 18"/></svg>',
  mail: '<svg viewBox="0 0 24 24"><rect x="3" y="5.5" width="18" height="13" rx="2"/><path d="M3.5 7l8.5 6 8.5-6"/></svg>',
};
export { ICON };
export const USER_TYPE_NAMES = { farmer: "Farmer", extension: "Extension officer", researcher: "Researcher", other: "Other" };

// ---------------------------------------------------------------- lookups
export const pixelIndex = (site, pixelId) => state.pixels[site]?.byId.get(pixelId) ?? null;
export const placeAt = (site, pixelId) => state.places.find((p) => p.site === site && p.pixel_id === pixelId) ?? null;
export const placeById = (id) => state.places.find((p) => p.id === id) ?? null;
export const homePlace = () => placeById(state.profile?.home_place) ?? state.places[0] ?? null;
export const cropName = (id) => state.catalogue.find((c) => c.id === id)?.name ?? id;
export function placeCoords(p) {
  const i = pixelIndex(p.site, p.pixel_id), P = state.pixels[p.site];
  return i == null ? null : { i, lat: P.lat[i], lon: P.lon[i], elev: P.elev[i] };
}

// crop advice for a square and issue day, cached (the same request serves the HUD and the drawer)
const adviceCache = new Map();
export function getAdvice(site, pixelId, issue, crops = state.crops) {
  const key = `${site}/${pixelId}/${issue}/${crops.join(",")}`;
  if (!adviceCache.has(key)) {
    adviceCache.set(key, api(`/api/advice/${site}/${encodeURIComponent(pixelId)}/${issue}?crops=${crops.join(",")}`)
      .catch((e) => { adviceCache.delete(key); throw e; }));
    if (adviceCache.size > 60) adviceCache.delete(adviceCache.keys().next().value);
  }
  return adviceCache.get(key);
}

// ---------------------------------------------------------------- toasts
export function toast(message, { kind = "info", action = null, timeout = 5000 } = {}) {
  const el = document.createElement("div");
  el.className = `toast ${kind}`;
  el.setAttribute("role", kind === "error" ? "alert" : "status");
  const text = document.createElement("span");
  text.textContent = message;
  el.append(text);
  const close = () => { el.classList.add("leaving"); setTimeout(() => el.remove(), 250); };
  if (action) {
    const b = document.createElement("button");
    b.textContent = action.label;
    b.addEventListener("click", () => { close(); action.run(); });
    el.append(b);
  }
  const x = document.createElement("button");
  x.className = "toast-x"; x.setAttribute("aria-label", "Dismiss"); x.innerHTML = ICON.close;
  x.addEventListener("click", close);
  el.append(x);
  $("#toasts").append(el);
  if (timeout) setTimeout(close, timeout);
  return close;
}

// ---------------------------------------------------------------- dialogs
let dialogClose = null;
export function dialog({ title, html, wide = false, onClose = null, dismissable = true }) {
  const m = $("#dialog");
  if (dialogClose) { const f = dialogClose; dialogClose = null; f(); }
  $("#dialog-title").textContent = title;
  $("#dialog-card").className = `modal-card${wide ? " wide" : ""}`;
  m.querySelector(".close-modal").hidden = !dismissable;
  m.dataset.locked = dismissable ? "" : "1";
  const body = $("#dialog-body");
  body.innerHTML = html;
  m.hidden = false;
  dialogClose = onClose;
  const first = body.querySelector("input:not([type=hidden]):not([type=radio]):not([type=checkbox]), button, [tabindex]");
  setTimeout(() => first?.focus(), 30);
  return body;
}
export function closeDialog() {
  $("#dialog").hidden = true;
  if (dialogClose) { const f = dialogClose; dialogClose = null; f(); }
}

export function confirmDialog({ title, text, confirm = "OK", danger = false }) {
  return new Promise((resolve) => {
    let done = false;
    const body = dialog({ title, html: `<p class="summary"></p><div class="dialog-actions">
      <button class="btn" data-act="no">Cancel</button><button class="btn ${danger ? "danger" : "primary"}" data-act="yes"></button></div>`,
      onClose: () => { if (!done) resolve(false); } });
    body.querySelector(".summary").textContent = text;
    body.querySelector("[data-act=yes]").textContent = confirm;
    body.querySelector("[data-act=no]").addEventListener("click", () => { closeDialog(); });
    body.querySelector("[data-act=yes]").addEventListener("click", () => { done = true; resolve(true); closeDialog(); });
  });
}

function formError(body, msg) {
  const box = body.querySelector(".form-error");
  box.textContent = msg.charAt(0).toUpperCase() + msg.slice(1);
  box.hidden = !msg;
}

// ---------------------------------------------------------------- places
export function savePlaceDialog(site, i, { name = "", note = "" } = {}) {
  const P = state.pixels[site];
  const existing = placeAt(site, P.ids[i]);
  if (existing) return renamePlaceDialog(existing);
  const suggestion = name || (state.places.length ? `Place ${state.places.length + 1}` : (state.profile.user_type === "farmer" ? "My farm" : "My place"));
  const body = dialog({ title: "Save this place", html: `
    <p class="summary">This saves the whole <b>5 km square</b> around ${fmtLatLon(P.lat[i], P.lon[i])} (${SITE_NAMES[site]}).
      Only you can see your saved places.</p>
    <form class="form" id="place-form">
      <div class="form-error" role="alert" hidden></div>
      <label for="pl-name">Name</label>
      <input id="pl-name" maxlength="60" required>
      <label for="pl-note">Note <span class="opt">(optional)</span></label>
      <input id="pl-note" maxlength="200" placeholder="e.g. potatoes near the river, 2 acres">
      <div class="dialog-actions"><button type="button" class="btn" data-act="cancel">Cancel</button>
        <button type="submit" class="btn primary">Save place</button></div>
    </form>` });
  body.querySelector("#pl-name").value = suggestion;
  body.querySelector("#pl-note").value = note;
  body.querySelector("#pl-name").select();
  body.querySelector("[data-act=cancel]").addEventListener("click", closeDialog);
  body.querySelector("#place-form").addEventListener("submit", async (e) => {
    e.preventDefault();
    try {
      const p = await send("/api/places", { site, pixel_id: P.ids[i], name: body.querySelector("#pl-name").value,
        note: body.querySelector("#pl-note").value });
      state.places.push(p);
      if (state.places.length === 1) state.profile = { ...state.profile, home_place: p.id, home_site: site };
      closeDialog();
      placesChanged();
      toast(`Saved “${p.name}”.${state.places.length === 1 ? " It is now your home place in At a glance." : ""}`, { kind: "good" });
    } catch (err) { formError(body, err.message); }
  });
}

export function renamePlaceDialog(place) {
  const body = dialog({ title: "Edit place", html: `
    <form class="form" id="place-form">
      <div class="form-error" role="alert" hidden></div>
      <label for="pl-name">Name</label><input id="pl-name" maxlength="60" required>
      <label for="pl-note">Note <span class="opt">(optional)</span></label><input id="pl-note" maxlength="200">
      <div class="dialog-actions"><button type="button" class="btn" data-act="cancel">Cancel</button>
        <button type="submit" class="btn primary">Save</button></div>
    </form>` });
  body.querySelector("#pl-name").value = place.name;
  body.querySelector("#pl-note").value = place.note || "";
  body.querySelector("[data-act=cancel]").addEventListener("click", closeDialog);
  body.querySelector("#place-form").addEventListener("submit", async (e) => {
    e.preventDefault();
    try {
      const p = await send(`/api/places/${place.id}`, { name: body.querySelector("#pl-name").value, note: body.querySelector("#pl-note").value });
      Object.assign(place, p);
      closeDialog();
      placesChanged();
    } catch (err) { formError(body, err.message); }
  });
}

export async function deletePlace(place) {
  const ok = await confirmDialog({ title: "Delete place?", text: `“${place.name}” will be removed from your saved places.`, confirm: "Delete", danger: true });
  if (!ok) return;
  try {
    await send(`/api/places/${place.id}`, {}, "DELETE");
    state.places = state.places.filter((p) => p.id !== place.id);
    if (state.profile.home_place === place.id) state.profile = { ...state.profile, home_place: null };
    placesChanged();
    toast(`Deleted “${place.name}”.`, { action: { label: "Undo", run: async () => {
      const p = await send("/api/places", { site: place.site, pixel_id: place.pixel_id, name: place.name, note: place.note });
      state.places.push(p);
      placesChanged();
    } } });
  } catch (err) { toast(err.message, { kind: "error" }); }
}

export async function setHome(place) {
  try {
    state.profile = await send("/api/profile", { home_place: place.id, home_site: place.site });
    placesChanged();
    toast(`“${place.name}” is now your home place.`, { kind: "good" });
  } catch (err) { toast(err.message, { kind: "error" }); }
}

function placesChanged() {
  renderMarkers();
  renderMenu();
  emit("places");
  if (state.selected != null) refreshDrawer();
}

let markers = [];
function renderMarkers() {
  markers.forEach((m) => m.remove());
  markers = [];
  const home = state.profile.home_place;
  for (const p of state.places.filter((x) => x.site === state.site)) {
    const c = placeCoords(p);
    if (!c) continue;
    const el = document.createElement("button");
    el.className = `pin${p.id === home ? " home" : ""}`;
    el.setAttribute("aria-label", `Saved place ${p.name}`);
    el.innerHTML = `${p.id === home ? ICON.star : ICON.pin}<span></span>`;
    el.querySelector("span").textContent = p.name;
    el.addEventListener("click", (e) => { e.stopPropagation(); goToSquare(p.site, c.i); });
    markers.push(new maplibregl.Marker({ element: el, anchor: "bottom" }).setLngLat([c.lon, c.lat]).addTo(state.map));
  }
}

// ---------------------------------------------------------------- crops
export function cropsDialog({ onDone = null, inOnboarding = false } = {}) {
  const chosen = new Set(state.crops);
  const html = `
    <p class="summary">Pick the crops you grow or advise on. You will get advice for them for every square you open and in
      <b>At a glance</b>. <span class="basis tested">Tested model</span> means the forecast was scored on unseen data;
      <span class="basis rule">Guidance rule</span> means simple advice built on the tested rain and humidity forecasts.</p>
    ${cropCardsHTML(chosen)}
    <div class="form-error" role="alert" hidden></div>
    <div class="dialog-actions">${inOnboarding ? "" : '<button class="btn" data-act="cancel">Cancel</button>'}
      <button class="btn primary" data-act="save">Save crops</button></div>`;
  const body = dialog({ title: "My crops", html, wide: true });
  wireCropCards(body, chosen);
  body.querySelector("[data-act=cancel]")?.addEventListener("click", closeDialog);
  body.querySelector("[data-act=save]").addEventListener("click", async () => {
    try {
      await saveCrops([...chosen]);
      closeDialog();
      toast(state.crops.length ? `Crop advice on for ${state.crops.map(cropName).join(", ")}.` : "No crops selected.", { kind: "good" });
      onDone?.();
    } catch (err) { formError(body, err.message); }
  });
}

export function cropCardsHTML(chosen) {
  return `<div class="crop-grid">${state.catalogue.map((c) => `
    <label class="crop-card${chosen.has(c.id) ? " on" : ""}">
      <input type="checkbox" value="${esc(c.id)}" ${chosen.has(c.id) ? "checked" : ""}>
      <span class="crop-top"><b>${esc(c.name)}</b><em>${esc(c.local)}</em><i class="tick">${ICON.leaf}</i></span>
      <span class="crop-where">${esc(c.where)}</span>
      <span class="crop-what">${esc(c.provides)}</span>
      <span class="basis ${c.basis}">${c.basis === "tested" ? "Tested model" : "Guidance rule"}</span>
    </label>`).join("")}</div>`;
}

export function wireCropCards(root, chosen) {
  root.querySelectorAll(".crop-card input").forEach((cb) => cb.addEventListener("change", () => {
    cb.checked ? chosen.add(cb.value) : chosen.delete(cb.value);
    cb.closest(".crop-card").classList.toggle("on", cb.checked);
  }));
}

export async function saveCrops(list) {
  const r = await send("/api/crops", { crops: list });
  state.crops = r.crops;
  renderMenu();
  emit("crops");
  if (state.selected != null) refreshDrawer();
}

// ---------------------------------------------------------------- account
function accountDialog() {
  const u = state.user, admin = u.role === "admin";
  const body = dialog({ title: "Account", html: `
    <div class="account-grid">
      <section>
        <h3>Profile</h3>
        <form class="form" id="acc-name">
          <div class="form-error" role="alert" hidden></div>
          <label for="acc-display">Your name</label><input id="acc-display" maxlength="60" required>
          <label>Username</label><div class="static" id="acc-user"></div>
          <label>Email</label><div class="static" id="acc-email"></div>
          <div class="dialog-actions left"><button class="btn primary" type="submit">Save name</button></div>
        </form>
      </section>
      <section>
        <h3>Change password</h3>
        <form class="form" id="acc-pw">
          <div class="form-error" role="alert" hidden></div>
          <label for="pw-cur">Current password</label><input id="pw-cur" type="password" autocomplete="current-password" required>
          <label for="pw-new">New password</label><input id="pw-new" type="password" autocomplete="new-password" minlength="12" required>
          <label for="pw-new2">Repeat new password</label><input id="pw-new2" type="password" autocomplete="new-password" required>
          <small class="hint">At least 12 characters. Changing it signs out your other devices.</small>
          <div class="dialog-actions left"><button class="btn primary" type="submit">Change password</button></div>
        </form>
      </section>
      <section>
        <h3>Devices</h3>
        <p class="hint">Signed in somewhere you no longer use? End every other session.</p>
        <button class="btn" id="acc-sessions">Sign out other devices</button>
      </section>
      <section class="danger-zone">
        <h3>Delete account</h3>
        ${admin ? `<p class="hint">Admin accounts are removed on the server with <code>python -m web.users remove</code>.</p>` : `
        <p class="hint">Deletes your account, saved places, crops and settings. This cannot be undone.</p>
        <form class="form" id="acc-del">
          <div class="form-error" role="alert" hidden></div>
          <label for="del-pw">Type your password to confirm</label><input id="del-pw" type="password" autocomplete="current-password" required>
          <div class="dialog-actions left"><button class="btn danger" type="submit">Delete my account</button></div>
        </form>`}
      </section>
    </div>`, wide: true });
  body.querySelector("#acc-display").value = u.display_name;
  body.querySelector("#acc-user").textContent = u.username;
  body.querySelector("#acc-email").textContent = u.email ? `${u.email} · ${u.email_verified ? "confirmed" : "not confirmed yet"}` : "none (account made by an administrator)";
  body.querySelector("#acc-name").addEventListener("submit", async (e) => {
    e.preventDefault();
    const f = e.target;
    try {
      const acc = await send("/api/account", { display_name: f.querySelector("#acc-display").value });
      state.user = { ...state.user, display_name: acc.display_name };
      renderMenu(); renderAvatar();
      formError(f, ""); toast("Name saved.", { kind: "good" });
    } catch (err) { formError(f, err.message); }
  });
  body.querySelector("#acc-pw").addEventListener("submit", async (e) => {
    e.preventDefault();
    const f = e.target;
    try {
      await send("/api/account/password", { current: f.querySelector("#pw-cur").value, new: f.querySelector("#pw-new").value,
        new2: f.querySelector("#pw-new2").value });
      f.reset(); formError(f, ""); toast("Password changed. Other devices were signed out.", { kind: "good" });
    } catch (err) { formError(f, err.message); }
  });
  body.querySelector("#acc-sessions").addEventListener("click", async () => {
    try {
      const r = await send("/api/account/sessions");
      toast(r.ended ? `Signed out ${r.ended} other session${r.ended > 1 ? "s" : ""}.` : "No other sessions were open.", { kind: "good" });
    } catch (err) { toast(err.message, { kind: "error" }); }
  });
  body.querySelector("#acc-del")?.addEventListener("submit", async (e) => {
    e.preventDefault();
    const f = e.target;
    const ok = await confirmDialog({ title: "Delete your account?", text: "Your account, saved places, crops and settings will be deleted permanently.",
      confirm: "Delete everything", danger: true });
    if (!ok) { accountDialog(); return; }
    try {
      await send("/api/account/delete", { password: f.querySelector("#del-pw").value });
      location.href = "/login?deleted=1";
    } catch (err) { accountDialog(); toast(err.message, { kind: "error" }); }
  });
}

async function resendVerify() {
  try {
    const r = await send("/api/account/verify");
    toast(r.sent ? `Confirmation link sent to ${state.user.email}.` : "Your email is already confirmed.", { kind: "good" });
  } catch (err) { toast(err.message, { kind: "error" }); }
}

// ---------------------------------------------------------------- settings
async function setSetting(key, value) {
  try {
    state.profile = await send("/api/profile", { [key]: value });
    if (key === "animations") applyFxSetting();
    emit("settings");
  } catch (err) { toast(err.message, { kind: "error" }); renderMenu(); }
}

// ---------------------------------------------------------------- menu
const initials = (name) => name.split(/[\s._-]+/).filter(Boolean).slice(0, 2).map((w) => w[0].toUpperCase()).join("") || "?";

function renderAvatar() {
  const b = $("#avatar-btn");
  b.textContent = initials(state.user.display_name || state.user.username);
  b.classList.toggle("dot", !!state.user.email && !state.user.email_verified);
}

function placeRow(p) {
  const c = placeCoords(p), home = p.id === state.profile.home_place;
  return `<li class="place-row" data-id="${p.id}">
    <button class="place-go" data-act="go" title="Show on the map">
      <span class="place-icon${home ? " home" : ""}">${home ? ICON.star : ICON.pin}</span>
      <span class="place-text"><span class="place-name">${esc(p.name)}</span>
        <span class="place-meta">${SITE_NAMES[p.site]}${c ? ` · ${fmtLatLon(c.lat, c.lon, 2)}` : ""}${p.note ? ` · ${esc(p.note)}` : ""}</span></span>
    </button>
    <span class="place-tools">
      <button class="icon-sm" data-act="home" title="${home ? "Home place" : "Make this my home place"}" aria-pressed="${home}">${ICON.star}</button>
      <button class="icon-sm" data-act="rename" title="Rename">${ICON.edit}</button>
      <button class="icon-sm" data-act="delete" title="Delete">${ICON.trash}</button>
    </span></li>`;
}

export function renderMenu() {
  const u = state.user, prof = state.profile, admin = u.role === "admin";
  const reduced = matchMedia("(prefers-reduced-motion: reduce)").matches;
  $("#menu").innerHTML = `
    <div class="menu-head">
      <div class="menu-user">
        <span class="avatar big">${esc(initials(u.display_name || u.username))}</span>
        <span><b>${esc(u.display_name)}</b><small>@${esc(u.username)} · ${USER_TYPE_NAMES[prof.user_type] ?? "User"}${admin ? ' · <span class="pill admin">admin</span>' : ""}</small></span>
      </div>
      <button class="icon" data-act="close-menu" aria-label="Close menu">${ICON.close}</button>
    </div>
    ${u.email && !u.email_verified ? `<div class="menu-banner">${ICON.mail}<span>Please confirm your email so you can reset your password if you forget it.</span>
      <button class="link-btn" data-act="verify">Resend link</button></div>` : ""}
    <div class="menu-scroll">
      <section class="menu-sec">
        <div class="menu-title"><span>My places</span><span class="count">${state.places.length}</span></div>
        ${state.places.length ? `<ul class="place-list">${state.places.map(placeRow).join("")}</ul>`
          : `<p class="menu-empty">No saved places yet. Click any square on the map and choose <b>Save place</b>, or find your farm:</p>`}
        <div class="menu-row-btns">
          <button class="btn small" data-act="locate">${ICON.locate}<span>Use my location</span></button>
          <button class="btn small" data-act="search">${ICON.search}<span>Search</span></button>
        </div>
      </section>
      <section class="menu-sec">
        <div class="menu-title"><span>My crops</span><button class="link-btn" data-act="crops">Edit</button></div>
        ${state.crops.length ? `<div class="chips">${state.crops.map((c) => `<span class="chip on">${ICON.leaf}${esc(cropName(c))}</span>`).join("")}</div>`
          : `<p class="menu-empty">Pick your crops to get late-blight risk, planting, spraying and harvest advice.</p>
             <button class="btn small" data-act="crops">${ICON.leaf}<span>Choose crops</span></button>`}
      </section>
      <section class="menu-sec">
        <div class="menu-title"><span>Explore</span></div>
        <button class="menu-item" data-act="skill">${ICON.chart}<span>How good is it?<small>Accuracy on 21 months the models never saw</small></span></button>
        <button class="menu-item" data-act="why">${ICON.bulb}<span>Why these forecasts?<small>What drives the models</small></span></button>
        <button class="menu-item" data-act="about">${ICON.info}<span>About &amp; caveats<small>Data, methods and limits</small></span></button>
      </section>
      <section class="menu-sec">
        <div class="menu-title"><span>Help &amp; settings</span></div>
        <button class="menu-item" data-act="tour">${ICON.play}<span>Take the tour<small>A one-minute walk-through</small></span></button>
        <label class="switch-row"><span>Tutorial when I sign in<small>Shows the tour at the start of each visit</small></span>
          <input type="checkbox" role="switch" data-setting="tour_on_login" ${prof.tour_on_login ? "checked" : ""}></label>
        <label class="switch-row"><span>Map animations<small>${reduced ? "Off because your device asks for reduced motion" : "Subtle rain, air and spore movement"}</small></span>
          <input type="checkbox" role="switch" data-setting="animations" ${prof.animations ? "checked" : ""} ${reduced ? "disabled" : ""}></label>
      </section>
      <section class="menu-sec">
        <div class="menu-title"><span>Account</span></div>
        <button class="menu-item" data-act="account">${ICON.user}<span>Account settings<small>Name, password, devices, delete</small></span></button>
        ${admin ? `<button class="menu-item" data-act="users">${ICON.users}<span>Users<small>Accounts and sign-in events</small></span></button>` : ""}
        <button class="menu-item" data-act="signout">${ICON.out}<span>Sign out</span></button>
      </section>
      <p class="menu-foot">Research prototype · Strathmore University. A replay of 2025–26 forecasts, not an official forecast:
        the Kenya Meteorological Department is the authority.</p>
    </div>`;
}

export function openMenu() {
  const m = $("#menu");
  document.body.classList.add("menu-open");
  m.setAttribute("aria-hidden", "false");
  $("#scrim").hidden = false;
  $("#menu-btn").setAttribute("aria-expanded", "true");
  setTimeout(() => m.focus(), 50);
}
export function closeMenu() {
  document.body.classList.remove("menu-open");
  $("#menu").setAttribute("aria-hidden", "true");
  $("#scrim").hidden = true;
  $("#menu-btn").setAttribute("aria-expanded", "false");
}

function wireMenu() {
  $("#menu-btn").addEventListener("click", openMenu);
  $("#avatar-btn").addEventListener("click", openMenu);
  $("#scrim").addEventListener("click", closeMenu);
  document.addEventListener("keydown", (e) => { if (e.key === "Escape" && document.body.classList.contains("menu-open")) closeMenu(); });
  $("#menu").addEventListener("change", (e) => {
    const key = e.target.dataset.setting;
    if (key) setSetting(key, e.target.checked);
  });
  $("#menu").addEventListener("click", (e) => {
    const b = e.target.closest("[data-act]");
    if (!b) return;
    const act = b.dataset.act, row = b.closest(".place-row"), place = row ? placeById(Number(row.dataset.id)) : null;
    const go = (fn) => { closeMenu(); fn(); };
    switch (act) {
      case "close-menu": closeMenu(); break;
      case "go": { const c = placeCoords(place); if (c) go(() => goToSquare(place.site, c.i)); break; }
      case "home": setHome(place); break;
      case "rename": renamePlaceDialog(place); break;
      case "delete": deletePlace(place); break;
      case "locate": go(() => locateMe({ offerSave: true })); break;
      case "search": go(focusSearch); break;
      case "crops": cropsDialog(); break;
      case "skill": go(openSkill); break;
      case "why": go(openWhy); break;
      case "about": go(openAbout); break;
      case "tour": go(() => startTour()); break;
      case "account": accountDialog(); break;
      case "users": go(openUsers); break;
      case "verify": resendVerify(); break;
      case "signout": signOut(); break;
      default: break;
    }
  });
}

// ---------------------------------------------------------------- first visit
function onboardingDialog() {
  const prof = state.profile, chosen = new Set(state.crops);
  let step = 0, userType = prof.user_type, site = prof.home_site || state.site;
  const first = (state.user.display_name || "").split(" ")[0];
  const steps = [
    () => `<div class="onb-step">
      <div class="onb-hero">${ICON.pin}</div>
      <p class="summary">Karibu${first ? `, <b>${esc(first)}</b>` : ""}! This app shows rain, humidity and crop-risk forecasts for
        every <b>5 km square</b> of Nakuru County and a Busia area, 1 to 7 days ahead. Three quick questions set it up for you.</p>
      <div class="onb-label">What describes you best?</div>
      <div class="onb-choices" role="radiogroup">${Object.entries(USER_TYPE_NAMES).map(([k, v]) => `
        <button class="onb-choice" role="radio" data-type="${k}" aria-checked="${k === userType}">${v}</button>`).join("")}</div>
      <div class="onb-label">Which area?</div>
      <div class="onb-choices" role="radiogroup">${Object.entries(SITE_NAMES).map(([k, v]) => `
        <button class="onb-choice" role="radio" data-site="${k}" aria-checked="${k === site}">${v}</button>`).join("")}</div></div>`,
    () => `<div class="onb-step"><p class="summary">Which crops do you ${userType === "extension" ? "advise on" : userType === "farmer" ? "grow" : "want advice for"}?
        You can change this any time from the menu.</p>${cropCardsHTML(chosen)}</div>`,
    () => `<div class="onb-step">
      <div class="onb-hero">${ICON.locate}</div>
      <p class="summary">Last step: find your ${userType === "farmer" ? "farm" : "place"} and save it. Its forecast then appears in
        <b>At a glance</b> every time you open the app. Saved places are private to your account.</p>
      <div class="onb-big-btns">
        <button class="btn primary" data-act="locate">${ICON.locate}<span>Use my location</span></button>
        <button class="btn" data-act="search">${ICON.search}<span>Search coordinates or a town</span></button>
        <button class="btn ghost" data-act="later"><span>Later: I will click on the map</span></button>
      </div></div>`,
  ];
  const render = () => {
    const body = dialog({ title: ["Welcome", "Your crops", "Your place"][step], dismissable: false, wide: step === 1, html: `
      <div class="onb-dots" aria-hidden="true">${steps.map((_, k) => `<span class="${k === step ? "on" : ""}"></span>`).join("")}</div>
      ${steps[step]()}
      <div class="form-error" role="alert" hidden></div>
      ${step < 2 ? `<div class="dialog-actions">${step > 0 ? '<button class="btn" data-act="back">Back</button>' : ""}
        <button class="btn primary" data-act="next">${step === 1 && !chosen.size ? "Skip" : "Continue"}</button></div>` : ""}` });
    body.querySelectorAll("[data-type]").forEach((b) => b.addEventListener("click", () => { userType = b.dataset.type; render(); }));
    body.querySelectorAll("[data-site]").forEach((b) => b.addEventListener("click", () => { site = b.dataset.site; render(); }));
    if (step === 1) {
      wireCropCards(body, chosen);
      body.querySelectorAll(".crop-card input").forEach((cb) => cb.addEventListener("change", () => {
        body.querySelector("[data-act=next]").textContent = chosen.size ? "Continue" : "Skip";
      }));
    }
    body.querySelector("[data-act=back]")?.addEventListener("click", () => { step -= 1; render(); });
    body.querySelector("[data-act=next]")?.addEventListener("click", async () => {
      try {
        if (step === 0) {
          state.profile = await send("/api/profile", { user_type: userType, home_site: site });
          if (site !== state.site) setState({ site });
          // researchers and the curious can skip crops
          step = userType === "researcher" ? 2 : 1;
        } else if (step === 1) {
          await saveCrops([...chosen]);
          step = 2;
        }
        render();
      } catch (err) { formError(body, err.message); }
    });
    const finish = async (then) => {
      try { state.profile = await send("/api/profile", { onboarded: true }); } catch { /* not fatal */ }
      closeDialog();
      renderMenu();
      if (state.profile.tour_on_login) startTour({ after: then });
      else then?.();
    };
    body.querySelector("[data-act=locate]")?.addEventListener("click", () => finish(() => locateMe({ offerSave: true })));
    body.querySelector("[data-act=search]")?.addEventListener("click", () => finish(focusSearch));
    body.querySelector("[data-act=later]")?.addEventListener("click", () => finish(null));
  };
  render();
}

function journey() {
  const q = new URLSearchParams(location.search);
  if (q.get("verified") === "1") toast("Email confirmed. Thank you!", { kind: "good" });
  if (q.get("verified") === "0") toast("That confirmation link has expired or was used. Resend one from the menu.", { kind: "error", timeout: 9000 });
  if ([...q.keys()].length) history.replaceState(null, "", location.pathname + location.hash);
  let shown = false;
  try { shown = sessionStorage.getItem("hl-tour-shown") === "1"; } catch { /* storage blocked: show once per load */ }
  if (!state.profile.onboarded) onboardingDialog();
  else if (state.profile.tour_on_login && !shown) startTour();
}

export async function initUser() {
  renderAvatar();
  renderMenu();
  wireMenu();
  renderMarkers();
  bus.addEventListener("change", () => {  // pins belong to the current site
    if (markersSite !== state.site) { markersSite = state.site; renderMarkers(); }
  });
  initSearch();
  initHud();
  initFx();
  // keep the dialog from closing on a backdrop click while it is locked (onboarding)
  $("#dialog").addEventListener("close", () => {
    if ($("#dialog").dataset.locked) { $("#dialog").hidden = false; return; }
    if (dialogClose) { const f = dialogClose; dialogClose = null; f(); }
  });
  journey();
}
let markersSite = null;
