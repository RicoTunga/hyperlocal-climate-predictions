// Subtle weather animations over the map. Not decoration only: particles are spawned in proportion to the
// value shown on the map, so busier areas mean a higher forecast. Kept sparse and faint on purpose.
//   rain layers -> thin blue streaks falling     humidity -> slow green wisps drifting
//   late blight -> faint orange spores rising     tomato blight -> soft red rings swelling and fading
//   ("dew" on leaves: long leaf wetness is what lets the disease spread)   "error" view -> off
// Off when the user turns it off (menu), when the device asks for reduced motion, while the map moves,
// and while the tab is hidden. Capped at 30 frames per second and a few hundred particles.
import { state, bus, valueOf } from "./app.js";

const MAX = { rain: 190, air: 46, spores: 70, dew: 55 };
const SPAWN_TRIES = { rain: 22, air: 3, spores: 5, dew: 3 };
let canvas, ctx, dpr = 1, parts = [], raf = null, last = 0, moving = false;
const reduced = matchMedia("(prefers-reduced-motion: reduce)");

function kind() {
  if (state.view === "error") return null;
  if (state.layer.startsWith("rain")) return "rain";
  if (state.layer === "rh") return "air";
  if (state.layer === "blight_tomato") return "dew";
  if (state.layer.startsWith("blight")) return "spores";
  return null;
}

/** 0..1 intensity of a square for the current layer and view. */
function intensity(H, i) {
  const v = valueOf(state.layer, state.view, H, i);
  if (v == null) return 0;
  if (state.view === "observed" && state.layer.startsWith("rain")) return Math.min(1, v / 15);
  switch (state.layer) {
    case "rain1": return v;
    case "rain10": return Math.min(1, v * 2);
    case "rain25": return Math.min(1, v * 4);
    case "rainmm": return Math.min(1, v / 15);
    case "rh": return Math.max(0, Math.min(1, (v - 55) / 40));
    default: return v;  // blight probability, or rule met (1) / not met (0)
  }
}

export function fxOn() {
  return !!state.profile?.animations && !reduced.matches && !!kind() && !document.hidden;
}

function resize() {
  dpr = Math.min(2, window.devicePixelRatio || 1);
  canvas.width = Math.round(innerWidth * dpr);
  canvas.height = Math.round(innerHeight * dpr);
  canvas.style.width = `${innerWidth}px`;
  canvas.style.height = `${innerHeight}px`;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
}

function spawn(k) {
  const H = state.day?.horizons[state.h], P = state.pixels[state.site];
  if (!H || !P) return;
  const zoom = state.map.getZoom();
  const cap = Math.round(MAX[k] * Math.min(1.4, Math.max(0.6, (zoom - 6) / 3)));
  const half = state.meta.sites[state.site].step / 2;
  for (let t = 0; t < SPAWN_TRIES[k] && parts.length < cap; t++) {
    const i = Math.floor(Math.random() * P.ids.length);
    const w = intensity(H, i);
    if (Math.random() > w * w * 0.95 + 0.02 * w) continue;  // squares near zero stay calm
    const lng = P.lon[i] + (Math.random() * 2 - 1) * half, lat = P.lat[i] + (Math.random() * 2 - 1) * half;
    const pt = state.map.project([lng, lat]);
    if (pt.x < 0 || pt.y < 0 || pt.x > innerWidth || pt.y > innerHeight) continue;
    if (k === "rain") {
      parts.push({ k, x: pt.x, y: pt.y - 14, vx: -0.9, vy: 6 + Math.random() * 3, age: 0, life: 14 + Math.random() * 14,
        len: 8 + Math.random() * 8, a: 0.26 + 0.34 * w });
    } else if (k === "air") {
      parts.push({ k, x: pt.x, y: pt.y, vx: 0.35 + Math.random() * 0.35, age: 0, life: 150 + Math.random() * 110,
        len: 28 + Math.random() * 26, ph: Math.random() * 6.28, a: 0.20 + 0.24 * w });
    } else if (k === "dew") {
      parts.push({ k, x: pt.x, y: pt.y, vx: 0, vy: 0, age: 0, life: 70 + Math.random() * 60,
        r: 3 + Math.random() * 4, a: 0.40 + 0.34 * w });
    } else {
      parts.push({ k, x: pt.x, y: pt.y, vx: 0, vy: -(0.18 + Math.random() * 0.3), age: 0, life: 110 + Math.random() * 90,
        r: 0.9 + Math.random() * 1.2, ph: Math.random() * 6.28, a: 0.32 + 0.33 * w });
    }
  }
}

function draw() {
  ctx.clearRect(0, 0, innerWidth, innerHeight);
  for (const p of parts) {
    const f = Math.sin(Math.PI * Math.min(1, p.age / p.life));  // fade in and out
    if (p.k === "rain") {
      ctx.strokeStyle = `rgba(186, 220, 255, ${p.a * f})`;  // pale blue: readable on the blue rain field
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(p.x, p.y);
      ctx.lineTo(p.x - p.vx * (p.len / p.vy), p.y - p.len);
      ctx.stroke();
    } else if (p.k === "air") {
      ctx.strokeStyle = `rgba(214, 255, 236, ${p.a * f})`;  // pale mint: readable on the green humidity field
      ctx.lineWidth = 1.4;
      ctx.lineCap = "round";
      ctx.beginPath();
      for (let s = 0; s <= 10; s++) {
        const x = p.x + (s / 10) * p.len, y = p.y + Math.sin(p.ph + s * 0.55 + p.age * 0.03) * 2.6;
        s ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
      }
      ctx.stroke();
    } else if (p.k === "dew") {
      // a small bead that swells into a ring and fades
      const t = p.age / p.life, rr = 1 + p.r * t;
      ctx.strokeStyle = `rgba(255, 222, 212, ${p.a * (1 - t)})`;  // near-white pink: readable on the red field
      ctx.lineWidth = 1.2;
      ctx.beginPath();
      ctx.arc(p.x, p.y, rr, 0, 6.283);
      ctx.stroke();
      ctx.fillStyle = `rgba(255, 196, 182, ${p.a * f * 0.9})`;
      ctx.beginPath();
      ctx.arc(p.x, p.y, 1.1, 0, 6.283);
      ctx.fill();
    } else {
      ctx.fillStyle = `rgba(255, 200, 150, ${p.a * f})`;
      ctx.beginPath();
      ctx.arc(p.x + Math.sin(p.ph + p.age * 0.05) * 2, p.y, p.r, 0, 6.283);
      ctx.fill();
    }
  }
}

function step() {
  for (const p of parts) {
    p.age += 1;
    p.x += p.vx;
    if (p.vy) p.y += p.vy;
  }
  parts = parts.filter((p) => p.age < p.life);
}

function frame(t) {
  raf = null;
  if (!fxOn()) { clear(); return; }
  if (t - last >= 33) {  // ~30 fps is plenty for this and saves battery
    last = t;
    if (!moving) spawn(kind());
    step();
    draw();
  }
  raf = requestAnimationFrame(frame);
}

function clear() {
  parts = [];
  ctx?.clearRect(0, 0, innerWidth, innerHeight);
}

function restart() {
  if (kindNow !== kind()) { kindNow = kind(); parts = []; }
  if (fxOn() && raf == null) raf = requestAnimationFrame(frame);
  if (!fxOn()) clear();
}
let kindNow = null;

export function applyFxSetting() { restart(); }

export function initFx() {
  canvas = document.getElementById("fx");
  ctx = canvas.getContext("2d");
  resize();
  addEventListener("resize", resize);
  // particles live in screen space, so they are cleared while the map pans or zooms and come back after
  state.map.on("movestart", () => { moving = true; clear(); });
  state.map.on("moveend", () => { moving = false; });
  document.addEventListener("visibilitychange", restart);
  reduced.addEventListener?.("change", restart);
  bus.addEventListener("change", restart);
  bus.addEventListener("settings", restart);
  restart();
}
