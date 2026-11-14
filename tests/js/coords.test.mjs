// Node test for the search box's coordinate parser. Run by tests/test_frontend.py, which passes the URL of an
// .mjs copy of web/static/coords.js (Node reads plain .js files outside a package as CommonJS).
import assert from "node:assert/strict";

const { parseCoords } = await import(process.argv[2]);

const near = (r, lat, lng) => r && Math.abs(r.lat - lat) < 1e-6 && Math.abs(r.lng - lng) < 1e-6;
const cases = [
  ["-0.33, 35.94", -0.33, 35.94],
  ["-0.33 35.94", -0.33, 35.94],
  ["0.33S 35.94E", -0.33, 35.94],            // trailing hemisphere letters
  ["0.33 S, 35.94 E", -0.33, 35.94],
  ["S 0.33 E 35.94", -0.33, 35.94],          // leading hemisphere letters
  ["35.94E 0.33S", -0.33, 35.94],            // longitude first, but labelled
  ["0°19'48\"S 35°56'24\"E", -0.33, 35.94],  // degrees, minutes, seconds
  ["0° 19.8′ S, 35° 56.4′ E", -0.33, 35.94], // degrees and decimal minutes, typographic marks
  ["35.94, -0.33", -0.33, 35.94],            // pasted as lon, lat (no labels)
  ["0.461N, 34.111E", 0.461, 34.111],
  ["https://www.google.com/maps/@-0.3301,35.9402,14z", -0.3301, 35.9402],
];
for (const [text, lat, lng] of cases) assert.ok(near(parseCoords(text), lat, lng), `${text} -> ${JSON.stringify(parseCoords(text))}`);
for (const text of ["", "Njoro", "1 2 3", "abc, def", "-0.33", "200, 300"]) assert.equal(parseCoords(text), null, text);
console.log(`coords: ${cases.length + 6} checks passed`);
