// Coordinate parsing for the search box. No imports and no page access, so tests can run it in Node
// (tests/js/coords.test.mjs, run by tests/test_frontend.py).

/** Parse coordinates typed or pasted in common formats; returns {lat, lng} or null.
 *  "-0.33, 35.94" · "-0.33 35.94" · "0.33S 35.94E" · "S 0.33 E 35.94" · "0°19'48\"S 35°56'24\"E" ·
 *  "35.94, -0.33" (swapped: Kenya's longitudes are far outside the latitude range) · a Google Maps link with @lat,lng */
export function parseCoords(text) {
  if (!text) return null;
  const at = text.match(/@(-?\d+(?:\.\d+)?),\s*(-?\d+(?:\.\d+)?)/) || text.match(/[?&]q=(-?\d+(?:\.\d+)?),\s*(-?\d+(?:\.\d+)?)/);
  if (at) return check(Number(at[1]), Number(at[2]));
  const s = text.toUpperCase().replace(/[°º]/g, "D").replace(/[′’']/g, "M").replace(/[″”"]/g, "X").replace(/,/g, " ").trim();
  if (/[^0-9NSEWDMX.\s+-]/.test(s)) return null;
  const leading = /^[NSEW]/.test(s);
  const num = "([+-]?\\d+(?:\\.\\d+)?)\\s*D?\\s*(?:(\\d+(?:\\.\\d+)?)\\s*M)?\\s*(?:(\\d+(?:\\.\\d+)?)\\s*X)?";
  const re = new RegExp(leading ? `([NSEW])\\s*${num}` : `${num}\\s*([NSEW])?`, "g");
  const parts = [];
  for (const m of s.matchAll(re)) {
    if (!m[0].trim()) continue;
    const [hemi, d, mi, se] = leading ? [m[1], m[2], m[3], m[4]] : [m[4], m[1], m[2], m[3]];
    let v = Math.abs(Number(d)) + (mi ? Number(mi) / 60 : 0) + (se ? Number(se) / 3600 : 0);
    if (d.startsWith("-") || hemi === "S" || hemi === "W") v = -v;
    parts.push({ v, hemi });
  }
  if (parts.length !== 2) return null;
  let lat, lng;
  const [a, b] = parts;
  if (a.hemi && "EW".includes(a.hemi) || b.hemi && "NS".includes(b.hemi)) [lat, lng] = [b.v, a.v];
  else [lat, lng] = [a.v, b.v];
  if (!a.hemi && !b.hemi && Math.abs(lat) > 20 && Math.abs(lng) <= 20) [lat, lng] = [lng, lat];  // pasted as lon, lat
  return check(lat, lng);
}
const check = (lat, lng) => (Number.isFinite(lat) && Number.isFinite(lng) && Math.abs(lat) <= 90 && Math.abs(lng) <= 180 ? { lat, lng } : null);

