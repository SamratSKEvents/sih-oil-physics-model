// The slick boundary: closed polygons whose vertices are material points advected by the flow.
import { DEG, EARTH_RADIUS_M, dLon, fromLocal, localDistance, safeCos, toLocal, wrapLon } from '../geo/geodesy';
import type { Rng } from '../rng/prng';

export interface CurveMetrics {
  vertexCount: number;
  areaM2: number; // signed, + counter-clockwise (east-north plane)
  centroidLat: number;
  centroidLon: number;
  perimeterM: number;
  bbox: { minLat: number; maxLat: number; minLon: number; maxLon: number }; // lon relative-unwrapped about centroid
}

export interface Curve {
  id: number;
  lat: number[];
  lon: number[];
  metrics: CurveMetrics;
}

/**
 * Radially perturbed closed curve:  r(θ) = r0 · (1 + Σ_{k=2..8} a_k sin(kθ + φ_k)).
 * a_k random with decay ~ k^-1.2, rescaled so Σ|a_k| is 20–35 % of r0.
 */
export function initialCurve(rng: Rng, lat0: number, lon0: number, r0: number, n = 200, id = 0): Curve {
  const a: number[] = [], ph: number[] = [];
  for (let k = 2; k <= 8; k++) {
    a.push(rng.range(0.3, 1) / Math.pow(k, 1.2));
    ph.push(rng.range(0, 2 * Math.PI));
  }
  const total = rng.range(0.2, 0.35);
  const sum = a.reduce((s, x) => s + x, 0);
  for (let i = 0; i < a.length; i++) a[i] *= total / sum;
  const rot = rng.range(0, 2 * Math.PI);
  const lat: number[] = [], lon: number[] = [];
  const c0 = safeCos(lat0);
  for (let i = 0; i < n; i++) {
    const th = (2 * Math.PI * i) / n;
    let r = 1;
    for (let k = 2; k <= 8; k++) r += a[k - 2] * Math.sin(k * th + ph[k - 2]);
    const [la, lo] = fromLocal(lat0, lon0, r0 * r * Math.cos(th + rot), r0 * r * Math.sin(th + rot), c0);
    lat.push(la);
    lon.push(lo);
  }
  const c: Curve = { id, lat, lon, metrics: null as unknown as CurveMetrics };
  c.metrics = computeMetrics(c.lat, c.lon);
  return c;
}

/** Project vertices to local equirectangular metres about (refLat, refLon). */
export function projectCurve(lat: number[], lon: number[], refLat: number, refLon: number): { x: Float64Array; y: Float64Array } {
  const n = lat.length;
  const x = new Float64Array(n), y = new Float64Array(n);
  const c = safeCos(refLat);
  for (let i = 0; i < n; i++) {
    const p = toLocal(refLat, refLon, lat[i], lon[i], c);
    x[i] = p[0];
    y[i] = p[1];
  }
  return { x, y };
}

export function shoelace(x: ArrayLike<number>, y: ArrayLike<number>): number {
  const n = x.length;
  let s = 0;
  for (let i = 0, j = n - 1; i < n; j = i++) s += x[j] * y[i] - x[i] * y[j];
  return s / 2;
}

/** Signed area, area centroid, perimeter and bbox on a local projection about the vertex mean. */
export function computeMetrics(lat: number[], lon: number[]): CurveMetrics {
  const n = lat.length;
  let mLat = 0, mLonOff = 0;
  for (let i = 0; i < n; i++) { mLat += lat[i]; mLonOff += dLon(lon[0], lon[i]); }
  mLat /= n;
  const mLon = wrapLon(lon[0] + mLonOff / n);
  const { x, y } = projectCurve(lat, lon, mLat, mLon);
  let a2 = 0, cx = 0, cy = 0, per = 0;
  let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
  for (let i = 0, j = n - 1; i < n; j = i++) {
    const cr = x[j] * y[i] - x[i] * y[j];
    a2 += cr;
    cx += (x[j] + x[i]) * cr;
    cy += (y[j] + y[i]) * cr;
    per += Math.hypot(x[i] - x[j], y[i] - y[j]);
    if (x[i] < minX) minX = x[i];
    if (x[i] > maxX) maxX = x[i];
    if (y[i] < minY) minY = y[i];
    if (y[i] > maxY) maxY = y[i];
  }
  let gx = 0, gy = 0;
  if (Math.abs(a2) > 1e-9) { gx = cx / (3 * a2); gy = cy / (3 * a2); }
  const c = safeCos(mLat);
  const [clat, clon] = fromLocal(mLat, mLon, gx, gy, c);
  const mPerDeg = DEG * EARTH_RADIUS_M;
  return {
    vertexCount: n,
    areaM2: a2 / 2,
    centroidLat: clat,
    centroidLon: clon,
    perimeterM: per,
    bbox: {
      minLat: mLat + minY / mPerDeg,
      maxLat: mLat + maxY / mPerDeg,
      minLon: mLon + minX / (mPerDeg * c),
      maxLon: mLon + maxX / (mPerDeg * c),
    },
  };
}

/** Centripetal Catmull-Rom point between P1 and P2 at u∈(0,1), in local metres. Centripetal avoids cusps and loops. */
function catmullRom(x0: number, y0: number, x1: number, y1: number, x2: number, y2: number, x3: number, y3: number, u: number): [number, number] {
  const eps = 1e-6;
  const t0 = 0;
  const t1 = t0 + Math.max(Math.sqrt(Math.hypot(x1 - x0, y1 - y0)), eps);
  const t2 = t1 + Math.max(Math.sqrt(Math.hypot(x2 - x1, y2 - y1)), eps);
  const t3 = t2 + Math.max(Math.sqrt(Math.hypot(x3 - x2, y3 - y2)), eps);
  const t = t1 + u * (t2 - t1);
  const lerp = (ax: number, ay: number, bx: number, by: number, ta: number, tb: number): [number, number] => {
    const w = (t - ta) / (tb - ta);
    return [ax + w * (bx - ax), ay + w * (by - ay)];
  };
  const A1 = lerp(x0, y0, x1, y1, t0, t1);
  const A2 = lerp(x1, y1, x2, y2, t1, t2);
  const A3 = lerp(x2, y2, x3, y3, t2, t3);
  const B1 = lerp(A1[0], A1[1], A2[0], A2[1], t0, t2);
  const B2 = lerp(A2[0], A2[1], A3[0], A3[1], t1, t3);
  return lerp(B1[0], B1[1], B2[0], B2[1], t1, t2);
}

export interface ResampleResult {
  inserted: number;
  deleted: number;
  capped: boolean;
}

/**
 * Curve maintenance. Insert Catmull-Rom points where adjacent vertices are > dMax apart (while the
 * vertex budget allows), then delete vertices closer than dMin to the previous kept vertex (only if
 * that does not open a gap > dMax). Mutates the curve in place.
 */
export function resampleCurve(c: Curve, dMin: number, dMax: number, budget: number): ResampleResult {
  const n = c.lat.length;
  const la = c.lat, lo = c.lon;
  const nlat: number[] = [], nlon: number[] = [];
  let inserted = 0, capped = false;
  let room = budget - n;
  for (let i = 0; i < n; i++) {
    const j = (i + 1) % n;
    nlat.push(la[i]);
    nlon.push(lo[i]);
    const d = localDistance(la[i], lo[i], la[j], lo[j]);
    if (d > dMax) {
      let m = Math.ceil(d / dMax) - 1;
      if (m > room) { m = Math.max(room, 0); capped = true; }
      if (m > 0) {
        const h = (i - 1 + n) % n, k2 = (j + 1) % n;
        const cRef = safeCos(la[i]);
        const p0 = toLocal(la[i], lo[i], la[h], lo[h], cRef);
        const p2 = toLocal(la[i], lo[i], la[j], lo[j], cRef);
        const p3 = toLocal(la[i], lo[i], la[k2], lo[k2], cRef);
        for (let s = 1; s <= m; s++) {
          const q = catmullRom(p0[0], p0[1], 0, 0, p2[0], p2[1], p3[0], p3[1], s / (m + 1));
          const g = fromLocal(la[i], lo[i], q[0], q[1], cRef);
          nlat.push(g[0]);
          nlon.push(g[1]);
        }
        room -= m;
        inserted += m;
      }
    }
  }
  // deletion pass
  const m = nlat.length;
  const olat: number[] = [nlat[0]], olon: number[] = [nlon[0]];
  let deleted = 0;
  for (let i = 1; i < m; i++) {
    const pl = olat[olat.length - 1], po = olon[olon.length - 1];
    const nx = (i + 1) % m;
    const remaining = olat.length + (m - i);
    if (remaining > 3 && localDistance(pl, po, nlat[i], nlon[i]) < dMin && localDistance(pl, po, nlat[nx], nlon[nx]) <= dMax) {
      deleted++;
      continue;
    }
    olat.push(nlat[i]);
    olon.push(nlon[i]);
  }
  // closing pair: last kept vertex vs the first
  if (olat.length > 3) {
    const L = olat.length - 1;
    if (localDistance(olat[L], olon[L], olat[0], olon[0]) < dMin && localDistance(olat[L - 1], olon[L - 1], olat[0], olon[0]) <= dMax) {
      olat.pop();
      olon.pop();
      deleted++;
    }
  }
  c.lat = olat;
  c.lon = olon;
  return { inserted, deleted, capped };
}

/** Point-in-polygon (even-odd) on local metres. */
export function pointInPolygon(px: number, py: number, x: ArrayLike<number>, y: ArrayLike<number>): boolean {
  let inside = false;
  const n = x.length;
  for (let i = 0, j = n - 1; i < n; j = i++) {
    if ((y[i] > py) !== (y[j] > py) && px < ((x[j] - x[i]) * (py - y[i])) / (y[j] - y[i]) + x[i]) inside = !inside;
  }
  return inside;
}
