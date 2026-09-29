// Mass-weighted Gaussian kernel density of the particle cloud on a local grid (fixed kernel radius per particle),
// the simulated slick derived from it (equivalent-thickness threshold, outlines, connected patches) and
// percentile contours for backward runs. All values are simulated, not observations.
import { dLon, fromLocal, safeCos, toLocal, wrapLon } from '../geo/geodesy';
import type { Particles } from './particles';

export interface DensityGrid {
  nx: number;
  ny: number;
  refLat: number;
  refLon: number;
  x0: number; // m, west edge (local projection about ref)
  y0: number; // m, south edge
  cellM: number; // m
  values: Float64Array; // kg of particle mass per m², row-major from south-west
  max: number;
  total: number; // kg
  bandwidthM: number;
  // lat/lon rectangle covered (exact: the local projection is linear in lat/lon)
  south: number;
  north: number;
  west: number;
  east: number;
}

/** Tunable assumption: Gaussian kernel radius carried by each particle, m (the reference engine uses ~250 m). */
export const KERNEL_M = 250;

export function computeDensity(p: Particles, kernelM = KERNEL_M, maxCells = 256): DensityGrid | null {
  const n = p.count;
  let W = 0;
  for (let i = 0; i < n; i++) W += p.mass[i];
  if (n < 1 || !(W > 0)) return null;
  // mass-weighted mean (lon unwrapped about particle 0)
  let mLat = 0, mLonOff = 0;
  for (let i = 0; i < n; i++) { mLat += p.mass[i] * p.lat[i]; mLonOff += p.mass[i] * dLon(p.lon[0], p.lon[i]); }
  const refLat = mLat / W, refLon = wrapLon(p.lon[0] + mLonOff / W);
  const cos0 = safeCos(refLat);
  const xs = new Float64Array(n), ys = new Float64Array(n);
  let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
  for (let i = 0; i < n; i++) {
    if (!(p.mass[i] > 0)) continue;
    const [x, y] = toLocal(refLat, refLon, p.lat[i], p.lon[i], cos0);
    xs[i] = x; ys[i] = y;
    if (x < minX) minX = x; if (x > maxX) maxX = x;
    if (y < minY) minY = y; if (y > maxY) maxY = y;
  }
  // the kernel can never be narrower than the grid resolves (a very spread cloud gets a coarser, wider kernel)
  const span = Math.max(maxX - minX, maxY - minY);
  const cellM = Math.max((span + 6 * kernelM) / (maxCells - 6), kernelM / 3);
  const h = Math.max(kernelM, 0.75 * cellM);
  const pad = 3 * h;
  const x0 = minX - pad, y0 = minY - pad;
  const nx = Math.max(2, Math.ceil((maxX - minX + 2 * pad) / cellM));
  const ny = Math.max(2, Math.ceil((maxY - minY + 2 * pad) / cellM));
  let g = new Float64Array(nx * ny);
  // cloud-in-cell deposit onto cell centres
  for (let i = 0; i < n; i++) {
    const m = p.mass[i];
    if (!(m > 0)) continue;
    const fx = (xs[i] - x0) / cellM - 0.5, fy = (ys[i] - y0) / cellM - 0.5;
    const ix = Math.max(0, Math.min(nx - 2, Math.floor(fx))), iy = Math.max(0, Math.min(ny - 2, Math.floor(fy)));
    const ax = Math.max(0, Math.min(1, fx - ix)), ay = Math.max(0, Math.min(1, fy - iy));
    const o = iy * nx + ix;
    g[o] += m * (1 - ax) * (1 - ay);
    g[o + 1] += m * ax * (1 - ay);
    g[o + nx] += m * (1 - ax) * ay;
    g[o + nx + 1] += m * ax * ay;
  }
  // separable Gaussian blur, σ = h in cells, truncated at 3σ, renormalised
  const sc = h / cellM;
  const r = Math.ceil(3 * sc);
  const ker = new Float64Array(2 * r + 1);
  let ks = 0;
  for (let k = -r; k <= r; k++) ks += ker[k + r] = Math.exp(-(k * k) / (2 * sc * sc));
  for (let k = 0; k < ker.length; k++) ker[k] /= ks;
  const tmp = new Float64Array(nx * ny);
  for (let y = 0; y < ny; y++)
    for (let x = 0; x < nx; x++) {
      const m = g[y * nx + x];
      if (m === 0) continue;
      for (let k = -r; k <= r; k++) { const xx = x + k; if (xx >= 0 && xx < nx) tmp[y * nx + xx] += ker[k + r] * m; }
    }
  const out = new Float64Array(nx * ny);
  for (let y = 0; y < ny; y++)
    for (let x = 0; x < nx; x++) {
      const m = tmp[y * nx + x];
      if (m === 0) continue;
      for (let k = -r; k <= r; k++) { const yy = y + k; if (yy >= 0 && yy < ny) out[yy * nx + x] += ker[k + r] * m; }
    }
  g = out;
  const area = cellM * cellM;
  let max = 0, total = 0;
  for (let k = 0; k < g.length; k++) { total += g[k]; g[k] /= area; if (g[k] > max) max = g[k]; }
  const [south, west] = fromLocal(refLat, refLon, x0, y0, cos0);
  const [north, east] = fromLocal(refLat, refLon, x0 + nx * cellM, y0 + ny * cellM, cos0);
  return { nx, ny, refLat, refLon, x0, y0, cellM, values: g, max, total, bandwidthM: h, south, north, west, east };
}

/**
 * Reference-engine look: every particle is a soft disc of radius R whose radial profile is the reference viewer's
 * canvas gradient (0.42 at the centre, 0.28 at 0.35 R, 0.08 at 0.7 R, 0 at R), added and clamped at 1. Each disc is
 * scaled by the particle's remaining mass fraction m/m0, so weathered oil fades. Values are coverage in [0, 1].
 */
export const SPLAT_RADIUS_M = 580; // reference: kernel 0.022 of a 12 km scene × 2.2
export const SPLAT_THRESHOLD = 0.06; // reference viewer's soft-threshold centre (± 0.05)

export function computeSplatField(p: Particles, radiusM = SPLAT_RADIUS_M, maxCells = 256): DensityGrid | null {
  const n = p.count;
  let W = 0;
  for (let i = 0; i < n; i++) W += p.mass[i];
  if (n < 1 || !(W > 0)) return null;
  let mLat = 0, mLonOff = 0;
  for (let i = 0; i < n; i++) { mLat += p.mass[i] * p.lat[i]; mLonOff += p.mass[i] * dLon(p.lon[0], p.lon[i]); }
  const refLat = mLat / W, refLon = wrapLon(p.lon[0] + mLonOff / W), cos0 = safeCos(refLat);
  const xs = new Float64Array(n), ys = new Float64Array(n);
  let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
  for (let i = 0; i < n; i++) {
    if (!(p.mass[i] > 0)) continue;
    const [x, y] = toLocal(refLat, refLon, p.lat[i], p.lon[i], cos0);
    xs[i] = x; ys[i] = y;
    if (x < minX) minX = x; if (x > maxX) maxX = x;
    if (y < minY) minY = y; if (y > maxY) maxY = y;
  }
  const R = radiusM;
  const cellM = Math.max((Math.max(maxX - minX, maxY - minY) + 2 * R) / (maxCells - 4), R / 12);
  const x0 = minX - R - cellM, y0 = minY - R - cellM;
  const nx = Math.ceil((maxX - minX + 2 * R) / cellM) + 2, ny = Math.ceil((maxY - minY + 2 * R) / cellM) + 2;
  const g = new Float64Array(nx * ny);
  const profile = (q: number) => (q < 0.35 ? 0.42 - (0.14 * q) / 0.35 : q < 0.7 ? 0.28 - (0.2 * (q - 0.35)) / 0.35 : q < 1 ? 0.08 - (0.08 * (q - 0.7)) / 0.3 : 0);
  const rc = Math.ceil(R / cellM);
  for (let i = 0; i < n; i++) {
    const m = p.mass[i];
    if (!(m > 0)) continue;
    const w = p.initialMass[i] > 0 ? m / p.initialMass[i] : 1;
    const ci = Math.floor((xs[i] - x0) / cellM), cj = Math.floor((ys[i] - y0) / cellM);
    for (let dj = -rc; dj <= rc; dj++) {
      const j = cj + dj;
      if (j < 0 || j >= ny) continue;
      const cy = y0 + (j + 0.5) * cellM - ys[i];
      for (let di = -rc; di <= rc; di++) {
        const ii = ci + di;
        if (ii < 0 || ii >= nx) continue;
        const cx = x0 + (ii + 0.5) * cellM - xs[i];
        const q = Math.sqrt(cx * cx + cy * cy) / R;
        if (q < 1) g[j * nx + ii] += w * profile(q);
      }
    }
  }
  let max = 0, total = 0;
  for (let k = 0; k < g.length; k++) { if (g[k] > 1) g[k] = 1; total += g[k] * cellM * cellM; if (g[k] > max) max = g[k]; }
  const [south, west] = fromLocal(refLat, refLon, x0, y0, cos0);
  const [north, east] = fromLocal(refLat, refLon, x0 + nx * cellM, y0 + ny * cellM, cos0);
  return { nx, ny, refLat, refLon, x0, y0, cellM, values: g, max, total, bandwidthM: R, south, north, west, east };
}

/** Slick in the reference style: splat coverage ≥ SPLAT_THRESHOLD; patch masses are summed from the particles inside. */
export function analyseSplatSlick(p: Particles, radiusM = SPLAT_RADIUS_M): SlickState | null {
  const grid = computeSplatField(p, radiusM);
  if (!grid) return null;
  let cells = 0;
  for (let k = 0; k < grid.values.length; k++) if (grid.values[k] >= SPLAT_THRESHOLD) cells++;
  const label = new Int32Array(grid.nx * grid.ny);
  const patches = findPatches(grid, SPLAT_THRESHOLD, PATCH_MIN_CELLS, PATCH_MIN_AREA_FRACTION, 0, label);
  const byLabel = new Map(patches.map((q) => { q.massKg = 0; return [q.label, q] as const; }));
  const cos0 = safeCos(grid.refLat);
  let W = 0;
  for (let i = 0; i < p.count; i++) {
    if (!(p.mass[i] > 0)) continue;
    W += p.mass[i];
    const [x, y] = toLocal(grid.refLat, grid.refLon, p.lat[i], p.lon[i], cos0);
    const ci = Math.floor((x - grid.x0) / grid.cellM), cj = Math.floor((y - grid.y0) / grid.cellM);
    const q = byLabel.get(label[cj * grid.nx + ci]);
    if (q) q.massKg += p.mass[i];
  }
  const kept = patches.filter((q) => q.massKg >= PATCH_MIN_MASS_FRACTION * W).sort((a, b) => b.massKg - a.massKg);
  return { grid, thresholdKgM2: SPLAT_THRESHOLD, areaM2: cells * grid.cellM * grid.cellM, maxThicknessM: NaN, patches: kept, outlines: [] };
}

export interface SlickState {
  grid: DensityGrid;
  thresholdKgM2: number; // sheen thickness × oil density
  areaM2: number; // cells at or above the threshold
  maxThicknessM: number;
  patches: Patch[];
  outlines: number[][]; // polylines, flat [lat, lon, ...]
}

/**
 * The simulated slick: equivalent thickness H = density / ρ_oil, slick = cells with H ≥ sheen thickness.
 * Treating deposited mass as a uniform film is a tunable assumption, not a measured thickness.
 */
export function analyseSlick(p: Particles, rhoOil: number, sheenM: number, kernelM = KERNEL_M): SlickState | null {
  const grid = computeDensity(p, kernelM);
  return grid ? slickFromGrid(grid, rhoOil, sheenM) : null;
}

/** Slick of any mass-per-area grid (particle density or an Eulerian thickness field × ρ_oil). */
export function slickFromGrid(grid: DensityGrid, rhoOil: number, sheenM: number): SlickState {
  const thr = sheenM * rhoOil;
  let cells = 0;
  for (let k = 0; k < grid.values.length; k++) if (grid.values[k] >= thr) cells++;
  return {
    grid,
    thresholdKgM2: thr,
    areaM2: cells * grid.cellM * grid.cellM,
    maxThicknessM: grid.max / rhoOil,
    patches: findPatches(grid, thr),
    outlines: cells ? stitch(marchingSquares(grid, thr, safeCos(grid.refLat))) : [],
  };
}

/** Join marching-squares segments into polylines by matching endpoints (closed loops repeat their first point). */
export function stitch(segs: number[]): number[][] {
  const key = (lat: number, lon: number) => `${Math.round(lat * 1e6)},${Math.round(lon * 1e6)}`;
  const ends = new Map<string, number[]>();
  const n = segs.length / 4;
  for (let s = 0; s < n; s++)
    for (const e of [0, 2]) {
      const k = key(segs[4 * s + e], segs[4 * s + e + 1]);
      const l = ends.get(k);
      if (l) l.push(s); else ends.set(k, [s]);
    }
  const used = new Uint8Array(n);
  const lines: number[][] = [];
  for (let s0 = 0; s0 < n; s0++) {
    if (used[s0]) continue;
    used[s0] = 1;
    const line = [segs[4 * s0], segs[4 * s0 + 1], segs[4 * s0 + 2], segs[4 * s0 + 3]];
    for (const forward of [true, false]) {
      for (;;) {
        const L = line.length;
        const [la, lo] = forward ? [line[L - 2], line[L - 1]] : [line[0], line[1]];
        const next = ends.get(key(la, lo))?.find((q) => !used[q]);
        if (next === undefined) break;
        used[next] = 1;
        const o = 4 * next;
        const startMatches = key(segs[o], segs[o + 1]) === key(la, lo);
        const [pla, plo] = startMatches ? [segs[o + 2], segs[o + 3]] : [segs[o], segs[o + 1]];
        if (forward) line.push(pla, plo); else line.unshift(pla, plo);
      }
    }
    lines.push(line);
  }
  return lines;
}

/** Tunable assumptions for patch detection (defaults mirror the reference engine's 2 % area / 1 % mass filters). */
export const PATCH_MIN_CELLS = 10;
export const PATCH_MIN_AREA_FRACTION = 0.02;
export const PATCH_MIN_MASS_FRACTION = 0.01;

export interface Patch {
  cells: number;
  areaM2: number;
  massKg: number;
}

/**
 * Connected patches of simulated particle density: cells ≥ threshold, 8-connected, kept if
 * cells ≥ max(minCells, areaFraction × all cells above threshold) and mass ≥ massFraction × total. Largest first.
 */
export function findPatches(d: DensityGrid, threshold: number, minCells = PATCH_MIN_CELLS, areaFraction = PATCH_MIN_AREA_FRACTION, massFraction = PATCH_MIN_MASS_FRACTION, label = new Int32Array(d.nx * d.ny)): (Patch & { label: number })[] {
  const { nx, ny, values: v } = d, area = d.cellM * d.cellM;
  label.fill(-1);
  const stack: number[] = [];
  const found: (Patch & { label: number })[] = [];
  let active = 0;
  for (let start = 0; start < v.length; start++) {
    if (v[start] < threshold || label[start] >= 0) continue;
    const p = { cells: 0, areaM2: 0, massKg: 0, label: found.length };
    label[start] = found.length;
    stack.push(start);
    while (stack.length) {
      const c = stack.pop()!, ci = c % nx, cj = (c - ci) / nx;
      p.cells++;
      p.massKg += v[c] * area;
      for (let dj = -1; dj <= 1; dj++)
        for (let di = -1; di <= 1; di++) {
          const i = ci + di, j = cj + dj, k = j * nx + i;
          if (i < 0 || j < 0 || i >= nx || j >= ny || label[k] >= 0 || v[k] < threshold) continue;
          label[k] = found.length;
          stack.push(k);
        }
    }
    p.areaM2 = p.cells * area;
    active += p.cells;
    found.push(p);
  }
  const minC = Math.max(minCells, areaFraction * active);
  return found.filter((p) => p.cells >= minC && p.massKg >= massFraction * d.total).sort((a, b) => b.massKg - a.massKg);
}

export interface PercentileContour {
  fraction: number; // e.g. 0.5 → region containing 50 % of particle mass
  threshold: number; // density level, kg/m²
  segments: number[]; // flat [lat1, lon1, lat2, lon2, ...]
}

/** Highest-density-region contours: the level enclosing `fraction` of the total mass, traced by marching squares. */
export function percentileContours(d: DensityGrid, fractions: number[]): PercentileContour[] {
  const sorted = Float64Array.from(d.values).sort().reverse();
  const area = d.cellM * d.cellM;
  const total = d.total;
  const cos0 = safeCos(d.refLat);
  return fractions.map((fraction) => {
    let cum = 0, thr = sorted[sorted.length - 1];
    for (let k = 0; k < sorted.length; k++) {
      cum += sorted[k] * area;
      if (cum >= fraction * total) { thr = sorted[k]; break; }
    }
    return { fraction, threshold: thr, segments: marchingSquares(d, thr, cos0) };
  });
}

export function marchingSquares(d: DensityGrid, level: number, cos0: number): number[] {
  const { nx, ny, values: v, cellM, x0, y0 } = d;
  const segs: number[] = [];
  const cx = (i: number) => x0 + (i + 0.5) * cellM;
  const cy = (j: number) => y0 + (j + 0.5) * cellM;
  const push = (ax: number, ay: number, bx: number, by: number) => {
    const a = fromLocal(d.refLat, d.refLon, ax, ay, cos0), b = fromLocal(d.refLat, d.refLon, bx, by, cos0);
    segs.push(a[0], a[1], b[0], b[1]);
  };
  for (let j = 0; j < ny - 1; j++) {
    for (let i = 0; i < nx - 1; i++) {
      const a = v[j * nx + i], b = v[j * nx + i + 1], c = v[(j + 1) * nx + i + 1], e = v[(j + 1) * nx + i];
      const idx = (a >= level ? 1 : 0) | (b >= level ? 2 : 0) | (c >= level ? 4 : 0) | (e >= level ? 8 : 0);
      if (idx === 0 || idx === 15) continue;
      const t = (p: number, q: number) => (p === q ? 0.5 : (level - p) / (q - p));
      const x = cx(i), y = cy(j), s = cellM;
      // edge points: bottom (a-b), right (b-c), top (e-c), left (a-e)
      const B = [x + t(a, b) * s, y], R = [x + s, y + t(b, c) * s], T = [x + t(e, c) * s, y + s], L = [x, y + t(a, e) * s];
      const seg = (p: number[], q: number[]) => push(p[0], p[1], q[0], q[1]);
      switch (idx) {
        case 1: case 14: seg(L, B); break;
        case 2: case 13: seg(B, R); break;
        case 3: case 12: seg(L, R); break;
        case 4: case 11: seg(R, T); break;
        case 6: case 9: seg(B, T); break;
        case 7: case 8: seg(L, T); break;
        case 5: seg(L, T); seg(B, R); break;
        case 10: seg(L, B); seg(R, T); break;
      }
    }
  }
  return segs;
}
