// User-drawn shapes for the slick models: forcing regions (local wind or current) and drawn slicks. Polygons are in
// world-local metres about the model's (lat0, lon0), the same coordinates the flow is sampled in.
import { pointInPolygon, shoelace } from '../curve/curve';
import type { Frame } from './grid';

export interface Polygon { x: Float64Array; y: Float64Array; minX: number; maxX: number; minY: number; maxY: number }

export function polygon(flat: ArrayLike<number>): Polygon {
  const n = flat.length >> 1, x = new Float64Array(n), y = new Float64Array(n);
  let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
  for (let i = 0; i < n; i++) {
    x[i] = flat[2 * i]; y[i] = flat[2 * i + 1];
    minX = Math.min(minX, x[i]); maxX = Math.max(maxX, x[i]); minY = Math.min(minY, y[i]); maxY = Math.max(maxY, y[i]);
  }
  return { x, y, minX, maxX, minY, maxY };
}

export const polygonArea = (p: Polygon) => Math.abs(shoelace(p.x, p.y));

export function centroid(p: Polygon): [number, number] {
  let a = 0, cx = 0, cy = 0;
  for (let i = 0, j = p.x.length - 1; i < p.x.length; j = i++) {
    const c = p.x[j] * p.y[i] - p.x[i] * p.y[j];
    a += c; cx += (p.x[j] + p.x[i]) * c; cy += (p.y[j] + p.y[i]) * c;
  }
  return Math.abs(a) > 1e-9 ? [cx / (3 * a), cy / (3 * a)] : [p.x[0], p.y[0]];
}

/** Distance from a point to the polygon's boundary. */
function edgeDistance(p: Polygon, px: number, py: number): number {
  let d = Infinity;
  for (let i = 0, j = p.x.length - 1; i < p.x.length; j = i++) {
    const ax = p.x[j], ay = p.y[j], bx = p.x[i] - ax, by = p.y[i] - ay, L = bx * bx + by * by;
    const t = L > 0 ? Math.max(0, Math.min(1, ((px - ax) * bx + (py - ay) * by) / L)) : 0;
    d = Math.min(d, Math.hypot(px - ax - t * bx, py - ay - t * by));
  }
  return d;
}

/** Signed distance: + inside, − outside. */
export function signedDistance(p: Polygon, px: number, py: number): number {
  const d = edgeDistance(p, px, py);
  return px >= p.minX && px <= p.maxX && py >= p.minY && py <= p.maxY && pointInPolygon(px, py, p.x, p.y) ? d : -d;
}

export type SlickProfile = 'uniform' | 'dome';

/**
 * Per-cell share of a drawn slick on a grid (sums to 1 over the cells it covers inside the grid, excluding land):
 * uniform thickness, or a dome thickest at the centre of the shape (∝ distance to the edge). 2×2 sub-samples per
 * cell smooth the edge. Returns the shares and the fraction of the drawn area that fell outside the grid or on land.
 */
export function rasterise(f: Frame, p: Polygon, profile: SlickProfile, land: Uint8Array | null): { share: Float64Array; lost: number } {
  const share = new Float64Array(f.nx * f.ny);
  const x0 = f.ox - (f.nx * f.dx) / 2, y0 = f.oy - (f.ny * f.dx) / 2;
  let kept = 0, all = 0;
  const maxD = profile === 'dome' ? Math.max(1, 0.5 * Math.min(p.maxX - p.minX, p.maxY - p.minY)) : 1;
  // cover the whole shape at grid spacing (also outside the grid) so the lost fraction is measured, not guessed
  for (let sy = p.minY + f.dx / 4; sy <= p.maxY; sy += f.dx / 2)
    for (let sx = p.minX + f.dx / 4; sx <= p.maxX; sx += f.dx / 2) {
      if (!pointInPolygon(sx, sy, p.x, p.y)) continue;
      const w = profile === 'dome' ? Math.min(1, edgeDistance(p, sx, sy) / maxD) : 1;
      all += w;
      const i = Math.floor((sx - x0) / f.dx), j = Math.floor((sy - y0) / f.dx);
      if (i < 0 || j < 0 || i >= f.nx || j >= f.ny || (land && land[j * f.nx + i])) continue;
      share[j * f.nx + i] += w;
      kept += w;
    }
  if (kept > 0) for (let k = 0; k < share.length; k++) share[k] /= kept;
  return { share, lost: all > 0 ? 1 - kept / all : 1 };
}

export type RegionKind = 'wind' | 'current';

/**
 * A drawn wind or current arrow: a stroke (polyline, drawn start to end) whose direction follows the stroke. Within
 * half the width of the stroke the forcing has the stroke's speed along it; it fades to nothing at the full width.
 */
export interface ForcingRegion {
  id: number;
  kind: RegionKind;
  path: Float64Array; // flat [x, y, ...] world-local metres, in drawing order
  widthM: number;
  speed: number; // m/s
  minX: number; maxX: number; minY: number; maxY: number;
}

export function stroke(id: number, kind: RegionKind, flat: ArrayLike<number>, widthM: number, speed: number): ForcingRegion {
  const path = Float64Array.from(flat);
  let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
  for (let i = 0; i < path.length; i += 2) { minX = Math.min(minX, path[i]); maxX = Math.max(maxX, path[i]); minY = Math.min(minY, path[i + 1]); maxY = Math.max(maxY, path[i + 1]); }
  return { id, kind, path, widthM, speed, minX, maxX, minY, maxY };
}

/**
 * Forcing of a stroke at (px, py): weight w ∈ [0, 1] and the unit direction. The direction blends the tangents of all
 * nearby segments (weighted by closeness), so it turns smoothly round the bends of the stroke.
 */
export function strokeAt(r: ForcingRegion, px: number, py: number, out: { w: number; ux: number; uy: number }) {
  out.w = 0; out.ux = 0; out.uy = 0;
  const W = Math.max(r.widthM, 1);
  if (px < r.minX - W || px > r.maxX + W || py < r.minY - W || py > r.maxY + W) return out;
  let dMin = Infinity, tx = 0, ty = 0;
  const P = r.path;
  for (let i = 0; i + 3 < P.length; i += 2) {
    const ax = P[i], ay = P[i + 1], bx = P[i + 2] - ax, by = P[i + 3] - ay, L2 = bx * bx + by * by;
    if (L2 <= 0) continue;
    const t = Math.max(0, Math.min(1, ((px - ax) * bx + (py - ay) * by) / L2));
    const ex = px - ax - t * bx, ey = py - ay - t * by;
    // clearly outside the width: skip before the (slow) hypot; the margin keeps the answer identical to d >= W
    if (ex * ex + ey * ey > W * W * 1.0001) continue;
    const d = Math.hypot(ex, ey);
    if (d >= W) continue;
    const k = (1 - d / W) ** 2, L = Math.sqrt(L2);
    tx += (k * bx) / L; ty += (k * by) / L;
    dMin = Math.min(dMin, d);
  }
  const n = Math.hypot(tx, ty);
  if (!(n > 0)) return out;
  // full strength within half the width, smooth fade to zero at the width
  const q = Math.max(0, Math.min(1, (dMin - W / 2) / (W / 2)));
  out.w = 1 - q * q * (3 - 2 * q);
  out.ux = tx / n; out.uy = ty / n;
  return out;
}
