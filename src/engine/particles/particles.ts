// Particle cloud (secondary representation). Struct-of-arrays for speed.
import { fromLocal, safeCos } from '../geo/geodesy';
import type { Rng } from '../rng/prng';
import { pointInPolygon, projectCurve, type Curve } from '../curve/curve';

export const STATUS_FLOATING = 0;
export const STATUS_STRANDED = 1;

export class Particles {
  readonly id: Int32Array;
  readonly lat: Float64Array;
  readonly lon: Float64Array;
  readonly mass: Float64Array; // kg
  readonly initialMass: Float64Array; // kg
  readonly ageSeconds: Float64Array; // signed integration time since release (negative when run backward)
  readonly status: Uint8Array;
  readonly evaporated: Float64Array; // cumulative evaporated fraction of initialMass

  constructor(readonly count: number) {
    this.evaporated = new Float64Array(count);
    this.id = new Int32Array(count);
    this.lat = new Float64Array(count);
    this.lon = new Float64Array(count);
    this.mass = new Float64Array(count);
    this.initialMass = new Float64Array(count);
    this.ageSeconds = new Float64Array(count);
    this.status = new Uint8Array(count);
  }

  totalMass(): number {
    let s = 0;
    for (let i = 0; i < this.count; i++) s += this.mass[i];
    return s;
  }
}

/** Seed `count` particles uniformly inside the curve (rejection sampling), sharing releaseMassKg equally. */
export function seedInsideCurve(rng: Rng, curve: Curve, count: number, releaseMassKg: number): Particles {
  const p = new Particles(count);
  const { centroidLat: cLat, centroidLon: cLon } = curve.metrics;
  const { x, y } = projectCurve(curve.lat, curve.lon, cLat, cLon);
  let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
  for (let i = 0; i < x.length; i++) {
    minX = Math.min(minX, x[i]); maxX = Math.max(maxX, x[i]);
    minY = Math.min(minY, y[i]); maxY = Math.max(maxY, y[i]);
  }
  const m = count > 0 ? releaseMassKg / count : 0;
  const cos0 = safeCos(cLat);
  let assigned = 0;
  for (let k = 0; k < count; ) {
    const px = rng.range(minX, maxX), py = rng.range(minY, maxY);
    if (!pointInPolygon(px, py, x, y)) continue;
    const [la, lo] = fromLocal(cLat, cLon, px, py, cos0);
    // last particle takes the rounding remainder so the sum equals the released mass
    const mk = k === count - 1 ? releaseMassKg - assigned : m;
    p.id[k] = k;
    p.lat[k] = la;
    p.lon[k] = lo;
    p.mass[k] = mk;
    p.initialMass[k] = mk;
    assigned += mk;
    k++;
  }
  return p;
}
