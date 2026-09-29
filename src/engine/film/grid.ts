// Shared pieces of the Eulerian slick models: a local square grid about a geographic point, conversion of a
// thickness field to the slick analysis used everywhere else, the Fingas evaporation curve, and the small model
// interface the /models/ page drives. Pure TypeScript.
import { fromLocal, safeCos } from '../geo/geodesy';
import { slickFromGrid, type DensityGrid, type SlickState } from '../particles/density';
import type { WeatheringParams } from '../particles/weathering';
import type { ForcingRegion, Polygon, SlickProfile } from './shapes';

export const G = 9.81;
export const RHO_WATER = 1025;

/** Local grid, nx × ny cells of dx metres, centred at (ox, oy) metres east/north of (lat0, lon0). */
export interface Frame { nx: number; ny: number; dx: number; lat0: number; lon0: number; ox: number; oy: number }

/** Thickness field h (m, row-major from the south-west) as mass per area (kg/m²) on its lat/lon rectangle. */
export function densityGrid(f: Frame, h: ArrayLike<number>, rhoOil: number): DensityGrid {
  const cos0 = safeCos(f.lat0);
  const x0 = f.ox - (f.nx * f.dx) / 2, y0 = f.oy - (f.ny * f.dx) / 2;
  const values = new Float64Array(f.nx * f.ny);
  let max = 0, total = 0;
  for (let k = 0; k < values.length; k++) {
    const v = Math.max(0, h[k]) * rhoOil;
    values[k] = v;
    total += v;
    if (v > max) max = v;
  }
  const [south, west] = fromLocal(f.lat0, f.lon0, x0, y0, cos0);
  const [north, east] = fromLocal(f.lat0, f.lon0, x0 + f.nx * f.dx, y0 + f.ny * f.dx, cos0);
  return { nx: f.nx, ny: f.ny, refLat: f.lat0, refLon: f.lon0, x0, y0, cellM: f.dx, values, max, total: total * f.dx * f.dx, bandwidthM: f.dx, south, north, west, east };
}

export function fieldSlick(f: Frame, h: ArrayLike<number>, rhoOil: number, sheenM: number): SlickState {
  return slickFromGrid(densityGrid(f, h, rhoOil), rhoOil, sheenM);
}

/** Fingas log-form evaporated fraction of the released oil at age t seconds (same curve as the particle model). */
export function evaporatedFraction(tSec: number, sst: number, w: WeatheringParams): number {
  if (!w.weathering || tSec <= 0) return 0;
  return Math.min(w.evapMax, Math.max(0, w.evapA + w.evapB * sst) * Math.log1p(tSec / 60));
}

/** Mass budget shared by every model, in m³ of oil. */
export class Budget {
  released = 0; evaporated = 0; dispersed = 0; stranded = 0; left = 0;
  trimmed = 0; // sparse field: sub-threshold oil dropped at the edge of the allocated blocks
  private evapF = 0;

  constructor(released: number) { this.released = released; }

  /** Oil added during a run; it weathers on the same age curve as the first release. */
  // ponytail: one oil age for the whole field; track per-release age fields if staggered releases matter
  add(vol: number) { this.released += vol; }

  /**
   * Evaporation to age t and first-order dispersion at rate k over dt, applied to the whole field (one release, so every
   * cell has the same age). Returns the factor to multiply thickness by.
   */
  weather(floating: number, t: number, dt: number, sst: number, k: number, w: WeatheringParams): number {
    if (!w.weathering || !(floating > 0)) return 1;
    const f = evaporatedFraction(t, sst, w);
    const ev = Math.min(floating, this.released * (f - this.evapF));
    this.evapF = f;
    const after = floating - ev;
    const di = after * (1 - Math.exp(-k * dt));
    this.evaporated += ev;
    this.dispersed += di;
    return (after - di) / floating;
  }

  /** Relative error of floating + removed − released. */
  error(floating: number): number {
    return Math.abs(floating + this.evaporated + this.dispersed + this.stranded + this.left + this.trimmed - this.released) / Math.max(this.released, 1e-300);
  }

  rows(floating: number): [string, string][] {
    const pct = (v: number) => `${((100 * v) / Math.max(this.released, 1e-300)).toFixed(1)}%`;
    return [
      ['oil budget', `floating ${pct(floating)} · evaporated ${pct(this.evaporated)} · dispersed ${pct(this.dispersed)}${this.stranded ? ` · stranded ${pct(this.stranded)}` : ''}${this.left ? ` · left grid ${pct(this.left)}` : ''}${this.trimmed ? ` · trimmed ${pct(this.trimmed)}` : ''}`],
      ['budget error', this.error(floating).toExponential(1)],
    ];
  }
}

export interface ParamSpec {
  key: string;
  label: string;
  unit: string;
  min: number;
  max: number;
  step: number;
  reset?: boolean; // takes effect on reset
  assumption?: boolean; // tunable assumption, not a physical constant
}

export interface ModelInfo {
  calculates: string;
  assumptions: string;
  master: string; // what the master model keeps from it, and why
}

interface ModelBase {
  readonly id: string;
  readonly title: string;
  readonly info: ModelInfo;
  readonly spec: ParamSpec[];
  readonly params: Record<string, number>;
  readonly speeds: number[]; // sim seconds per wall second offered by the UI
  t: number;
  step(dt: number): void;
  reset(): void;
  stats(): [string, string][];
  warnings(): string[];
}

/** A plan-view model: a thickness field on a moving or fixed local grid. */
export interface MapModel extends ModelBase {
  readonly kind: 'map';
  readonly frame: Frame;
  readonly rhoOil: number;
  readonly sheenM: number;
  thickness(): Float64Array; // m
  land(): Uint8Array | null; // 1 = land, same grid
  arrows(): number[]; // flat [x, y, u, v] in frame metres (grid centre at ox, oy), m/s
  readonly supportsRegions: boolean;
  setRegions(regions: ForcingRegion[]): void;
  /** Put a drawn slick of `volumeM3` into the run (or restart with it). Returns volume placed and the share lost off-grid/on land. */
  addOil(poly: Polygon, volumeM3: number, profile: SlickProfile, replace: boolean): { placed: number; lost: number };
}

/** A vertical x–z section model. */
export interface SectionModel extends ModelBase {
  readonly kind: 'section';
  readonly nx: number;
  readonly nz: number;
  readonly dx: number;
  readonly zBottom: number;
  oil: Float64Array; // oil volume fraction per cell, row-major from the bottom-left
  surface(x: number): number; // free-surface elevation, m
}

export type SlickModel = MapModel | SectionModel;

/** Mean over the field of a thickness array, as volume (m³). */
export function volume(h: ArrayLike<number>, dx: number): number {
  let s = 0;
  for (let k = 0; k < h.length; k++) s += h[k];
  return s * dx * dx;
}

/** Initial disc of radius r (m) holding vol m³, with a ±noise relative ragged edge from the seeded generator. */
export function discRelease(f: Frame, h: Float64Array, vol: number, r: number, noise: number, rnd: () => number, cx = 0, cy = 0): void {
  h.fill(0);
  const harmonics = Array.from({ length: 6 }, (_, n) => ({ a: (noise * (rnd() - 0.5) * 2) / (n + 1), p: rnd() * 2 * Math.PI }));
  let sum = 0;
  for (let j = 0; j < f.ny; j++)
    for (let i = 0; i < f.nx; i++) {
      const x = (i + 0.5 - f.nx / 2) * f.dx - cx, y = (j + 0.5 - f.ny / 2) * f.dx - cy;
      const th = Math.atan2(y, x);
      let edge = 1;
      harmonics.forEach((q, n) => { edge += q.a * Math.cos((n + 2) * th + q.p); });
      const rr = Math.hypot(x, y) / (r * edge);
      if (rr < 1) { const v = (1 - rr * rr) * (1 + noise * 0.2 * (rnd() - 0.5)); h[j * f.nx + i] = v; sum += v; }
    }
  const s = vol / (sum * f.dx * f.dx);
  for (let k = 0; k < h.length; k++) h[k] *= s;
}
