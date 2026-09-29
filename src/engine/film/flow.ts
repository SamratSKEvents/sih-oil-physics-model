// Synthetic surface flow for the Eulerian slick models (local metres, not the global gridded environment):
//   mean drift  = uniform current + windage × wind (windage includes wave-induced drift)
//   eddies      = random divergence-free Fourier modes over four octaves below the eddy scale, speed per mode ∝ k^(−1/3)
//                 (a Kolmogorov −5/3 energy spectrum), so slick edges are torn at many sizes (optionally periodic)
//   streaks     = convergence streaks from Langmuir circulation. Real windrows are a few to ~50 m apart, below the grid,
//                 so they are not drawn as regular stripes: three families of lines at different spacings and slightly
//                 different angles to the wind, each broken into segments of random length that fade in and out over
//                 ~20 min, so lines end, merge and wander. Crosswind speed ∝ wind, plus a downwind jet on the lines;
//                 they start above ~3 m/s wind
//   oil drift   = helpers for thickness-dependent drift (thin sheen lags thick oil, giving a thick downwind head and a
//                 sheen tail) and along-wind shear dispersion; both scale with the wind
//   arrows      = drawn wind or current strokes: along a stroke (within its width) the wind or current follows the
//                 stroke's direction at its speed; the local wind sets windage and windrow direction there
// All amplitudes and scales are tunable assumptions. Nothing here is observed data.
import { Rng } from '../rng/prng';
import { strokeAt, type ForcingRegion } from './shapes';

export interface FlowParams {
  driftU: number; // m/s east
  driftV: number; // m/s north
  windSpeed: number; // m/s
  windDirDeg: number; // direction the wind blows towards, degrees clockwise from north
  windage: number; // fraction of wind speed
  rowSpacingM: number;
  rowSpeedFrac: number; // crosswind convergence speed / wind speed
  rowJetFrac: number; // extra downwind speed on convergence lines / wind speed
  eddySpeed: number; // rms, m/s
  eddyScaleM: number; // largest eddy size; modes go down to 1/8 of it
  shearDispFrac: number; // along-wind shear dispersion K_along = shearDispFrac × wind speed, m²/s per m/s
  sheenLagFrac: number; // how much slower than thick oil a vanishing sheen drifts, × wind speed
  sheenLagRefUm: number; // thickness at which the lag is half, µm
}

export const DEFAULT_FLOW: FlowParams = { driftU: 0.05, driftV: 0.02, windSpeed: 7, windDirDeg: 60, windage: 0.03, rowSpacingM: 320, rowSpeedFrac: 0.006, rowJetFrac: 0.006, eddySpeed: 0.025, eddyScaleM: 2500, shearDispFrac: 1, sheenLagFrac: 0.008, sheenLagRefUm: 50 };

/** Per-row hash in [0, 1) (deterministic, no generator state). */
export const hash = (m: number, k: number) => { const s = Math.sin(m * 127.1 + k * 311.7) * 43758.5453; return s - Math.floor(s); };

/** Streak families: spacing and angle relative to the wind, strength weight. */
export const FAMILIES = [{ spacing: 1, angle: 0, weight: 0.6 }, { spacing: 0.62, angle: 0.21, weight: 0.45 }, { spacing: 1.47, angle: -0.16, weight: 0.5 }].map((f) => ({ ...f, cos: Math.cos(f.angle), sin: Math.sin(f.angle) }));

interface Mode { kx: number; ky: number; amp: number; phase: number; omega: number }

/** Current override at a point: current = (1 − w)·base + (u, v). */
export interface CurrentOverride { w: number; u: number; v: number }

export class SurfaceFlow {
  private modes: Mode[] = [];
  regions: ForcingRegion[] = [];
  version = 0; // bumps when regions change, so cached velocity fields refresh
  private s = { w: 0, ux: 0, uy: 0 };

  constructor(readonly p: FlowParams, seed: number, periodicM = 0, nModes = 24) {
    const rng = new Rng(seed);
    const base = (2 * Math.PI) / Math.max(p.eddyScaleM, 1);
    const amps: number[] = [];
    for (let n = 0; n < nModes; n++) {
      // log-spaced over four octaves (eddy scale down to 1/8 of it), random within each slot
      const ang = rng.next() * 2 * Math.PI, mag = base * 2 ** ((4 * (n + rng.next())) / nModes);
      let kx = mag * Math.cos(ang), ky = mag * Math.sin(ang);
      if (periodicM > 0) {
        const k1 = (2 * Math.PI) / periodicM;
        kx = Math.round(kx / k1) * k1;
        ky = Math.round(ky / k1) * k1;
        if (kx === 0 && ky === 0) kx = k1;
      }
      const km = Math.hypot(kx, ky), speed = (km / base) ** (-1 / 3); // Kolmogorov: mode speed ∝ k^(−1/3)
      amps.push(speed);
      this.modes.push({ kx, ky, amp: speed / km, phase: rng.next() * 2 * Math.PI, omega: 0.3 * p.eddySpeed * km * speed * (rng.next() < 0.5 ? -1 : 1) });
    }
    // rms speed of random-phase modes with speed amplitudes a_k is sqrt(Σ a_k² / 2): scale it to eddySpeed
    const norm = p.eddySpeed / Math.sqrt(amps.reduce((s, a) => s + a * a, 0) / 2 || 1);
    for (const m of this.modes) m.amp *= norm;
    // flat copies for the hot loop: [kx, ky, phase, omega, amp·ky, amp·kx] per mode
    this.flat = new Float64Array(6 * this.modes.length);
    this.modes.forEach((m, i) => this.flat.set([m.kx, m.ky, m.phase, m.omega, m.amp * m.ky, m.amp * m.kx], 6 * i));
  }
  /** Eddy modes, [kx, ky, phase, omega, amp·ky, amp·kx] each. */
  readonly flat: Float64Array;
  private hashCache = new Map<number, Float64Array>();
  /** The three per-line hashes of a streak family, memoised (lines repeat across millions of samples). */
  private lineHash(line: number, fam: number): Float64Array {
    const key = line * 8 + fam;
    let v = this.hashCache.get(key);
    if (!v) { v = Float64Array.of(hash(line, fam), hash(line, fam + 7), hash(line, fam + 3)); this.hashCache.set(key, v); }
    return v;
  }

  /** Unit along-wind direction and wind speed (the global wind). */
  alongWind(): [number, number, number] {
    const th = (this.p.windDirDeg * Math.PI) / 180;
    return [Math.sin(th), Math.cos(th), this.p.windSpeed];
  }

  wind(): [number, number] {
    const th = (this.p.windDirDeg * Math.PI) / 180;
    return [this.p.windSpeed * Math.sin(th), this.p.windSpeed * Math.cos(th)];
  }

  /** Uniform part: current + windage × wind. */
  mean(): [number, number] {
    const [wx, wy] = this.wind();
    return [this.p.driftU + this.p.windage * wx, this.p.driftV + this.p.windage * wy];
  }

  setRegions(regions: ForcingRegion[]) {
    this.regions = regions;
    this.version++;
  }

  /** Wind at (x, y) after the drawn wind arrows. */
  localWind(x: number, y: number): [number, number] {
    let [wx, wy] = this.wind();
    const s = this.s;
    for (const r of this.regions) {
      if (r.kind !== 'wind' || strokeAt(r, x, y, s).w <= 0) continue;
      wx += s.w * (r.speed * s.ux - wx); wy += s.w * (r.speed * s.uy - wy);
    }
    return [wx, wy];
  }

  /** Drawn current arrows at (x, y), composed in drawing order (later arrows on top). */
  currentOverride(x: number, y: number, out: CurrentOverride): CurrentOverride {
    out.w = 0; out.u = 0; out.v = 0;
    const s = this.s;
    for (const r of this.regions) {
      if (r.kind !== 'current' || strokeAt(r, x, y, s).w <= 0) continue;
      const w = s.w;
      out.u = (1 - w) * out.u + w * r.speed * s.ux;
      out.v = (1 - w) * out.v + w * r.speed * s.uy;
      out.w = 1 - (1 - w) * (1 - out.w);
    }
    return out;
  }

  /**
   * Spatially varying part at (x, y) m, time t s: eddies, windrows and the windage change from local wind regions
   * (current regions are applied by the caller, which knows the base current). Adds into out.
   */
  anomaly(x: number, y: number, t: number, out: { u: number; v: number }, rows = true): void {
    let u = 0, v = 0;
    const M = this.flat;
    for (let i = 0; i < M.length; i += 6) {
      const s = Math.sin(M[i] * x + M[i + 1] * y + M[i + 2] + M[i + 3] * t);
      u -= M[i + 4] * s;
      v += M[i + 5] * s;
    }
    let W = this.p.windSpeed, ax = 0, ay = 0;
    if (this.regions.length) {
      const [gx, gy] = this.wind(), [lx, ly] = this.localWind(x, y);
      u += this.p.windage * (lx - gx);
      v += this.p.windage * (ly - gy);
      W = Math.hypot(lx, ly);
      ax = W > 0 ? lx / W : 0; ay = W > 0 ? ly / W : 1;
    } else {
      const th = (this.p.windDirDeg * Math.PI) / 180;
      ax = Math.sin(th); ay = Math.cos(th);
    }
    if (rows && W > 2 && this.p.rowSpeedFrac > 0) {
      const onset = Math.min(1, (W - 2) / 2);
      for (let fam = 0; fam < FAMILIES.length; fam++) {
        const F = FAMILIES[fam], lam = this.p.rowSpacingM * F.spacing;
        const ca = F.cos, sa = F.sin, bx = ax * ca + ay * sa, by = -ax * sa + ay * ca; // rotated wind axis
        const s = x * bx + y * by;
        const n = x * by - y * bx + 0.25 * lam * Math.sin((2 * Math.PI * s) / (4 * lam) + t / 1800 + fam);
        // each line (between two divergence lines) is broken into segments: a random segment length and phase per
        // line, and a slow fade in and out, so lines start, stop and hand over to neighbours
        const line = Math.round(n / lam), H = this.lineHash(line, fam), segL = lam * (3 + 5 * H[0]);
        const env = Math.max(0, Math.sin((2 * Math.PI * s) / segL + 2 * Math.PI * H[1] + t / (1200 * (1 + H[2]))));
        const w = onset * F.weight * env * env;
        if (w <= 0) continue;
        const c = -w * this.p.rowSpeedFrac * W * Math.sin((2 * Math.PI * n) / lam);
        const jet = w * this.p.rowJetFrac * W * Math.cos((2 * Math.PI * n) / lam); // + on convergence lines, − between
        u += c * by + jet * bx;
        v += -c * bx + jet * by;
      }
    }
    out.u += u;
    out.v += v;
  }
}

/** Flow parameters as UI specs (shared by the map models). */
export const FLOW_SPEC = [
  { key: 'windSpeed', label: 'Wind speed', unit: 'm/s', min: 0, max: 20, step: 0.5 },
  { key: 'windDirDeg', label: 'Wind towards', unit: '°', min: 0, max: 359, step: 1 },
  { key: 'windage', label: 'Windage (incl. wave drift)', unit: '×wind', min: 0, max: 0.05, step: 0.001, assumption: true },
  { key: 'rowSpacingM', label: 'Streak spacing (mean)', unit: 'm', min: 120, max: 1000, step: 10, assumption: true },
  { key: 'rowSpeedFrac', label: 'Streak convergence speed', unit: '×wind', min: 0, max: 0.02, step: 0.0005, assumption: true },
  { key: 'rowJetFrac', label: 'Streak downwind jet', unit: '×wind', min: 0, max: 0.02, step: 0.0005, assumption: true },
  { key: 'eddySpeed', label: 'Eddy rms speed', unit: 'm/s', min: 0, max: 0.3, step: 0.005, assumption: true, reset: true },
  { key: 'eddyScaleM', label: 'Largest eddy size', unit: 'm', min: 300, max: 8000, step: 100, assumption: true, reset: true },
  { key: 'shearDispFrac', label: 'Along-wind shear dispersion', unit: 'm²/s per m/s', min: 0, max: 3, step: 0.05, assumption: true },
  { key: 'sheenLagFrac', label: 'Sheen lag behind thick oil', unit: '×wind', min: 0, max: 0.02, step: 0.0005, assumption: true },
  { key: 'sheenLagRefUm', label: 'Lag half-thickness', unit: 'µm', min: 1, max: 500, step: 1, assumption: true },
] as const;
