// Volume-of-fluid (VOF) section: oil volume fraction in a vertical x–z slice through a wave train, two wavelengths long
// and periodic in x.
//
//   air/water:  free surface η = a cos(kx − ωt) of a deep-water linear wave (prescribed, not solved); a cell's water
//               capacity is the fraction of it below η
//   velocity:   linear wave orbital velocity, Wheeler-stretched to follow the surface + a plunging-breaker roller on the front
//               face of breaking crests (Lamb–Oseen vortex, divergence-free) + breaker turbulence as diffusivity
//   oil:        volume fraction C transported by the Hirt–Nichols donor–acceptor scheme (directionally split), bounded by
//               each cell's water capacity; droplets rise at a buoyant slip speed
//
// Kinematic VOF: the water motion is prescribed, not a Navier–Stokes solution, so it shows how breakers push a surface
// film under and how droplets resurface, not wave generation. The breaker share, strength, turbulence and rise speed
// are tunable assumptions. Its output for the master model is an entrainment rate (s⁻¹) per sea state.
import { Rng, subSeed } from '../rng/prng';
import { G, type ParamSpec, type SectionModel } from './grid';

export const VOF_DEFAULTS = {
  waveHeightM: 1.0,
  wavelengthM: 12,
  breakingFraction: 0.5, // share of crests that break in a given wave period
  breakerStrength: 0.8, // roller circulation / (c·H)
  turbulence: 0.015, // breaker eddy diffusivity / (c·H)
  riseMmS: 15, // droplet rise speed
  slickMm: 20,
  resolution: 100, // cells per wavelength
  seed: 7,
};

export type VofParams = typeof VOF_DEFAULTS;

export class VofModel implements SectionModel {
  readonly kind = 'section';
  readonly id = 'vof';
  readonly title = 'Volume of fluid (section)';
  readonly info = {
    calculates: 'Oil, water and air volume fractions in a vertical slice through breaking waves: how a surface film is pushed under by plunging breakers, spread by breaker turbulence, and resurfaces as droplets rise.',
    assumptions: 'Waves are prescribed linear deep-water kinematics (not solved); breakers are a vortex roller on steep crests. Breaker share, strength, turbulence and droplet rise speed are tunable assumptions. Resolving a millimetre film would need far finer cells; here the film is a fraction of one cell.',
    master: 'Kept as a sub-grid table: the master runs this section offline for each wind speed and uses the resulting entrainment rate as natural dispersion in every surface cell. A 50 m map cell cannot resolve breakers, so running it per cell would be redundant cost.',
  };
  readonly spec: ParamSpec[] = [
    { key: 'waveHeightM', label: 'Wave height H', unit: 'm', min: 0.1, max: 4, step: 0.05 },
    { key: 'wavelengthM', label: 'Wavelength λ', unit: 'm', min: 4, max: 60, step: 1, reset: true },
    { key: 'breakingFraction', label: 'Share of crests breaking', unit: '', min: 0, max: 1, step: 0.05, assumption: true },
    { key: 'breakerStrength', label: 'Breaker roller strength', unit: '×c·H', min: 0, max: 2, step: 0.05, assumption: true },
    { key: 'turbulence', label: 'Breaker turbulence', unit: '×c·H', min: 0, max: 0.05, step: 0.001, assumption: true },
    { key: 'riseMmS', label: 'Droplet rise speed', unit: 'mm/s', min: 0, max: 100, step: 1, assumption: true },
    { key: 'slickMm', label: 'Slick thickness', unit: 'mm', min: 1, max: 100, step: 1, reset: true },
    { key: 'resolution', label: 'Cells per wavelength', unit: '', min: 30, max: 200, step: 10, reset: true },
    { key: 'seed', label: 'Seed', unit: '', min: 1, max: 99, step: 1, reset: true },
  ];
  readonly params: VofParams & Record<string, number> = { ...VOF_DEFAULTS };
  readonly speeds = [0.1, 0.25, 0.5, 1, 2];
  t = 0;
  nx = 0; nz = 0; dx = 0; zBottom = 0;
  oil = new Float64Array(0);
  private cap = new Float64Array(0);
  private flux = new Float64Array(0);
  private outR = new Float64Array(0);
  private inR = new Float64Array(0);
  private released = 0;
  private sweepOrder = 0;
  private faceCos = new Float64Array(0); private faceSin = new Float64Array(0); private cellCos = new Float64Array(0); private cellSin = new Float64Array(0);
  subSum = 0; // time integral of the submerged share
  depthSum = 0; // time integral of share × mean submerged depth
  lastDt = 0;

  constructor(params: Partial<VofParams> = {}) {
    Object.assign(this.params, params);
    this.reset();
  }

  get k() { return (2 * Math.PI) / this.params.wavelengthM; }
  get omega() { return Math.sqrt(G * this.k); }
  get period() { return (2 * Math.PI) / this.omega; }

  reset() {
    const p = this.params;
    this.t = 0;
    this.subSum = 0;
    this.depthSum = 0;
    this.nx = 2 * Math.round(p.resolution);
    this.dx = p.wavelengthM / p.resolution;
    // 0.12 λ of air above still water, 0.63 λ of water below (orbital motion < 2 % at the bottom)
    this.nz = Math.round(0.75 * p.resolution);
    this.zBottom = 0.12 * p.wavelengthM - this.nz * this.dx;
    const N = this.nx * this.nz;
    this.oil = new Float64Array(N);
    this.cap = new Float64Array(N);
    this.flux = new Float64Array(N + this.nx + this.nz);
    this.outR = new Float64Array(N);
    this.inR = new Float64Array(N);
    this.updateCapacity();
    // film of slickMm on the surface: fill downwards from η
    for (let i = 0; i < this.nx; i++) {
      let need = (p.slickMm * 1e-3) / this.dx;
      for (let j = this.nz - 1; j >= 0 && need > 0; j--) {
        const k = j * this.nx + i, put = Math.min(need, this.cap[k]);
        this.oil[k] = put;
        need -= put;
      }
    }
    this.released = this.oil.reduce((a, b) => a + b, 0);
  }

  surface(x: number): number {
    return (this.params.waveHeightM / 2) * Math.cos(this.k * x - this.omega * this.t);
  }

  private updateCapacity() {
    const { nx, nz, dx } = this;
    for (let i = 0; i < nx; i++) {
      const eta = this.surface((i + 0.5) * dx);
      for (let j = 0; j < nz; j++) {
        const zb = this.zBottom + j * dx;
        this.cap[j * nx + i] = Math.max(0, Math.min(1, (eta - zb) / dx));
      }
    }
  }

  /** Breakers active now: [x centre, z centre, core radius, circulation, turbulence amplitude] per breaking crest. */
  breakers(): number[][] {
    const p = this.params, lam = p.wavelengthM, H = p.waveHeightM, c = this.omega / this.k;
    const T = this.period, per = Math.floor(this.t / T), s = Math.sin(Math.PI * ((this.t / T) % 1));
    const out: number[][] = [];
    for (let j = 0; j < 2; j++) {
      const rng = new Rng(subSeed(p.seed, per * 2 + j));
      if (rng.next() >= p.breakingFraction) continue;
      // crest j sits where kx − ωt = 2πm; roller on its front face, a quarter height down
      const xc = ((((this.omega * this.t) / this.k + j * lam) % (2 * lam)) + 2 * lam) % (2 * lam);
      out.push([(xc + lam / 8) % (2 * lam), -H / 4, Math.max(H / 2, 2 * this.dx), p.breakerStrength * s * c * H, p.turbulence * s * c * H]);
    }
    return out;
  }

  /** cth, sth: cos and sin of the wave phase kx − ωt at x (computed once per column by the caller). */
  private velocity(x: number, z: number, cth: number, sth: number, br: number[][], out: { u: number; w: number; K: number }) {
    const p = this.params, a = p.waveHeightM / 2, k = this.k, om = this.omega, L = 2 * p.wavelengthM;
    // Wheeler stretching: orbital velocity measured from the moving surface, so a film at η moves exactly with it
    const e = a * om * Math.exp(k * Math.min(z - a * cth, 0));
    let u = e * cth, w = e * sth, K = 0;
    for (let q = 0; q < br.length; q++) {
      const B = br[q], xv = B[0], zv = B[1], R = B[2], gam = B[3], turb = B[4];
      let rx = x - xv;
      rx -= L * Math.round(rx / L);
      const rz = z - zv, r2 = rx * rx + rz * rz;
      if (r2 > 0) {
        const f = (gam / (2 * Math.PI * r2)) * (1 - Math.exp(-r2 / (R * R)));
        u += f * rz; // clockwise: forward above the core, down in front of it
        w -= f * rx;
      }
      K += turb * Math.exp(-(rx * rx) / (p.wavelengthM * p.wavelengthM * 0.0625) - Math.max(0, zv - z) / (p.waveHeightM / 2));
    }
    out.u = u; out.w = w; out.K = K;
  }

  step(dt: number) {
    let done = 0;
    while (done < dt - 1e-12) {
      const br = this.breakers();
      const p = this.params, c = this.omega / this.k;
      const umax = (Math.PI * p.waveHeightM) / this.period * Math.exp(this.k * p.waveHeightM / 2) + p.breakerStrength * c * 0.35 + p.riseMmS * 1e-3;
      const Kmax = p.turbulence * c * p.waveHeightM;
      const sub = Math.min(dt - done, (0.4 * this.dx) / umax, Kmax > 0 ? (0.2 * this.dx * this.dx) / Kmax : Infinity);
      this.sweepOrder ^= 1;
      if (this.sweepOrder) { this.sweep(true, sub, br); this.sweep(false, sub, br); }
      else { this.sweep(false, sub, br); this.sweep(true, sub, br); }
      this.t += sub;
      this.updateCapacity();
      this.settle();
      const [share, depth] = this.submerged();
      this.subSum += share * sub;
      this.depthSum += share * depth * sub;
      done += sub;
      this.lastDt = sub;
    }
  }

  /** Oil above a cell's water capacity (the surface moved down past it) goes to the cell below. */
  private settle() {
    const { nx, nz, oil, cap } = this;
    for (let i = 0; i < nx; i++) {
      let excess = 0;
      for (let j = nz - 1; j >= 0; j--) {
        const k = j * nx + i;
        oil[k] += excess;
        excess = Math.max(0, oil[k] - cap[k]);
        oil[k] -= excess;
      }
      oil[i] += excess; // bottom row keeps any remainder (never reached in practice)
    }
  }

  /** One Hirt–Nichols donor–acceptor sweep along x (periodic) or z (closed top and bottom), with diffusion and rise. */
  private sweep(xdir: boolean, dt: number, br: number[][]) {
    const { nx, nz, dx, oil: C, cap, flux, outR, inR } = this;
    const rise = this.params.riseMmS * 1e-3, v = { u: 0, w: 0, K: 0 };
    const nFaces = xdir ? nx : nz + 1, nLines = xdir ? nz : nx;
    const cell = (line: number, a: number) => (xdir ? line * nx + ((a % nx) + nx) % nx : a * nx + line);
    const valid = (a: number) => xdir || (a >= 0 && a < nz);
    // wave phase trig per column, at faces (x = a·dx) and at cell centres (x = (i + ½)·dx)
    if (this.faceCos.length !== nx) { this.faceCos = new Float64Array(nx); this.faceSin = new Float64Array(nx); this.cellCos = new Float64Array(nx); this.cellSin = new Float64Array(nx); }
    const kk = this.k, om = this.omega;
    for (let i = 0; i < nx; i++) {
      const tf = kk * (i * dx) - om * this.t, tc = kk * ((i + 0.5) * dx) - om * this.t;
      this.faceCos[i] = Math.cos(tf); this.faceSin[i] = Math.sin(tf); this.cellCos[i] = Math.cos(tc); this.cellSin[i] = Math.sin(tc);
    }
    // face a sits between cells a−1 and a
    for (let line = 0; line < nLines; line++)
      for (let a = 0; a < nFaces; a++) {
        const fi = line * nFaces + a;
        if (!valid(a - 1) || !valid(a)) { flux[fi] = 0; continue; }
        const kL = cell(line, a - 1), kR = cell(line, a);
        const x = xdir ? a * dx : (line + 0.5) * dx, z = this.zBottom + (xdir ? (line + 0.5) * dx : a * dx);
        const col = xdir ? a : line;
        this.velocity(x, z, xdir ? this.faceCos[col] : this.cellCos[col], xdir ? this.faceSin[col] : this.cellSin[col], br, v);
        let vel = xdir ? v.u : v.w;
        if (!xdir && cap[kL] > 0.999) vel += rise; // buoyant slip upwards out of submerged cells
        const V = (vel * dt) / dx;
        let f = 0;
        if (V !== 0) {
          const kD = V > 0 ? kL : kR, kA = V > 0 ? kR : kL, aD = V > 0 ? a - 1 : a;
          const CD = C[kD], CA = C[kA];
          if (CD > 0) {
            // interface orientation at the donor decides donor or acceptor value (Hirt & Nichols 1981)
            const gAlong = Math.abs(valid(aD + 1) && valid(aD - 1) ? C[cell(line, aD + 1)] - C[cell(line, aD - 1)] : 0);
            const kd = kD;
            const gAcross = xdir
              ? Math.abs((line + 1 < nz ? C[kd + nx] : 0) - (line > 0 ? C[kd - nx] : 0))
              : Math.abs(C[aD * nx + ((line + 1) % nx)] - C[aD * nx + ((line - 1 + nx) % nx)]);
            const CAD = gAlong > gAcross && CA > 0 ? CA : CD;
            const Vd = Math.abs(V), capD = Math.max(cap[kD], 1e-12);
            const CF = Math.max((capD - CAD) * Vd - (capD - CD), 0);
            f = Math.sign(V) * Math.min(CAD * Vd + CF, CD);
          }
        }
        const K = 0.5 * v.K;
        if (K > 0) f -= (K * dt * (C[kR] - C[kL])) / (dx * dx);
        flux[fi] = f;
      }
    // limit: no cell gives more than it has or receives more than its free water capacity
    outR.fill(0); inR.fill(0);
    for (let line = 0; line < nLines; line++)
      for (let a = 0; a < nFaces; a++) {
        const f = flux[line * nFaces + a];
        if (f === 0) continue;
        const kL = cell(line, a - 1), kR = cell(line, a);
        if (f > 0) { outR[kL] += f; inR[kR] += f; } else { outR[kR] -= f; inR[kL] -= f; }
      }
    for (let k = 0; k < C.length; k++) {
      const o = outR[k], i = inR[k];
      outR[k] = o > C[k] ? C[k] / o : 1;
      inR[k] = i > 0 ? Math.min(1, Math.max(0, cap[k] - C[k] + Math.min(o, C[k])) / i) : 1;
    }
    for (let line = 0; line < nLines; line++)
      for (let a = 0; a < nFaces; a++) {
        const fi = line * nFaces + a, f = flux[fi];
        if (f === 0) continue;
        const kL = cell(line, a - 1), kR = cell(line, a);
        const s = f > 0 ? Math.min(outR[kL], inR[kR]) : Math.min(outR[kR], inR[kL]);
        C[kL] -= f * s;
        C[kR] += f * s;
      }
    for (let k = 0; k < C.length; k++) if (C[k] < 0) C[k] = 0;
  }

  total(): number { return this.oil.reduce((a, b) => a + b, 0); }

  /** Share of the oil deeper than 3 cells below the local surface, and its mean depth (m). */
  submerged(): [number, number] {
    const { nx, nz, dx } = this;
    let sub = 0, all = 0, depth = 0;
    for (let i = 0; i < nx; i++) {
      const eta = this.surface((i + 0.5) * dx);
      for (let j = 0; j < nz; j++) {
        const c = this.oil[j * nx + i];
        if (c <= 0) continue;
        all += c;
        const d = eta - (this.zBottom + (j + 0.5) * dx);
        if (d > 3 * dx) { sub += c; depth += c * d; }
      }
    }
    return [all > 0 ? sub / all : 0, sub > 0 ? depth / sub : 0];
  }

  /**
   * Gross entrainment rate, s⁻¹ of the oil: with droplets resurfacing at the rise speed w from mean depth d, a
   * quasi-steady submerged share S needs an entrainment flux S·w/d to sustain it (time averages over the run).
   */
  entrainmentRate(): number {
    if (!(this.t > 0) || !(this.subSum > 0)) return 0;
    const S = this.subSum / this.t, d = this.depthSum / this.subSum, w = Math.max(this.params.riseMmS * 1e-3, 1e-4);
    return (S * w) / Math.max(d, this.dx);
  }

  maxDepth(): number {
    const { nx, nz, dx } = this;
    let d = 0;
    for (let i = 0; i < nx; i++) {
      const eta = this.surface((i + 0.5) * dx);
      for (let j = 0; j < nz; j++) if (this.oil[j * nx + i] > 1e-3) d = Math.max(d, eta - (this.zBottom + (j + 0.5) * dx));
    }
    return d;
  }

  stats(): [string, string][] {
    const p = this.params, tot = this.total();
    return [
      ['elapsed', `${this.t.toFixed(1)} s (${(this.t / this.period).toFixed(1)} wave periods of ${this.period.toFixed(2)} s)`],
      ['steepness H/λ', `${(p.waveHeightM / p.wavelengthM).toFixed(3)} (linear-wave breaking limit ≈ 0.142)`],
      ['phase speed c', `${(this.omega / this.k).toFixed(2)} m/s`],
      ['breakers now', `${this.breakers().length} of 2 crests`],
      ['submerged oil', `${(100 * this.submerged()[0]).toFixed(1)}% deeper than 3 cells (${(3 * this.dx).toFixed(2)} m), mean depth ${this.submerged()[1].toFixed(2)} m`],
      ['deepest droplets', `${this.maxDepth().toFixed(2)} m below the surface`],
      ['entrainment rate', `${this.entrainmentRate().toExponential(2)} s⁻¹ (submerged share × rise speed ÷ mean depth, time-averaged)`],
      ['grid', `${this.nx} × ${this.nz}, cell ${(this.dx * 100).toFixed(1)} cm, dt ${(this.lastDt * 1000).toFixed(1)} ms`],
      ['volume error', `${(Math.abs(tot - this.released) / Math.max(this.released, 1e-300)).toExponential(1)}`],
    ];
  }

  warnings(): string[] {
    const p = this.params, w: string[] = [];
    if (p.waveHeightM / p.wavelengthM > 0.142) w.push('Steepness above the Stokes limit (0.142): linear kinematics are not valid for such waves.');
    const film = (p.slickMm * 1e-3) / this.dx;
    if (film < 1) w.push(`The ${p.slickMm} mm film is ${(100 * film).toFixed(0)}% of one cell: its thickness is not resolved, only its volume.`);
    const err = Math.abs(this.total() - this.released) / Math.max(this.released, 1e-300);
    if (err > 1e-6) w.push(`Oil volume error ${err.toExponential(1)}.`);
    return w;
  }
}

/** Whitecap coverage (Monahan & O'Muircheartaigh 1980) → share of crests breaking, assuming a breaker covers 10 % of a wavelength. */
export const breakingShare = (U10: number) => Math.min(1, (3.84e-6 * Math.max(0, U10) ** 3.41) / 0.1);
/** Fully developed significant wave height (Pierson–Moskowitz), m. */
export const seaHeight = (U10: number) => (0.21 * U10 * U10) / G;

/**
 * Breaker-driven entrainment rate (s⁻¹) for a fully developed sea at wind U10, from two coarse offline VOF runs of
 * `periods` wave periods: every crest breaking, minus no crest breaking (removes the scheme's own numerical mixing of
 * the film), scaled by the share of crests that break at that wind.
 * ponytail: linear in the breaking share (breakers assumed independent); run a long random-breaker section if they interact.
 */
export const VOF_TABLE_WINDS = [4, 6, 8, 10, 12, 15, 20];

export function vofEntrainment(U10: number, periods = 4, steepness = 0.08, resolution = 50, riseMmS = VOF_DEFAULTS.riseMmS): number {
  const H = Math.max(0.05, seaHeight(U10)), lam = H / steepness;
  const run = (fb: number) => {
    const m = new VofModel({ waveHeightM: H, wavelengthM: lam, breakingFraction: fb, resolution, riseMmS, slickMm: Math.max(1, 250 * (lam / resolution)) });
    m.step(periods * m.period);
    return m.entrainmentRate();
  };
  const fb = breakingShare(U10);
  return fb > 0 ? fb * Math.max(0, run(1) - run(0)) : 0;
}
