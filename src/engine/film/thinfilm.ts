// Eulerian thin-film slick: oil thickness h(x, y, t) on a grid that moves with the mean surface drift.
//
//   ∂h/∂t + ∇·(u h) = ∇·(D(h) ∇h) − ∇·(ν₄ w(h) ∇∇²h) − sinks
//   D(h) = g′h²/C_i + K_h − D_lens·χ(h)
//
//   g′h²/C_i   gravity–viscous spreading: reduced gravity g′ = g(ρw − ρo)/ρw against interfacial drag C_i (m/s)
//   K_h        sub-grid turbulent diffusion
//   D_lens·χ   film rupture: below the terminal thickness h_t a film stops spreading and gathers into lenses, modelled as
//              negative diffusion (χ = (1 − h/h_t) for h < h_t, faded in over the thinnest 5 %) with hyperdiffusion ν₄ choosing the fastest-growing
//              wavelength λ* = 2π·sqrt(2ν₄/D_lens). This is a grid-scale parameterisation, a tunable assumption.
//   K_along    along-wind shear dispersion (adds K_along·a aᵀ to the diffusion tensor, a = wind direction)
//   lag        thin oil drifts slower than thick oil along the wind: extra velocity −L·h_ref/(h + h_ref)·a, so the
//              thick oil forms a downwind head and the sheen a tail (the grid moves with thick oil)
//   sinks      Fingas evaporation and first-order natural dispersion, as in the particle model
//
// Finite volume, MUSCL (minmod) upwind advection, explicit sub-steps, positivity-preserving flux limiter.
import { Rng, subSeed } from '../rng/prng';
import { DEFAULT_WEATHERING } from '../particles/weathering';
import { Budget, G, RHO_WATER, discRelease, fieldSlick, volume, type Frame, type MapModel, type ParamSpec } from './grid';
import { DEFAULT_FLOW, FLOW_SPEC, SurfaceFlow } from './flow';
import { centroid, rasterise, type ForcingRegion, type Polygon, type SlickProfile } from './shapes';

export interface FilmFluxParams {
  gPrime: number; // m/s²
  spreadC: number; // interfacial drag velocity C_i, m/s
  Kh: number; // m²/s
  hTerminal: number; // m
  lensD: number; // m²/s
  lensWavelengthM: number;
  ax: number; // unit along-wind direction
  ay: number;
  KAlong: number; // m²/s
  lag: number; // m/s, lag of a vanishing film
  lagRef: number; // m
}

/** Along-wind direction, shear dispersion and sheen lag from the flow's wind (shared by the thin film and the master). */
// ponytail: uses the global wind direction; drawn wind arrows steer drift and streaks but not this anisotropy
export function oilDrift(flow: SurfaceFlow): Pick<FilmFluxParams, 'ax' | 'ay' | 'KAlong' | 'lag' | 'lagRef'> {
  const [ax, ay, W] = flow.alongWind();
  return { ax, ay, KAlong: flow.p.shearDispFrac * W, lag: flow.p.sheenLagFrac * W, lagRef: flow.p.sheenLagRefUm * 1e-6 };
}

export const minmod = (a: number, b: number) => (a * b <= 0 ? 0 : Math.abs(a) < Math.abs(b) ? a : b);

/** Face fluxes and their update for a thickness field on an nx × ny grid; reused by the master model. */
export class FilmFlux {
  readonly fx: Float64Array; // (nx + 1) × ny, volume flux per unit width, m²/s, positive east
  readonly fy: Float64Array; // nx × (ny + 1), positive north
  private ratio: Float64Array;
  private pad: Float64Array;

  constructor(readonly nx: number, readonly ny: number) {
    this.fx = new Float64Array((nx + 1) * ny);
    this.fy = new Float64Array(nx * (ny + 1));
    this.ratio = new Float64Array(nx * ny);
    this.pad = new Float64Array((nx + 4) * (ny + 4));
  }

  static nu4(p: FilmFluxParams): number {
    return (p.lensD * (p.lensWavelengthM / (2 * Math.PI)) ** 2) / 2;
  }

  /** Largest stable explicit dt for the diffusive terms. */
  dtLimit(hMax: number, dx: number, p: FilmFluxParams, spreading: boolean): number {
    const D = p.Kh + p.KAlong + p.lensD + (spreading ? (p.gPrime * hMax * hMax) / p.spreadC : 0);
    const n4 = FilmFlux.nu4(p);
    return Math.min(D > 0 ? (0.2 * dx * dx) / D : Infinity, n4 > 0 ? (0.8 * dx ** 4) / (64 * n4) : Infinity);
  }

  /**
   * One explicit step of film transport (plus advection when face velocities are given). Faces touching land carry no
   * flux; the outer boundary is open (zero thickness outside). Returns volume (m³) that left the grid.
   */
  apply(h: Float64Array, dx: number, dt: number, p: FilmFluxParams, spreading: boolean, land: Uint8Array | null, ufx: Float64Array | null, vfy: Float64Array | null): number {
    if (!this.window(h)) return 0;
    this.faces(h, dx, p, spreading, land, ufx, vfy);
    return this.limitAndUpdate(h, dx, dt);
  }

  // active window: cells i0..i1, j0..j1 (inclusive)
  private i0 = 0; private i1 = -1; private j0 = 0; private j1 = -1;

  /**
   * Bounding box of the non-zero thickness, grown by 3 cells: every face with a non-zero flux has a stencil cell with
   * oil (so lies within 2 cells of it) and so does its donor, so faces, ratios and updates outside the box are exactly
   * zero or no-ops and can be skipped. Returns false when there is no oil at all.
   */
  private window(h: Float64Array): boolean {
    const { nx, ny } = this;
    let i0 = nx, i1 = -1, j0 = ny, j1 = -1;
    for (let j = 0; j < ny; j++) {
      const r = j * nx;
      let a = -1, b = -1;
      for (let i = 0; i < nx; i++) if (h[r + i] !== 0) { a = i; break; }
      if (a < 0) continue;
      for (let i = nx - 1; i >= a; i--) if (h[r + i] !== 0) { b = i; break; }
      if (a < i0) i0 = a;
      if (b > i1) i1 = b;
      if (j < j0) j0 = j;
      j1 = j;
    }
    if (i1 < 0) return false;
    const M = 3;
    this.i0 = Math.max(0, i0 - M); this.i1 = Math.min(nx - 1, i1 + M);
    this.j0 = Math.max(0, j0 - M); this.j1 = Math.min(ny - 1, j1 + M);
    return true;
  }

  /** Face fluxes into fx, fy (no limiting). */
  private faces(h: Float64Array, dx: number, p: FilmFluxParams, spreading: boolean, land: Uint8Array | null, ufx: Float64Array | null, vfy: Float64Array | null): void {
    const { nx, ny, fx, fy, pad } = this;
    const n4 = FilmFlux.nu4(p), iht = 1 / p.hTerminal, idx = 1 / dx, n4dx3 = n4 / (dx * dx * dx);
    const kh = p.Kh, lens = p.lensD, spread = spreading ? p.gPrime / p.spreadC : 0;
    const Ka = p.KAlong, lag = p.lag, lagRef = p.lagRef, pax = p.ax, pay = p.ay;
    // thickness with a 2-cell border of open water (0)
    const W = nx + 4, NX1 = nx + 1;
    const { i0, i1, j0, j1 } = this;
    // pad rows of the window and 2 rows around it (stencils read ±2 cells); the rest of pad is never read
    for (let j = Math.max(-2, j0 - 2); j <= Math.min(ny + 1, j1 + 2); j++) {
      const r = (j + 2) * W;
      if (j < 0 || j >= ny) { pad.fill(0, r, r + W); continue; }
      pad[r] = 0; pad[r + 1] = 0; pad[r + W - 2] = 0; pad[r + W - 1] = 0;
      pad.set(h.subarray(j * nx, (j + 1) * nx), r + 2);
    }
    // Face flux, written out once per direction (no per-face function call). With LL, L | R, RR the cells along the
    // face normal, u the normal velocity, dT the along-face gradient and (an, at) the wind direction's normal and
    // tangential components:
    //   lag:   donor-cell upwind on the (always upwind) lag velocity
    //   shear: −K_along (a·∇h)(a·n)
    //   advection: MUSCL minmod upwind
    //   film diffusion D(h) = K_h + g′h²/C_i − rupture, plus hyperdiffusion where rupture acts (fading out by 3 h_t)
    // A face whose whole stencil is empty water carries no flux: skipped (most of the grid).
    for (let sweep = 0; sweep < 2; sweep++) {
      const xs = sweep === 0;
      // faces of the window: x-faces i0..i1+1 on rows j0..j1, y-faces j0..j1+1 on columns i0..i1
      const jA = j0, jB = xs ? j1 : j1 + 1, iA = i0, iB = xs ? i1 + 1 : i1;
      const step = xs ? 1 : W, side = xs ? W : 1; // along the face normal, along the face
      const an = xs ? pax : pay, at = xs ? pay : pax;
      const out = xs ? fx : fy, vel = xs ? ufx : vfy, rowW = xs ? NX1 : nx;
      for (let j = jA; j <= jB; j++) {
        for (let i = iA; i <= iB; i++) {
          const fi = j * rowW + i;
          const c = xs ? (j + 2) * W + i + 2 : (j + 2) * W + i + 2;
          const LL = pad[c - 2 * step], L = pad[c - step], R = pad[c], RR = pad[c + step];
          const lu = pad[c - step + side], ld = pad[c - step - side], ru = pad[c + side], rd = pad[c - side];
          if (LL === 0 && L === 0 && R === 0 && RR === 0 && lu === 0 && ld === 0 && ru === 0 && rd === 0) { out[fi] = 0; continue; }
          if (land !== null) {
            const blocked = xs
              ? (i > 0 && land[j * nx + i - 1] === 1) || (i < nx && land[j * nx + i] === 1)
              : (j > 0 && land[(j - 1) * nx + i] === 1) || (j < ny && land[j * nx + i] === 1);
            if (blocked) { out[fi] = 0; continue; }
          }
          const u = vel ? vel[fi] : 0;
          let f = 0;
          if (lag > 0 && an !== 0) { const vn = -lag * an, hd = vn > 0 ? L : R; f += (vn * hd * lagRef) / (hd + lagRef); }
          if (Ka > 0) f -= Ka * an * (an * (R - L) * idx + at * ((lu - ld + ru - rd) * 0.25 * idx));
          if (u > 0) { const a1 = L - LL, b1 = R - L; f += u * (L + (a1 * b1 <= 0 ? 0 : 0.5 * (Math.abs(a1) < Math.abs(b1) ? a1 : b1))); }
          else if (u < 0) { const a1 = R - L, b1 = RR - R; f += u * (R - (a1 * b1 <= 0 ? 0 : 0.5 * (Math.abs(a1) < Math.abs(b1) ? a1 : b1))); }
          const hm = 0.5 * (L + R);
          if (hm > 0) {
            const q = hm * iht;
            let D = kh + spread * hm * hm;
            if (q < 1) D -= lens * (1 - q) * (q < 0.05 ? q / 0.05 : 1);
            f -= D * (R - L) * idx;
            if (q < 3) f += n4dx3 * (q < 0.05 ? q / 0.05 : q < 1.5 ? 1 : (3 - q) / 1.5) * (RR - 3 * R + 3 * L - LL);
          }
          out[fi] = f;
        }
      }
    }
  }

  private limitAndUpdate(h: Float64Array, dx: number, dt: number): number {
    const { nx, ny, fx, fy, ratio, i0, i1, j0, j1 } = this, NX1 = nx + 1;
    // positivity: scale every cell's outgoing fluxes so it cannot give more than it holds
    for (let j = j0; j <= j1; j++) {
      const r0 = j * NX1, r1 = j * nx, r2 = (j + 1) * nx;
      for (let i = i0; i <= i1; i++) {
        const k = r1 + i, e = fx[r0 + i + 1], w = fx[r0 + i], n = fy[r2 + i], sdn = fy[k];
        const o = (e > 0 ? e : 0) + (w < 0 ? -w : 0) + (n > 0 ? n : 0) + (sdn < 0 ? -sdn : 0);
        ratio[k] = o * dt > h[k] * dx ? Math.max(0, (h[k] * dx) / (o * dt)) : 1;
      }
    }
    // scale each face by its donor's ratio; outside the grid there is no oil to give (ratio 0)
    let left = 0;
    for (let j = j0; j <= j1; j++) {
      const r0 = j * NX1, r1 = j * nx;
      for (let i = i0; i <= i1 + 1; i++) {
        const f = r0 + i, v = fx[f];
        if (v === 0) continue;
        const sc = v > 0 ? (i > 0 ? ratio[r1 + i - 1] : 0) : i < nx ? ratio[r1 + i] : 0;
        const nv = v * sc;
        fx[f] = nv;
        if (i === 0 && nv < 0) left -= nv;
        if (i === nx && nv > 0) left += nv;
      }
    }
    for (let j = j0; j <= j1 + 1; j++) {
      const r0 = j * nx;
      for (let i = i0; i <= i1; i++) {
        const f = r0 + i, v = fy[f];
        if (v === 0) continue;
        const sc = v > 0 ? (j > 0 ? ratio[r0 - nx + i] : 0) : j < ny ? ratio[r0 + i] : 0;
        const nv = v * sc;
        fy[f] = nv;
        if (j === 0 && nv < 0) left -= nv;
        if (j === ny && nv > 0) left += nv;
      }
    }
    const cdt = dt / dx;
    for (let j = j0; j <= j1; j++) {
      const r0 = j * NX1, r1 = j * nx, r2 = (j + 1) * nx;
      for (let i = i0; i <= i1; i++) {
        const k = r1 + i;
        const d = fx[r0 + i + 1] - fx[r0 + i] + fy[r2 + i] - fy[k];
        if (d !== 0) h[k] = Math.max(0, h[k] - cdt * d);
      }
    }
    return left * dx * dt;
  }
}

export const THIN_FILM_DEFAULTS = {
  ...DEFAULT_FLOW,
  releaseM3: 350,
  radiusM: 700,
  oilDensity: 860,
  spreadC: 2e-5,
  Kh: 0.3,
  hTerminalUm: 40,
  lensD: 1.5,
  lensWavelengthM: 320,
  sst: 20,
  dispersionRate: DEFAULT_WEATHERING.dispersionRate,
  evapMax: DEFAULT_WEATHERING.evapMax,
  sheenUm: 1,
  seed: 7,
};

export const OIL_SPEC: ParamSpec[] = [
  { key: 'releaseM3', label: 'Released volume', unit: 'm³', min: 10, max: 2000, step: 10, reset: true },
  { key: 'radiusM', label: 'Initial radius', unit: 'm', min: 100, max: 2000, step: 50, reset: true },
  { key: 'oilDensity', label: 'Oil density', unit: 'kg/m³', min: 700, max: 1000, step: 5 },
  { key: 'sst', label: 'Sea temperature', unit: '°C', min: 0, max: 32, step: 1 },
  { key: 'evapMax', label: 'Max evaporable fraction', unit: '', min: 0, max: 0.8, step: 0.05, assumption: true },
  { key: 'dispersionRate', label: 'Natural dispersion rate', unit: 's⁻¹', min: 0, max: 2e-5, step: 5e-7, assumption: true },
  { key: 'sheenUm', label: 'Slick edge (sheen) threshold', unit: 'µm', min: 0.05, max: 50, step: 0.05 },
  { key: 'seed', label: 'Seed', unit: '', min: 1, max: 99, step: 1, reset: true },
];

export const RUPTURE_SPEC: ParamSpec[] = [
  { key: 'hTerminalUm', label: 'Terminal (rupture) thickness h_t', unit: 'µm', min: 1, max: 200, step: 1, assumption: true },
  { key: 'lensD', label: 'Rupture strength D_lens', unit: 'm²/s', min: 0, max: 5, step: 0.1, assumption: true },
  { key: 'lensWavelengthM', label: 'Rupture wavelength λ*', unit: 'm', min: 160, max: 1200, step: 20, assumption: true },
];

export class ThinFilmModel implements MapModel {
  readonly kind = 'map';
  readonly id = 'thinfilm';
  readonly title = 'Eulerian thin film';
  readonly info = {
    calculates: 'Oil thickness per surface cell and the film fluxes between neighbouring cells: drift and eddy advection, windrow convergence, gravity–viscous spreading, turbulent diffusion, film rupture below a terminal thickness, evaporation and natural dispersion.',
    assumptions: 'Rupture (terminal thickness, strength, wavelength), interfacial drag, K_h, windrows and eddies are tunable assumptions. The grid moves with the mean drift, so the slick stays in view; the flow is synthetic.',
    master: 'Kept: rupture, K_h and weathering. Its spreading term is dropped in the master because the oil-layer momentum of the two-layer model reproduces it (in the drag-dominated limit the momentum equation reduces to this flux).',
  };
  readonly spec: ParamSpec[] = [...OIL_SPEC, { key: 'spreadC', label: 'Interfacial drag velocity C_i', unit: 'm/s', min: 2e-6, max: 2e-4, step: 2e-6, assumption: true }, { key: 'Kh', label: 'Turbulent diffusivity K_h', unit: 'm²/s', min: 0, max: 10, step: 0.1, assumption: true }, ...RUPTURE_SPEC, ...FLOW_SPEC];
  readonly params: typeof THIN_FILM_DEFAULTS & Record<string, number> = { ...THIN_FILM_DEFAULTS };
  readonly speeds = [60, 300, 900, 1800, 3600];
  readonly frame: Frame = { nx: 256, ny: 256, dx: 40, lat0: 25, lon0: -60, ox: 0, oy: 0 };
  t = 0;
  h = new Float64Array(256 * 256);
  flow!: SurfaceFlow;
  budget!: Budget;
  private flux = new FilmFlux(256, 256);
  private ufx = new Float64Array(257 * 256);
  private vfy = new Float64Array(256 * 257);
  private cornerU = new Float64Array(257 * 257);
  private cornerV = new Float64Array(257 * 257);
  private velT = -Infinity;
  private velVersion = -1;
  private umax = 0;
  private regions: ForcingRegion[] = [];
  readonly supportsRegions = true;
  lastDt = 0;
  substeps = 0;

  get rhoOil() { return this.params.oilDensity; }
  get sheenM() { return this.params.sheenUm * 1e-6; }

  constructor(params: Partial<typeof THIN_FILM_DEFAULTS> = {}) {
    Object.assign(this.params, params);
    this.reset();
  }

  reset() {
    const p = this.params, f = this.frame;
    this.t = 0;
    f.ox = 0; f.oy = 0;
    this.velT = -Infinity;
    this.flow = new SurfaceFlow(p, subSeed(p.seed, 1));
    this.flow.setRegions(this.regions);
    const rng = new Rng(subSeed(p.seed, 2));
    discRelease(f, this.h, p.releaseM3, p.radiusM, 0.25, () => rng.next());
    this.budget = new Budget(p.releaseM3);
  }

  setRegions(regions: ForcingRegion[]) {
    this.regions = regions;
    this.flow.setRegions(regions);
  }

  addOil(poly: Polygon, volumeM3: number, profile: SlickProfile, replace: boolean) {
    const f = this.frame;
    if (replace) {
      this.reset();
      [f.ox, f.oy] = centroid(poly);
      this.h.fill(0);
      this.budget = new Budget(0);
    }
    const { share, lost } = rasterise(f, poly, profile, null);
    const placed = volumeM3 * (1 - lost);
    for (let k = 0; k < share.length; k++) this.h[k] += (share[k] * placed) / (f.dx * f.dx);
    this.budget.add(placed);
    this.velT = -Infinity;
    return { placed, lost };
  }

  /** Flow relative to the moving grid at (x, y): eddies, windrows, local wind and current regions. */
  private relative(x: number, y: number, o: { u: number; v: number }) {
    o.u = 0; o.v = 0;
    this.flow.anomaly(x, y, this.t, o);
    if (this.regions.length) {
      const c = this.flow.currentOverride(x, y, this.cur);
      o.u += c.u - c.w * this.params.driftU;
      o.v += c.v - c.w * this.params.driftV;
    }
  }
  private cur = { w: 0, u: 0, v: 0 };

  filmParams(): FilmFluxParams {
    const p = this.params;
    return { gPrime: (G * (RHO_WATER - p.oilDensity)) / RHO_WATER, spreadC: p.spreadC, Kh: p.Kh, hTerminal: p.hTerminalUm * 1e-6, lensD: p.lensD, lensWavelengthM: p.lensWavelengthM, ...oilDrift(this.flow) };
  }

  /** Flow at grid corners, averaged onto faces: one sample per corner instead of two per face (half the cost). */
  private faceVelocities() {
    const { nx, ny, dx, ox, oy } = this.frame, o = { u: 0, v: 0 }, cu = this.cornerU, cv = this.cornerV, W = nx + 1;
    for (let j = 0; j <= ny; j++)
      for (let i = 0; i <= nx; i++) {
        this.relative(ox + (i - nx / 2) * dx, oy + (j - ny / 2) * dx, o);
        cu[j * W + i] = o.u; cv[j * W + i] = o.v;
      }
    let um = 0;
    for (let j = 0; j < ny; j++)
      for (let i = 0; i <= nx; i++) {
        const u = 0.5 * (cu[j * W + i] + cu[(j + 1) * W + i]);
        this.ufx[j * (nx + 1) + i] = u;
        um = Math.max(um, Math.abs(u));
      }
    for (let j = 0; j <= ny; j++)
      for (let i = 0; i < nx; i++) {
        const v = 0.5 * (cv[j * W + i] + cv[j * W + i + 1]);
        this.vfy[j * nx + i] = v;
        um = Math.max(um, Math.abs(v));
      }
    this.umax = um;
    this.velT = this.t;
    this.velVersion = this.flow.version;
  }

  step(dt: number) {
    const p = this.params, f = this.frame, fp = this.filmParams();
    let done = 0;
    this.substeps = 0;
    while (done < dt - 1e-9) {
      if (Math.abs(this.t - this.velT) >= 600 || this.velVersion !== this.flow.version) this.faceVelocities(); // ponytail: flow refreshed every 600 s sim; eddies evolve over days, windrows over ~40 min
      let hMax = 0;
      for (let k = 0; k < this.h.length; k++) if (this.h[k] > hMax) hMax = this.h[k];
      const sub = Math.min(dt - done, 120, (0.45 * f.dx) / Math.max(this.umax + fp.lag, 1e-9), this.flux.dtLimit(hMax, f.dx, fp, true));
      this.budget.left += this.flux.apply(this.h, f.dx, sub, fp, true, null, this.ufx, this.vfy);
      const [mu, mv] = this.flow.mean();
      f.ox += mu * sub; f.oy += mv * sub;
      this.t += sub;
      const vol = volume(this.h, f.dx);
      const factor = this.budget.weather(vol, this.t, sub, p.sst, p.dispersionRate, { ...DEFAULT_WEATHERING, evapMax: p.evapMax });
      if (factor !== 1) for (let k = 0; k < this.h.length; k++) this.h[k] *= factor;
      done += sub;
      this.lastDt = sub;
      this.substeps++;
    }
  }

  thickness() { return this.h; }
  land() { return null; }

  arrows(): number[] {
    const { nx, ny, dx, ox, oy } = this.frame, o = { u: 0, v: 0 }, a: number[] = [];
    for (let j = 8; j < ny; j += 16)
      for (let i = 8; i < nx; i += 16) {
        const x = ox + (i - nx / 2) * dx, y = oy + (j - ny / 2) * dx;
        this.relative(x, y, o);
        a.push(x, y, o.u, o.v);
      }
    return a;
  }

  stats(): [string, string][] {
    const vol = volume(this.h, this.frame.dx);
    const [mu, mv] = this.flow.mean();
    return [
      ...slickRows(this),
      ['frame drift', `${mu.toFixed(3)}, ${mv.toFixed(3)} m/s (grid follows it)`],
      ['sub-steps', `${this.substeps} last frame, dt ${this.lastDt.toFixed(1)} s`],
      ['rupture λ* / ν₄', `${this.params.lensWavelengthM} m / ${FilmFlux.nu4(this.filmParams()).toFixed(0)} m⁴/s`],
      ...this.budget.rows(vol),
    ];
  }

  warnings(): string[] {
    return filmWarnings(this, this.budget, volume(this.h, this.frame.dx), this.params.lensWavelengthM);
  }
}

/** Bonn Agreement appearance classes (thickness, m), used by the readout and the renderer. */
export const BONN = [
  { name: 'sheen', min: 0.04e-6 },
  { name: 'rainbow', min: 0.3e-6 },
  { name: 'metallic', min: 5e-6 },
  { name: 'discontinuous true colour', min: 50e-6 },
  { name: 'continuous true colour', min: 200e-6 },
] as const;

export function slickRows(m: MapModel, s = fieldSlick(m.frame, m.thickness(), m.rhoOil, m.sheenM)): [string, string][] {
  const total = s.patches.reduce((a, q) => a + q.massKg, 0);
  const area = new Array(BONN.length).fill(0);
  for (const v of m.thickness()) for (let c = BONN.length - 1; c >= 0; c--) if (v >= BONN[c].min) { area[c] += m.frame.dx * m.frame.dx; break; }
  return [
    ['elapsed', `${(m.t / 3600).toFixed(2)} h`],
    ['slick patches', `${s.patches.length}${s.patches.length ? ' — ' + s.patches.slice(0, 6).map((q) => `${(q.areaM2 / 1e6).toFixed(2)} km² (${((100 * q.massKg) / Math.max(total, 1e-300)).toFixed(0)}%)`).join(' / ') + (s.patches.length > 6 ? ' …' : '') : ''}`],
    ['slick area', `${(s.areaM2 / 1e6).toFixed(2)} km² with ≥ ${m.sheenM * 1e6} µm`],
    ['max thickness', `${(s.maxThicknessM * 1e6).toFixed(1)} µm`],
    ['Bonn classes', BONN.map((b, c) => `${b.name} ${(area[c] / 1e6).toFixed(2)}`).join(' · ') + ' km²'],
  ];
}

export function filmWarnings(m: MapModel, b: Budget, floating: number, lambda: number): string[] {
  const w: string[] = [];
  if (b.left > 1e-3 * b.released) w.push(`${((100 * b.left) / b.released).toFixed(1)}% of the oil has left the grid (open boundary).`);
  if (b.error(floating) > 1e-6) w.push(`Mass budget error ${b.error(floating).toExponential(1)}.`);
  // phase-field resolution rule: a patch and its gap are each half a wavelength and need ≥ 3 cells
  if (lambda < 6 * m.frame.dx) w.push(`Rupture wavelength ${lambda} m is under 6 grid cells (${6 * m.frame.dx} m): patch edges are not resolved, patches are grid noise.`);
  return w;
}
