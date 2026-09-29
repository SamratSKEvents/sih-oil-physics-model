// Two-layer shallow-water slick model on a synthetic coast.
//
// Water layer (lower, 128² C-grid, linear shallow water, periodic alongshore, open sponge offshore):
//   ∂η/∂t + ∇·(H u) = 0
//   ∂u/∂t = −g∇η − f ẑ×u − r u/H + τ_wind/(ρw H) + G(t) ŷ − (ρo/ρw)·τ_oil/H
//   G(t): alongshore tidal pressure gradient (M2), amplitude set so offshore tidal current ≈ U_tide
// Oil layer (upper, 256² finite volume, reduced-gravity shallow water, Rusanov fluxes, MUSCL (MC limiter), SSP-RK2):
//   ∂h/∂t + ∇·(h u) = 0
//   ∂(h u)/∂t + ∇·(h u u) + ∇(½ g′h²) = C_i (u_s − u)          (interfacial drag, point-implicit)
//   u_s = water velocity + windage × wind + windrows (the surface drift the oil is dragged towards)
// Coast: reflective for both layers; oil moving onshore strands until the shoreline retention capacity is full.
// Bathymetry, coast, tide, drag, capacity are tunable assumptions on a synthetic domain.
import { Rng, subSeed } from '../rng/prng';
import { DEFAULT_WEATHERING } from '../particles/weathering';
import { Budget, G, RHO_WATER, discRelease, volume, type Frame, type MapModel, type ParamSpec } from './grid';
import { DEFAULT_FLOW, FLOW_SPEC, SurfaceFlow } from './flow';
import { OIL_SPEC, filmWarnings, slickRows } from './thinfilm';
import { rasterise, type ForcingRegion, type Polygon, type SlickProfile } from './shapes';

const OMEGA_M2 = (2 * Math.PI) / (12.42 * 3600);
const RHO_AIR = 1.2, CD_AIR = 1.3e-3;

export interface Coast { x0: number; headland: number; headY: number; headW: number; hMin: number; slope: number; hMax: number }
export const DEFAULT_COAST: Coast = { x0: 3200, headland: 2000, headY: 1200, headW: 1400, hMin: 2, slope: 0.005, hMax: 30 };
export const coastX = (c: Coast, y: number) => c.x0 - c.headland * Math.exp(-(((y - c.headY) / c.headW) ** 2));
export const depthAt = (c: Coast, x: number, y: number) => Math.min(c.hMax, c.hMin + c.slope * Math.max(0, coastX(c, y) - x));

/** Linear shallow-water shelf on an n × n C-grid of spacing dx centred on the frame origin. */
export class ShelfWater {
  eta: Float64Array;
  u: Float64Array; // (n + 1) × n, x-faces
  v: Float64Array; // n × n, y-faces (periodic in y: v[j] is the face south of row j)
  H: Float64Array; // cell depth
  land: Uint8Array;
  oilTx: Float64Array; // oil drag on water per cell, m²/s² (accumulated by the oil layer, applied once per step)
  oilTy: Float64Array;
  t = 0;

  constructor(readonly n: number, readonly dx: number, readonly coast: Coast) {
    this.eta = new Float64Array(n * n);
    this.u = new Float64Array((n + 1) * n);
    this.v = new Float64Array(n * n);
    this.H = new Float64Array(n * n);
    this.land = new Uint8Array(n * n);
    this.oilTx = new Float64Array(n * n);
    this.oilTy = new Float64Array(n * n);
    for (let j = 0; j < n; j++)
      for (let i = 0; i < n; i++) {
        const x = (i + 0.5 - n / 2) * dx, y = (j + 0.5 - n / 2) * dx, k = j * n + i;
        this.land[k] = x > coastX(coast, y) ? 1 : 0;
        this.H[k] = depthAt(coast, x, y);
      }
  }

  stableDt(): number {
    return (0.45 * this.dx) / Math.sqrt(2 * G * this.coast.hMax);
  }

  /** Velocity at cell (i, j) centre. */
  cellVel(i: number, j: number): [number, number] {
    const n = this.n;
    return [0.5 * (this.u[j * (n + 1) + i] + this.u[j * (n + 1) + i + 1]), 0.5 * (this.v[j * n + i] + this.v[((j + 1) % n) * n + i])];
  }

  step(dt: number, wind: [number, number], uTide: number, friction: number, lat: number) {
    const { n, dx, eta, u, v, H, land } = this;
    const f = 2 * 7.2921e-5 * Math.sin((lat * Math.PI) / 180);
    const ramp = Math.min(1, this.t / 3600);
    const Href = this.coast.hMax * 0.7;
    const Gt = ramp * uTide * Math.hypot(OMEGA_M2, friction / Href) * Math.cos(OMEGA_M2 * this.t);
    const W = Math.hypot(wind[0], wind[1]);
    const tx = (RHO_AIR * CD_AIR * W * wind[0]) / RHO_WATER, ty = (RHO_AIR * CD_AIR * W * wind[1]) / RHO_WATER;
    const oTx = this.oilTx, oTy = this.oilTy;
    // u (x-faces); i = 0 is the open offshore edge (η = 0 outside), i = n is closed
    for (let j = 0; j < n; j++) {
      const jn = (j + 1 === n ? 0 : j + 1) * n;
      for (let i = 1; i < n; i++) {
        const kl = j * n + i - 1, kr = kl + 1, fu = j * (n + 1) + i;
        if (land[kl] || land[kr]) { u[fu] = 0; continue; }
        const h = 0.5 * (H[kl] + H[kr]);
        const vbar = 0.25 * (v[kl] + v[kr] + v[jn + i - 1] + v[jn + i]);
        const rhs = (-G * (eta[kr] - eta[kl])) / dx + f * vbar + (tx + 0.5 * (oTx[kl] + oTx[kr])) / h;
        u[fu] = (u[fu] + dt * rhs) / (1 + (dt * friction) / h);
      }
    }
    for (let j = 0; j < n; j++) {
      const k = j * n;
      const rhs = (-G * (eta[k] - 0)) / dx;
      u[j * (n + 1)] = land[k] ? 0 : (u[j * (n + 1)] + dt * rhs) / (1 + (dt * friction) / H[k]);
    }
    for (let j = 0; j < n; j++) {
      const jm = j === 0 ? n - 1 : j - 1;
      for (let i = 0; i < n; i++) {
        const ks = jm * n + i, kn = j * n + i;
        if (land[ks] || land[kn]) { v[kn] = 0; continue; }
        const h = 0.5 * (H[ks] + H[kn]);
        const ubar = 0.25 * (u[jm * (n + 1) + i] + u[jm * (n + 1) + i + 1] + u[j * (n + 1) + i] + u[j * (n + 1) + i + 1]);
        const rhs = (-G * (eta[kn] - eta[ks])) / dx - f * ubar + Gt + (ty + 0.5 * (oTy[ks] + oTy[kn])) / h;
        v[kn] = (v[kn] + dt * rhs) / (1 + (dt * friction) / h);
      }
    }
    // continuity; offshore sponge relaxes η towards 0 over the outer 12 cells
    for (let j = 0; j < n; j++) {
      const jm = (j === 0 ? n - 1 : j - 1) * n, jp = (j + 1 === n ? 0 : j + 1) * n;
      for (let i = 0; i < n; i++) {
        const k = j * n + i;
        if (land[k]) continue;
        const Hw = i > 0 ? 0.5 * (H[k - 1] + H[k]) : H[k], He = i < n - 1 ? 0.5 * (H[k] + H[k + 1]) : H[k];
        const Hs = 0.5 * (H[jm + i] + H[k]), Hn = 0.5 * (H[k] + H[jp + i]);
        const div = (He * u[j * (n + 1) + i + 1] - Hw * u[j * (n + 1) + i] + Hn * v[jp + i] - Hs * v[k]) / dx;
        eta[k] -= dt * div;
        if (i < 12) eta[k] *= 1 - (dt / 600) * ((12 - i) / 12) ** 2;
      }
    }
    this.t += dt;
  }
}

/** Reduced-gravity shallow-water oil layer on an nx × ny finite-volume grid; land cells are walls. */
export class OilLayer {
  h: Float64Array;
  qx: Float64Array;
  qy: Float64Array;
  stranded: Float64Array; // m³ held by each land cell
  private h1: Float64Array; private qx1: Float64Array; private qy1: Float64Array;
  private dh: Float64Array; private dqx: Float64Array; private dqy: Float64Array;
  private vx: Float64Array; private vy: Float64Array;
  static readonly DRY = 1e-9;

  constructor(readonly nx: number, readonly ny: number, readonly dx: number, readonly land: Uint8Array) {
    const N = nx * ny;
    this.vx = new Float64Array(N); this.vy = new Float64Array(N);
    this.h = new Float64Array(N); this.qx = new Float64Array(N); this.qy = new Float64Array(N);
    this.h1 = new Float64Array(N); this.qx1 = new Float64Array(N); this.qy1 = new Float64Array(N);
    this.dh = new Float64Array(N); this.dqx = new Float64Array(N); this.dqy = new Float64Array(N);
    this.stranded = new Float64Array(N);
    // water cells with a land neighbour, in row order: the only cells that can strand oil
    const shore: number[] = [];
    for (let j = 0; j < ny; j++)
      for (let i = 0; i < nx; i++) {
        const k = j * nx + i;
        if (!land[k] && ((i + 1 < nx && land[k + 1]) || (i > 0 && land[k - 1]) || (j + 1 < ny && land[k + nx]) || (j > 0 && land[k - nx]))) shore.push(k);
      }
    this.shore = Int32Array.from(shore);
  }
  private shore: Int32Array;

  // active window (inclusive cell range); empty when wi1 < wi0
  private wi0 = 0; private wi1 = -1; private wj0 = 0; private wj1 = -1;

  private window(): boolean {
    const { nx, ny, h } = this;
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
    if (i1 < 0) { this.wi1 = -1; this.wj1 = -1; return false; }
    const M = 6;
    this.wi0 = Math.max(0, i0 - M); this.wi1 = Math.min(nx - 1, i1 + M);
    this.wj0 = Math.max(0, j0 - M); this.wj1 = Math.min(ny - 1, j1 + M);
    return true;
  }

  maxSpeed(gp: number): number {
    let s = 0;
    for (let k = 0; k < this.h.length; k++) {
      const h = this.h[k];
      if (h <= OilLayer.DRY) continue;
      s = Math.max(s, Math.hypot(this.qx[k], this.qy[k]) / h + Math.sqrt(gp * h));
    }
    return s;
  }

  /** Flux divergence of (h, qx, qy) into dh, dqx, dqy (per second). Returns volume flux out of the open edges (m³/s). */
  private rhs(h: Float64Array, qx: Float64Array, qy: Float64Array, gp: number): number {
    const { nx, ny, dx, land, dh, dqx, dqy, wi0, wi1, wj0, wj1 } = this;
    let out = 0;
    const DRY = OilLayer.DRY, { vx, vy } = this;
    // derivatives are written only inside the window; velocities cover the window and the 2 cells the stencils read
    for (let j = wj0; j <= wj1; j++) { const r = j * nx; dh.fill(0, r + wi0, r + wi1 + 1); dqx.fill(0, r + wi0, r + wi1 + 1); dqy.fill(0, r + wi0, r + wi1 + 1); }
    for (let j = Math.max(0, wj0 - 2); j <= Math.min(ny - 1, wj1 + 2); j++)
      for (let i = Math.max(0, wi0 - 2); i <= Math.min(nx - 1, wi1 + 2); i++) { const k = j * nx + i, d = h[k] > DRY; vx[k] = d ? qx[k] / h[k] : 0; vy[k] = d ? qy[k] / h[k] : 0; }
    // monotonized-central limiter: sharper than minmod, so windrow streaks survive advection at full drift speed
    const mm = (a: number, b: number) => (a * b <= 0 ? 0 : Math.sign(a) * Math.min(2 * Math.abs(a), 2 * Math.abs(b), 0.5 * Math.abs(a + b)));
    // one sweep over faces normal to x (xdir) or y
    const sweep = (xdir: boolean) => {
      const ni = xdir ? nx : ny, step = xdir ? 1 : nx;
      const un = xdir ? vx : vy, ut = xdir ? vy : vx, dqn = xdir ? dqx : dqy, dqt = xdir ? dqy : dqx;
      const bA = xdir ? wj0 : wi0, bB = xdir ? wj1 : wi1, aA = xdir ? wi0 : wj0, aB = (xdir ? wi1 : wj1) + 1;
      for (let b = bA; b <= bB; b++) {
        const base = xdir ? b * nx : b;
        for (let a = aA; a <= aB; a++) {
          // face between cells a−1 (L) and a (R); a wall is a land cell, an open edge is outside the grid
          const kL = base + (a - 1) * step, kR = base + a * step;
          // fast path: interior face with water on all four stencil cells (the same arithmetic as the general path)
          if (a >= 2 && a <= ni - 2) {
            const kLL = kL - step, kRR = kR + step;
            if (land[kLL] === 0 && land[kL] === 0 && land[kR] === 0 && land[kRR] === 0) {
              const HLL = h[kLL], HL = h[kL], HR = h[kR], HRR = h[kRR];
              if (HLL === 0 && HL === 0 && HR === 0 && HRR === 0) continue; // dry: every flux is exactly zero
              let d1 = HL - HLL, d2 = HR - HL;
              const hL = Math.max(0, HL + 0.5 * (d1 * d2 <= 0 ? 0 : Math.sign(d1) * Math.min(2 * Math.abs(d1), 2 * Math.abs(d2), 0.5 * Math.abs(d1 + d2))));
              d1 = HR - HL; d2 = HRR - HR;
              const hR = Math.max(0, HR - 0.5 * (d1 * d2 <= 0 ? 0 : Math.sign(d1) * Math.min(2 * Math.abs(d1), 2 * Math.abs(d2), 0.5 * Math.abs(d1 + d2))));
              const NLL = un[kLL], NL = un[kL], NR = un[kR], NRR = un[kRR];
              let uL = 0, uR = 0;
              if (hL > DRY) { d1 = NL - NLL; d2 = NR - NL; uL = NL + 0.5 * (d1 * d2 <= 0 ? 0 : Math.sign(d1) * Math.min(2 * Math.abs(d1), 2 * Math.abs(d2), 0.5 * Math.abs(d1 + d2))); }
              if (hR > DRY) { d1 = NR - NL; d2 = NRR - NR; uR = NR - 0.5 * (d1 * d2 <= 0 ? 0 : Math.sign(d1) * Math.min(2 * Math.abs(d1), 2 * Math.abs(d2), 0.5 * Math.abs(d1 + d2))); }
              const TLL = ut[kLL], TL = ut[kL], TR = ut[kR], TRR = ut[kRR];
              d1 = TL - TLL; d2 = TR - TL;
              const tL = TL + 0.5 * (d1 * d2 <= 0 ? 0 : Math.sign(d1) * Math.min(2 * Math.abs(d1), 2 * Math.abs(d2), 0.5 * Math.abs(d1 + d2)));
              d1 = TR - TL; d2 = TRR - TR;
              const tR = TR - 0.5 * (d1 * d2 <= 0 ? 0 : Math.sign(d1) * Math.min(2 * Math.abs(d1), 2 * Math.abs(d2), 0.5 * Math.abs(d1 + d2)));
              const s = Math.max(Math.abs(uL) + Math.sqrt(gp * hL), Math.abs(uR) + Math.sqrt(gp * hR));
              const FhL = hL * uL, FhR = hR * uR;
              const Fh = 0.5 * (FhL + FhR) - 0.5 * s * (hR - hL);
              const Fn = 0.5 * (FhL * uL + 0.5 * gp * hL * hL + FhR * uR + 0.5 * gp * hR * hR) - 0.5 * s * (FhR - FhL);
              const Ft = 0.5 * (FhL * tL + FhR * tR) - 0.5 * s * (hR * tR - hL * tL);
              dh[kL] -= Fh / dx; dqn[kL] -= Fn / dx; dqt[kL] -= Ft / dx;
              dh[kR] += Fh / dx; dqn[kR] += Fn / dx; dqt[kR] += Ft / dx;
              continue;
            }
          }
          const wetL = a > 0 && !land[kL], wetR = a < ni && !land[kR];
          if (!wetL && !wetR) continue;
          if ((a > 0 && !wetL) || (a < ni && !wetR)) {
            // wall: zero mass flux, pressure only
            const k = wetL ? kL : kR;
            const hw = h[k];
            if (wetL) dqn[kL] -= (0.5 * gp * hw * hw) / dx; else dqn[kR] += (0.5 * gp * hw * hw) / dx;
            continue;
          }
          // neighbour lookup: open edges copy the nearest cell, land mirrors it (normal velocity reversed)
          const kc = wetL ? kL : kR;
          const nb = (m: number, from: number): number => (m < 0 || m >= ni ? from : land[base + m * step] ? -1 - from : base + m * step);
          const iLL = wetL ? nb(a - 2, kL) : kc, iL = wetL ? kL : kc, iR = wetR ? kR : kc, iRR = wetR ? nb(a + 1, kR) : kc;
          const H = (i: number) => h[i < 0 ? -1 - i : i];
          const N = (i: number) => (i < 0 ? -un[-1 - i] : un[i]);
          const T = (i: number) => ut[i < 0 ? -1 - i : i];
          const hL = Math.max(0, H(iL) + 0.5 * mm(H(iL) - H(iLL), H(iR) - H(iL))), hR = Math.max(0, H(iR) - 0.5 * mm(H(iR) - H(iL), H(iRR) - H(iR)));
          const uL = hL > DRY ? N(iL) + 0.5 * mm(N(iL) - N(iLL), N(iR) - N(iL)) : 0, uR = hR > DRY ? N(iR) - 0.5 * mm(N(iR) - N(iL), N(iRR) - N(iR)) : 0;
          const tL = T(iL) + 0.5 * mm(T(iL) - T(iLL), T(iR) - T(iL)), tR = T(iR) - 0.5 * mm(T(iR) - T(iL), T(iRR) - T(iR));
          const s = Math.max(Math.abs(uL) + Math.sqrt(gp * hL), Math.abs(uR) + Math.sqrt(gp * hR));
          const FhL = hL * uL, FhR = hR * uR;
          let Fh = 0.5 * (FhL + FhR) - 0.5 * s * (hR - hL);
          let Fn = 0.5 * (FhL * uL + 0.5 * gp * hL * hL + FhR * uR + 0.5 * gp * hR * hR) - 0.5 * s * (FhR - FhL);
          let Ft = 0.5 * (FhL * tL + FhR * tR) - 0.5 * s * (hR * tR - hL * tL);
          if (a === 0 || a === ni) {
            // open edge: outflow only
            if (a === 0 && Fh > 0) { Fh = 0; Fn = 0.5 * gp * hR * hR; Ft = 0; }
            if (a === ni && Fh < 0) { Fh = 0; Fn = 0.5 * gp * hL * hL; Ft = 0; }
            out += Math.abs(Fh);
          }
          if (wetL) { dh[kL] -= Fh / dx; dqn[kL] -= Fn / dx; dqt[kL] -= Ft / dx; }
          if (wetR) { dh[kR] += Fh / dx; dqn[kR] += Fn / dx; dqt[kR] += Ft / dx; }
        }
      }
    };
    sweep(true);
    sweep(false);
    return out * dx;
  }

  /**
   * Advance by dt (caller keeps dt within the CFL limit): SSP-RK2 fluxes, then drag towards the surface drift (tu, tv)
   * at rate C_i/h, then stranding. Returns volumes that left the grid and stranded.
   */
  step(dt: number, gp: number, Ci: number, tu: Float64Array, tv: Float64Array, capacityM3PerM: number): { left: number; stranded: number } {
    const { h, qx, qy, h1, qx1, qy1, dh, dqx, dqy, land, nx, ny, dx } = this;
    // Window: oil bounding box grown by 6 cells. Stage-1 fluxes reach 2 cells beyond the oil, stage 2 another 2, and the
    // stencils read 2 more, so nothing outside changes. The stage copy h1 is cleared where the previous window was.
    for (let j = this.wj0; j <= this.wj1; j++) { const r = j * nx; h1.fill(0, r + this.wi0, r + this.wi1 + 1); qx1.fill(0, r + this.wi0, r + this.wi1 + 1); qy1.fill(0, r + this.wi0, r + this.wi1 + 1); }
    if (!this.window()) return { left: 0, stranded: 0 };
    const { wi0, wi1, wj0, wj1 } = this;
    let left = this.rhs(h, qx, qy, gp) * dt * 0.5;
    for (let j = wj0; j <= wj1; j++)
      for (let k = j * nx + wi0; k <= j * nx + wi1; k++) { h1[k] = h[k] + dt * dh[k]; qx1[k] = qx[k] + dt * dqx[k]; qy1[k] = qy[k] + dt * dqy[k]; }
    left += this.rhs(h1, qx1, qy1, gp) * dt * 0.5;
    for (let j = wj0; j <= wj1; j++)
    for (let k = j * nx + wi0; k <= j * nx + wi1; k++) {
      if (land[k]) continue;
      let hn = 0.5 * (h[k] + h1[k] + dt * dh[k]);
      let ax = 0.5 * (qx[k] + qx1[k] + dt * dqx[k]), ay = 0.5 * (qy[k] + qy1[k] + dt * dqy[k]);
      if (hn <= OilLayer.DRY) {
        left += Math.min(0, hn) * dx * dx; // clamping a negative cell adds volume: book it against the budget
        hn = Math.max(0, hn); ax = hn * tu[k]; ay = hn * tv[k];
      }
      else {
        // drag: d(hu)/dt = C_i (h u_s − h u)/h, point-implicit
        const r = (dt * Ci) / hn;
        ax = (ax + r * hn * tu[k]) / (1 + r);
        ay = (ay + r * hn * tv[k]) / (1 + r);
      }
      h[k] = hn; qx[k] = ax; qy[k] = ay;
    }
    // stranding: onshore volume flux into each adjacent land cell, up to its capacity per metre of shore
    let str = 0;
    const cap = capacityM3PerM * dx;
    for (const k of this.shore) {
        const i = k % nx, j = (k - i) / nx;
        if (h[k] <= OilLayer.DRY) continue;
        const u = qx[k] / h[k], v = qy[k] / h[k];
        const tryShore = (kk: number, un: number) => {
          if (kk < 0 || !land[kk] || un <= 0 || this.stranded[kk] >= cap) return;
          const dV = Math.min(h[k] * un * dx * dt, cap - this.stranded[kk], h[k] * dx * dx);
          this.stranded[kk] += dV;
          const keep = (h[k] * dx * dx - dV) / (h[k] * dx * dx);
          h[k] *= keep; qx[k] *= keep; qy[k] *= keep;
          str += dV;
        };
        tryShore(i + 1 < nx ? k + 1 : -1, u);
        tryShore(i > 0 ? k - 1 : -1, -u);
        tryShore(j + 1 < ny ? k + nx : -1, v);
        tryShore(j > 0 ? k - nx : -1, -v);
    }
    return { left, stranded: str };
  }
}

export const TWO_LAYER_DEFAULTS = {
  ...DEFAULT_FLOW,
  // windrows twice the thin-film default: the fixed 50 m coastal grid, advected at full drift speed, smears streaks more
  driftU: 0, driftV: 0, eddySpeed: 0.02, eddyScaleM: 2000, windDirDeg: 70, rowSpeedFrac: 0.012,
  releaseM3: 350, radiusM: 600, oilDensity: 860, sst: 20,
  dispersionRate: DEFAULT_WEATHERING.dispersionRate, evapMax: DEFAULT_WEATHERING.evapMax, sheenUm: 1, seed: 7,
  releaseX: -2600, releaseY: -2200,
  uTide: 0.25, friction: 1e-3, spreadC: 2e-5, shoreCapacity: 0.02,
};

export const SHELF_SPEC: ParamSpec[] = [
  { key: 'uTide', label: 'Offshore tidal current (M2)', unit: 'm/s', min: 0, max: 1, step: 0.01, assumption: true },
  { key: 'friction', label: 'Bottom friction r', unit: 'm/s', min: 1e-4, max: 5e-3, step: 1e-4, assumption: true },
  { key: 'spreadC', label: 'Interfacial drag velocity C_i', unit: 'm/s', min: 2e-6, max: 2e-4, step: 2e-6, assumption: true },
  { key: 'shoreCapacity', label: 'Shoreline retention capacity', unit: 'm³/m', min: 0, max: 0.2, step: 0.005, assumption: true },
  { key: 'releaseX', label: 'Release east of centre', unit: 'm', min: -5500, max: 1500, step: 100, reset: true },
  { key: 'releaseY', label: 'Release north of centre', unit: 'm', min: -5500, max: 5500, step: 100, reset: true },
];

/** Shared machinery of the two-layer and master models: coast, water, oil layer, surface-drift targets. */
export class CoastalCore {
  readonly frame: Frame = { nx: 256, ny: 256, dx: 50, lat0: 25, lon0: -60, ox: 0, oy: 0 };
  water!: ShelfWater;
  oil!: OilLayer;
  flow!: SurfaceFlow;
  budget!: Budget;
  t = 0;
  private tu = new Float64Array(256 * 256);
  private tv = new Float64Array(256 * 256);
  private anomU = new Float64Array(256 * 256);
  private anomV = new Float64Array(256 * 256);
  private anomT = -Infinity;
  private anomVersion = -1;
  private curW = new Float64Array(256 * 256); // current-region weight and weighted vector per oil cell
  private curU = new Float64Array(256 * 256);
  private curV = new Float64Array(256 * 256);
  private cellU = new Float64Array(128 * 128); // water cell-centre velocity, scratch for targets()
  private cellV = new Float64Array(128 * 128);
  private colI0 = new Int32Array(256);
  private colFx = new Float64Array(256);
  regions: ForcingRegion[] = [];
  lastDt = 0; substeps = 0; waterSubsteps = 0;

  constructor(readonly params: typeof TWO_LAYER_DEFAULTS & Record<string, number>) {}

  reset() {
    const p = this.params, f = this.frame;
    this.t = 0;
    this.anomT = -Infinity;
    this.water = new ShelfWater(128, 100, DEFAULT_COAST);
    const land = new Uint8Array(f.nx * f.ny);
    for (let j = 0; j < f.ny; j++)
      for (let i = 0; i < f.nx; i++) land[j * f.nx + i] = (i + 0.5 - f.nx / 2) * f.dx > coastX(DEFAULT_COAST, (j + 0.5 - f.ny / 2) * f.dx) ? 1 : 0;
    this.oil = new OilLayer(f.nx, f.ny, f.dx, land);
    this.flow = new SurfaceFlow(p, subSeed(p.seed, 1));
    this.flow.setRegions(this.regions);
    const rng = new Rng(subSeed(p.seed, 2));
    discRelease(f, this.oil.h, p.releaseM3, p.radiusM, 0.25, () => rng.next(), p.releaseX, p.releaseY);
    for (let k = 0; k < land.length; k++) if (land[k]) this.oil.h[k] = 0;
    const v0 = volume(this.oil.h, f.dx);
    for (let k = 0; k < land.length; k++) this.oil.h[k] *= p.releaseM3 / v0;
    this.budget = new Budget(p.releaseM3);
  }

  gPrime() { return (G * (RHO_WATER - this.params.oilDensity)) / RHO_WATER; }

  /** Surface drift at oil cell centres: water (bilinear from the coarse grid) + windage × wind + windrows/eddies. */
  targets() {
    const { nx, ny, dx } = this.frame, w = this.water, p = this.params;
    if (Math.abs(this.t - this.anomT) >= 600 || this.anomVersion !== this.flow.version) {
      const o = { u: 0, v: 0 }, c = { w: 0, u: 0, v: 0 };
      for (let j = 0; j < ny; j++)
        for (let i = 0; i < nx; i++) {
          const x = (i + 0.5 - nx / 2) * dx, y = (j + 0.5 - ny / 2) * dx, k = j * nx + i;
          o.u = 0; o.v = 0;
          this.flow.anomaly(x, y, this.t, o);
          this.anomU[k] = o.u; this.anomV[k] = o.v;
          this.flow.currentOverride(x, y, c);
          this.curW[k] = c.w; this.curU[k] = c.u; this.curV[k] = c.v;
        }
      this.anomT = this.t;
      this.anomVersion = this.flow.version;
    }
    const [mu, mv] = this.flow.mean();
    const [lax, lay, lW] = this.flow.alongWind(), lag = this.flow.p.sheenLagFrac * lW, lagRef = this.flow.p.sheenLagRefUm * 1e-6;
    const r = w.dx / dx, wn = w.n, WU = w.u, WV = w.v;
    // water cell-centre velocity, as cellVel(i, j) = [½(u[i] + u[i+1]), ½(v[j] + v[j+1])], once per water cell
    const CU = this.cellU, CV = this.cellV;
    for (let j = 0; j < wn; j++)
      for (let i = 0; i < wn; i++) {
        CU[j * wn + i] = 0.5 * (WU[j * (wn + 1) + i] + WU[j * (wn + 1) + i + 1]);
        CV[j * wn + i] = 0.5 * (WV[j * wn + i] + WV[((j + 1) % wn) * wn + i]);
      }
    // bilinear weights per oil column, the same for every row
    const I0 = this.colI0, FX = this.colFx;
    for (let i = 0; i < nx; i++) {
      const gx = Math.min(wn - 1.001, Math.max(0, (i + 0.5) / r - 0.5));
      I0[i] = Math.floor(gx); FX[i] = gx - I0[i];
    }
    const h = this.oil.h, tu = this.tu, tv = this.tv, curW = this.curW, curU = this.curU, curV = this.curV, anomU = this.anomU, anomV = this.anomV;
    for (let j = 0; j < ny; j++) {
      const gy = Math.min(wn - 1.001, Math.max(0, (j + 0.5) / r - 0.5)), j0 = Math.floor(gy), fy = gy - j0;
      const r0 = j0 * wn, r1 = (j0 + 1) * wn;
      for (let i = 0; i < nx; i++) {
        const k = j * nx + i;
        const i0 = I0[i], fx = FX[i];
        // current regions replace the water velocity in the surface drift (the water layer itself is not forced by them)
        const wu = (1 - fy) * ((1 - fx) * CU[r0 + i0] + fx * CU[r0 + i0 + 1]) + fy * ((1 - fx) * CU[r1 + i0] + fx * CU[r1 + i0 + 1]);
        const wv = (1 - fy) * ((1 - fx) * CV[r0 + i0] + fx * CV[r0 + i0 + 1]) + fy * ((1 - fx) * CV[r1 + i0] + fx * CV[r1 + i0 + 1]);
        // sheen lag: thin oil is dragged towards a slower downwind drift than thick oil
        const lagK = (lag * lagRef) / (h[k] + lagRef);
        tu[k] = (1 - curW[k]) * wu + curU[k] + mu + anomU[k] - lagK * lax;
        tv[k] = (1 - curW[k]) * wv + curV[k] + mv + anomV[k] - lagK * lay;
      }
    }
    void p;
  }

  /** Oil drag on the water, averaged onto the coarse cells (momentum per unit water column, m²/s²). */
  private dragOnWater() {
    const { nx, ny } = this.frame, w = this.water, r = Math.round(w.dx / this.frame.dx), o = this.oil, Ci = this.params.spreadC;
    const rho = this.params.oilDensity / RHO_WATER;
    w.oilTx.fill(0); w.oilTy.fill(0);
    for (let j = 0; j < ny; j++)
      for (let i = 0; i < nx; i++) {
        const k = j * nx + i, h = o.h[k];
        if (h <= OilLayer.DRY) continue;
        const kw = Math.floor(j / r) * w.n + Math.floor(i / r);
        w.oilTx[kw] += (rho * Ci * (o.qx[k] / h - this.tu[k])) / (r * r);
        w.oilTy[kw] += (rho * Ci * (o.qy[k] / h - this.tv[k])) / (r * r);
      }
  }

  /** Advance water and oil by dt; `after` runs after each oil sub-step (master adds film fluxes there). Returns sub-dt list length. */
  advance(dt: number, dispersionRate: (t: number) => number, after?: (sub: number) => void) {
    const p = this.params, f = this.frame, gp = this.gPrime();
    let done = 0;
    this.substeps = 0; this.waterSubsteps = 0;
    while (done < dt - 1e-9) {
      this.targets();
      const sub = Math.min(dt - done, 60, (0.3 * f.dx) / Math.max(this.oil.maxSpeed(gp), 1e-6));
      // water sub-cycles to the same time
      const nW = Math.ceil(sub / this.water.stableDt());
      this.dragOnWater();
      for (let s = 0; s < nW; s++) this.water.step(sub / nW, this.flow.wind(), p.uTide, p.friction, f.lat0);
      this.waterSubsteps += nW;
      const r = this.oil.step(sub, gp, p.spreadC, this.tu, this.tv, p.shoreCapacity);
      this.budget.left += r.left;
      this.budget.stranded += r.stranded;
      after?.(sub);
      this.t += sub;
      const vol = volume(this.oil.h, f.dx);
      const factor = this.budget.weather(vol, this.t, sub, p.sst, dispersionRate(this.t), { ...DEFAULT_WEATHERING, evapMax: p.evapMax });
      if (factor !== 1) for (let k = 0; k < this.oil.h.length; k++) { this.oil.h[k] *= factor; this.oil.qx[k] *= factor; this.oil.qy[k] *= factor; }
      done += sub;
      this.lastDt = sub;
      this.substeps++;
    }
  }

  /** Surface drift the oil is dragged towards (water + windage + windrows + regions), every 16 oil cells. */
  arrows(): number[] {
    const { nx, ny, dx } = this.frame, a: number[] = [];
    this.targets();
    for (let j = 8; j < ny; j += 16)
      for (let i = 8; i < nx; i += 16) {
        const k = j * nx + i;
        if (this.oil.land[k]) continue;
        a.push((i + 0.5 - nx / 2) * dx, (j + 0.5 - ny / 2) * dx, this.tu[k], this.tv[k]);
      }
    return a;
  }

  setRegions(regions: ForcingRegion[]) {
    this.regions = regions;
    this.flow.setRegions(regions);
  }

  addOil(poly: Polygon, volumeM3: number, profile: SlickProfile, replace: boolean) {
    const f = this.frame;
    if (replace) {
      this.reset(); // builds a new oil layer: take the reference after this
      this.oil.h.fill(0);
      this.budget = new Budget(0);
    }
    const o = this.oil;
    this.targets();
    const { share, lost } = rasterise(f, poly, profile, o.land);
    const placed = volumeM3 * (1 - lost);
    for (let k = 0; k < share.length; k++) {
      const dh = (share[k] * placed) / (f.dx * f.dx);
      if (dh <= 0) continue;
      o.h[k] += dh; o.qx[k] += dh * this.tu[k]; o.qy[k] += dh * this.tv[k]; // new oil starts at the surface drift
    }
    this.budget.add(placed);
    return { placed, lost };
  }

  rows(): [string, string][] {
    let shore = 0, umax = 0;
    for (const s of this.oil.stranded) if (s > 0) shore++;
    for (let j = 0; j < this.water.n; j++) for (let i = 0; i < this.water.n; i++) umax = Math.max(umax, Math.hypot(...this.water.cellVel(i, j)));
    return [
      ['tide phase', `${(((OMEGA_M2 * this.t) / (2 * Math.PI)) % 1 * 12.42).toFixed(1)} h of 12.42 h M2 cycle`],
      ['max water speed', `${umax.toFixed(2)} m/s`],
      ['oiled shoreline', `${((shore * this.frame.dx) / 1000).toFixed(2)} km of cells holding stranded oil`],
      ['sub-steps', `oil ${this.substeps} (dt ${this.lastDt.toFixed(1)} s), water ${this.waterSubsteps} last frame`],
    ];
  }
}

export class TwoLayerModel implements MapModel {
  readonly kind = 'map';
  readonly id = 'twolayer';
  readonly title = 'Two-layer shallow water';
  readonly info = {
    calculates: 'A coastal water layer (sea level and depth-averaged current from tide, wind, Coriolis and friction around a headland) and an oil layer on top with its own thickness and momentum: reduced-gravity spreading, drag towards the surface drift, oil-front movement and stranding on the shore.',
    assumptions: 'Coast, bathymetry, tide amplitude, friction, interfacial drag and shoreline capacity are synthetic, tunable assumptions. Water is linear (no eddy shedding); oil drag on the water is included but tiny.',
    master: 'Kept whole: it is the master\'s transport core (water layer, oil momentum and fronts, coast, stranding).',
  };
  readonly spec: ParamSpec[] = [...OIL_SPEC, ...SHELF_SPEC, ...FLOW_SPEC];
  readonly params = { ...TWO_LAYER_DEFAULTS } as typeof TWO_LAYER_DEFAULTS & Record<string, number>;
  readonly speeds = [60, 300, 900, 1800];
  readonly core = new CoastalCore(this.params);
  readonly supportsRegions = true;

  get frame() { return this.core.frame; }
  get t() { return this.core.t; }
  set t(v: number) { this.core.t = v; }
  get rhoOil() { return this.params.oilDensity; }
  get sheenM() { return this.params.sheenUm * 1e-6; }

  constructor(params: Partial<typeof TWO_LAYER_DEFAULTS> = {}) {
    Object.assign(this.params, params);
    this.reset();
  }

  reset() { this.core.reset(); }
  setRegions(r: ForcingRegion[]) { this.core.setRegions(r); }
  addOil(poly: Polygon, v: number, profile: SlickProfile, replace: boolean) { return this.core.addOil(poly, v, profile, replace); }
  step(dt: number) { this.core.advance(dt, () => this.params.dispersionRate); }
  thickness() { return this.core.oil.h; }
  land() { return this.core.oil.land; }
  arrows() { return this.core.arrows(); }

  stats(): [string, string][] {
    return [...slickRows(this), ...this.core.rows(), ...this.core.budget.rows(volume(this.core.oil.h, this.frame.dx))];
  }

  warnings(): string[] {
    return filmWarnings(this, this.core.budget, volume(this.core.oil.h, this.frame.dx), Infinity);
  }
}
