// The slick model as WebGPU compute kernels: a line-for-line port of the CPU engine's step (src/engine/film:
// CoastalCore.advance, OilLayer, ShelfWater, SurfaceFlow, FilmFlux and the master's film/weathering hooks), in 32-bit
// floats. Every field lives in one f32 buffer (F) and one u32 buffer (M) at fixed offsets, so no kernel needs more
// than three bindings. No atomics and every sum runs in a fixed order, so a run is bit-identical on the same GPU.

export const N = 256;
export const WN = 128;
const N2 = N * N, NF = (N + 1) * N, W2 = WN * WN, WUN = (WN + 1) * WN;
export const MAX_MODES = 64, MAX_STROKES = 64, MAX_PTS = 8192;

/** f32 fields: name → length, laid out in this order. */
const F_FIELDS: [string, number][] = [
  ['H', N2], ['QX', N2], ['QY', N2], ['H1', N2], ['QX1', N2], ['QY1', N2],
  ['TU', N2], ['TV', N2],
  ['ANU', N2], ['ANV', N2], ['CW', N2], ['CU', N2], ['CV', N2],
  ['STR', N2], ['BEF', N2], ['RAT', N2], ['ADD', N2], ['SCL', N2],
  // budget terms gathered per cell during a step and summed once at its end: left the grid, stranded, held by booms
  ['LACC', N2], ['SACC', N2], ['HACC', N2], ['EACC', 4 * N],
  ['REQ', 4 * N2],
  ['FXH', NF], ['FXN', NF], ['FXT', NF], ['FYH', NF], ['FYN', NF], ['FYT', NF], ['FFX', NF], ['FFY', NF],
  ['ETA', W2], ['WU', WUN], ['WV', W2], ['WH', W2], ['OTX', W2], ['OTY', W2],
  ['STATS', 64], ['DRIFT', 512],
  ['MODES', 6 * MAX_MODES], ['STK', 12 * MAX_STROKES], ['PTS', 2 * MAX_PTS],
];
const M_FIELDS: [string, number][] = [['LAND', N2], ['SHORE', N2], ['WLAND', W2], ['NEWL', N2], ['ROWH', N], ['HASH', 4]];

function layout(fields: [string, number][]) {
  const at: Record<string, number> = {};
  let o = 0;
  for (const [name, len] of fields) { at[name] = o; o += len; }
  return { at, size: o };
}
export const FL = layout(F_FIELDS);
export const ML = layout(M_FIELDS);

/** Slots of the STATS block. */
export const S = { FLOAT: 0, MAXS: 1, EVAP: 2, DISP: 3, STRAND: 4, LEFT: 5, HELD: 6, EVAPF: 7, FACTOR: 8, TMP: 9 } as const;

/** Uniform record: 48 floats/uints, one per dispatch group, at 256-byte aligned dynamic offsets. */
export const P_FIELDS = [
  'dt', 'gp', 'ci', 'cap', 't', 'gt', 'fcor', 'fric', 'tx', 'ty', 'wx', 'wy', 'mu', 'mv', 'lax', 'lay',
  'lag', 'lagRef', 'windage', 'wspeed', 'kh', 'ka', 'lens', 'n4dx3', 'iht', 'rowSp', 'rowSpd', 'rowJet', 'rho', 'a0', 'a1', 'a2',
] as const;
export const P_UINTS = ['u0', 'u1', 'nModes', 'nStrokes'] as const;
export const P_BYTES = 256;

const consts = [...Object.entries(FL.at).map(([k, v]) => `const O_${k}: i32 = ${v};`), ...Object.entries(ML.at).map(([k, v]) => `const M_${k}: i32 = ${v};`),
  ...Object.entries(S).map(([k, v]) => `const S_${k}: i32 = ${v};`)].join('\n');

export const WGSL = /* wgsl */ `
${consts}
const N: i32 = ${N};
const WN: i32 = ${WN};
const DX: f32 = 50.0;
const WDX: f32 = 100.0;
const DRY: f32 = 1e-9;
const G: f32 = 9.81;
const TAU: f32 = 6.283185307179586;

struct P {
  ${P_FIELDS.map((f) => `${f}: f32,`).join(' ')}
  ${P_UINTS.map((f) => `${f}: u32,`).join(' ')}
}

@group(0) @binding(0) var<storage, read_write> F: array<f32>;
@group(0) @binding(1) var<storage, read_write> M: array<u32>;
@group(0) @binding(2) var<uniform> p: P;

fn land(k: i32) -> bool { return M[M_LAND + k] != 0u; }

/* ---------------------------------------------------------------- water layer (ShelfWater.step) */

fn waterU(g: i32, dt: f32) {
  let j = g / WN; let i = g % WN; let n = WN;
  if (i == 0) {
    let k = j * n; let fu = j * (n + 1);
    let rhs = -G * (F[O_ETA + k] - 0.0) / WDX;
    if (M[M_WLAND + k] != 0u) { F[O_WU + fu] = 0.0; }
    else { F[O_WU + fu] = (F[O_WU + fu] + dt * rhs) / (1.0 + (dt * p.fric) / F[O_WH + k]); }
    return;
  }
  let jn = select(j + 1, 0, j + 1 == n) * n;
  let kl = j * n + i - 1; let kr = kl + 1; let fu = j * (n + 1) + i;
  if (M[M_WLAND + kl] != 0u || M[M_WLAND + kr] != 0u) { F[O_WU + fu] = 0.0; return; }
  let h = 0.5 * (F[O_WH + kl] + F[O_WH + kr]);
  let vbar = 0.25 * (F[O_WV + kl] + F[O_WV + kr] + F[O_WV + jn + i - 1] + F[O_WV + jn + i]);
  let rhs = (-G * (F[O_ETA + kr] - F[O_ETA + kl])) / WDX + p.fcor * vbar + (p.tx + 0.5 * (F[O_OTX + kl] + F[O_OTX + kr])) / h;
  F[O_WU + fu] = (F[O_WU + fu] + dt * rhs) / (1.0 + (dt * p.fric) / h);
}

fn waterV(g: i32, dt: f32, gt: f32) {
  let j = g / WN; let i = g % WN; let n = WN;
  let jm = select(j - 1, n - 1, j == 0);
  let ks = jm * n + i; let kn = j * n + i;
  if (M[M_WLAND + ks] != 0u || M[M_WLAND + kn] != 0u) { F[O_WV + kn] = 0.0; return; }
  let h = 0.5 * (F[O_WH + ks] + F[O_WH + kn]);
  let ubar = 0.25 * (F[O_WU + jm * (n + 1) + i] + F[O_WU + jm * (n + 1) + i + 1] + F[O_WU + j * (n + 1) + i] + F[O_WU + j * (n + 1) + i + 1]);
  let rhs = (-G * (F[O_ETA + kn] - F[O_ETA + ks])) / WDX - p.fcor * ubar + gt + (p.ty + 0.5 * (F[O_OTY + ks] + F[O_OTY + kn])) / h;
  F[O_WV + kn] = (F[O_WV + kn] + dt * rhs) / (1.0 + (dt * p.fric) / h);
}

fn waterEta(g: i32, dt: f32) {
  let j = g / WN; let i = g % WN; let n = WN;
  let k = j * n + i;
  if (M[M_WLAND + k] != 0u) { return; }
  let jm = select(j - 1, n - 1, j == 0) * n; let jp = select(j + 1, 0, j + 1 == n) * n;
  let Hk = F[O_WH + k];
  let Hw = select(Hk, 0.5 * (F[O_WH + k - 1] + Hk), i > 0);
  let He = select(Hk, 0.5 * (Hk + F[O_WH + k + 1]), i < n - 1);
  let Hs = 0.5 * (F[O_WH + jm + i] + Hk); let Hn = 0.5 * (Hk + F[O_WH + jp + i]);
  let div = (He * F[O_WU + j * (n + 1) + i + 1] - Hw * F[O_WU + j * (n + 1) + i] + Hn * F[O_WV + jp + i] - Hs * F[O_WV + k]) / WDX;
  var e = F[O_ETA + k] - dt * div;
  if (i < 12) { let q = f32(12 - i) / 12.0; e *= 1.0 - (dt / 600.0) * q * q; }
  F[O_ETA + k] = e;
}

/*
 * Every water sub-step of one oil sub-step in a single dispatch: one workgroup, barriers between the u, v and eta
 * sweeps (each reads what the one before wrote). u1 sub-steps of dt from water time a0; a2 is the tidal pressure
 * gradient's amplitude (ShelfWater: ramped in over the first hour, M2 period).
 */
override WATER_WG: u32 = 256u;
@compute @workgroup_size(WATER_WG) fn water(@builtin(local_invocation_index) t: u32) {
  let omega = TAU / (12.42 * 3600.0);
  for (var w = 0u; w < p.u1; w++) {
    let wt = p.a0 + f32(w) * p.dt;
    let gt = min(1.0, wt / 3600.0) * p.a2 * cos(omega * wt);
    for (var g = i32(t); g < WN * WN; g += i32(WATER_WG)) { waterU(g, p.dt); }
    storageBarrier(); workgroupBarrier();
    for (var g = i32(t); g < WN * WN; g += i32(WATER_WG)) { waterV(g, p.dt, gt); }
    storageBarrier(); workgroupBarrier();
    for (var g = i32(t); g < WN * WN; g += i32(WATER_WG)) { waterEta(g, p.dt); }
    storageBarrier(); workgroupBarrier();
  }
}

/* Oil drag on the water, averaged onto the coarse cells (CoastalCore.dragOnWater), in the CPU's cell order. */
@compute @workgroup_size(256) fn drag(@builtin(global_invocation_id) id: vec3<u32>) {
  let g = i32(id.x);
  if (g >= WN * WN) { return; }
  let jw = g / WN; let iw = g % WN;
  var sx = 0.0; var sy = 0.0;
  for (var dj = 0; dj < 2; dj++) {
    for (var di = 0; di < 2; di++) {
      let k = (2 * jw + dj) * N + 2 * iw + di; let h = F[O_H + k];
      if (h <= DRY) { continue; }
      sx += (p.rho * p.ci * (F[O_QX + k] / h - F[O_TU + k])) / 4.0;
      sy += (p.rho * p.ci * (F[O_QY + k] / h - F[O_TV + k])) / 4.0;
    }
  }
  F[O_OTX + g] = sx; F[O_OTY + g] = sy;
}

/* ---------------------------------------------------------------- surface flow (SurfaceFlow) */

/* A drawn stroke at (px, py): weight and unit direction (shapes.ts strokeAt). */
fn strokeAt(s: i32, px: f32, py: f32) -> vec3<f32> {
  let b = O_STK + s * 12;
  let W = max(F[b + 1], 1.0);
  if (px < F[b + 3] - W || px > F[b + 4] + W || py < F[b + 5] - W || py > F[b + 6] + W) { return vec3<f32>(0.0); }
  let p0 = i32(F[b + 7]); let cnt = i32(F[b + 8]);
  var dMin = 3.0e38; var tx = 0.0; var ty = 0.0;
  for (var q = 0; q + 1 < cnt; q++) {
    let ax = F[O_PTS + 2 * (p0 + q)]; let ay = F[O_PTS + 2 * (p0 + q) + 1];
    let bx = F[O_PTS + 2 * (p0 + q + 1)] - ax; let by = F[O_PTS + 2 * (p0 + q + 1) + 1] - ay;
    let L2 = bx * bx + by * by;
    if (L2 <= 0.0) { continue; }
    let t = clamp(((px - ax) * bx + (py - ay) * by) / L2, 0.0, 1.0);
    let d = length(vec2<f32>(px - ax - t * bx, py - ay - t * by));
    if (d >= W) { continue; }
    let k = (1.0 - d / W) * (1.0 - d / W); let L = sqrt(L2);
    tx += (k * bx) / L; ty += (k * by) / L;
    dMin = min(dMin, d);
  }
  let n = length(vec2<f32>(tx, ty));
  if (!(n > 0.0)) { return vec3<f32>(0.0); }
  let q = clamp((dMin - W / 2.0) / (W / 2.0), 0.0, 1.0);
  return vec3<f32>(1.0 - q * q * (3.0 - 2.0 * q), tx / n, ty / n);
}

fn pcg(v: u32) -> u32 {
  let s = v * 747796405u + 2891336453u;
  let w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
  return (w >> 22u) ^ w;
}
/* Per-line hash in [0, 1): the CPU's sin-hash has no f32 equivalent, so an integer hash plays the same role. */
fn lineHash(line: i32, k: i32) -> f32 { return f32(pcg(bitcast<u32>(line) * 1664525u + u32(k) * 1013904223u + 12345u) >> 8u) / 16777216.0; }

/* Eddies, windrows and wind-arrow windage (SurfaceFlow.anomaly), and the current-arrow override (currentOverride). */
@compute @workgroup_size(256) fn anomaly(@builtin(global_invocation_id) id: vec3<u32>) {
  let k = i32(id.x);
  if (k >= N * N) { return; }
  let x = (f32(k % N) + 0.5 - f32(N / 2)) * DX; let y = (f32(k / N) + 0.5 - f32(N / 2)) * DX; let t = p.t;
  var u = 0.0; var v = 0.0;
  for (var m = 0; m < i32(p.nModes); m++) {
    let b = O_MODES + 6 * m;
    let s = sin(F[b] * x + F[b + 1] * y + F[b + 2] + F[b + 3] * t);
    u -= F[b + 4] * s; v += F[b + 5] * s;
  }
  var W = p.wspeed; var ax = p.lax; var ay = p.lay;
  if (p.nStrokes > 0u) {
    var lx = p.wx; var ly = p.wy;
    for (var s = 0; s < i32(p.nStrokes); s++) {
      let b = O_STK + s * 12;
      if (F[b] != 0.0) { continue; }
      let r = strokeAt(s, x, y);
      if (r.x <= 0.0) { continue; }
      lx += r.x * (F[b + 2] * r.y - lx); ly += r.x * (F[b + 2] * r.z - ly);
    }
    u += p.windage * (lx - p.wx); v += p.windage * (ly - p.wy);
    W = length(vec2<f32>(lx, ly));
    ax = select(0.0, lx / W, W > 0.0); ay = select(1.0, ly / W, W > 0.0);
  }
  if (W > 2.0 && p.rowSpd > 0.0) {
    let onset = min(1.0, (W - 2.0) / 2.0);
    let sp = array<f32, 3>(1.0, 0.62, 1.47); let an = array<f32, 3>(0.0, 0.21, -0.16); let wt = array<f32, 3>(0.6, 0.45, 0.5);
    for (var fam = 0; fam < 3; fam++) {
      let lam = p.rowSp * sp[fam]; let ca = cos(an[fam]); let sa = sin(an[fam]);
      let bx = ax * ca + ay * sa; let by = -ax * sa + ay * ca;
      let s = x * bx + y * by;
      let n = x * by - y * bx + 0.25 * lam * sin((TAU * s) / (4.0 * lam) + t / 1800.0 + f32(fam));
      let line = i32(floor(n / lam + 0.5));
      let h0 = lineHash(line, fam); let h1 = lineHash(line, fam + 7); let h2 = lineHash(line, fam + 3);
      let segL = lam * (3.0 + 5.0 * h0);
      let env = max(0.0, sin((TAU * s) / segL + TAU * h1 + t / (1200.0 * (1.0 + h2))));
      let w = onset * wt[fam] * env * env;
      if (w <= 0.0) { continue; }
      let c = -w * p.rowSpd * W * sin((TAU * n) / lam);
      let jet = w * p.rowJet * W * cos((TAU * n) / lam);
      u += c * by + jet * bx; v += -c * bx + jet * by;
    }
  }
  var cw = 0.0; var cu = 0.0; var cv = 0.0;
  for (var s = 0; s < i32(p.nStrokes); s++) {
    let b = O_STK + s * 12;
    if (F[b] != 1.0) { continue; }
    let r = strokeAt(s, x, y);
    if (r.x <= 0.0) { continue; }
    cu = (1.0 - r.x) * cu + r.x * F[b + 2] * r.y;
    cv = (1.0 - r.x) * cv + r.x * F[b + 2] * r.z;
    cw = 1.0 - (1.0 - r.x) * (1.0 - cw);
  }
  F[O_ANU + k] = u; F[O_ANV + k] = v; F[O_CW + k] = cw; F[O_CU + k] = cu; F[O_CV + k] = cv;
}

/* Surface drift the oil is dragged towards (CoastalCore.targets). */
@compute @workgroup_size(256) fn targets(@builtin(global_invocation_id) id: vec3<u32>) {
  let k = i32(id.x);
  if (k >= N * N) { return; }
  let i = k % N; let j = k / N;
  let gx = min(f32(WN) - 1.001, max(0.0, (f32(i) + 0.5) / 2.0 - 0.5)); let i0 = i32(floor(gx)); let fx = gx - f32(i0);
  let gy = min(f32(WN) - 1.001, max(0.0, (f32(j) + 0.5) / 2.0 - 0.5)); let j0 = i32(floor(gy)); let fy = gy - f32(j0);
  let wu = (1.0 - fy) * ((1.0 - fx) * cu(i0, j0) + fx * cu(i0 + 1, j0)) + fy * ((1.0 - fx) * cu(i0, j0 + 1) + fx * cu(i0 + 1, j0 + 1));
  let wv = (1.0 - fy) * ((1.0 - fx) * cv(i0, j0) + fx * cv(i0 + 1, j0)) + fy * ((1.0 - fx) * cv(i0, j0 + 1) + fx * cv(i0 + 1, j0 + 1));
  let lagK = (p.lag * p.lagRef) / (F[O_H + k] + p.lagRef);
  F[O_TU + k] = (1.0 - F[O_CW + k]) * wu + F[O_CU + k] + p.mu + F[O_ANU + k] - lagK * p.lax;
  F[O_TV + k] = (1.0 - F[O_CW + k]) * wv + F[O_CV + k] + p.mv + F[O_ANV + k] - lagK * p.lay;
}
fn cu(i: i32, j: i32) -> f32 { return 0.5 * (F[O_WU + j * (WN + 1) + i] + F[O_WU + j * (WN + 1) + i + 1]); }
fn cv(i: i32, j: i32) -> f32 { return 0.5 * (F[O_WV + j * WN + i] + F[O_WV + ((j + 1) % WN) * WN + i]); }

/* New oil at the surface drift (CoastalCore.addOil): ADD holds the thickness to add per cell. */
@compute @workgroup_size(256) fn add_oil(@builtin(global_invocation_id) id: vec3<u32>) {
  let k = i32(id.x);
  if (k >= N * N) { return; }
  let dh = F[O_ADD + k];
  if (dh <= 0.0) { return; }
  F[O_H + k] += dh; F[O_QX + k] += dh * F[O_TU + k]; F[O_QY + k] += dh * F[O_TV + k];
}

/* ---------------------------------------------------------------- oil layer (OilLayer.step) */

fn oh(which: u32) -> i32 { return select(O_H, O_H1, which == 1u); }
/* Velocity component (x when xc) of cell k in state "which" (0: h, 1: the RK stage), zero where dry. */
fn velo(which: u32, k: i32, xc: bool) -> f32 {
  let h = F[oh(which) + k];
  let q = F[select(select(O_QY, O_QY1, which == 1u), select(O_QX, O_QX1, which == 1u), xc) + k];
  return select(0.0, q / h, h > DRY);
}

fn mm(a: f32, b: f32) -> f32 {
  if (a * b <= 0.0) { return 0.0; }
  return sign(a) * min(min(2.0 * abs(a), 2.0 * abs(b)), 0.5 * abs(a + b));
}

/* A neighbour along the sweep: open edges copy the nearest cell, land mirrors it (flag 1: normal velocity reversed). */
fn nb(m: i32, near: i32, base: i32, step: i32) -> vec2<i32> {
  if (m < 0 || m >= N) { return vec2<i32>(near, 0); }
  let k = base + m * step;
  if (land(k)) { return vec2<i32>(near, 1); }
  return vec2<i32>(k, 0);
}

/* Rusanov flux with MUSCL (monotonized-central) states on the face between a-1 and a: (Fh, Fn, Ft, outflow). */
fn face(xdir: bool, a: i32, b: i32) -> vec4<f32> {
  let H = oh(p.u0); let gp = p.gp;
  let step = select(N, 1, xdir); let base = select(b, b * N, xdir);
  let kL = base + (a - 1) * step; let kR = base + a * step;
  // Every cell the stencil can read is one of LL, L, R, RR (mirrors and edge copies reuse them): all dry, no flux.
  var dry = true;
  for (var m = a - 2; m <= a + 1; m++) { if (m >= 0 && m < N && F[H + base + m * step] != 0.0) { dry = false; } }
  if (dry) { return vec4<f32>(0.0); }
  let wetL = a > 0 && !land(kL); let wetR = a < N && !land(kR);
  if (!wetL && !wetR) { return vec4<f32>(0.0); }
  if ((a > 0 && !wetL) || (a < N && !wetR)) {
    let hw = F[H + select(kR, kL, wetL)];
    return vec4<f32>(0.0, 0.5 * gp * hw * hw, 0.0, 0.0);
  }
  let kc = select(kR, kL, wetL);
  var iLL = vec2<i32>(kc, 0); var iRR = vec2<i32>(kc, 0);
  if (wetL) { iLL = nb(a - 2, kL, base, step); }
  if (wetR) { iRR = nb(a + 1, kR, base, step); }
  let iL = vec2<i32>(select(kc, kL, wetL), 0);
  let iR = vec2<i32>(select(kc, kR, wetR), 0);
  let HLL = F[H + iLL.x]; let HL = F[H + iL.x]; let HR = F[H + iR.x]; let HRR = F[H + iRR.x];
  if (HLL == 0.0 && HL == 0.0 && HR == 0.0 && HRR == 0.0) { return vec4<f32>(0.0); } // dry: every flux is zero
  let w = p.u0;
  let NLL = select(1.0, -1.0, iLL.y == 1) * velo(w, iLL.x, xdir); let NL = velo(w, iL.x, xdir); let NR = velo(w, iR.x, xdir);
  let NRR = select(1.0, -1.0, iRR.y == 1) * velo(w, iRR.x, xdir);
  let TLL = velo(w, iLL.x, !xdir); let TL = velo(w, iL.x, !xdir); let TR = velo(w, iR.x, !xdir); let TRR = velo(w, iRR.x, !xdir);
  let hL = max(0.0, HL + 0.5 * mm(HL - HLL, HR - HL)); let hR = max(0.0, HR - 0.5 * mm(HR - HL, HRR - HR));
  let uL = select(0.0, NL + 0.5 * mm(NL - NLL, NR - NL), hL > DRY);
  let uR = select(0.0, NR - 0.5 * mm(NR - NL, NRR - NR), hR > DRY);
  let tL = TL + 0.5 * mm(TL - TLL, TR - TL); let tR = TR - 0.5 * mm(TR - TL, TRR - TR);
  let s = max(abs(uL) + sqrt(gp * hL), abs(uR) + sqrt(gp * hR));
  let FhL = hL * uL; let FhR = hR * uR;
  var Fh = 0.5 * (FhL + FhR) - 0.5 * s * (hR - hL);
  var Fn = 0.5 * (FhL * uL + 0.5 * gp * hL * hL + FhR * uR + 0.5 * gp * hR * hR) - 0.5 * s * (FhR - FhL);
  var Ft = 0.5 * (FhL * tL + FhR * tR) - 0.5 * s * (hR * tR - hL * tL);
  var out = 0.0;
  if (a == 0 || a == N) {
    if (a == 0 && Fh > 0.0) { Fh = 0.0; Fn = 0.5 * gp * hR * hR; Ft = 0.0; }
    if (a == N && Fh < 0.0) { Fh = 0.0; Fn = 0.5 * gp * hL * hL; Ft = 0.0; }
    out = abs(Fh);
  }
  return vec4<f32>(Fh, Fn, Ft, out);
}

/*
 * Both face families in one dispatch, both stored row-major so neighbouring threads touch neighbouring cells:
 * x-face i of row j at j*(N+1)+i, y-face j (south of row j) of column i at j*N+i.
 */
@compute @workgroup_size(256) fn faces(@builtin(global_invocation_id) id: vec3<u32>) {
  var g = i32(id.x);
  if (g >= 2 * (N + 1) * N) { return; }
  let xdir = g < (N + 1) * N;
  if (!xdir) { g -= (N + 1) * N; }
  // a: index along the face normal, b: the row (x-faces) or column (y-faces)
  let a = select(g / N, g % (N + 1), xdir); let b = select(g % N, g / (N + 1), xdir);
  let f = face(xdir, a, b);
  if (xdir) { F[O_FXH + g] = f.x; F[O_FXN + g] = f.y; F[O_FXT + g] = f.z; }
  else { F[O_FYH + g] = f.x; F[O_FYN + g] = f.y; F[O_FYT + g] = f.z; }
  // outflow at the open edges, booked as left the grid (a0 = dx * dt / 2 per RK stage)
  let side = select(2, 0, xdir);
  if (a == 0) { F[O_EACC + side * N + b] += f.w * p.a0; }
  if (a == N) { F[O_EACC + (side + 1) * N + b] += f.w * p.a0; }
}

/* Flux divergence of cell k: (dh, dqx, dqy) per second. */
fn divergence(k: i32) -> vec3<f32> {
  let i = k % N; let j = k / N;
  let xw = j * (N + 1) + i; let ys = j * N + i; let yn = ys + N;
  return vec3<f32>(
    (F[O_FXH + xw] - F[O_FXH + xw + 1] + F[O_FYH + ys] - F[O_FYH + yn]) / DX,
    (F[O_FXN + xw] - F[O_FXN + xw + 1] + F[O_FYT + ys] - F[O_FYT + yn]) / DX,
    (F[O_FXT + xw] - F[O_FXT + xw + 1] + F[O_FYN + ys] - F[O_FYN + yn]) / DX);
}

@compute @workgroup_size(256) fn stage1(@builtin(global_invocation_id) id: vec3<u32>) {
  let k = i32(id.x);
  if (k >= N * N) { return; }
  let d = divergence(k);
  F[O_H1 + k] = F[O_H + k] + p.dt * d.x;
  F[O_QX1 + k] = F[O_QX + k] + p.dt * d.y;
  F[O_QY1 + k] = F[O_QY + k] + p.dt * d.z;
}

/* SSP-RK2 combine, then drag towards the surface drift at rate C_i/h, point-implicit. */
@compute @workgroup_size(256) fn stage2(@builtin(global_invocation_id) id: vec3<u32>) {
  let k = i32(id.x);
  if (k >= N * N) { return; }
  if (land(k)) { return; }
  let d = divergence(k);
  var hn = 0.5 * (F[O_H + k] + F[O_H1 + k] + p.dt * d.x);
  var ax = 0.5 * (F[O_QX + k] + F[O_QX1 + k] + p.dt * d.y);
  var ay = 0.5 * (F[O_QY + k] + F[O_QY1 + k] + p.dt * d.z);
  if (hn <= DRY) {
    F[O_LACC + k] += min(0.0, hn) * DX * DX; // clamping a negative cell adds volume: booked against the budget
    hn = max(0.0, hn); ax = hn * F[O_TU + k]; ay = hn * F[O_TV + k];
  } else {
    let r = (p.dt * p.ci) / hn;
    ax = (ax + r * hn * F[O_TU + k]) / (1.0 + r);
    ay = (ay + r * hn * F[O_TV + k]) / (1.0 + r);
  }
  F[O_H + k] = hn; F[O_QX + k] = ax; F[O_QY + k] = ay;
}

/*
 * Stranding on the coast: onshore flux into land up to each land cell's capacity. The CPU walks shore cells in
 * order and fills capacity as it goes; here every shore cell asks (A), every land cell scales the asks it received
 * to what it can hold (B), and every shore cell gives what was granted (C). Same totals, no race.
 */
fn dirK(k: i32, d: i32) -> i32 {
  let i = k % N; let j = k / N;
  if (d == 0) { return select(-1, k + 1, i + 1 < N); }
  if (d == 1) { return select(-1, k - 1, i > 0); }
  if (d == 2) { return select(-1, k + N, j + 1 < N); }
  return select(-1, k - N, j > 0);
}

@compute @workgroup_size(256) fn strand_ask(@builtin(global_invocation_id) id: vec3<u32>) {
  let k = i32(id.x);
  if (k >= N * N) { return; }
  for (var d = 0; d < 4; d++) { F[O_REQ + 4 * k + d] = 0.0; }
  var h = F[O_H + k];
  if (M[M_SHORE + k] == 0u || h <= DRY) { return; }
  let u = F[O_QX + k] / h; let v = F[O_QY + k] / h;
  let un = array<f32, 4>(u, -u, v, -v);
  for (var d = 0; d < 4; d++) {
    let kk = dirK(k, d);
    if (kk < 0 || !land(kk) || un[d] <= 0.0 || F[O_STR + kk] >= p.cap) { continue; }
    let dV = min(min(h * un[d] * DX * p.dt, p.cap - F[O_STR + kk]), h * DX * DX);
    F[O_REQ + 4 * k + d] = dV;
    h *= (h * DX * DX - dV) / (h * DX * DX);
  }
}

@compute @workgroup_size(256) fn strand_grant(@builtin(global_invocation_id) id: vec3<u32>) {
  let k = i32(id.x);
  if (k >= N * N) { return; }
  F[O_SCL + k] = 1.0;
  if (!land(k)) { return; }
  // asks towards k come from its west neighbour going east (d 0), east going west (1), south going north (2), north going south (3)
  var inc = 0.0;
  let w = dirK(k, 1); let e = dirK(k, 0); let s = dirK(k, 3); let n = dirK(k, 2);
  if (w >= 0) { inc += F[O_REQ + 4 * w + 0]; }
  if (e >= 0) { inc += F[O_REQ + 4 * e + 1]; }
  if (s >= 0) { inc += F[O_REQ + 4 * s + 2]; }
  if (n >= 0) { inc += F[O_REQ + 4 * n + 3]; }
  if (inc <= 0.0) { return; }
  let sc = min(1.0, max(0.0, p.cap - F[O_STR + k]) / inc);
  F[O_SCL + k] = sc; F[O_STR + k] += inc * sc; F[O_SACC + k] += inc * sc;
}

@compute @workgroup_size(256) fn strand_give(@builtin(global_invocation_id) id: vec3<u32>) {
  let k = i32(id.x);
  if (k >= N * N) { return; }
  if (M[M_SHORE + k] == 0u) { return; }
  for (var d = 0; d < 4; d++) {
    let r = F[O_REQ + 4 * k + d];
    if (r <= 0.0) { continue; }
    let h = F[O_H + k];
    let a = min(r * F[O_SCL + dirK(k, d)], h * DX * DX);
    let keep = select(0.0, (h * DX * DX - a) / (h * DX * DX), h > 0.0);
    F[O_H + k] = h * keep; F[O_QX + k] *= keep; F[O_QY + k] *= keep;
  }
}

/* ---------------------------------------------------------------- film transport (FilmFlux, master's after()) */

fn pad(i: i32, j: i32) -> f32 {
  if (i < 0 || j < 0 || i >= N || j >= N) { return 0.0; }
  return F[O_H + j * N + i];
}

fn filmFlux(LL: f32, L: f32, R: f32, RR: f32, lu: f32, ld: f32, ru: f32, rd: f32, an: f32, at: f32) -> f32 {
  if (LL == 0.0 && L == 0.0 && R == 0.0 && RR == 0.0 && lu == 0.0 && ld == 0.0 && ru == 0.0 && rd == 0.0) { return 0.0; }
  let idx = 1.0 / DX;
  var f = 0.0;
  if (p.ka > 0.0) { f -= p.ka * an * (an * (R - L) * idx + at * ((lu - ld + ru - rd) * 0.25 * idx)); }
  let hm = 0.5 * (L + R);
  if (hm > 0.0) {
    let q = hm * p.iht;
    var D = p.kh;
    if (q < 1.0) { D -= p.lens * (1.0 - q) * select(1.0, q / 0.05, q < 0.05); }
    f -= D * (R - L) * idx;
    if (q < 3.0) { f += p.n4dx3 * select(select((3.0 - q) / 1.5, 1.0, q < 1.5), q / 0.05, q < 0.05) * (RR - 3.0 * R + 3.0 * L - LL); }
  }
  return f;
}

@compute @workgroup_size(256) fn film_faces(@builtin(global_invocation_id) id: vec3<u32>) {
  var g = i32(id.x);
  if (g >= 2 * (N + 1) * N) { return; }
  if (g < (N + 1) * N) {
    let i = g % (N + 1); let j = g / (N + 1);
    if ((i > 0 && land(j * N + i - 1)) || (i < N && land(j * N + i))) { F[O_FFX + g] = 0.0; return; }
    F[O_FFX + g] = filmFlux(pad(i - 2, j), pad(i - 1, j), pad(i, j), pad(i + 1, j), pad(i - 1, j + 1), pad(i - 1, j - 1), pad(i, j + 1), pad(i, j - 1), p.lax, p.lay);
    return;
  }
  g -= (N + 1) * N;
  let i = g % N; let j = g / N;
  if ((j > 0 && land((j - 1) * N + i)) || (j < N && land(j * N + i))) { F[O_FFY + g] = 0.0; return; }
  F[O_FFY + g] = filmFlux(pad(i, j - 2), pad(i, j - 1), pad(i, j), pad(i, j + 1), pad(i + 1, j - 1), pad(i - 1, j - 1), pad(i + 1, j), pad(i - 1, j), p.lay, p.lax);
}

/* Positivity: a cell's outgoing fluxes are scaled so it cannot give more than it holds. */
@compute @workgroup_size(256) fn film_ratio(@builtin(global_invocation_id) id: vec3<u32>) {
  let k = i32(id.x);
  if (k >= N * N) { return; }
  let i = k % N; let j = k / N;
  let e = F[O_FFX + j * (N + 1) + i + 1]; let w = F[O_FFX + j * (N + 1) + i];
  let n = F[O_FFY + (j + 1) * N + i]; let s = F[O_FFY + k];
  let o = max(e, 0.0) + max(-w, 0.0) + max(n, 0.0) + max(-s, 0.0);
  let h = F[O_H + k];
  F[O_RAT + k] = select(1.0, max(0.0, (h * DX) / (o * p.dt)), o * p.dt > h * DX);
}

/* Each face scaled by its donor's ratio (outside the grid there is no oil to give); outflow booked, a0 = dx * dt. */
@compute @workgroup_size(256) fn film_scale(@builtin(global_invocation_id) id: vec3<u32>) {
  var g = i32(id.x);
  if (g >= 2 * (N + 1) * N) { return; }
  if (g < (N + 1) * N) {
    let i = g % (N + 1); let j = g / (N + 1);
    let v = F[O_FFX + g];
    let sc = select(select(0.0, F[O_RAT + j * N + i], i < N), select(0.0, F[O_RAT + j * N + i - 1], i > 0), v > 0.0);
    let nv = v * sc;
    F[O_FFX + g] = nv;
    if (i == 0) { F[O_EACC + j] += max(-nv, 0.0) * p.a0; }
    if (i == N) { F[O_EACC + N + j] += max(nv, 0.0) * p.a0; }
    return;
  }
  g -= (N + 1) * N;
  let i = g % N; let j = g / N;
  let v = F[O_FFY + g];
  let sc = select(select(0.0, F[O_RAT + j * N + i], j < N), select(0.0, F[O_RAT + (j - 1) * N + i], j > 0), v > 0.0);
  let nv = v * sc;
  F[O_FFY + g] = nv;
  if (j == 0) { F[O_EACC + 2 * N + i] += max(-nv, 0.0) * p.a0; }
  if (j == N) { F[O_EACC + 3 * N + i] += max(nv, 0.0) * p.a0; }
}

@compute @workgroup_size(256) fn film_update(@builtin(global_invocation_id) id: vec3<u32>) {
  let k = i32(id.x);
  if (k >= N * N) { return; }
  let i = k % N; let j = k / N;
  let d = F[O_FFX + j * (N + 1) + i + 1] - F[O_FFX + j * (N + 1) + i] + F[O_FFY + (j + 1) * N + i] - F[O_FFY + k];
  if (d != 0.0) { F[O_H + k] = max(0.0, F[O_H + k] - (p.dt / DX) * d); }
}

@compute @workgroup_size(256) fn keep_before(@builtin(global_invocation_id) id: vec3<u32>) {
  let k = i32(id.x);
  if (k >= N * N) { return; }
  F[O_BEF + k] = F[O_H + k];
}

/* Film fluxes move mass, not momentum: velocity is kept, so momentum follows the moved mass. */
@compute @workgroup_size(256) fn film_momentum(@builtin(global_invocation_id) id: vec3<u32>) {
  let k = i32(id.x);
  if (k >= N * N) { return; }
  let b = F[O_BEF + k];
  if (b > DRY) { let r = F[O_H + k] / b; F[O_QX + k] *= r; F[O_QY + k] *= r; }
  else { F[O_QX + k] = 0.0; F[O_QY + k] = 0.0; }
}

/* ---------------------------------------------------------------- weathering (Budget.weather) */

var<workgroup> red: array<f32, 256>;
var<workgroup> red2: array<f32, 256>;
var<workgroup> red3: array<f32, 256>;

/* Fixed-order tree over the workgroup's 256 partials (red and red3 summed, red2 the max). */
fn tree(t: u32) {
  for (var s = 128u; s > 0u; s >>= 1u) {
    workgroupBarrier();
    if (t < s) { red[t] += red[t + s]; red2[t] = max(red2[t], red2[t + s]); red3[t] += red3[t + s]; }
  }
  workgroupBarrier();
}

/* Floating volume, then evaporation to age t (a0 = evaporated fraction there) and dispersion at rate a2 over dt; a1 = released. */
@compute @workgroup_size(256) fn weather(@builtin(local_invocation_index) t: u32) {
  var s = 0.0;
  for (var i = 0; i < N; i++) { s += F[O_H + i32(t) * N + i]; }
  red[t] = s; red2[t] = 0.0; red3[t] = 0.0;
  tree(t);
  if (t != 0u) { return; }
  let floating = red[0] * DX * DX;
  F[O_STATS + S_FACTOR] = 1.0;
  if (!(floating > 0.0)) { return; }
  let ev = min(floating, p.a1 * (p.a0 - F[O_STATS + S_EVAPF]));
  F[O_STATS + S_EVAPF] = p.a0;
  let after = floating - ev;
  let di = after * (1.0 - exp(-p.a2 * p.dt));
  F[O_STATS + S_EVAP] += ev; F[O_STATS + S_DISP] += di;
  F[O_STATS + S_FACTOR] = (after - di) / floating;
}

@compute @workgroup_size(256) fn weather_scale(@builtin(global_invocation_id) id: vec3<u32>) {
  let k = i32(id.x);
  if (k >= N * N) { return; }
  let f = F[O_STATS + S_FACTOR];
  if (f == 1.0) { return; }
  F[O_H + k] *= f; F[O_QX + k] *= f; F[O_QY + k] *= f;
}

/* ---------------------------------------------------------------- booms, speed, drift, sums, fingerprint */

/* A boom laid over oil holds it: that oil counts as stranded. Every changed cell starts empty. */
@compute @workgroup_size(256) fn land_update(@builtin(global_invocation_id) id: vec3<u32>) {
  let k = i32(id.x);
  if (k >= N * N) { return; }
  let nl = M[M_NEWL + k];
  if (nl == M[M_LAND + k]) { return; }
  let h = F[O_H + k];
  if (nl != 0u && h > 0.0) { let v = h * DX * DX; F[O_STR + k] += v; F[O_SACC + k] += v; F[O_HACC + k] += v; }
  F[O_H + k] = 0.0; F[O_QX + k] = 0.0; F[O_QY + k] = 0.0;
  M[M_LAND + k] = nl;
}

/* Surface drift at every 16th cell from cell 8, zero over land: what the page's tracers follow. */
@compute @workgroup_size(256) fn drift(@builtin(global_invocation_id) id: vec3<u32>) {
  let g = i32(id.x);
  if (g >= 256) { return; }
  let k = (8 + 16 * (g / 16)) * N + 8 + 16 * (g % 16);
  let on = !land(k);
  F[O_DRIFT + 2 * g] = select(0.0, F[O_TU + k], on); F[O_DRIFT + 2 * g + 1] = select(0.0, F[O_TV + k], on);
}

/*
 * End of a step, one workgroup: floating volume, fastest oil speed (next steps' CFL limit), the budget terms gathered
 * per cell (then cleared), and the fingerprint: FNV-1a over the raw bits of thickness and momentum, column by column
 * (so neighbouring threads read neighbouring cells), then over the column hashes in order.
 */
@compute @workgroup_size(256) fn finish(@builtin(local_invocation_index) t: u32) {
  let j = i32(t);
  var a = 2166136261u; var s = 0.0; var mx = 0.0; var left = 0.0; var str = 0.0; var held = 0.0;
  for (var r = 0; r < N; r++) {
    let k = r * N + j; let h = F[O_H + k]; let qx = F[O_QX + k]; let qy = F[O_QY + k];
    a = (a ^ bitcast<u32>(h)) * 16777619u;
    a = (a ^ bitcast<u32>(qx)) * 16777619u;
    a = (a ^ bitcast<u32>(qy)) * 16777619u;
    s += h;
    if (h > DRY) { mx = max(mx, length(vec2<f32>(qx, qy)) / h + sqrt(p.gp * h)); }
    left += F[O_LACC + k]; str += F[O_SACC + k]; held += F[O_HACC + k];
    F[O_LACC + k] = 0.0; F[O_SACC + k] = 0.0; F[O_HACC + k] = 0.0;
  }
  for (var e = 0; e < 4; e++) { left += F[O_EACC + e * N + j]; F[O_EACC + e * N + j] = 0.0; }
  M[M_ROWH + j] = a;
  red[t] = s; red2[t] = mx; red3[t] = left;
  tree(t);
  let sSum = red[0]; let sMax = red2[0]; let sLeft = red3[0];
  workgroupBarrier();
  red[t] = str; red2[t] = 0.0; red3[t] = held;
  tree(t);
  storageBarrier();
  if (t != 0u) { return; }
  F[O_STATS + S_FLOAT] = sSum * DX * DX;
  F[O_STATS + S_MAXS] = sMax;
  F[O_STATS + S_LEFT] += sLeft;
  F[O_STATS + S_STRAND] += red[0];
  F[O_STATS + S_HELD] += red3[0];
  var hsh = 2166136261u;
  for (var r = 0; r < N; r++) { hsh = (hsh ^ M[M_ROWH + r]) * 16777619u; }
  M[M_HASH] = hsh;
}
`;
