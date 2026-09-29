// The GPU backend: the same run as Runner (src/runner.ts), with the model's step on WebGPU compute kernels. The CPU
// keeps what is cheap and must stay exact: the clock, the water layer's time, the flow-field refresh schedule, the
// released volume and the rasterising of new oil. Everything per cell is on the GPU.
//
// Each simulated minute is one submission ending in a small readback (budget, fastest oil speed, fingerprint, drift).
// Minute c sizes its substeps from the fastest speed at the end of minute c − 2, so two minutes are in flight at once
// and the readback latency is hidden. The rule depends only on step numbers, never on timing, so a replay makes the
// same choices and lands on the same bits.
import { MASTER_DEFAULTS, entrainmentAt } from '../engine/film/master';
import { SurfaceFlow } from '../engine/film/flow';
import { subSeed } from '../engine/rng/prng';
import { Budget, evaporatedFraction, type Frame } from '../engine/film/grid';
import { rasterise, type Polygon, type SlickProfile } from '../engine/film/shapes';
import { slickRows, filmWarnings } from '../engine/film/thinfilm';
import { DEFAULT_WEATHERING } from '../engine/particles/weathering';
import { boomCells, circle } from '../runner';
import { CHUNK, DX, N, landMask, type Edit, type Scenario, type Scene } from '../world';
import type { Budget as BudgetMsg } from '../protocol';
import { FL, ML, MAX_MODES, MAX_PTS, MAX_STROKES, P_BYTES, P_FIELDS, P_UINTS, S, WGSL, WN } from './kernels';

const DEPTH = 12;
const G = 9.81, RHO_WATER = 1025, RHO_AIR = 1.2, CD_AIR = 1.3e-3;
const OMEGA_M2 = (2 * Math.PI) / (12.42 * 3600);
const WATER_DT = (0.45 * 100) / Math.sqrt(2 * G * DEPTH);
const FRAME: Frame = { nx: N, ny: N, dx: DX, lat0: 25, lon0: -60, ox: 0, oy: 0 };
/** Headroom on a fastest speed that is two minutes old. */
const CFL_MARGIN = 1.5;
/** Uniform records per submission, and submissions that can be in flight (each gets its own slice and readback). */
const SLOTS = 1024;
const RING = 4;
const KERNELS = ['water', 'drag', 'anomaly', 'targets', 'add_oil', 'faces', 'stage1', 'stage2', 'strand_ask', 'strand_grant', 'strand_give',
  'film_faces', 'film_ratio', 'film_scale', 'film_update', 'keep_before', 'film_momentum', 'weather', 'weather_scale', 'land_update', 'drift', 'finish'] as const;
type Kernel = (typeof KERNELS)[number];
type Rec = Partial<Record<(typeof P_FIELDS)[number] | (typeof P_UINTS)[number], number>>;

const N2 = N * N, NF = (N + 1) * N;
/** Readback block: 64 stats floats, the fingerprint, then the 16 × 16 drift. */
const READ_BYTES = (80 + 512) * 4;

let devicePromise: Promise<GPUDevice | null> | undefined;
/** One device per worker, or null when this browser has no WebGPU. */
export function gpuDevice(): Promise<GPUDevice | null> {
  devicePromise ??= (async () => {
    if (typeof navigator === 'undefined' || !navigator.gpu) return null;
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) return null;
    const limit = Math.min(adapter.limits.maxStorageBufferBindingSize, 256 * 2 ** 20);
    const wg = Math.min(1024, adapter.limits.maxComputeInvocationsPerWorkgroup, adapter.limits.maxComputeWorkgroupSizeX);
    return adapter.requestDevice({
      requiredLimits: { maxStorageBufferBindingSize: limit, maxBufferSize: Math.min(adapter.limits.maxBufferSize, limit), maxComputeInvocationsPerWorkgroup: wg, maxComputeWorkgroupSizeX: wg },
    });
  })().catch(() => null);
  return devicePromise;
}

interface Pipes { layout: GPUBindGroupLayout; pipes: Record<Kernel, GPUComputePipeline> }
const pipeCache = new WeakMap<GPUDevice, Pipes>();
function pipelines(device: GPUDevice): Pipes {
  let c = pipeCache.get(device);
  if (c) return c;
  const layout = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform', hasDynamicOffset: true } },
    ],
  });
  const module = device.createShaderModule({ code: WGSL });
  void module.getCompilationInfo().then((info) => { for (const m of info.messages) if (m.type === 'error') console.error(`WGSL ${m.lineNum}:${m.linePos} ${m.message}`); });
  const pl = device.createPipelineLayout({ bindGroupLayouts: [layout] });
  // the water layer runs in one workgroup: as wide as the device allows (a power of two, at least 256)
  const wg = 2 ** Math.floor(Math.log2(Math.min(1024, device.limits.maxComputeInvocationsPerWorkgroup, device.limits.maxComputeWorkgroupSizeX)));
  const pipes = Object.fromEntries(KERNELS.map((k) => [k, device.createComputePipeline({
    layout: pl, compute: { module, entryPoint: k, constants: k === 'water' ? { WATER_WG: wg } : {} },
  })])) as Record<Kernel, GPUComputePipeline>;
  c = { layout, pipes };
  pipeCache.set(device, c);
  return c;
}

/** One submission's dispatches, with their uniform records packed into one upload to this submission's slice. */
class Encoder {
  private data = new ArrayBuffer(SLOTS * P_BYTES);
  private f = new Float32Array(this.data);
  private u = new Uint32Array(this.data);
  private slots = 0;
  private enc: GPUCommandEncoder;
  private pass: GPUComputePassEncoder;

  constructor(private sim: GpuSim, private slice: number) {
    this.enc = sim.device.createCommandEncoder();
    this.pass = this.enc.beginComputePass();
  }

  /** A uniform record: the run's current parameters, with `over` on top. Returns its dynamic offset. */
  rec(over: Rec = {}): number {
    if (this.slots >= SLOTS) throw new Error('GPU step needs more uniform slots');
    const r = { ...this.sim.base(), ...over }, at = this.slots++, o = at * (P_BYTES / 4);
    P_FIELDS.forEach((k, i) => { this.f[o + i] = r[k] ?? 0; });
    P_UINTS.forEach((k, i) => { this.u[o + P_FIELDS.length + i] = r[k] ?? 0; });
    return (this.slice * SLOTS + at) * P_BYTES;
  }

  run(k: Kernel, items: number, rec: number) {
    this.pass.setPipeline(this.sim.pipes.pipes[k]);
    this.pass.setBindGroup(0, this.sim.bind, [rec]);
    this.pass.dispatchWorkgroups(Math.ceil(items / 256));
  }

  /** One workgroup: the kernels that loop or reduce inside a single group. */
  one(k: Kernel, rec: number) { this.run(k, 1, rec); }

  /** Ends the pass and copies the end-of-step block into `read`. */
  finish(read: GPUBuffer): GPUCommandBuffer {
    this.pass.end();
    const s = this.sim;
    this.enc.copyBufferToBuffer(s.F, FL.at.STATS * 4, read, 0, 64 * 4);
    this.enc.copyBufferToBuffer(s.M, ML.at.HASH * 4, read, 64 * 4, 16);
    this.enc.copyBufferToBuffer(s.F, FL.at.DRIFT * 4, read, 80 * 4, 512 * 4);
    if (this.slots) s.device.queue.writeBuffer(s.uni, this.slice * SLOTS * P_BYTES, this.data, 0, this.slots * P_BYTES);
    return this.enc.finish();
  }
}

interface Pending { chunk: number; read: GPUBuffer; done: Promise<void> }

export class GpuSim {
  readonly pipes: Pipes;
  readonly F: GPUBuffer;
  readonly M: GPUBuffer;
  readonly uni: GPUBuffer;
  readonly bind: GPUBindGroup;
  private reads: GPUBuffer[];
  private editRead: GPUBuffer;
  private hRead: GPUBuffer;

  chunk = 0;
  t = 0;
  scene: Scene = { booms: [], strokes: [], sources: [] };
  drift = new Float32Array(512);
  readonly params: Record<string, number>;
  private waterT = 0;
  private anomT = -Infinity;
  private flowVersion = 0;
  private anomVersion = -1;
  private released = 0;
  private stats = new Float32Array(64);
  private hash = '';
  /** Fastest oil speed at the end of each step (by the step's index), and after edits at each step boundary. */
  private endSpeed = new Map<number, number>();
  private editSpeed = new Map<number, number>();
  private pending: Pending[] = [];
  private fresh: [number, string][] = [];
  private readonly base0: Uint8Array;
  private land: Uint8Array;
  private nStrokes = 0;
  private nModes = 0;

  private constructor(readonly device: GPUDevice, readonly sc: Scenario) {
    this.pipes = pipelines(device);
    this.params = { ...MASTER_DEFAULTS, ...sc.params } as Record<string, number>;
    const S_ = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
    const R_ = GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST;
    this.F = device.createBuffer({ size: FL.size * 4, usage: S_ });
    this.M = device.createBuffer({ size: ML.size * 4, usage: S_ });
    this.uni = device.createBuffer({ size: (RING + 1) * SLOTS * P_BYTES, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.reads = Array.from({ length: RING }, () => device.createBuffer({ size: READ_BYTES, usage: R_ }));
    this.editRead = device.createBuffer({ size: READ_BYTES, usage: R_ });
    this.hRead = device.createBuffer({ size: N2 * 4, usage: R_ });
    this.bind = device.createBindGroup({
      layout: this.pipes.layout,
      entries: [{ binding: 0, resource: { buffer: this.F } }, { binding: 1, resource: { buffer: this.M } }, { binding: 2, resource: { buffer: this.uni, size: P_BYTES } }],
    });
    this.base0 = landMask(sc);
    this.land = this.base0.slice();
  }

  static async create(device: GPUDevice, sc: Scenario): Promise<GpuSim> {
    const s = new GpuSim(device, sc);
    await s.init();
    return s;
  }

  private async init() {
    const q = this.device.queue, b = this.base0;
    // water: a cell is land when most of its four oil cells are; a 12 m shelf everywhere
    const wl = new Uint32Array(WN * WN), wh = new Float32Array(WN * WN).fill(DEPTH);
    for (let j = 0; j < WN; j++)
      for (let i = 0; i < WN; i++) { const k = 2 * j * N + 2 * i; wl[j * WN + i] = b[k] + b[k + 1] + b[k + N] + b[k + N + 1] >= 2 ? 1 : 0; }
    // shore: water cells of the scenario's coast with a land neighbour (booms hold oil, they do not strand it)
    const shore = new Uint32Array(N2);
    for (let j = 0; j < N; j++)
      for (let i = 0; i < N; i++) {
        const k = j * N + i;
        if (!b[k] && ((i + 1 < N && b[k + 1]) || (i > 0 && b[k - 1]) || (j + 1 < N && b[k + N]) || (j > 0 && b[k - N]))) shore[k] = 1;
      }
    q.writeBuffer(this.M, ML.at.LAND * 4, Uint32Array.from(b));
    q.writeBuffer(this.M, ML.at.SHORE * 4, shore);
    q.writeBuffer(this.M, ML.at.WLAND * 4, wl);
    q.writeBuffer(this.F, FL.at.WH * 4, wh);
    this.uploadModes();
    await this.apply({ k: 'scene', scene: this.sc.scene });
    for (const spill of this.sc.spills) await this.apply({ k: 'spill', spill });
  }

  /** The uniform record every kernel starts from: parameters as the CPU engine derives them. */
  base(): Rec {
    const p = this.params, th = (p.windDirDeg * Math.PI) / 180, W = p.windSpeed;
    const wx = W * Math.sin(th), wy = W * Math.cos(th), n4 = (p.lensD * (p.lensWavelengthM / (2 * Math.PI)) ** 2) / 2;
    return {
      gp: (G * (RHO_WATER - p.oilDensity)) / RHO_WATER, ci: p.spreadC, cap: p.shoreCapacity * DX, rho: p.oilDensity / RHO_WATER, t: this.t,
      fcor: 2 * 7.2921e-5 * Math.sin((FRAME.lat0 * Math.PI) / 180), fric: p.friction,
      tx: (RHO_AIR * CD_AIR * W * wx) / RHO_WATER, ty: (RHO_AIR * CD_AIR * W * wy) / RHO_WATER, wx, wy,
      mu: p.driftU + p.windage * wx, mv: p.driftV + p.windage * wy, lax: Math.sin(th), lay: Math.cos(th),
      lag: p.sheenLagFrac * W, lagRef: p.sheenLagRefUm * 1e-6, windage: p.windage, wspeed: W,
      kh: p.Kh, ka: p.shearDispFrac * W, lens: p.lensD, n4dx3: n4 / DX ** 3, iht: 1 / (p.hTerminalUm * 1e-6),
      rowSp: p.rowSpacingM, rowSpd: p.rowSpeedFrac, rowJet: p.rowJetFrac, nModes: this.nModes, nStrokes: this.nStrokes,
    };
  }

  private uploadModes() {
    const flat = new SurfaceFlow(this.params as never, subSeed(this.params.seed, 1)).flat;
    this.nModes = Math.min(MAX_MODES, flat.length / 6);
    this.device.queue.writeBuffer(this.F, FL.at.MODES * 4, Float32Array.from(flat.subarray(0, this.nModes * 6)));
    this.flowVersion++;
  }

  private uploadStrokes() {
    const stk = new Float32Array(12 * MAX_STROKES), pts: number[] = [];
    let n = 0;
    for (const s of this.scene.strokes.slice(0, MAX_STROKES)) {
      if (pts.length / 2 + s.path.length > MAX_PTS) break;
      const xs = s.path.map((q) => q[0]), ys = s.path.map((q) => q[1]);
      stk.set([s.kind === 'wind' ? 0 : 1, s.width, s.speed, Math.min(...xs), Math.max(...xs), Math.min(...ys), Math.max(...ys), pts.length / 2, s.path.length], 12 * n++);
      for (const [x, y] of s.path) pts.push(x, y);
    }
    this.nStrokes = n;
    this.device.queue.writeBuffer(this.F, FL.at.STK * 4, stk);
    if (pts.length) this.device.queue.writeBuffer(this.F, FL.at.PTS * 4, Float32Array.from(pts));
    this.flowVersion++;
  }

  /** targets() of the CPU engine: refresh the flow field when it is 10 minutes old or the forcing changed. */
  private targets(e: Encoder) {
    if (Math.abs(this.t - this.anomT) >= 600 || this.anomVersion !== this.flowVersion) {
      e.run('anomaly', N2, e.rec({ t: this.t }));
      this.anomT = this.t;
      this.anomVersion = this.flowVersion;
    }
    e.run('targets', N2, e.rec());
  }

  /** New oil in a polygon, rasterised on the current coast and booms, added to `add` (m of thickness per cell). */
  private rasterOil(add: Float32Array, poly: Polygon, volumeM3: number, profile: SlickProfile) {
    const { share, lost } = rasterise(FRAME, poly, profile, this.land);
    const placed = volumeM3 * (1 - lost);
    for (let k = 0; k < N2; k++) if (share[k] > 0) add[k] += (share[k] * placed) / (DX * DX);
    this.released += placed;
  }

  /** The new oil arrives at the local surface drift (CoastalCore.addOil). */
  private addOil(e: Encoder, add: Float32Array) {
    this.device.queue.writeBuffer(this.F, FL.at.ADD * 4, add);
    this.targets(e);
    e.run('add_oil', N2, e.rec());
  }

  /** An edit lands between steps. It waits for everything in flight, so it sees (and books) the state it changes. */
  async apply(ed: Edit) {
    await this.drain();
    const e = new Encoder(this, RING);
    if (ed.k === 'param') {
      this.params[ed.key] = ed.value;
      if (ed.key === 'eddySpeed' || ed.key === 'eddyScaleM') this.uploadModes();
      else this.flowVersion++; // a new wind shows in the flow field now, not at its next refresh
    } else if (ed.k === 'spill') {
      const add = new Float32Array(N2);
      this.rasterOil(add, circle(ed.spill.at, ed.spill.radius), ed.spill.volume, 'dome');
      this.addOil(e, add);
    } else {
      this.scene = structuredClone(ed.scene);
      this.uploadStrokes();
      const booms = boomCells(this.scene.booms), next = new Uint32Array(N2);
      for (let k = 0; k < N2; k++) { next[k] = this.base0[k] | booms[k]; this.land[k] = next[k]; }
      this.device.queue.writeBuffer(this.M, ML.at.NEWL * 4, next);
      e.run('land_update', N2, e.rec());
    }
    e.one('finish', e.rec());
    this.device.queue.submit([e.finish(this.editRead)]);
    await this.editRead.mapAsync(GPUMapMode.READ);
    this.take(this.editRead);
    this.editSpeed.set(this.chunk, Math.max(this.editSpeed.get(this.chunk) ?? 0, this.stats[S.MAXS]));
  }

  /** One simulated minute: leaks, then the engine's advance(60 s) with the master's film and weathering. */
  async step() {
    const c = this.chunk;
    // step c − 2 must be read before its speed can size this one (anything older is read by then too)
    while (this.pending.length && this.pending[0].chunk <= c - 2) await this.settle();
    const e = new Encoder(this, c % RING), p = this.params;
    if (this.scene.sources.length) {
      const add = new Float32Array(N2);
      for (const s of this.scene.sources) this.rasterOil(add, circle(s.at, 90), (s.rate * CHUNK) / 3600, 'uniform');
      this.addOil(e, add);
    }
    const speed = Math.max(this.endSpeed.get(c - 2) ?? 0, this.editSpeed.get(c) ?? 0, this.editSpeed.get(c - 1) ?? 0, 1e-6) * CFL_MARGIN;
    const nOil = Math.max(1, Math.ceil(CHUNK / Math.min(CHUNK, (0.3 * DX) / speed))), sub = CHUNK / nOil;
    const nW = Math.ceil(sub / WATER_DT), wdt = sub / nW;
    const n4 = (p.lensD * (p.lensWavelengthM / (2 * Math.PI)) ** 2) / 2, D = p.Kh + p.shearDispFrac * p.windSpeed + p.lensD;
    const filmLimit = Math.min(D > 0 ? (0.2 * DX * DX) / D : Infinity, n4 > 0 ? (0.8 * DX ** 4) / (64 * n4) : Infinity);
    const nFilm = Math.max(1, Math.ceil(sub / filmLimit)), fdt = sub / nFilm;
    const disp = p.permanentShare * entrainmentAt(p.windSpeed);
    const tide = p.uTide * Math.hypot(OMEGA_M2, p.friction / (DEPTH * 0.7));

    for (let s = 0; s < nOil; s++) {
      this.targets(e);
      e.run('drag', WN * WN, e.rec());
      e.one('water', e.rec({ dt: wdt, u1: nW, a0: this.waterT, a2: tide }));
      this.waterT += sub;
      const r0 = e.rec({ dt: sub, u0: 0, a0: DX * sub * 0.5 }), r1 = e.rec({ dt: sub, u0: 1, a0: DX * sub * 0.5 });
      e.run('faces', 2 * NF, r0); e.run('stage1', N2, r0);
      e.run('faces', 2 * NF, r1); e.run('stage2', N2, r0);
      e.run('strand_ask', N2, r0); e.run('strand_grant', N2, r0); e.run('strand_give', N2, r0);
      // film rupture and diffusion as mass fluxes after the momentum step (MasterModel.step's `after`)
      const rf = e.rec({ dt: fdt, a0: DX * fdt });
      e.run('keep_before', N2, rf);
      for (let f = 0; f < nFilm; f++) {
        e.run('film_faces', 2 * NF, rf); e.run('film_ratio', N2, rf); e.run('film_scale', 2 * NF, rf); e.run('film_update', N2, rf);
      }
      e.run('film_momentum', N2, rf);
      this.t += sub;
      const f = evaporatedFraction(this.t, p.sst, { ...DEFAULT_WEATHERING, evapMax: p.evapMax });
      const rw = e.rec({ dt: sub, a0: f, a1: this.released, a2: disp });
      e.one('weather', rw); e.run('weather_scale', N2, rw);
    }
    e.run('drift', 256, e.rec());
    e.one('finish', e.rec());
    const read = this.reads[c % RING];
    this.device.queue.submit([e.finish(read)]);
    this.pending.push({ chunk: c, read, done: read.mapAsync(GPUMapMode.READ) });
    this.chunk++;
  }

  /** Take the oldest step in flight: its budget, speed, drift and fingerprint. */
  private async settle() {
    const q = this.pending.shift()!;
    await q.done;
    this.take(q.read);
    this.endSpeed.set(q.chunk, this.stats[S.MAXS]);
    this.endSpeed.delete(q.chunk - 4);
    this.editSpeed.delete(q.chunk - 4);
    this.fresh.push([q.chunk + 1, this.hash]);
  }

  private take(read: GPUBuffer) {
    const buf = read.getMappedRange();
    this.stats = new Float32Array(buf.slice(0, 64 * 4));
    this.hash = (new Uint32Array(buf.slice(64 * 4, 64 * 4 + 4))[0] >>> 0).toString(16).padStart(8, '0');
    this.drift = new Float32Array(buf.slice(80 * 4, READ_BYTES));
    read.unmap();
  }

  /** Wait for every step in flight. */
  async drain() { while (this.pending.length) await this.settle(); }

  /** Fingerprints of the steps finished since the last call, as [step, hash]. */
  takeHashes(): [number, string][] { const f = this.fresh; this.fresh = []; return f; }

  /** Fingerprint of the latest finished state (after drain(), the current one). */
  fingerprint() { return this.hash; }

  budget(): BudgetMsg {
    const s = this.stats;
    return { released: this.released, floating: s[S.FLOAT], evaporated: s[S.EVAP], dispersed: s[S.DISP], stranded: s[S.STRAND], left: s[S.LEFT], held: s[S.HELD] };
  }

  /** Thickness after the last step submitted. */
  async thickness(): Promise<Float32Array> {
    const enc = this.device.createCommandEncoder();
    enc.copyBufferToBuffer(this.F, FL.at.H * 4, this.hRead, 0, N2 * 4);
    this.device.queue.submit([enc.finish()]);
    await this.hRead.mapAsync(GPUMapMode.READ);
    const h = new Float32Array(this.hRead.getMappedRange().slice(0));
    this.hRead.unmap();
    return h;
  }

  /** The engine readout, from a thickness read back. */
  statsRows(h: Float32Array): [string, string][] {
    const m = { frame: FRAME, thickness: () => h, rhoOil: this.params.oilDensity, sheenM: this.params.sheenUm * 1e-6, t: this.t };
    return [...slickRows(m as never), ['engine', 'WebGPU compute, 32-bit floats'], ...this.budgetObj().rows(this.budget().floating).filter(([k]) => k !== 'budget error')];
  }

  warnings(): string[] {
    return filmWarnings({ frame: FRAME } as never, this.budgetObj(), this.budget().floating, this.params.lensWavelengthM).filter((w) => !w.startsWith('Mass budget error'));
  }

  private budgetObj() {
    const b = new Budget(this.released), s = this.budget();
    b.evaporated = s.evaporated; b.dispersed = s.dispersed; b.stranded = s.stranded; b.left = s.left;
    return b;
  }

  destroy() { for (const b of [this.F, this.M, this.uni, this.hRead, this.editRead, ...this.reads]) b.destroy(); }
}
